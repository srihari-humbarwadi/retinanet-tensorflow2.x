import tensorflow as tf

from retinanet.dataloader.utils import (convert_to_xywh, normalize_image,
                                        random_flip_horizontal)


class PreprocessingPipelineV2:
    def __init__(self, input_shape, params):

        if not input_shape[0] == input_shape[1]:
            raise AssertionError('Non square inputs are not supported, got {}'
                                 .format(input_shape))

        self.target_size = input_shape[0]
        self.preprocessing_params = params.preprocessing
        self.augmentation_params = params.augmentations

    def _resize_with_pad(self, image):
        target_size = self.target_size
        image_size = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
        scale = tf.reduce_min(target_size / image_size)

        scaled_shape = tf.round(scale * image_size)
        resize_scale = scaled_shape / image_size
        scaled_shape = tf.cast(scaled_shape, dtype=tf.int32)
        image = tf.image.resize(image, size=scaled_shape)
        image = tf.image.pad_to_bounding_box(image, 0, 0, target_size,
                                             target_size)
        return image, image_size, resize_scale

    def _rescale_labels(self,
                        boxes,
                        class_ids,
                        image_size,
                        resize_scale,
                        offset,
                        target_size):
        boxes = tf.stack(
            [
                boxes[:, 0] * image_size[1] * resize_scale[1] - offset[1],
                boxes[:, 1] * image_size[0] * resize_scale[0] - offset[0],
                boxes[:, 2] * image_size[1] * resize_scale[1] - offset[1],
                boxes[:, 3] * image_size[0] * resize_scale[0] - offset[0],
            ],
            axis=-1,
        )
        boxes = tf.clip_by_value(
            boxes, 0.0,
            [target_size[1], target_size[0], target_size[1], target_size[0]])

        boxes = convert_to_xywh(boxes)
        idx = tf.where(
            tf.logical_and(tf.greater(boxes[:, 2], 0.), tf.greater(boxes[:, 3],
                                                                   0.)))[:, 0]
        boxes = tf.gather(boxes, idx)
        class_ids = tf.gather(class_ids, idx)
        return boxes, class_ids

    def _random_rescale_image(self, image):
        image_size = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
        scale = tf.random.uniform([],
                                  self.augmentation_params.scale_jitter.min_scale,
                                  self.augmentation_params.scale_jitter.max_scale)

        scaled_shape = tf.cast(tf.round(scale * image_size), dtype=tf.int32)
        image = tf.image.resize(image, size=scaled_shape)
        return image

    def _random_crop_image_and_labels(self, image, boxes, class_ids):
        image_size = tf.shape(image)
        min_object_covered = tf.random.uniform((), minval=0.1, maxval=0.9)
        boxes_transposed = tf.stack(
            [boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=-1)

        offset, size, _ = tf.image.sample_distorted_bounding_box(
            image_size=image_size,
            bounding_boxes=tf.expand_dims(boxes_transposed, axis=0),
            min_object_covered=min_object_covered,
            aspect_ratio_range=[0.8, 1.25],
            area_range=[0.1, 1.0],
            max_attempts=100,
            name='random_crop_image_and_labels')

        image = tf.slice(image, offset, size)
        image, _, resize_scale = self._resize_with_pad(image)

        offset = tf.cast(offset, dtype=tf.float32)
        size = tf.cast(size, dtype=tf.float32)
        image_size = tf.cast(image_size, dtype=tf.float32)

        boxes, class_ids = self._rescale_labels(
            boxes=boxes,
            class_ids=class_ids,
            image_size=image_size[:2],
            resize_scale=[1.0, 1.0],
            offset=offset[:2],
            target_size=size[:2] - 1.0)

        boxes = boxes * tf.stack([resize_scale[1], resize_scale[0],
                                  resize_scale[1], resize_scale[0]], axis=-1)
        return image, boxes, class_ids

    def __call__(self, sample):
        image = normalize_image(
            sample["image"],
            offset=self.preprocessing_params.offset,
            scale=self.preprocessing_params.scale)

        boxes = sample["objects"]["bbox"]
        class_ids = tf.cast(sample["objects"]["label"], dtype=tf.int32)

        if self.augmentation_params.use_augmentation \
                and self.augmentation_params.horizontal_flip:
            image, boxes = random_flip_horizontal(image, boxes)

        if tf.random.uniform([]) < 0.5 or \
                not self.augmentation_params.use_augmentation:
            image, image_shape, resize_scale = self._resize_with_pad(image)
            boxes, class_ids = self._rescale_labels(
                boxes=boxes,
                class_ids=class_ids,
                image_size=image_shape,
                resize_scale=resize_scale,
                offset=[0.0, 0.0],
                target_size=[self.target_size, self.target_size])
        else:
            if tf.random.uniform([]) < 0.5 \
                    and self.augmentation_params.use_augmentation:
                image = self._random_rescale_image(image)

            image, boxes, class_ids = self._random_crop_image_and_labels(
                image,
                boxes,
                class_ids)

        image.set_shape([self.target_size, self.target_size, 3])
        return image, boxes, class_ids

    def preprocess_val_sample(self, sample):
        image = normalize_image(
            sample["image"],
            offset=self.preprocessing_params.offset,
            scale=self.preprocessing_params.scale)

        image, _, resize_scale = self._resize_with_pad(image)
        return {
            'image': image,
            'image_id': sample['image_id'],
            'resize_scale': resize_scale
        }
