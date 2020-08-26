import tensorflow as tf

from retinanet.dataloader.utils import (convert_to_xywh, normalize_image,
                                        random_flip_horizontal)


class PreprocessingPipeline:
    def __init__(self, input_shape, params):
        self.input_shape = input_shape
        self.preprocessing_params = params.preprocessing
        self.augmentation_params = params.augmentations

    def _prepare_image(self, image, jitter=[None, None]):
        target_shape = self.input_shape
        image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
        scaled_shape = target_shape

        if self.augmentation_params.use_augmentation:
            jitter = [
                self.augmentation_params.scale_jitter.min_scale,
                self.augmentation_params.scale_jitter.max_scale
            ]

        if jitter[0]:
            random_scale = tf.random.uniform([], jitter[0], jitter[1])
            scaled_shape = random_scale * target_shape

        scale = tf.minimum(scaled_shape[0] / image_shape[0],
                           scaled_shape[1] / image_shape[1])

        scaled_shape = tf.round(image_shape * scale)
        image_scale = scaled_shape / image_shape

        offset = tf.zeros([2], tf.int32)
        if jitter[0]:
            max_offset = scaled_shape - target_shape
            max_offset = tf.where(tf.less(max_offset, 0.), 0., max_offset)
            offset = max_offset * tf.random.uniform([2], 0, 1)
            offset = tf.cast(offset, tf.int32)

        scaled_shape = tf.cast(scaled_shape, dtype=tf.int32)
        resized_image = tf.image.resize(image, size=scaled_shape)

        if jitter[0]:
            resized_image = resized_image[offset[0]:offset[0] +
                                          target_shape[0],
                                          offset[1]:offset[1] +
                                          target_shape[1], :]

        resized_image = tf.image.pad_to_bounding_box(resized_image, 0, 0,
                                                     target_shape[0],
                                                     target_shape[1])
        return resized_image, image_scale, offset, image_shape

    def _prepare_labels(self, boxes, class_ids):
        target_shape = tf.cast(self.input_shape, dtype=tf.float32)
        boxes = tf.clip_by_value(
            boxes, 0.,
            tf.tile(tf.expand_dims(target_shape, axis=0), multiples=[1, 2]))
        boxes = convert_to_xywh(boxes)
        idx = tf.where(
            tf.logical_and(tf.greater(boxes[:, 2], 0.),
                           tf.greater(boxes[:, 3], 0.)))
        idx = idx[:, 0]
        return tf.gather(boxes, idx), tf.gather(class_ids, idx)

    def __call__(self, sample):
        image = normalize_image(
            sample["image"],
            offset=self.preprocessing_params.offset,
            scale=self.preprocessing_params.scale)
        bbox = sample["objects"]["bbox"]
        class_ids = tf.cast(sample["objects"]["label"], dtype=tf.int32)

        if self.augmentation_params.use_augmentation \
                and self.augmentation_params.horizontal_flip:
            image, bbox = random_flip_horizontal(image, bbox)

        image, scale, offset, image_shape = self._prepare_image(image)
        offset = tf.cast(offset, dtype=tf.float32)
        bbox = tf.stack(
            [
                bbox[:, 0] * image_shape[1] * scale[1] - offset[1],
                bbox[:, 1] * image_shape[0] * scale[0] - offset[0],
                bbox[:, 2] * image_shape[1] * scale[1] - offset[1],
                bbox[:, 3] * image_shape[0] * scale[0] - offset[0],
            ],
            axis=-1,
        )
        bbox, class_ids = self._prepare_labels(bbox, class_ids)
        return image, bbox, class_ids, scale

    def resize_val_image(self, image):
        target_shape = self.input_shape
        image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
        scaled_shape = tf.round(image_shape * tf.minimum(
            target_shape[0] / image_shape[0], target_shape[1] / image_shape[1]))

        image_scale = scaled_shape / image_shape
        scaled_shape = tf.cast(scaled_shape, dtype=tf.int32)

        resized_image = tf.image.resize(image, size=scaled_shape)
        resized_image = tf.image.pad_to_bounding_box(resized_image, 0, 0,
                                                     target_shape[0],
                                                     target_shape[1])
        return resized_image, image_scale

    def preprocess_val_sample(self, sample, return_labels=False):
        image = normalize_image(
            sample["image"],
            offset=self.preprocessing_params.offset,
            scale=self.preprocessing_params.scale)

        image, scale = self.resize_val_image(image)
        return {
            'image': image,
            'image_id': sample['image_id'],
            'resize_scale': scale
        }
