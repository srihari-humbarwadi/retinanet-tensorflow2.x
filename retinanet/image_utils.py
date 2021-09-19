import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from absl import logging


def read_image(path):
    image_bytes = tf.io.read_file(path)
    image = tf.image.decode_image(image_bytes)
    image.set_shape([None, None, None])

    if image.get_shape().as_list()[-1] == 1:
        image = tf.image.grayscale_to_rgb(image)

    return tf.cast(image, dtype=tf.float32)


def read_image_cv2(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def resize_with_pad_cv2(image, target_shape):
    target_shape = np.array(target_shape, dtype=np.float32)
    image_shape = np.array(image.shape[:2], dtype=np.float32)
    scaled_shape = np.round(
        image_shape * np.minimum(
            target_shape[0] / image_shape[0],
            target_shape[1] / image_shape[1]))

    scaled_shape = np.int32(scaled_shape)
    scaled_h, scaled_w = scaled_shape

    padded_image = np.zeros(
        shape=(int(target_shape[0]), int(target_shape[1]), 3),
        dtype=image.dtype)

    resized_image = cv2.resize(image, (scaled_w, scaled_h))
    padded_image[:scaled_h, :scaled_w, :] = resized_image
    return padded_image


def normalize_image_cv2(image, mean, stddev, pixel_scale):
    mean = np.reshape(mean, [1, 1, 3])
    stddev = np.reshape(stddev, [1, 1, 3])

    image = image / pixel_scale
    image = (image - mean) / stddev
    return image


def prepare_image_cv2(image, target_shape, mean, stddev, pixel_scale):
    input_image = image.copy()
    input_image = normalize_image_cv2(input_image, mean, stddev, pixel_scale)
    input_image = resize_with_pad_cv2(input_image, target_shape)
    input_image = np.expand_dims(input_image, axis=0)
    input_image = np.float32(input_image)

    image_height, image_width, _ = image.shape
    scale = np.maximum(image_height, image_width) / np.array(target_shape)
    scale = np.expand_dims(scale, axis=0)
    scale = np.tile(scale, [1, 2])

    return input_image, scale


def imshow(image, figsize=(16, 9), title=None):
    plt.figure(figsize=figsize)
    plt.axis('off')
    plt.imshow(np.uint8(image))

    if title:
        plt.title(title)


def visualize_detections(image,
                         boxes,
                         classes,
                         scores,
                         figsize=(12, 12),
                         linewidth=1,
                         color=[0, 0, 1],
                         title=None,
                         score_threshold=0.25,
                         show_labels=True,
                         save=False,
                         filename=None):
    """Visualize Detections"""
    image = np.array(image, dtype=np.uint8)
    plt.figure(figsize=figsize)

    if title:
        plt.title(title)

    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()
    for box, _cls, score in zip(boxes, classes, scores):

        if score < score_threshold:
            continue

        text = "{}: {:.2f}".format(_cls, score)
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        patch = plt.Rectangle([x1, y1],
                              w,
                              h,
                              fill=False,
                              edgecolor=color,
                              linewidth=linewidth)
        ax.add_patch(patch)

        if show_labels:
            ax.text(
                x1,
                y1,
                text,
                bbox={
                    "facecolor": color,
                    "alpha": 0.4
                },
                clip_box=ax.clipbox,
                clip_on=True,
            )

    if save:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()


def visualize_detections_cv2(image, boxes, classes, scores, score_threshold,
                             save=False, filename='output.png'):
    image = np.uint8(image)
    boxes = np.array(boxes, dtype=np.int32)
    for _box, _cls, _score in zip(boxes, classes, scores):

        if _score < score_threshold:
            continue

        text = '{} | {:.2f}'.format(_cls, _score)
        text_orig = (_box[0] + 5, _box[1] - 6)
        image = cv2.putText(image,
                            text,
                            text_orig,
                            cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            .35, [0, 0, 0],
                            4,
                            lineType=cv2.LINE_AA)
        image = cv2.putText(image,
                            text,
                            text_orig,
                            cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            .35, [255, 255, 255],
                            1,
                            lineType=cv2.LINE_AA)
        image = cv2.rectangle(image, (_box[0], _box[1]), (_box[2], _box[3]),
                              [0, 0, 255], 1)

        if save:
            cv2.imwrite(filename, image[:, :, ::-1])
    return image


class ImageGenerator:
    def __init__(
            self,
            image_paths,
            max_images,
            batch_size,
            target_shape,
            channel_mean,
            channel_stddev,
            pixel_scale):

        self._image_paths = sorted(image_paths)
        logging.info('Found {} image paths'.format(len(image_paths)))

        self._max_images = max_images
        self._batch_size = batch_size
        self._target_shape = target_shape
        self._channel_mean = channel_mean
        self._channel_stddev = channel_stddev
        self._pixel_scale = pixel_scale

        self._num_batches = min(len(image_paths), max_images) // batch_size
        self.num_images = self._num_batches * batch_size

        logging.info('Using {}/{} images'.format(self.num_images, len(image_paths)))

    def get_input_spec(self):
        return [self._batch_size, *self._target_shape, 3]

    def get_batches(self):
        for idx in range(self._num_batches):
            batch = np.zeros(
                shape=self.get_input_spec(),
                dtype=np.float32)

            start_idx = idx * self._batch_size
            end_idx = (idx + 1) * self._batch_size
            image_paths = self._image_paths[start_idx:end_idx]

            for i, path in enumerate(image_paths):
                image = read_image_cv2(path)
                input_image, _ = prepare_image_cv2(
                    image,
                    self._target_shape,
                    self._channel_mean,
                    self._channel_stddev,
                    self._pixel_scale)
                batch[i] = input_image
            yield batch
