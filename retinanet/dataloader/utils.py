import tensorflow as tf


def swap_xy(boxes):
    return tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]],
                    axis=-1)


def convert_to_xywh(boxes):
    return tf.concat(
        [(boxes[..., :2] + boxes[..., 2:]) / 2.0,
         boxes[..., 2:] - boxes[..., :2]],
        axis=-1,
    )


def convert_to_corners(boxes):
    return tf.concat(
        [
            boxes[..., :2] - boxes[..., 2:] / 2.0,
            boxes[..., :2] + boxes[..., 2:] / 2.0
        ],
        axis=-1,
    )


def compute_iou(boxes1, boxes2, pair_wise=True):
    boxes1_corners = convert_to_corners(boxes1)
    boxes2_corners = convert_to_corners(boxes2)

    if pair_wise:
        boxes1_corners = tf.expand_dims(boxes1_corners, axis=1)

    lu = tf.maximum(boxes1_corners[..., :2], boxes2_corners[:, :2])
    rd = tf.minimum(boxes1_corners[...,  2:], boxes2_corners[:, 2:])
    intersection = tf.maximum(0.0, rd - lu)
    intersection_area = intersection[..., 0] * intersection[..., 1]
    boxes1_area = boxes1[:, 2] * boxes1[:, 3]
    boxes2_area = boxes2[:, 2] * boxes2[:, 3]

    if pair_wise:
        boxes1_area = tf.expand_dims(boxes1_area, axis=1)

    union_area = tf.maximum(
        boxes1_area + boxes2_area - intersection_area, 1e-8)
    return tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)


def random_flip_horizontal(image, boxes, seed=0):
    if tf.random.uniform((), seed=seed) > 0.5:
        image = tf.image.flip_left_right(image)
        boxes = tf.stack(
            [1 - boxes[:, 2], boxes[:, 1], 1 - boxes[:, 0], boxes[:, 3]],
            axis=-1)
    return image, boxes


def normalize_image(image, mean, stddev, pixel_scale):
    pixel_scale = tf.constant(pixel_scale)
    mean = tf.reshape(tf.constant(mean), shape=[1, 1, 3])
    stddev = tf.reshape(tf.constant(stddev), shape=[1, 1, 3])

    image = image / pixel_scale
    image = (image - mean) / stddev

    return image
