import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def read_image(path):
    image_bytes = tf.io.read_file(path)
    image = tf.image.decode_image(image_bytes)
    image.set_shape([None, None, None])

    if image.get_shape().as_list()[-1] == 1:
        image = tf.image.grayscale_to_rgb(image)

    return tf.cast(image, dtype=tf.float32)


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
