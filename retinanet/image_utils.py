import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def read_image(path):
    image_bytes = tf.io.read_file(path)
    image = tf.image.decode_image(image_bytes)
    return tf.cast(image, dtype=tf.float32)


def visualize_detections(image,
                         boxes,
                         classes,
                         scores,
                         figsize=(12, 12),
                         linewidth=1,
                         color=[0, 0, 1],
                         score_threshold=0.25):
    """Visualize Detections"""
    image = np.array(image, dtype=np.uint8)
    plt.figure(figsize=figsize)
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
    plt.show()
    return ax
