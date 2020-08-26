import tensorflow as tf


def parse_example(example_proto):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'image_id': tf.io.FixedLenFeature([], tf.int64),
        'xmins': tf.io.VarLenFeature(tf.float32),
        'ymins': tf.io.VarLenFeature(tf.float32),
        'xmaxs': tf.io.VarLenFeature(tf.float32),
        'ymaxs': tf.io.VarLenFeature(tf.float32),
        'classes': tf.io.VarLenFeature(tf.int64),
    }

    parsed_example = tf.io.parse_single_example(example_proto,
                                                feature_description)

    image = tf.io.decode_image(parsed_example['image'], channels=3)
    image = tf.cast(image, dtype=tf.float32)
    image.set_shape([None, None, 3])

    image_id = parsed_example['image_id']

    bbox = tf.stack([
        tf.sparse.to_dense(parsed_example['xmins']),
        tf.sparse.to_dense(parsed_example['ymins']),
        tf.sparse.to_dense(parsed_example['xmaxs']),
        tf.sparse.to_dense(parsed_example['ymaxs']),
    ],
        axis=-1)
    label = tf.sparse.to_dense(parsed_example['classes'])

    sample = {
        'image': image,
        'image_id': image_id,
        'objects': {
            'bbox': bbox,
            'label': label
        }
    }
    return sample
