import os

from absl import app, flags, logging
import numpy as np
import tensorflow as tf

from retinanet.dataset_utils.coco_parser import CocoParser
from retinanet.dataset_utils.tfrecord_writer import TFrecordWriter

flags.DEFINE_string('download_path',
                    default=None,
                    help='Path to the downloaded and unzipped COCO files.')

flags.DEFINE_integer('num_shards',
                     default=256,
                     help='Number of tfrecord files required.')

flags.DEFINE_string('output_dir',
                    default='./coco_tfrecords',
                    help='Path to store the generated tfrecords in.')

FLAGS = flags.FLAGS


def write_tfrecords(data, num_shards, output_dir, split_name):
    tfrecord_writer = TFrecordWriter(n_samples=len(data),
                                     n_shards=num_shards,
                                     output_dir=output_dir,
                                     prefix=split_name)
    bad_samples = 0
    for sample in data:
        try:
            with tf.io.gfile.GFile(sample['image'], 'rb') as fp:
                image = fp.read()
                h, w, _ = tf.image.decode_image(image).shape.as_list()
        except Exception:
            bad_samples += 1
            continue

        tfrecord_writer.push(
            image,
            np.array(sample['label']['boxes'], dtype=np.float32) /
            np.array([w, h, w, h]),
            np.array(sample['label']['classes'], dtype=np.int32))
    tfrecord_writer.flush_last()
    logging.warning('Skipped {} corrupted samples from {} data'.format(
        bad_samples, split_name))


def main(_):
    if not os.path.exists(FLAGS.output_dir):
        os.mkdir(FLAGS.output_dir)

    coco_parser = CocoParser(FLAGS.download_path)

    write_tfrecords(coco_parser.dataset['train'], FLAGS.num_shards,
                    FLAGS.output_dir, 'train')
    write_tfrecords(coco_parser.dataset['val'], 32, FLAGS.output_dir, 'val')


if __name__ == '__main__':
    app.run(main)
