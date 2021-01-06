import os

import tensorflow as tf
from absl import logging


class TFrecordWriter:
    def __init__(self, n_samples, n_shards, output_dir='', prefix=''):
        self.n_samples = n_samples
        self.n_shards = n_shards
        self._step_size = self.n_samples // self.n_shards
        self.prefix = prefix
        self.output_dir = output_dir
        self._buffer = []
        self._file_count = 1

        logging.info(
            'writing {} samples in each tfrecord'.format(self._step_size))

        if self.n_samples % self.n_shards:
            remainder = self.n_samples - (self._step_size * self.n_shards)
            logging.warn(
                'writing {} remaining samples in last tfrecord'.format(
                    remainder))

    @staticmethod
    def _make_example(image, boxes, classes, image_id):
        feature = {
            'image':
            tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
            'image_id':
            tf.train.Feature(int64_list=tf.train.Int64List(value=[image_id])),
            'xmins':
            tf.train.Feature(float_list=tf.train.FloatList(value=boxes[:, 0])),
            'ymins':
            tf.train.Feature(float_list=tf.train.FloatList(value=boxes[:, 1])),
            'xmaxs':
            tf.train.Feature(float_list=tf.train.FloatList(value=boxes[:, 2])),
            'ymaxs':
            tf.train.Feature(float_list=tf.train.FloatList(value=boxes[:, 3])),
            'classes':
            tf.train.Feature(int64_list=tf.train.Int64List(value=classes))
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def _write_tfrecord(self, tfrecord_path):
        logging.info('writing {} samples in {}'.format(len(self._buffer),
                                                       tfrecord_path))
        with tf.io.TFRecordWriter(tfrecord_path) as writer:
            for (image, boxes, classes, image_id) in self._buffer:
                example = TFrecordWriter._make_example(
                    image, boxes, classes, image_id)
                writer.write(example.SerializeToString())

    def push(self, image, boxes, classes, image_id):
        self._buffer.append([image, boxes, classes, image_id])
        if len(self._buffer) == self._step_size:
            fname = self.prefix + '-{:04.0f}'.format(
                self._file_count) + '.tfrecord'
            tfrecord_path = os.path.join(self.output_dir, fname)
            self._write_tfrecord(tfrecord_path)
            self._clear_buffer()
            self._file_count += 1

    def flush_last(self):
        if self._buffer:
            fname = self.prefix + '-{:04.0f}'.format(
                self._file_count) + '.tfrecord'
            tfrecord_path = os.path.join(self.output_dir, fname)
            self._write_tfrecord(tfrecord_path)

    def _clear_buffer(self):
        self._buffer = []
