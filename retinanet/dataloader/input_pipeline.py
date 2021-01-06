import tensorflow as tf
from absl import logging

from retinanet.dataloader.label_encoder import LabelEncoder
from retinanet.dataloader.tfrecord_parser import parse_example


class InputPipeline:
    _SUPPORTED_RUN_MODES = ['train', 'val']

    def __init__(self, run_mode, params, is_multi_host, num_replicas):
        if run_mode not in InputPipeline._SUPPORTED_RUN_MODES:
            raise AssertionError('Unsupported run mode requested, available run modes: {}'.format(
                    InputPipeline._SUPPORTED_RUN_MODES))
        self.run_mode = run_mode
        self.is_multi_host = is_multi_host
        self.num_replicas = num_replicas
        self.batch_size = params.training.batch_size[run_mode]
        self.tfrecord_files = params.dataloader_params.tfrecords[run_mode]
        self.label_encoder = LabelEncoder(params)

    def __call__(self, input_context=None):
        options = tf.data.Options()
        options.experimental_deterministic = False
        autotune = tf.data.experimental.AUTOTUNE

        dataset = tf.data.Dataset.list_files(self.tfrecord_files)

        logging.info('Found {} {} tfrecords matching {}'.format(
            len(dataset), self.run_mode, self.tfrecord_files))

        batch_size = self.batch_size
        if self.is_multi_host and input_context is not None:
            batch_size = input_context.get_per_replica_batch_size(
                self.batch_size)
            dataset = dataset.shard(input_context.num_input_pipelines,
                                    input_context.input_pipeline_id)

        dataset = dataset.cache()

        if not self.run_mode == 'val':
            dataset = dataset.repeat()

        dataset = dataset.interleave(map_func=tf.data.TFRecordDataset,
                                     cycle_length=32,
                                     num_parallel_calls=autotune)
        if self.run_mode == 'val':
            preprocess_fn = \
                self.label_encoder.preprocessing_pipeline.preprocess_val_sample
            dataset = dataset.map(
                map_func=lambda x: preprocess_fn(
                    parse_example(x)),
                num_parallel_calls=autotune)
            dataset = dataset.batch(
                batch_size=batch_size,
                drop_remainder=False)
            dataset = dataset.prefetch(autotune)
            return dataset

        dataset = dataset.with_options(options)
        dataset = dataset.shuffle(1024)
        dataset = dataset.map(map_func=lambda x: self.label_encoder.
                              encode_sample(parse_example(x)),
                              num_parallel_calls=autotune)
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
        dataset = dataset.prefetch(autotune)
        return dataset
