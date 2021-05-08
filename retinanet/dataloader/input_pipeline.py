import tensorflow as tf
from absl import logging

from retinanet.dataloader.label_encoder import LabelEncoder
from retinanet.dataloader.tfrecord_parser import parse_example


class InputPipeline:
    _SUPPORTED_RUN_MODES = ['train', 'val']
    _RANDOM_SEED = 1337

    def __init__(self, run_mode, params, is_multi_host, num_replicas):

        if run_mode not in InputPipeline._SUPPORTED_RUN_MODES:
            raise AssertionError('Unsupported run mode requested,\
                 available run modes: {}'.format(
                InputPipeline._SUPPORTED_RUN_MODES))

        self.run_mode = run_mode
        self.is_multi_host = is_multi_host
        self.num_replicas = num_replicas
        self.batch_size = params.training.batch_size[run_mode]
        self.shuffle_buffer_size = params.dataloader_params.shuffle_buffer_size
        self.tfrecord_files = params.dataloader_params.tfrecords[run_mode]
        self.label_encoder = LabelEncoder(params)

    def __call__(self, input_context=None):
        batch_size = self.batch_size
        autotune = tf.data.experimental.AUTOTUNE

        matched_tfrecord_files = tf.io.gfile.glob(self.tfrecord_files)
        num_files = len(matched_tfrecord_files)

        logging.info('Found {} {} tfrecords matching {}'.format(
            num_files, self.run_mode, self.tfrecord_files))

        dataset = tf.data.Dataset.from_tensor_slices(matched_tfrecord_files)

        if not self.run_mode == 'val':
            dataset = dataset.shuffle(
                num_files,
                seed=InputPipeline._RANDOM_SEED,
                reshuffle_each_iteration=True)
            dataset = dataset.repeat()

        if self.is_multi_host and input_context is not None:
            batch_size = input_context.get_per_replica_batch_size(
                self.batch_size)
            dataset = dataset.shard(input_context.num_input_pipelines,
                                    input_context.input_pipeline_id)

            logging.info(
                '[Worker ID {}] Using per_replica batch_size of {} for {}'
                .format(
                    input_context.input_pipeline_id,
                    batch_size,
                    self.run_mode))

        dataset = dataset.cache()

        dataset = dataset.interleave(
            map_func=tf.data.TFRecordDataset,
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
                batch_size=self.num_replicas,
                num_parallel_calls=autotune,
                deterministic=False,
                drop_remainder=False)
            dataset = dataset.prefetch(autotune)
            return dataset

        dataset = dataset.shuffle(self.shuffle_buffer_size)

        dataset = dataset.map(
            map_func=lambda x: self.label_encoder.
            encode_sample(parse_example(x)),
            num_parallel_calls=autotune)

        dataset = dataset.batch(
            batch_size=batch_size,
            num_parallel_calls=autotune,
            deterministic=False, drop_remainder=True)
        dataset = dataset.prefetch(autotune)
        return dataset
