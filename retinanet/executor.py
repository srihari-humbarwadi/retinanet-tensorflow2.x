import json
import os
from time import sleep, time

import numpy as np
import tensorflow as tf
from absl import logging
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

from retinanet.eval import COCOEvaluator


class Executor:

    _RUN_MODES = [
        'train',
        'val',
        'train_val',
        'continuous_eval',
        'export'
    ]

    def __init__(self,
                 params,
                 strategy,
                 run_mode,
                 model_builder,
                 train_input_fn,
                 val_input_fn=None,
                 is_multi_host=False,
                 enable_weights_info=False,
                 resume_from=None):
        self.params = params
        self.distribute_strategy = strategy
        self.run_mode = run_mode
        self.model_builder = model_builder
        self.restore_checkpoint = params.training.restore_checkpoint
        self.train_input_fn = train_input_fn
        self.val_input_fn = val_input_fn
        self.is_multi_host = is_multi_host
        self.enable_weights_info = enable_weights_info
        self.resume_from = resume_from
        self.train_steps = params.training.train_steps
        self.validation_samples = params.training.validation_samples
        self.val_freq = params.training.validation_freq
        self.steps_per_execution = params.training.steps_per_execution
        self.batch_size = params.training.batch_size['train']
        self.model_dir = os.path.join(
            params.experiment.model_dir, params.experiment.name)
        self.save_every = params.training.save_every
        self.summary_dir = params.experiment.tensorboard_dir
        self.name = params.experiment.name

        self.val_steps = self.validation_samples // params.training.batch_size['val']

        self.num_replicas = self.distribute_strategy.num_replicas_in_sync
        self.restore_status = None
        self.use_float16 = False
        self._summary_writers = {}
        self._run_evaluation_at_end = params.training.validation_freq < 1

        if self.run_mode not in Executor._RUN_MODES:
            raise AssertionError(
                'Invalid run mode, aborting!\n Supported run models {}'
                .format(Executor._RUN_MODES))

        self._setup()

    def run(self):
        if 'train' in self.run_mode:
            self.train()

        elif self.run_mode == 'val':
            self.evaluate()

        elif self.run_mode == 'continuous_eval':
            self.continuous_evaluate()

    def _setup_model(self):
        logging.info('Setting up model for {}'.format(self.run_mode))
        self._model = self.model_builder()
        self.optimizer = self._model.optimizer
        self._created_optimizer_weights = False

        if self.params.fine_tuning.fine_tune:
            logging.info(
                'Loading pretrained weights for fine-tuning from {}'.format(
                    self.params.fine_tuning.pretrained_checkpoint))
            self._model.load_weights(self.params.fine_tuning.pretrained_checkpoint,
                                     skip_mismatch=True,
                                     by_name=True)

        logging.info('Trainable variables: {}'.format(
            len(self._model.trainable_variables)))

        _has_frozen_layers = self._maybe_freeze_layers()

        if _has_frozen_layers:
            logging.info(
                'Trainable variables after freezing: {}'
                .format(len(self._model.trainable_variables)))

        if isinstance(
            self.optimizer,
                tf.keras.mixed_precision.experimental.LossScaleOptimizer):
            self.use_float16 = True

        if 'val' in self.run_mode:
            self._eval_model = \
                self.model_builder.add_post_processing_stage(self._model)

            flops = self.get_flops()
            logging.info('Total flops (multiplyadds): {:,}'.format(flops))

        self._model.summary(print_fn=logging.debug)
        logging.info('Total trainable parameters: {:,}'.format(
            sum([
                tf.keras.backend.count_params(x)
                for x in self._model.trainable_variables
            ])))

        self._weight_decay_vars = self._get_weight_decay_variables()
        logging.info('Initial weight decay loss: {}'.format(
            np.round(self.weight_decay(), 4)))

    def _maybe_freeze_layers(self):
        _freeze_patterns = self.params.training.freeze_variables
        _layers = []

        if not _freeze_patterns:
            return False

        for _pattern in _freeze_patterns:
            _regex = self.model_builder.FREEZE_VARS_REGEX[_pattern]

            logging.warning(
                'Freezing layers with variables that match pattern: {}'
                .format(_regex.pattern))

            for layer in self._model.layers:
                _layers.extend(Executor._maybe_flatten_layers(layer))

            for layer in _layers:
                for weight in layer.weights:
                    if _regex.search(weight.name) and layer.trainable:
                        layer.trainable = False
                        logging.debug('Freezing layer: {}'.format(layer.name))
        return True

    def _setup_dataset(self):
        if 'val' in self.run_mode:
            logging.info('Setting up val dataset')

            if self.is_multi_host:
                self._val_dataset = \
                    self.distribute_strategy.distribute_datasets_from_function(
                        self.val_input_fn
                    )
            else:
                self._val_dataset = \
                    self.distribute_strategy.experimental_distribute_dataset(
                        self.val_input_fn())

        if 'train' in self.run_mode:
            logging.info('Setting up train dataset')

            if self.is_multi_host:
                self._train_dataset = \
                    self.distribute_strategy.distribute_datasets_from_function(
                        self.train_input_fn)
            else:
                self._train_dataset = \
                    self.distribute_strategy.experimental_distribute_dataset(
                        self.train_input_fn())

    def _setup_summary_writers(self):
        logging.info('Setting up summary writers')

        if 'train' in self.run_mode:
            train_summary_dir = os.path.join(self.summary_dir, self.name, 'train')
            logging.info('Writing train summaries to {}'.format(train_summary_dir))

            self._summary_writers['train'] = \
                tf.summary.create_file_writer(train_summary_dir)

        if 'val' in self.run_mode:
            eval_summary_dir = os.path.join(self.summary_dir, self.name, 'eval')
            logging.info('Writing eval summaries to {}'.format(eval_summary_dir))

            self._summary_writers['eval'] = \
                tf.summary.create_file_writer(eval_summary_dir)

    def _restore_checkpoint(self, checkpoint=None):

        if checkpoint is not None:
            latest_checkpoint = checkpoint

        elif self.resume_from is not None:
            latest_checkpoint = os.path.join(self.model_dir, self.resume_from)

        else:
            logging.info('Looking for existing checkpoints in {}'
                         .format(self.model_dir))
            latest_checkpoint = tf.train.latest_checkpoint(self.model_dir)

        if latest_checkpoint is not None:
            logging.info(
                'Found existing checkpoint {}, restoring model and optimizer state from checkpoint'  # noqa: E501
                .format(latest_checkpoint))

            if not self._created_optimizer_weights:
                logging.info('Initializing optimizer slots/weights')
                self.optimizer._create_all_weights(self._model.trainable_variables)
                self._created_optimizer_weights = True

            self.restore_status = self._model.load_weights(latest_checkpoint)
            return

        if 'export' in self.run_mode:
            raise AssertionError(
                'No checkpoints found in {}, aborting.'.format(self.model_dir))

        logging.warning(
            'No existing checkpoints found in {}, running model in {} mode with random weights initialization!'  # noqa: E501
            .format(self.model_dir, self.run_mode))

    def _setup(self):
        if not tf.io.gfile.exists(self.model_dir):
            tf.io.gfile.makedirs(self.model_dir)

        config_path = os.path.join(self.model_dir, '{}.json'.format(self.name))

        with tf.io.gfile.GFile(config_path, 'w') as f:
            logging.info('Dumping config to {}'.format(config_path))
            f.write(json.dumps(self.params, indent=4))

        with self.distribute_strategy.scope():
            self._setup_model()
            self._setup_dataset()

            if self.restore_checkpoint:
                self._restore_checkpoint()

    @staticmethod
    def _maybe_flatten_layers(layer):
        if hasattr(layer, 'layers'):
            return layer.layers
        return [layer]

    def weight_decay(self):
        _alpha = self.params.training.weight_decay_alpha
        return tf.math.add_n([_alpha * tf.nn.l2_loss(x)
                              for x in self._weight_decay_vars])

    def _get_weight_decay_variables(self):
        weight_decay_vars = []
        _layers = []

        for layer in self._model.layers:
            _layers.extend(Executor._maybe_flatten_layers(layer))

        for layer in _layers:
            if layer.trainable:
                if isinstance(layer, tf.keras.layers.Conv2D):
                    weight_decay_vars.append(layer.kernel)

                elif isinstance(layer, tf.keras.layers.SeparableConv2D):
                    weight_decay_vars.append(layer.depthwise_kernel)
                    weight_decay_vars.append(layer.pointwise_kernel)

        return weight_decay_vars

    @tf.function
    def _write_weights_info(self, step, write_histogram=False):
        with self._summary_writers['train'].as_default():
            with tf.name_scope('weights'):
                for weight in self._model.trainable_weights:
                    tf.summary.scalar(
                        name='{}_norm'.format(weight.name),
                        data=tf.norm(weight),
                        step=step)

                    if write_histogram:
                        tf.summary.histogram(
                            name='{}_histogram'.format(weight.name),
                            data=weight,
                            step=step)

        self._summary_writers['train'].flush()

    @tf.function
    def _write_train_summaries(self, loss_dict, step):
        with self._summary_writers['train'].as_default():
            with tf.name_scope('losses'):
                for k in ['box-loss',
                          'class-loss',
                          'weighted-loss',
                          'total-loss']:
                    v = loss_dict[k]
                    tf.summary.scalar(name=k, data=v, step=step)

                if self.params.training.use_weight_decay:
                    tf.summary.scalar(
                        'l2-regularization',
                        data=loss_dict['l2-regularization'],
                        step=step)

            with tf.name_scope('metrics'):
                for k in ['learning-rate',
                          'gradient-norm',
                          'num-anchors-matched',
                          'execution-time']:
                    v = loss_dict[k]
                    tf.summary.scalar(name=k, data=v, step=step)

        self._summary_writers['train'].flush()

    @tf.function
    def _write_eval_summaries(self, scores, step):
        with self._summary_writers['eval'].as_default():
            with tf.name_scope('evaluation'):
                for k in ['AP-IoU=0.50:0.95',
                          'AP-IoU=0.50',
                          'AP-IoU=0.75',
                          'AR-(all)-IoU=0.50:0.95',
                          'AR-(L)-IoU=0.50:0.95']:
                    v = scores[k]
                    tf.summary.scalar(name=k, data=v, step=step)

        self._summary_writers['eval'].flush()

    def _eval_step(self, data):
        detections = self._eval_model(data['image'], training=False)
        return {
            'image_id': data['image_id'],
            'detections': detections,
            'resize_scale': data['resize_scale']
        }

    @tf.function
    def distributed_eval_step(self, data):
        results = self.distribute_strategy.run(self._eval_step,
                                               args=(data,))
        results = tf.nest.map_structure(
            lambda x: self.distribute_strategy.gather(x, axis=0), results)
        return results

    def _train_step(self, data):
        images, targets = data

        with tf.GradientTape() as tape:
            predictions = self._model(images, training=True)
            loss = self._model.loss(targets, predictions)
            loss['total-loss'] = loss['weighted-loss']

            if self.params.training.use_weight_decay:
                loss['l2-regularization'] = self.weight_decay()
                loss['total-loss'] += loss['l2-regularization']

            per_replica_loss = loss['total-loss'] / self.num_replicas

            if self.use_float16:
                per_replica_loss = self.optimizer.get_scaled_loss(
                    per_replica_loss)

        gradients = tape.gradient(per_replica_loss,
                                  self._model.trainable_variables)
        if self.use_float16:
            gradients = self.optimizer.get_unscaled_gradients(gradients)

        self.optimizer.apply_gradients(
            zip(gradients, self._model.trainable_variables))

        loss['num-anchors-matched'] /= tf.cast(tf.shape(images)[0], dtype=tf.float32)
        loss['gradient-norm'] = tf.linalg.global_norm(gradients) * self.num_replicas
        return loss

    @tf.function
    def distributed_train_step(self, iterator, num_steps):
        loss_dict = self.distribute_strategy.run(self._train_step,
                                                 args=(next(iterator),))
        for _ in tf.range(num_steps - 1):
            loss_dict = self.distribute_strategy.run(self._train_step,
                                                     args=(next(iterator),))
        loss_dict = tf.nest.map_structure(
            lambda x: self.distribute_strategy.reduce(
                tf.distribute.ReduceOp.MEAN, x, axis=None), loss_dict)
        return loss_dict

    def continuous_evaluate(self, sleep_time=250):
        current_checkpoint = None

        while True:
            latest_checkpoint = tf.train.latest_checkpoint(self.model_dir)

            if latest_checkpoint and latest_checkpoint != current_checkpoint:
                self._restore_checkpoint(latest_checkpoint)
                self.restore_status.assert_consumed()
                self.evaluate()
                current_checkpoint = latest_checkpoint

            logging.info(
                'Sleeping for {} secs before checking for new checkpoint'
                .format(sleep_time))
            sleep(sleep_time // 4)

    def evaluate(self):

        if 'eval' not in self._summary_writers:
            self._setup_summary_writers()

        total_steps = self.val_steps
        dataset_iterator = iter(self._val_dataset)
        global_step = tf.convert_to_tensor(
            self.optimizer.iterations, dtype=tf.int64)

        evaluator = COCOEvaluator(
            input_shape=self.params.input.input_shape,
            annotation_file_path=self.params.training.annotation_file_path,
            prediction_file_path=self.name + '.json')

        logging.info('Evaluating at step {} for {} eval steps'
                     .format(global_step.numpy(), total_steps))

        for i, data in enumerate(dataset_iterator):
            start = time()
            results = self.distributed_eval_step(data)
            evaluator.accumulate_results(results)
            end = time()
            execution_time = np.round(end - start, 2)
            images_per_second = self.num_replicas / execution_time

            secs = (total_steps - (i + 1)) * execution_time
            eta = []
            for interval in [3600, 60, 1]:
                eta += ['{:02}'.format(int(secs // interval))]
                secs %= interval
            eta = ':'.join(eta)

            logging.info('[eval_step {}/{}] [ETA: {}] [{:.2f} imgs/s]'
                         .format(
                             i + 1,
                             total_steps,
                             eta,
                             images_per_second))

            if (i + 1) == total_steps:
                break

        scores = evaluator.evaluate()
        self._write_eval_summaries(
            scores,
            global_step)

        logging.info(
            '[global_step {}] evaluation results: {}'
            .format(
                global_step,
                {k: np.round(v, 2) for k, v in scores.items()}))
        return scores

    def train(self):
        if self.restore_checkpoint and self.restore_status is not None:
            self.restore_status.assert_consumed()

        start_step = int(self.optimizer.iterations.numpy())
        current_step = start_step

        if 'val' in self.run_mode:
            logging.info('Running evaluation every {} steps'.format(self.val_freq))

        if current_step >= self.train_steps:
            logging.info('Training completed at step {}'.format(current_step))
            return

        logging.info(
            'Starting training from step {} for {} steps with {} steps per execution'  # noqa: E501
            .format(start_step, self.train_steps, self.steps_per_execution))
        logging.info('Saving checkpoints every {} steps in {}'.format(
            self.save_every, self.model_dir))

        if 'train' not in self._summary_writers:
            self._setup_summary_writers()

        if self.use_float16:
            logging.info('Training with AMP turned on!')

        self._add_graph_trace()

        dataset_iterator = iter(self._train_dataset)

        for _ in range(start_step, self.train_steps, self.steps_per_execution):
            start = time()

            loss_dict = self.distributed_train_step(
                dataset_iterator,
                tf.convert_to_tensor(self.steps_per_execution))
            current_step = int(self.optimizer.iterations.numpy())

            end = time()

            if self.use_float16:
                learning_rate = self.optimizer.inner_optimizer._decayed_lr('float')
            else:
                learning_rate = self.optimizer._decayed_lr('float')

            loss_dict['execution-time'] = np.round(end - start, 2)
            loss_dict['learning-rate'] = learning_rate

            steps_per_second = \
                self.steps_per_execution / loss_dict['execution-time']
            images_per_second = steps_per_second * self.batch_size

            secs = (self.train_steps - current_step) / steps_per_second
            eta = []
            for interval in [3600, 60, 1]:
                eta += ['{:02}'.format(int(secs // interval))]
                secs %= interval
            eta = ':'.join(eta)

            if current_step % self.save_every == 0:
                logging.info(
                    'Saving checkpoint at step {}'.format(current_step))
                self._model.save_weights(
                    os.path.join(self.model_dir,
                                 'weights_step_{}'.format(current_step)))

            self._write_train_summaries(
                loss_dict,
                tf.convert_to_tensor(current_step, dtype=tf.int64))

            if self.enable_weights_info:
                write_histogram = current_step % (50 * self.steps_per_execution) == 0

                self._write_weights_info(
                    step=tf.convert_to_tensor(current_step, dtype=tf.int64),
                    write_histogram=write_histogram)

            logging.info('[global_step {}/{}] [ETA: {}] [{:.2f} imgs/s] {}'
                         .format(
                             current_step,
                             self.train_steps,
                             eta,
                             images_per_second,
                             {k: np.round(v, 4)
                              for k, v in loss_dict.items()}))

            if (current_step % self.val_freq == 0) \
                    and (not self._run_evaluation_at_end) \
                    and ('val' in self.run_mode):
                self.evaluate()

        logging.info('Saving final checkpoint at step {}'.format(current_step))
        self._model.save_weights(
            os.path.join(self.model_dir,
                         'final_weights_step_{}'.format(current_step)))

        if self._run_evaluation_at_end and 'val' in self.run_mode:
            self.evaluate()

    def _add_graph_trace(self):
        @tf.function(input_signature=[
            tf.TensorSpec(shape=[1, *self._model.input_shape[1:]],
                          dtype=tf.float32, name='image')])
        def _forward_pass(image):
            return self._model(image, training=False)

        logging.warning('Collecting graph trace')
        concrete_fn = _forward_pass.get_concrete_function()

        with self._summary_writers['train'].as_default():
            tf.summary.graph(concrete_fn.graph)

        logging.warning('Successfully wrote model graph summary')

    def get_flops(self):
        @tf.function(input_signature=[
            tf.TensorSpec(
                shape=(1,) + self._eval_model.input_shape[1:],
                dtype=tf.float32, name='images')])
        def inference_function(images):
            return self._eval_model(images, training=False)

        concrete_funtion = inference_function.get_concrete_function()
        graph_info = profile(concrete_funtion.graph,
                             options=ProfileOptionBuilder.float_operation())
        flops = graph_info.total_float_ops // 2
        return flops

    @property
    def model(self):
        return self._model

    @property
    def weight_decay_variables(self):
        return self._weight_decay_vars
