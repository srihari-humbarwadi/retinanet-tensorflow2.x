import os

import tensorflow as tf
from absl import app, flags, logging
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from retinanet.cfg import Config
from retinanet.dataloader import InputPipeline
from retinanet.distribute import get_strategy
from retinanet.model import model_builder
from retinanet.trainer import Trainer

tf.get_logger().propagate = False
tf.config.set_soft_device_placement(True)

flags.DEFINE_string('config_path',
                    default=None,
                    help='Path to the config file')

flags.DEFINE_boolean('xla', default=False, help='Compile with XLA JIT')

flags.DEFINE_boolean('gpu_memory_allow_growth',
                     default=False,
                     help='If enabled, the runtime doesn\'t allocate all of the available memory')  # noqa: E501


flags.DEFINE_boolean(
    'is_multi_host',
    default=0,
    help='Set this to true if running on TPU Pods or MultiWorker setup')

flags.DEFINE_boolean('debug', default=False, help='Print debugging info')

FLAGS = flags.FLAGS
SUPPORTED_RUN_MODES = ['train', 'val', 'train_val']


def set_precision(precision):
    policy = mixed_precision.Policy(precision)
    mixed_precision.set_policy(policy)

    logging.info('Compute dtype: {}'.format(policy.compute_dtype))
    logging.info('Variable dtype: {}'.format(policy.variable_dtype))


def main(_):
    logging.set_verbosity(logging.DEBUG if FLAGS.debug else logging.INFO)

    params = Config(FLAGS.config_path).params

    if FLAGS.log_dir and (not os.path.exists(FLAGS.log_dir)):
        os.mkdir(FLAGS.log_dir)

    logging.get_absl_handler().use_absl_log_file(params.experiment.name)

    if FLAGS.xla:
        tf.config.optimizer.set_jit(True)

    if FLAGS.gpu_memory_allow_growth:
        physical_devices = tf.config.list_physical_devices('GPU')
        [tf.config.experimental.set_memory_growth(x, True)
         for x in physical_devices]

    set_precision(params.floatx.precision)
    strategy = get_strategy(params.training.strategy)

    run_mode = params.experiment.run_mode
    assert run_mode in SUPPORTED_RUN_MODES, \
        'Unsupported run mode requested, available run modes: {}'.format(
            SUPPORTED_RUN_MODES)

    train_input_fn = None
    val_input_fn = None
    if 'train' in run_mode:
        train_input_fn = InputPipeline(
            run_mode='train',
            params=params,
            is_multi_host=FLAGS.is_multi_host,
            num_replicas=strategy.num_replicas_in_sync)

    if 'val' in run_mode:
        val_input_fn = InputPipeline(
            run_mode='val',
            params=params,
            is_multi_host=FLAGS.is_multi_host,
            num_replicas=strategy.num_replicas_in_sync)

    model_fn = model_builder(params)
    train_batch_size = params.training.batch_size.train

    trainer = Trainer(  # noqa: F841
        strategy=strategy,
        run_mode=run_mode,
        model_fn=model_fn,
        train_input_fn=train_input_fn,
        val_input_fn=val_input_fn,
        train_steps=params.training.train_steps,
        val_steps=params.training.validation_steps,
        val_freq=params.training.validation_freq,
        steps_per_execution=params.training.steps_per_execution,
        batch_size=train_batch_size if run_mode == 'train' else None,
        model_dir=params.experiment.model_dir,
        save_every=params.training.save_every,
        restore_checkpoint=params.training.restore_checkpoint,
        summary_dir=params.experiment.tensorboard_dir,
        name=params.experiment.name
    )


if __name__ == '__main__':
    app.run(main)
