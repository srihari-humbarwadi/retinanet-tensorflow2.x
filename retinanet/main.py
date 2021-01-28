import os

import tensorflow as tf
from absl import app, flags, logging

from retinanet.cfg import Config
from retinanet.dataloader import InputPipeline
from retinanet.core.utils import get_strategy, set_precision
from retinanet.model.builder import ModelBuilder
from retinanet.trainer import Trainer

tf.get_logger().propagate = False
tf.config.set_soft_device_placement(True)


flags.DEFINE_string('config_path',
                    default=None,
                    help='Path to the config file')

flags.DEFINE_string('hparams',
                    default=None,
                    help='hparams overrides the configuration file i.e config_path.')

flags.DEFINE_string('model_dir',
                    default=None,
                    help='Overides `model_dir` specified in the config')

flags.DEFINE_string('resume_from',
                    default=None,
                    help='Overides latest_checkpoint')

flags.DEFINE_boolean('run_evaluation',
                     default=False,
                     help='Overides `run_mode` specified in the config')

flags.DEFINE_boolean('run_continuous_evaluation',
                     default=False,
                     help='Overides `run_mode` specified in the config')

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

SUPPORTED_RUN_MODES = ['train', 'val', 'train_val', 'continuous_eval']


def main(_):
    logging.set_verbosity(logging.DEBUG if FLAGS.debug else logging.INFO)

    config = Config(FLAGS.config_path)
    params = config.params

    # override config params with hparams string.
    if FLAGS.hparams:
        params = config.override(FLAGS.hparams)

    if FLAGS.log_dir and (not os.path.exists(FLAGS.log_dir)):
        os.makedirs(FLAGS.log_dir, exist_ok=True)

    logging.get_absl_handler().use_absl_log_file(params.experiment.name)

    if FLAGS.is_multi_host:
        logging.warning('Running in multi_host mode')

    if FLAGS.xla:
        tf.config.optimizer.set_jit(True)

    if FLAGS.gpu_memory_allow_growth:
        physical_devices = tf.config.list_physical_devices('GPU')
        [tf.config.experimental.set_memory_growth(x, True)
         for x in physical_devices]

    set_precision(params.floatx.precision)

    strategy = get_strategy(params.training.strategy)
    logging.info('Running on {} replicas'.format(strategy.num_replicas_in_sync))

    run_mode = params.experiment.run_mode

    if FLAGS.run_evaluation:
        logging.warning('Overiding `run_mode` from {} to evaluation only'.format(
            run_mode))
        run_mode = 'val'

    if FLAGS.run_continuous_evaluation:
        logging.warning('Overiding `run_mode` from {} to continuous evaluation'
                        .format(run_mode))
        run_mode = 'continuous_eval'

    if run_mode not in SUPPORTED_RUN_MODES:
        raise AssertionError(
            'Unsupported run mode requested, available run modes: {}'.format(
                SUPPORTED_RUN_MODES))

    if FLAGS.model_dir is not None:
        logging.warning('Overiding `model_dir` from {} to {}'.format(
            params.experiment.model_dir, FLAGS.model_dir))
        params.experiment.model_dir = FLAGS.model_dir

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

    model_builder = ModelBuilder(params)

    trainer = Trainer(  # noqa: F841
        params=params,
        strategy=strategy,
        run_mode=run_mode,
        model_builder=model_builder,
        train_input_fn=train_input_fn,
        val_input_fn=val_input_fn,
        is_multi_host=FLAGS.is_multi_host,
        resume_from=FLAGS.resume_from)

    trainer.run()


if __name__ == '__main__':
    app.run(main)
