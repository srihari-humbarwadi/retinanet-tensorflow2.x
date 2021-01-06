import os

import tensorflow as tf
from absl import app, flags, logging

from retinanet.cfg import Config
from retinanet.model import make_inference_model, model_builder
from retinanet.trainer import Trainer

tf.get_logger().propagate = False
tf.config.set_soft_device_placement(True)

flags.DEFINE_string('config_path',
                    default=None,
                    help='Path to the config file')

flags.DEFINE_string('export_dir',
                    default='export',
                    help='Path to store the model artefacts')

flags.DEFINE_boolean('export_saved_model',
                     default=False,
                     help='Export weights as a `saved_model`')

flags.DEFINE_boolean('export_h5',
                     default=False,
                     help='Export weights as an h5 file (can be used for fine tuning)')  # noqa: E501

flags.DEFINE_boolean('debug', default=False, help='Print debugging info')

FLAGS = flags.FLAGS


def main(_):
    logging.set_verbosity(logging.DEBUG if FLAGS.debug else logging.INFO)

    params = Config(FLAGS.config_path).params

    if FLAGS.log_dir and (not os.path.exists(FLAGS.log_dir)):
        os.mkdir(FLAGS.log_dir)

    logging.get_absl_handler().use_absl_log_file('export_' + params.experiment.name)

    run_mode = 'export'

    train_input_fn = None
    val_input_fn = None
    model_fn = model_builder(params)

    trainer = Trainer(
        strategy=tf.distribute.OneDeviceStrategy(device='/cpu:0'),
        run_mode=run_mode,
        model_fn=model_fn,
        train_input_fn=train_input_fn,
        val_input_fn=val_input_fn,
        train_steps=params.training.train_steps,
        val_steps=params.training.validation_steps,
        val_freq=params.training.validation_freq,
        steps_per_execution=params.training.steps_per_execution,
        batch_size=None,
        model_dir=params.experiment.model_dir,
        save_every=params.training.save_every,
        restore_checkpoint=True,
        summary_dir=params.experiment.tensorboard_dir,
        name=params.experiment.name
    )

    trainer.restore_status.assert_consumed()

    inference_model = make_inference_model(trainer.model, params)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None] + params.input.input_shape +
                      [params.input.channels],
                      name='images',
                      dtype=tf.float32)
    ])
    def serving_fn(images):
        detections = inference_model.call(images, training=False)
        return {
            'boxes': detections.nmsed_boxes,
            'scores': detections.nmsed_scores,
            'classes': detections.nmsed_classes,
            'num_detections': detections.valid_detections
        }

    if FLAGS.export_h5:
        export_dir = os.path.join(FLAGS.export_dir, trainer.name)

        if not os.path.exists(export_dir):
            os.makedirs(export_dir, exist_ok=True)

        latest_checkpoint = os.path.basename(
            tf.train.latest_checkpoint(trainer.model_dir))

        export_filename = os.path.join(export_dir, latest_checkpoint + '.h5')

        logging.info(
            'Exporting `weights in h5 format` to {}'.format(export_filename))

        trainer.model.save_weights(export_filename)

    if FLAGS.export_saved_model:

        logging.info('Exporting `saved_model` to {}'.format(FLAGS.export_dir))

        tf.saved_model.save(
            inference_model,
            os.path.join(FLAGS.export_dir, params.experiment.name),
            signatures={'serving_default': serving_fn.get_concrete_function()})


if __name__ == '__main__':
    app.run(main)
