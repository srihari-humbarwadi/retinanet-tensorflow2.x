import tensorflow as tf
from absl import app, flags, logging

from retinanet.cfg import Config
from retinanet.model import model_builder, make_inference_model
from retinanet.trainer import Trainer


tf.get_logger().propagate = False
tf.config.set_soft_device_placement(True)

flags.DEFINE_string('config_path',
                    default=None,
                    help='Path to the config file')

flags.DEFINE_string('export_dir',
                    default='export',
                    help='Path to store the `saved_model`')


flags.DEFINE_boolean('debug', default=False, help='Print debugging info')

FLAGS = flags.FLAGS


def main(_):
    logging.set_verbosity(logging.DEBUG if FLAGS.debug else logging.INFO)

    params = Config(FLAGS.config_path).params

    run_mode = 'export'

    train_input_fn = None
    val_input_fn = None
    model_fn = model_builder(params)

    trainer = Trainer(  # noqa: F841
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
        restore_checkpoint=params.training.restore_checkpoint,
        summary_dir=params.experiment.tensorboard_dir,
        name=params.experiment.name
    )

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

    tf.saved_model.save(
        inference_model,
        FLAGS.export_dir,
        signatures={'serving_default': serving_fn.get_concrete_function()})


if __name__ == '__main__':
    app.run(main)
