import os

import tensorflow as tf
from absl import app, flags, logging
from tensorflow.python.framework.convert_to_constants import \
    convert_variables_to_constants_v2_as_graph

from retinanet import Executor
from retinanet.cfg import Config
from retinanet.dataloader.preprocessing_pipeline import PreprocessingPipeline
from retinanet.model import ModelBuilder

tf.get_logger().propagate = False
tf.config.set_soft_device_placement(True)

flags.DEFINE_string(
    name='config_path',
    default=None,
    help='Path to the config file')

flags.DEFINE_string(
    name='export_dir',
    default='export',
    help='Path to store the model artefacts')

flags.DEFINE_boolean(
    name='export_saved_model',
    default=False,
    help='Export weights as a `saved_model`')

flags.DEFINE_boolean(
    name='export_h5',
    default=False,
    help='Export weights as an h5 file (can be used for fine tuning)')

flags.DEFINE_string(
    name='checkpoint_name',
    default='latest',
    help='Restores model weights from `checkpoint_name`. Default behaviours uses the latest available checkpoint')  # noqa: E501

flags.DEFINE_boolean(
    name='ignore_moving_average_weights',
    default=False,
    help='Loads non averaged weights if `use_moving_average` is set to True')

flags.DEFINE_string(
    name='model_dir',
    default=None,
    help='Overides `model_dir` specified in the config')

flags.DEFINE_boolean(
    name='disable_pre_nms_top_k',
    default=False,
    help='Skip top k filtering before applying nms')

flags.DEFINE_boolean(
    name='debug',
    default=False,
    help='Print debugging info')

FLAGS = flags.FLAGS


def main(_):
    logging.set_verbosity(logging.DEBUG if FLAGS.debug else logging.INFO)

    params = Config(FLAGS.config_path).params

    if FLAGS.log_dir and (not os.path.exists(FLAGS.log_dir)):
        os.makedirs(FLAGS.log_dir, exist_ok=True)

    logging.get_absl_handler().use_absl_log_file(
        'export_' + params.experiment.name)

    if not FLAGS.model_dir == 'null':
        params.experiment.model_dir = FLAGS.model_dir
        logging.warning('Using local path {} as `model_dir`'.format(
            params.experiment.model_dir))

    if FLAGS.disable_pre_nms_top_k:
        params.inference.pre_nms_top_k = -1
        logging.warning('Disabled pre nms top k filtering')

    if FLAGS.checkpoint_name == 'latest':
        checkpoint_name = None

    else:
        checkpoint_name = FLAGS.checkpoint_name

    run_mode = 'export'

    # skip loading pretrained backbone weights
    params.architecture.backbone.checkpoint = ''

    train_input_fn = None
    val_input_fn = None
    model_builder = ModelBuilder(params)

    executor = Executor(
        params=params,
        strategy=tf.distribute.OneDeviceStrategy(device='/cpu:0'),
        run_mode=run_mode,
        model_builder=model_builder,
        train_input_fn=train_input_fn,
        val_input_fn=val_input_fn,
        resume_from=checkpoint_name
    )

    executor.restore_status.assert_consumed()

    if params.training.optimizer.use_moving_average:
        non_averaged_weights = executor.assign_moving_averaged_weights()

        if FLAGS.ignore_moving_average_weights:
            logging.warning('Loading back non averaged weights into model')
            executor.model.set_weights(non_averaged_weights)

    if FLAGS.export_h5:
        export_dir = os.path.join(FLAGS.export_dir, executor.name)

        if not os.path.exists(export_dir):
            os.makedirs(export_dir, exist_ok=True)

        latest_checkpoint = os.path.basename(
            tf.train.latest_checkpoint(executor.model_dir))

        export_filename = os.path.join(export_dir, latest_checkpoint + '.h5')

        logging.info(
            'Exporting `weights in h5 format` to {}'.format(export_filename))

        executor.model.save_weights(export_filename)

    if FLAGS.export_saved_model:
        logging.info('Exporting `saved_model` to {}'.format(FLAGS.export_dir))

        serving_fn_input_signature = {
            'image':
                tf.TensorSpec(
                    shape=[None] + params.input.input_shape + [params.input.channels],  # noqa: E501
                    name='image',
                    dtype=tf.float32),
            'image_id':
                tf.TensorSpec(shape=[], name='image_id', dtype=tf.int32),
            'resize_scale':
                tf.TensorSpec(shape=[1, 4], name='resize_scale', dtype=tf.float32),
        }

        preprocessing_fn_input_signature = {
            'image':
                tf.TensorSpec(shape=[None, None, 3], name='image', dtype=tf.float32),
            'image_id':
                tf.TensorSpec(shape=[], name='image_id', dtype=tf.int32)
        }
        preprocessing_pipeling = PreprocessingPipeline(
            params.input.input_shape, params.dataloader_params)

        @tf.function
        def prepare_image(sample):
            image_dict = preprocessing_pipeling.preprocess_val_sample(sample)
            input_shape = tf.constant(params.input.input_shape, dtype=tf.float32)
            resize_scale = image_dict['resize_scale'] / input_shape
            resize_scale = tf.tile(tf.expand_dims(resize_scale, axis=0),
                                   multiples=[1, 2])
            return {
                'image': tf.expand_dims(image_dict['image'], axis=0),
                'image_id': sample['image_id'],
                'resize_scale': resize_scale
            }

        @tf.function()
        def serving_fn(sample):
            detections = inference_model.call(sample['image'], training=False)

            return {
                'image_id': sample['image_id'],
                'boxes': detections['boxes'] / sample['resize_scale'],
                'scores': detections['scores'],
                'classes': tf.cast(detections['classes'], dtype=tf.int32),
                'valid_detections': detections['valid_detections']
            }

        inference_model = model_builder.prepare_model_for_export(executor.model)

        frozen_serving_fn, _ = convert_variables_to_constants_v2_as_graph(
            serving_fn.get_concrete_function(serving_fn_input_signature),
            aggressive_inlining=True)

        class InferenceModule(tf.Module):
            def __init__(self, inference_function):
                super(InferenceModule, self).__init__(name='inference_module')
                self.inference_function = inference_function

            @tf.function
            def run_inference(self, sample):
                outputs = self.inference_function(**sample)
                return {
                    'image_id': outputs[2],
                    'boxes': outputs[0],
                    'scores': outputs[3],
                    'classes': outputs[1],
                    'valid_detections': outputs[4]
                }

        inference_module = InferenceModule(frozen_serving_fn)

        tf.saved_model.save(
            inference_module,
            os.path.join(FLAGS.export_dir, params.experiment.name),
            signatures={
                'serving_default':
                inference_module.run_inference.get_concrete_function(
                    serving_fn_input_signature),
                'prepare_image': prepare_image.get_concrete_function(
                    preprocessing_fn_input_signature)
            })


if __name__ == '__main__':
    app.run(main)
