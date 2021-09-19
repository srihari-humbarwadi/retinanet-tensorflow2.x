from glob import glob
import os

import tensorflow as tf
from absl import app, flags, logging
from tensorflow.python.framework.convert_to_constants import \
    convert_variables_to_constants_v2_as_graph

from retinanet import Executor, onnx_utils
from retinanet.cfg import Config
from retinanet.dataloader.preprocessing_pipeline import PreprocessingPipeline
from retinanet.image_utils import ImageGenerator
from retinanet.model import ModelBuilder
from retinanet.tensorrt.builder import TensorRTBuilder
from retinanet.tensorrt.calibrator import get_calibrator

tf.get_logger().propagate = False
tf.config.set_soft_device_placement(True)
tf.config.optimizer.set_jit(True)

flags.DEFINE_string(
    name='config_path',
    default=None,
    required=True,
    help='Path to the config file')

flags.DEFINE_enum(
    name='mode',
    default=None,
    required=True,
    enum_values=[
        'tf', 'tf_tensorrt', 'onnx', 'onnx_tensorrt'],
    help='Export mode for `saved_models`. Controls skipping decoding/NMS stages')

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

flags.DEFINE_boolean(
    name='export_checkpoint',
    default=False,
    help='Export weights in tensorflow object checkpoint format')

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
    name='skip_prepare_image_fn',
    default=False,
    help='Skip exporting `prepare_image` signature')

flags.DEFINE_boolean(
    name='debug',
    default=False,
    help='Print debugging info')

flags.DEFINE_string(
    name='calibration_images_dir',
    default='coco/val2017',
    help='Calibration images dir')

flags.DEFINE_enum(
    name='calibration_method',
    default='entropy',
    enum_values=['entropy', 'minmax'],
    help='INT8 Calibration method')

flags.DEFINE_integer(
    name='calibration_batch_size',
    default=8,
    help='Batch size for calibration')

flags.DEFINE_integer(
    name='num_calibration_images',
    default=5000,
    help='Number of images used in calibration')

flags.DEFINE_string(
    name='precision',
    default='fp32',
    help='Execution precision for TensorRT Engines')

FLAGS = flags.FLAGS


def main(_):
    logging.set_verbosity(logging.DEBUG if FLAGS.debug else logging.INFO)
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        logging.info('Found {} GPU(s)'.format(len(gpus)))
        [tf.config.experimental.set_memory_growth(device, True) for device in gpus]

    params = Config(FLAGS.config_path).params

    if FLAGS.log_dir and (not os.path.exists(FLAGS.log_dir)):
        os.makedirs(FLAGS.log_dir, exist_ok=True)

    logging.get_absl_handler().use_absl_log_file(
        'export_' + params.experiment.name)

    if FLAGS.model_dir:
        params.experiment.model_dir = FLAGS.model_dir
        logging.warning('Using local path {} as `model_dir`'.format(
            params.experiment.model_dir))

    if FLAGS.checkpoint_name == 'latest':
        checkpoint_name = None

    else:
        checkpoint_name = FLAGS.checkpoint_name

    run_mode = 'export'

    # skip loading pretrained backbone weights
    params.architecture.backbone.checkpoint = ''

    train_input_fn = None
    val_input_fn = None
    model_builder = ModelBuilder(params, run_mode=run_mode)

    executor = Executor(
        params=params,
        strategy=tf.distribute.OneDeviceStrategy(device='/cpu:0'),
        run_mode=run_mode,
        model_builder=model_builder,
        train_input_fn=train_input_fn,
        val_input_fn=val_input_fn,
        resume_from=checkpoint_name
    )
    export_dir = os.path.join(FLAGS.export_dir, params.experiment.name)
    saved_model_export_dir = os.path.join(export_dir, FLAGS.mode)

    if not tf.io.gfile.exists(export_dir):
        tf.io.gfile.makedirs(export_dir)

    if tf.io.gfile.exists(saved_model_export_dir):
        logging.warning('Found existing artefacts in {}, clearing old files')
        tf.io.gfile.rmtree(saved_model_export_dir)

    executor.dump_config(
        os.path.normpath(os.path.join(export_dir, 'config.json')))

    executor.restore_status.assert_consumed()

    if params.training.optimizer.use_moving_average:
        non_averaged_weights = executor.assign_moving_averaged_weights()

        if FLAGS.ignore_moving_average_weights:
            logging.warning('Loading back non averaged weights into model')
            executor.model.set_weights(non_averaged_weights)

    if FLAGS.export_h5:
        latest_checkpoint = os.path.basename(
            tf.train.latest_checkpoint(executor.model_dir))
        export_file_path = os.path.join(export_dir, latest_checkpoint + '.h5')
        logging.info(
            'Exporting weights in h5 format to {}'.format(export_file_path))
        executor.model.save_weights(export_file_path)

    if FLAGS.export_checkpoint:
        latest_checkpoint = os.path.basename(
            tf.train.latest_checkpoint(executor.model_dir))
        export_file_path = os.path.join(export_dir, latest_checkpoint)
        logging.info(
            'Exporting weights in tensorflow checkpoint format to {}'
            .format(export_file_path))
        executor.model.save_weights(export_file_path)

    if FLAGS.export_saved_model:
        logging.info('Exporting `saved_model` to {}'.format(FLAGS.export_dir))

        serving_fn_input_signature = {
            'image':
                tf.TensorSpec(
                    shape=[
                        params.inference.batch_size,
                        *params.input.input_shape,
                        params.input.channels],
                    name='image',
                    dtype=tf.float32)
            }

        preprocessing_fn_input_signature = {
            'image':
                tf.TensorSpec(shape=[None, None, 3], name='image', dtype=tf.float32)
        }

        preprocessing_pipeling = PreprocessingPipeline(
            params.input.input_shape,
            params.dataloader_params)

        inference_model = model_builder.prepare_model_for_export(
            executor.model, mode=FLAGS.mode)

        @tf.function
        def prepare_image(sample):
            image_dict = preprocessing_pipeling.normalize_and_resize_with_pad(
                image=sample['image'])
            return {
                'image': tf.expand_dims(image_dict['image'], axis=0)
            }

        @tf.function
        def serving_fn(sample):
            return inference_model.call(sample['image'], training=False)

        frozen_serving_fn, _ = convert_variables_to_constants_v2_as_graph(
            serving_fn.get_concrete_function(serving_fn_input_signature),
            aggressive_inlining=True)

        class InferenceModule(tf.Module):
            def __init__(self, inference_function, skip_nms):
                super(InferenceModule, self).__init__(name='inference_module')
                self.inference_function = inference_function
                self.skip_nms = skip_nms

            @tf.function
            def run_inference(self, sample):
                raw_outputs = self.inference_function(**sample)
                outputs = {}
                if not self.skip_nms:
                    outputs.update({
                        'boxes': raw_outputs[0],
                        'scores': raw_outputs[2],
                        'classes': raw_outputs[1],
                        'valid_detections': raw_outputs[3]})
                else:
                    outputs.update({
                        'boxes': raw_outputs[0],
                        'scores': raw_outputs[1]})
                return outputs

        # NMS is skipped for onnx_tensorrt mode,
        # since we run NMS withe TensorRT Plugin
        inference_module = InferenceModule(
            inference_function=frozen_serving_fn,
            skip_nms=FLAGS.mode == 'onnx_tensorrt')

        signatures = {
            'serving_default': inference_module.run_inference.get_concrete_function(
                serving_fn_input_signature)
        }

        if not FLAGS.skip_prepare_image_fn and 'tf' in FLAGS.mode:
            signatures['prepare_image'] = prepare_image.get_concrete_function(
                preprocessing_fn_input_signature)
        else:
            logging.warning('Skipping `prepare_image` signature in `saved_model`')

        for _signature_name, _concrete_fn in signatures.items():
            input_shapes = {x.name.split(':')[0]: x.shape.as_list()
                            for x in _concrete_fn.inputs}

            output_shapes = {k: v.as_list()
                             for k, v in _concrete_fn.output_shapes.items()}
            logging.info(
                '\nSignature: {}\n Input Shapes:\n {}\nOutput Shapes:\n{}'.format(
                    _signature_name,
                    input_shapes,
                    output_shapes))

        if 'tf' in FLAGS.mode:
            tf.saved_model.save(
                obj=inference_module,
                export_dir=saved_model_export_dir,
                signatures=signatures,
                options=tf.saved_model.SaveOptions(
                    experimental_custom_gradients=False))

        if 'onnx' in FLAGS.mode:
            onnx_utils.save_concrete_function(
                function=inference_module.run_inference,
                input_signature=[serving_fn_input_signature],
                add_nms_plugin=FLAGS.mode == 'onnx_tensorrt',
                model_params=params,
                output_dir=saved_model_export_dir,
                opset=13,
                simplify=True,
                debug=FLAGS.debug)

            if 'tensorrt' in FLAGS.mode:
                if FLAGS.precision == 'int8':
                    image_params = params.dataloader_params.preprocessing
                    calibration_image_paths = []
                    for ext in ['.png', '.jpg', '.jpeg']:
                        calibration_image_paths.extend(
                            glob(os.path.join(
                                FLAGS.calibration_images_dir, '*', ext)))

                    image_generator = ImageGenerator(
                        image_paths=calibration_image_paths,
                        max_images=FLAGS.calibration,
                        batch_size=FLAGS.calibration_batch_size,
                        target_shape=params.input.input_shape,
                        channel_mean=image_params.mean,
                        channel_stddev=image_params.stddev,
                        pixel_scale=image_params.pixel_scale)
                    calibrator = get_calibrator(
                        method=FLAGS.calibration_method,
                        image_generator=image_generator,
                        cache_file_path=os.path.join(
                            saved_model_export_dir, 'trt.cache'))
                else:
                    calibrator = None

                trt_builder = TensorRTBuilder(
                    onnx_path=os.path.join(
                        saved_model_export_dir, 'model.onnx'),
                    engine_path=os.path.join(
                        saved_model_export_dir, 'model.trt'),
                    workspace=4,
                    precision=FLAGS.precision,
                    calibrator=calibrator,
                    debug=FLAGS.debug)
                trt_builder.build_engine()


if __name__ == '__main__':
    app.run(main)
