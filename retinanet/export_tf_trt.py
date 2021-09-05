import os
from glob import glob

import tensorflow as tf
from absl import app, flags, logging
from tqdm import tqdm

from retinanet.cfg import Config
from retinanet.dataloader.preprocessing_pipeline import PreprocessingPipeline
from retinanet.image_utils import read_image

tf.get_logger().propagate = False
tf.config.set_soft_device_placement(True)
tf.config.optimizer.set_jit(True)

flags.DEFINE_string(
    name='config_path',
    default=None,
    required=True,
    help='Path to the config file')

flags.DEFINE_string(
    name='saved_model_path',
    default=None,
    required=True,
    help='Path to the `saved_model`')

flags.DEFINE_string(
    name='precision',
    default=None,
    required=True,
    help='Execution precision for TensorRT Engines')

flags.DEFINE_string(
    name='calibration_images',
    default='coco/val2017',
    help='Calibration data when exporting in INT8 precision')

flags.DEFINE_integer(
    name='calibration_steps',
    default=100,
    help='Number of calibration steps')

flags.DEFINE_string(
    name='export_dir',
    default='model_files/tensorrt',
    help='Path to store the TensorRT model artefacts')

flags.DEFINE_integer(
    name='minimum_segment_size',
    default=3,
    help='minimum number of nodes required for a subgraph to be replaced by '
         'TRTEngineOp')

flags.DEFINE_boolean(
    name='debug',
    default=False,
    help='Print debugging info')

FLAGS = flags.FLAGS


def get_input_fn(config_params, image_dir, steps=None):
    preprocessing_pipeling = PreprocessingPipeline(
        config_params.input.input_shape,
        config_params.dataloader_params)

    def _input_fn():
        image_paths = glob(os.path.join(image_dir, '*'))
        num_iterations = steps or len(image_paths)
        logging.info('Using {} out of {} images'.format(
            num_iterations, len(image_paths)))

        for idx in tqdm(range(num_iterations)):
            image = read_image(image_paths[idx])
            image_dict = preprocessing_pipeling.normalize_and_resize_with_pad(
                image=image)
            image = tf.expand_dims(image_dict['image'], axis=0)
            yield (image,)
    return _input_fn


def main(_):
    params = Config(FLAGS.config_path).params
    export_dir = os.path.join(FLAGS.export_dir, params.experiment.name)

    logging.info('Using `saved_model` from: {}'.format(FLAGS.saved_model_path))

    if FLAGS.debug:
        os.environ['TF_CPP_VMODULE'] = \
            'trt_logger=2,trt_engine_op=2,convert_nodes=2,convert_graph=2,segment=2,trt_shape_optimization_profiles=2,trt_engine_resource_ops=2'  # noqa: E501

    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        logging.info('Found {} GPU(s)'.format(len(gpus)))
        [tf.config.experimental.set_memory_growth(device, True) for device in gpus]

        import tensorrt as trt
        trt.init_libnvinfer_plugins(None, '')

        # Starting from tensorflow==2.5.0, TensorRT ignores the conversion
        # of combined_nms op if number of anchors is more than 4096.
        # Setting`TF_TRT_ALLOW_NMS_TOPK_OVERRIDE=1` will allow TensorRT to run
        # top_k filtering with k=4096 prior to running NMS.
        # refer https://github.com/tensorflow/tensorflow/issues/46453
        # and https://github.com/tensorflow/tensorflow/pull/47698
        os.environ['TF_TRT_ALLOW_NMS_TOPK_OVERRIDE'] = '1'

        logging.info('Successfully loaded TensorRT!!!')

    else:
        raise AssertionError('No GPU\'s found, cannot proceed')

    logging.info('Setting precision to {}'.format(FLAGS.precision))

    conversion_params = tf.experimental.tensorrt.ConversionParams(
        minimum_segment_size=FLAGS.minimum_segment_size,
        precision_mode=FLAGS.precision)

    converter = tf.experimental.tensorrt.Converter(
        input_saved_model_dir=FLAGS.saved_model_path,
        use_dynamic_shape=False,
        conversion_params=conversion_params)

    if FLAGS.precision == 'INT8':
        # To be used for INT8 calibration
        calibration_input_fn = get_input_fn(
            config_params=params,
            image_dir=FLAGS.calibration_images,
            steps=FLAGS.calibration_steps)
    else:
        calibration_input_fn = None

    # To be used for engine building
    input_fn = get_input_fn(
        config_params=params,
        image_dir=FLAGS.calibration_images,
        steps=10)

    converter.convert(calibration_input_fn=calibration_input_fn)
    logging.info('Done converting `saved_model`')

    converter.build(input_fn=input_fn)
    logging.info('Done building TensorRT engines')

    logging.info(
        'Saving {} converted saved_model and TensorRT engines to: {}'.format(
            FLAGS.precision,
            export_dir))

    converter.save(export_dir)


if __name__ == '__main__':
    app.run(main)
