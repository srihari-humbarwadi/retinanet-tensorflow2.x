import os
from glob import glob

from absl import app, flags, logging
from retinanet.cfg import Config
from retinanet.image_utils import ImageGenerator
from retinanet.tensorrt.builder import TensorRTBuilder
from retinanet.tensorrt.calibrator import get_calibrator

flags.DEFINE_string(
    name='config_path',
    default=None,
    required=True,
    help='Path to the config file')


flags.DEFINE_string(
    name='export_dir',
    default='export',
    help='Path to store the model artefacts')

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

flags.DEFINE_integer(
    name='dla_core',
    default=None,
    help='DLA Core to run the engine on')

flags.DEFINE_string(
    name='precision',
    default='fp32',
    help='Execution precision for TensorRT Engines')

FLAGS = flags.FLAGS


def main(_):
    logging.set_verbosity(logging.DEBUG if FLAGS.debug else logging.INFO)

    params = Config(FLAGS.config_path).params

    export_dir = os.path.join(
        FLAGS.export_dir, params.experiment.name, 'onnx_tensorrt')

    if FLAGS.precision == 'int8':
        image_params = params.dataloader_params.preprocessing
        calibration_image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            calibration_image_paths.extend(
                glob(os.path.join(
                    FLAGS.calibration_images_dir, ext)))

        image_generator = ImageGenerator(
            image_paths=calibration_image_paths,
            max_images=FLAGS.num_calibration_images,
            batch_size=FLAGS.calibration_batch_size,
            target_shape=params.input.input_shape,
            channel_mean=image_params.mean,
            channel_stddev=image_params.stddev,
            pixel_scale=image_params.pixel_scale)

        calibrator = get_calibrator(
            method=FLAGS.calibration_method,
            image_generator=image_generator,
            cache_file_path=os.path.join(export_dir, 'trt.cache'))
    else:
        calibrator = None

    trt_builder = TensorRTBuilder(
        onnx_path=os.path.join(
            export_dir, 'model.onnx'),
        engine_path=os.path.join(
            export_dir, 'model.trt'),
        workspace=4,
        precision=FLAGS.precision,
        calibrator=calibrator,
        dla_core=FLAGS.dla_core,
        debug=FLAGS.debug)
    trt_builder.build_engine()


if __name__ == '__main__':
    app.run(main)
