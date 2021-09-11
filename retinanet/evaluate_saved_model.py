import os
from time import time

import numpy as np
import tensorflow as tf
from absl import app, flags, logging

from retinanet.dataset_utils.coco_parser import CocoParser
from retinanet.eval import COCOEvaluator
from retinanet.image_utils import read_image
from retinanet.utils import AverageMeter, format_eta

os.environ['TF_XLA_FLAGS'] = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
tf.config.optimizer.set_jit(True)

flags.DEFINE_string(
    name='saved_model_path',
    default=None,
    help='Path to `saved_model`')

flags.DEFINE_boolean(
    name='debug',
    default=False,
    help='Print debugging info')

flags.DEFINE_string(
    name='coco_data_directory',
    default='.',
    help='Path to annotations json')

flags.DEFINE_string(
    name='prediction_file_path',
    default='predictions.json',
    help='Path to dump predictions json')

flags.DEFINE_boolean(
    name='remap_class_ids',
    default=False,
    help='Enables remapping of class ids. Use only if the model was trained '
    'with `remap_class_ids=True` in dataset parser')


FLAGS = flags.FLAGS


def evaluate(
        prepare_image_fn,
        serving_fn,
        coco_data_directory,
        prediction_file_path,
        return_predictions_only=False):

    coco_parser = CocoParser(coco_data_directory, only_val=True)
    coco_evaluator = COCOEvaluator(
        input_shape=None,
        annotation_file_path=coco_parser.val_annotations_path,
        prediction_file_path=prediction_file_path,
        remap_class_ids=FLAGS.remap_class_ids)

    fps_meter = AverageMeter(name='fps', momentum=0.975)
    num_samples = len(coco_parser.dataset['val'])

    for idx, sample in enumerate(coco_parser.dataset['val']):
        t1 = time()
        image = read_image(tf.constant(sample['image']))
        t2 = time()
        serving_input = prepare_image_fn(image=image)
        t3 = time()
        detections = serving_fn(
            image=serving_input['image'])
        t4 = time()
        fps_meter.accumulate(1 / (t4 - t3))
        fps = fps_meter.averaged_value

        detections['boxes'] *= tf.cast(tf.reduce_max(image.shape), dtype=tf.float32)

        coco_evaluator.accumulate_results(
            {
                'image_id': [sample['image_id']],
                'detections': detections,
                'resize_scale': None
            },
            rescale_detections=False)

        eta = format_eta((num_samples - (idx + 1)) / fps)
        completed = int((idx + 1) / num_samples * 10)
        progress = '=' * completed + ' ' * (10 - completed)
        print(
            '\r[{}] Processed {:04}/{} images | ETA: {} | FPS(detection): {:.1f}/sec | forward_pass_with_nms: {:03} ms | image_reading: {:03} ms | image_preprocessing: {:03} ms'  # noqa: E501
            .format(
                progress,
                idx + 1,
                num_samples,
                eta,
                fps,
                int(np.round(1000 * (t4 - t3))),
                int(np.round(1000 * (t2 - t1))),
                int(np.round(1000 * (t3 - t2)))),
            end='')

    if return_predictions_only:
        return coco_evaluator.processed_detections

    print()
    coco_evaluator.evaluate()


def main(_):
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        logging.info('Found {} GPU(s)'.format(len(gpus)))
        [tf.config.experimental.set_memory_growth(device, True) for device in gpus]
        try:
            # Attempt to load tensorrt, only used if the saved_model contains
            # TensorRT engines.
            import tensorrt as trt
            trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.WARNING), '')

            # Starting from tensorflow==2.5.0, TensorRT ignores the conversion
            # of combined_nms op if number of anchors is more than 4096.
            # Setting`TF_TRT_ALLOW_NMS_TOPK_OVERRIDE=1` will allow TensorRT to run
            # top_k filtering with k=4096 prior to running NMS.
            # refer https://github.com/tensorflow/tensorflow/issues/46453
            # and https://github.com/tensorflow/tensorflow/pull/47698
            os.environ['TF_TRT_ALLOW_NMS_TOPK_OVERRIDE'] = '1'
            logging.info(
                'Successfully loaded TensorRT {}'.format(trt.__version__))

        except ImportError:
            logging.warning('TensorRT not installed')
    else:
        logging.warning('No GPU\'s found, running on CPU')

    model = tf.saved_model.load(FLAGS.saved_model_path)
    prepare_image_fn = model.signatures['prepare_image']
    serving_fn = model.signatures['serving_default']

    logging.info('Successfully loaded `saved_model` from: {}'.format(
        FLAGS.saved_model_path))

    evaluate(
        prepare_image_fn,
        serving_fn,
        FLAGS.coco_data_directory,
        FLAGS.prediction_file_path)


if __name__ == '__main__':
    app.run(main)
