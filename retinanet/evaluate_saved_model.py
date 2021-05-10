import os
from time import time

import numpy as np
import tensorflow as tf
from absl import app, flags, logging

from retinanet.dataset_utils.coco_parser import CocoParser
from retinanet.eval import COCOEvaluator
from retinanet.image_utils import read_image
from retinanet.utils import AverageMeter, format_eta

flags.DEFINE_string(
    name='saved_model_path',
    default=None,
    help='Path to `saved_model`')

flags.DEFINE_boolean(
    name='debug',
    default=False,
    help='Print debugging info')

flags.DEFINE_string(
    name='annotation_file_path',
    default='annotations/instances_val2017.json',
    help='Path to annotations json')

flags.DEFINE_string(
    name='prediction_file_path',
    default='predictions.json',
    help='Path to predictions json')


try:
    import tensorrt as trt
    trt.init_libnvinfer_plugins(None, '')
    os.environ['TF_TRT_ALLOW_NMS_TOPK_OVERRIDE'] = '1'

except ImportError:
    logging.warning('TensorRT not installed')


gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print('Found {} GPU(s)'.format(len(gpus)))
    [tf.config.experimental.set_memory_growth(device, True) for device in gpus]
else:
    logging.warning('No GPU\'s found, running CPU')

FLAGS = flags.FLAGS


def main(_):
    model = tf.saved_model.load(FLAGS.saved_model_path)
    prepare_image_fn = model.signatures['prepare_image']
    serving_fn = model.signatures['serving_default']

    coco_parser = CocoParser('.', only_val=True)
    coco_evaluator = COCOEvaluator(None,
                                   FLAGS.annotation_file_path,
                                   FLAGS.prediction_file_path)

    fps_meter = AverageMeter(name='fps', momentum=0.975)
    num_samples = len(coco_parser.dataset['val'])

    for idx, sample in enumerate(coco_parser.dataset['val']):
        t1 = time()
        image = read_image(tf.constant(sample['image']))
        t2 = time()
        serving_input = prepare_image_fn(image=image,
                                         image_id=tf.constant(sample['image_id'],
                                                              dtype=tf.int32))
        t3 = time()
        detections = serving_fn(**serving_input)
        t4 = time()
        fps_meter.accumulate(1 / (t4 - t3))
        fps = fps_meter.averaged_value

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
    print()
    coco_evaluator.evaluate()


if __name__ == '__main__':
    app.run(main)