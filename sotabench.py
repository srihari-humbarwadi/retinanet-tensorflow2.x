import os
from zipfile import ZipFile

import tensorflow as tf
from absl import app, logging, flags
from sotabencheval.object_detection import COCOEvaluator
from sotabencheval.utils import is_server

from retinanet.evaluate_saved_model import evaluate

flags.DEFINE_string(
    name='model_name',
    default='mscoco-retinanet-resnet50-640x640-30x-64',
    help='Model to benchmark')

_MODEL_ZOO = {
    'mscoco-retinanet-resnet50-640x640-30x-64': 'https://github.com/srihari-humbarwadi/retinanet-tensorflow2.x/releases/download/v0.1.0/mscoco-retinanet-resnet50-640x640-30x-64.zip',
    'mscoco-retinanet-resnet50-640x640-3x-256': 'https://github.com/srihari-humbarwadi/retinanet-tensorflow2.x/releases/download/v0.1.0/mscoco-retinanet-resnet50-640x640-3x-256.zip'
}

_MODEL_DIR = os.path.join(os.getcwd(), 'model_files', 'saved_models')

FLAGS = flags.FLAGS


def format_model_name(model_name):
    *_, backbone, resolution, schedule, _ = model_name.split('-')
    return ' '.join([backbone, resolution, schedule])


def maybe_download_saved_model(model_name):
    try:
        url = _MODEL_ZOO[model_name]

    except KeyError:
        raise ValueError('Invalid model_name: {} requested, avaliable models are:\n{}'
                         .format(model_name, list(_MODEL_ZOO.keys())))

    if not os.path.exists(_MODEL_DIR):
        logging.info('Unable to find MODEL_DIR, creating at {}'.format(_MODEL_DIR))
        os.makedirs(_MODEL_DIR)

    saved_model_dir = os.path.join(_MODEL_DIR, model_name)

    if not os.path.exists(saved_model_dir):
        filename = os.path.normpath(os.path.join(_MODEL_DIR, os.path.basename(url)))
        tf.keras.utils.get_file(filename, origin=url)
        logging.info('Download file: {} from: {}'.format(filename, url))

    with ZipFile(filename, 'r') as z_f:
        z_f.extractall(_MODEL_DIR)
    logging.info('Extracted file: {} to: {}'.format(filename, saved_model_dir))


def main(_):
    logging.info('Benchmarking model: {}'.format(FLAGS.model_name))

    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        print('Found {} GPU(s)'.format(len(gpus)))
        [tf.config.experimental.set_memory_growth(device, True) for device in gpus]
    else:
        logging.warning('No GPU\'s found, running CPU')

    if is_server():
        data_root = './.data/vision/coco'

    else:
        data_root = '../../../datasets/coco2017'

    maybe_download_saved_model(FLAGS.model_name)

    evaluator = COCOEvaluator(
        root=data_root,
        model_name='RetinaNet {}'.format(format_model_name(FLAGS.model_name)),
        paper_arxiv_id='1703.06870')

    model = tf.saved_model.load(os.path.join(_MODEL_DIR, FLAGS.model_name))

    prepare_image_fn = model.signatures['prepare_image']
    serving_fn = model.signatures['serving_default']

    predictions = evaluate(
        prepare_image_fn,
        serving_fn,
        coco_data_directory=data_root,
        prediction_file_path='predictions.json',
        return_predictions_only=True)

    evaluator.detections = []
    evaluator.add(predictions)
    evaluator.save()


if __name__ == '__main__':
    app.run(main)
