import os

from absl import app, flags, logging
from google.cloud import storage
from retinanet.cfg import Config

flags.DEFINE_string(
    name='config_path',
    default=None,
    help='Path to the config file')

flags.DEFINE_boolean(
    name='debug',
    default=False,
    help='Print debugging info')

flags.DEFINE_string(
    name='checkpointed_at',
    default='final',
    help='Step to download checkpoint from')

flags.DEFINE_string(
    name='local_dir',
    default='null',
    help='Local directory to download the weights into')

FLAGS = flags.FLAGS


def main(_):
    logging.set_verbosity(logging.DEBUG if FLAGS.debug else logging.INFO)

    params = Config(FLAGS.config_path).params

    local_dir = os.path.join(FLAGS.local_dir, params.experiment.name)
    gcs_dir = '/'.join([params.experiment.model_dir, params.experiment.name])

    *_, bucket_name, weights_dir, model_dir = gcs_dir.split('/')
    logging.info('Looking for weights in {}'.format(gcs_dir))

    if not os.path.exists(local_dir):
        logging.info('Creating local directory: {}'.format(local_dir))
        os.makedirs(local_dir)

    logging.info('Downloading weights for experiment: {} from {} to {}'.format(
        params.experiment.name, gcs_dir, local_dir))

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    for blob in bucket.list_blobs(prefix=weights_dir + '/' + model_dir):
        if 'checkpoint' in blob.name \
            or FLAGS.checkpointed_at in blob.name \
                or blob.name.endswith('.json'):
            file_name = os.path.join(local_dir, os.path.basename(blob.name))
            try:
                blob.download_to_filename(file_name)
                logging.info('Successfully downloaded remote file: {} to {}'.format(
                    blob.name, file_name))
            except:  # noqa: E722
                logging.warning('Failed download remote file: {}'.format(blob.name))


if __name__ == '__main__':
    app.run(main)
