from absl import app, flags

from retinanet.dataset_utils.coco_converter import COCOConverter

flags.DEFINE_string('parsed_dataset_json',
                    default=None,
                    help='Path to the parsed dataset json')

flags.DEFINE_string('label_map',
                    default=None,
                    help='Path to the parsed label map')

flags.DEFINE_string('output_dir',
                    default='./converted_dataset',
                    help='Path to store the converted jsons in.')

flags.DEFINE_boolean('only_val',
                     default=False,
                     help='Only convert validation split from dataset')

FLAGS = flags.FLAGS


def main(_):
    converter = COCOConverter(
        FLAGS.parsed_dataset_json,
        FLAGS.label_map,
        FLAGS.output_dir,
        FLAGS.only_val
    )
    converter.convert()


if __name__ == '__main__':
    app.run(main)
