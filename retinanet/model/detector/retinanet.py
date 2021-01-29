import tensorflow as tf

from retinanet.model.builder import DETECTOR


@DETECTOR.register_module('retinanet')
class RetinaNet(tf.keras.Model):
    """ RetinaNet detector class. """

    def __init__(self, backbone, neck, box_head, class_head, **kwargs):
        image_inputs = tf.keras.Input(shape=backbone.input.shape[1:], name="images")

        features = backbone(image_inputs)
        neck_features = neck(features)
        box_outputs = box_head(neck_features)
        class_outputs = class_head(neck_features)

        outputs = {
            'class-predictions': class_outputs,
            'box-predictions': box_outputs
        }

        super(RetinaNet, self).__init__(
            inputs=[image_inputs],
            outputs=outputs,
            name='retinanet', **kwargs)
