import numpy as np
import tensorflow as tf


class AnchorBoxGenerator:
    '''Generates anchor boxes.

    This class has operations to generate anchor boxes for feature maps at
    strides `[8, 16, 32, 64, 128]`. Where each anchor each box is of the
    format `[x, y, width, height]`.

    Attributes:
      aspect_ratios: A list of float values representing the aspect ratios of
        the anchor boxes at each location on the feature map
      scales: A list of float values representing the scale of the anchor boxes
        at each location on the feature map.
      num_anchors: The number of anchor boxes at each location on feature map
      areas: A list of float values representing the areas of the anchor
        boxes for each feature map in the feature pyramid.
      strides: A list of float value representing the strides for each feature
        map in the feature pyramid.
    '''

    def __init__(self, img_h, img_w, min_level, max_level, params):
        self.image_height = img_h
        self.image_width = img_w

        self.areas = params.areas
        self.aspect_ratios = params.aspect_ratios
        self.scales = params.scales

        self._num_anchors = len(params.aspect_ratios) * len(params.scales)
        self._min_level = min_level
        self._max_level = max_level

        self._strides = [2**i for i in range(min_level, max_level+1)]
        self._anchor_dims = self._compute_dims()

        self._anchor_boundaries = self._compute_anchor_boundaries()
        self._boxes = self.get_anchors()

    def _compute_anchor_boundaries(self):
        boundaries = [0]
        for i in range(self._min_level, self._max_level + 1):
            num_anchors = int(
                np.ceil(self.image_height / 2**i) *
                np.ceil(self.image_width / 2**i) * self._num_anchors)
            boundaries += [boundaries[-1] + num_anchors]
        return boundaries

    def _compute_dims(self):
        '''Computes anchor dims for all ratios and scales at all levels'''
        anchor_dims_all = []
        for area in self.areas:
            anchor_dims = []
            for ratio in self.aspect_ratios:
                h = tf.math.sqrt(area / ratio)
                w = area / h
                wh = tf.reshape(tf.stack([w, h], axis=-1), [1, 1, 2])
                for scale in self.scales:
                    anchor_dims.append(scale * wh)
            anchor_dims_all.append(tf.stack(anchor_dims, axis=-2))
        return anchor_dims_all

    def _get_anchors(self, feature_height, feature_width, level):
        '''Generates anchor boxes for a given feature map size and level

        Arguments:
          feature_height: An integer representing the height of the feature map.
          feature_width: An integer representing the width of the feature map.
          level: An integer representing the level of the feature map in the
            feature pyramid.

        Returns:
          anchor boxes with the shape
          `(feature_height * feature_width * num_anchors, 4)`
        '''
        rx = tf.range(feature_width, dtype=tf.float32) + 0.5
        ry = tf.range(feature_height, dtype=tf.float32) + 0.5
        centers = tf.stack(tf.meshgrid(rx, ry), axis=-1) * \
            self._strides[level - self._min_level]
        centers = tf.expand_dims(centers, axis=-2)
        centers = tf.tile(centers, [1, 1, self._num_anchors, 1])
        wh = tf.tile(self._anchor_dims[level - self._min_level],
                     [feature_height, feature_width, 1, 1])
        anchors = tf.concat([centers, wh], axis=-1)
        return tf.reshape(
            anchors, [feature_height * feature_width * self._num_anchors, 4])

    def get_anchors(self):
        '''Generates anchor boxes for all the feature maps of the feature pyramid.

        Returns:
          anchor boxes for all the feature maps, stacked as a single tensor
            with shape `(total_anchors, 4)`
        '''
        anchors = [
            self._get_anchors(
                tf.math.ceil(self.image_height / 2**i),
                tf.math.ceil(self.image_width / 2**i),
                i,
            ) for i in range(self._min_level, self._max_level + 1)
        ]
        return tf.concat(anchors, axis=0)

    @property
    def anchor_boundaries(self):
        return self._anchor_boundaries

    @property
    def boxes(self):
        return self._boxes
