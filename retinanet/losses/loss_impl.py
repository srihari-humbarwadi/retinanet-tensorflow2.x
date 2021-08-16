import tensorflow as tf


class FocalLoss(tf.losses.Loss):

    def __init__(self, alpha, gamma, label_smoothing):
        self._alpha = alpha
        self._gamma = gamma
        self._label_smoothing = label_smoothing

        super(FocalLoss, self).__init__(
            name='focal_loss',
            reduction='sum')

    def call(self, y_true, y_pred):
        y_true_smoothened = (
            y_true * (1.0 - self._label_smoothing) +
            0.5 * self._label_smoothing)
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_true_smoothened,
            logits=y_pred)

        probs = tf.nn.sigmoid(y_pred)
        alpha = tf.where(tf.equal(y_true, 1.0), self._alpha,
                         (1.0 - self._alpha))
        pt = tf.where(tf.equal(y_true, 1.0), probs, 1.0 - probs)
        loss = alpha * tf.pow(1.0 - pt, self._gamma) * cross_entropy
        return loss


class ClassLoss:

    def __init__(self, num_classes, params):
        self._focal_loss = FocalLoss(
            params.alpha,
            params.gamma,
            params.label_smoothing)

        self._num_classes = num_classes

    def __call__(self, targets, predictions):
        loss = []
        for key in targets.keys():
            N, H, W, num_anchors = \
                targets[key].get_shape().as_list()
            one_hot_shape = [N, H, W, num_anchors * self._num_classes]

            # targets[key].shape  (N, H, W, num_anchors)
            # y_true.shape == (N, H, W, num_anchors * num_classes)
            # y_pred.shape == (N, H, W, num_anchors * num_classes)
            # sample_weight.shape == (N, H, W, num_anchors * num_classes)
            y_true = tf.one_hot(
                tf.cast(targets[key], dtype=tf.int32),
                depth=self._num_classes)
            y_true = tf.reshape(y_true,
                                shape=one_hot_shape)
            y_pred = predictions[key]

            # ignore_mask.shape == (N, H, W, num_anchors)
            ignore_mask = tf.cast(
                tf.not_equal(targets[key], -2.0),
                dtype=tf.float32)
            # ignore_mask.shape == (N, H, W, num_anchors, 1)
            ignore_mask = tf.expand_dims(ignore_mask, axis=-1)

            # ignore_mask.shape == (N, H, W, num_anchors, self._num_classes)
            ignore_mask = tf.tile(ignore_mask,
                                  multiples=[1, 1, 1, 1, self._num_classes])
            # ignore_mask.shape == (N, H, W, num_anchors * self._num_classes)
            ignore_mask = tf.reshape(ignore_mask, shape=one_hot_shape)

            current_level_loss = self._focal_loss(
                y_true=y_true,
                y_pred=y_pred,
                sample_weight=ignore_mask)
            loss.append(current_level_loss)
        return tf.math.add_n(loss)


class BoxLoss:

    def __init__(self, params):
        self._smooth_l1_loss = tf.losses.Huber(
            delta=params.delta,
            name='smooth_l1_loss',
            reduction='sum')

    def __call__(self, targets, predictions):
        loss = []
        for key in targets.keys():
            # y_true.shape == (N, H, W, num_anchors * 4, 1)
            # y_pred.shape == (N, H, W, num_anchors * 4, 1)
            # sample_weight.shape == (N, H, W, num_anchors * 4, 1)
            y_true = tf.expand_dims(targets[key], axis=-1)
            y_pred = tf.expand_dims(predictions[key], axis=-1)
            sample_weight = tf.not_equal(y_true, 0.0)

            current_level_loss = self._smooth_l1_loss(
                y_true=y_true,
                y_pred=y_pred,
                sample_weight=sample_weight)
            loss.append(current_level_loss)

        # averged loss across (4) box coordinates
        return tf.math.add_n(loss) / 4.0
