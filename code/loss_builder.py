import tensorflow as tf


def loss_center(y_true, y_pred):
    classification_loss = -1 * y_true[:, :, :, 2] * tf.log(tf.clip_by_value(y_pred[:, :, :, 0], 1e-10, 1.0)) + \
                          -1 * (1 - y_true[:, :, :, 2]) * tf.log(tf.clip_by_value(1 - y_pred[:, :, :, 0], 1e-10, 1.0))

    positives = y_true[:, :, :, 2]
    negatives = y_true[:, :, :, 1] - y_true[:, :, :, 2]

    foreground_weight = positives * ((1.0 - y_pred[:, :, :, 0]) ** 2.0)
    background_weight = negatives * ((1.0 - y_true[:, :, :, 0]) ** 4.0) * (y_pred[:, :, :, 0] ** 2.0)

    assigned_boxes = tf.reduce_sum(y_true[:, :, :, 2])
    pos_loss = tf.reduce_sum(foreground_weight * classification_loss) / tf.maximum(1.0, assigned_boxes)
    neg_loss = tf.reduce_sum(background_weight * classification_loss) / tf.maximum(1.0, assigned_boxes)

    out_loss = 1 * pos_loss + neg_loss
    debug_info = pos_loss

    return out_loss, debug_info


def loss_scale(y_true, y_pred):
    absolute_loss = tf.abs(y_true[:, :, :, :2] - y_pred[:, :, :, :])
    square_loss = 0.5 * ((y_true[:, :, :, :2] - y_pred[:, :, :, :]) ** 2)
    loss = y_true[:, :, :, 2] * tf.reduce_sum(tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5), axis=-1)
    assigned_boxes = tf.reduce_sum(y_true[:, :, :, 2])
    out_loss = tf.reduce_sum(loss) / tf.maximum(1.0, assigned_boxes)

    return out_loss


def loss_landmark(y_true, y_pred):
    absolute_loss = tf.abs(y_true[:, :, :, :10] - y_pred[:, :, :, :])
    square_loss = 0.5 * ((y_true[:, :, :, :10] - y_pred[:, :, :, :]) ** 2)
    loss = y_true[:, :, :, 10] * tf.reduce_sum(tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5), axis=-1)
    assigned_boxes = tf.reduce_sum(y_true[:, :, :, 10])
    out_loss = tf.reduce_sum(loss) / tf.maximum(1.0, assigned_boxes)

    return out_loss


def loss_offset(y_true, y_pred):
    absolute_loss = tf.abs(y_true[:, :, :, :2] - y_pred[:, :, :, :])
    square_loss = 0.5 * ((y_true[:, :, :, :2] - y_pred[:, :, :, :]) ** 2)
    loss = y_true[:, :, :, 2] * tf.reduce_sum(tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5), axis=-1)
    assigned_boxes = tf.reduce_sum(y_true[:, :, :, 2])
    out_loss = tf.reduce_sum(loss) / tf.maximum(1.0, assigned_boxes)

    return out_loss
