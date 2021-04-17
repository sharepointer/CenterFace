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

    out_loss = 3 * pos_loss + neg_loss
    debug_info = pos_loss

    return out_loss, debug_info


def loss_semantic(y_true, y_pred):
    y_true = y_true[:, :, :, 3:]

    y_true = tf.cast(y_true, dtype=tf.int32)
    class_num = int(y_pred.shape.as_list()[3])
    y_true = tf.reshape(y_true, shape=[-1, ])
    y_weight = tf.cast(tf.not_equal(y_true, 255), tf.float32)
    y_temp = tf.zeros_like(y_true)
    y_true = tf.where(tf.not_equal(y_true, 255), y_true, y_temp)
    y_pred = tf.reshape(y_pred, shape=[-1, class_num])

    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred) * y_weight
    all_num = tf.reduce_sum(y_weight)
    pos_mask = tf.cast(tf.greater(y_true, 0), dtype=tf.float32)
    pos_num = tf.reduce_sum(pos_mask)
    neg_num = all_num - pos_num
    loss_pos = tf.reduce_mean(loss_) * (neg_num / all_num)
    loss_neg = tf.reduce_mean(loss_) * (pos_num / all_num)
    loss_value_0 = loss_pos + loss_neg

    y_true_one = tf.cast(tf.one_hot(y_true, class_num, on_value=1, off_value=0), tf.float32)
    y_pred_score = tf.nn.softmax(y_pred)
    loss_all = ((1 - y_pred_score) ** 2) * (-1 * tf.math.log(tf.clip_by_value(y_pred_score, 1e-10, 1.0)) * y_true_one)
    loss_value_1 = tf.reduce_mean(tf.reduce_sum(loss_all, axis=-1) * y_weight)

    return (loss_value_0 + loss_value_1) * 10


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
