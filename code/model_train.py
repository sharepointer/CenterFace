import tensorflow as tf
import tf_slim as slim
import numpy as np
import pickle
import os
import time
import argparse

import net_factory
import loss_builder
from dataset_provider import DataProviderSet
from config import _C as CONFIG

"""
parameter
"""
cfg_dataset_root = CONFIG.TRAIN.DATA_ROOT
cfg_train_input_size = CONFIG.TRAIN.INPUT_SIZE
cfg_train_epoch_size = CONFIG.TRAIN.EPOCH_SIZE
cfg_train_batch_size = CONFIG.TRAIN.BATCH_SIZE
cfg_train_angle_list = CONFIG.TRAIN.ANGLE_LIST
cfg_train_learning_rate = CONFIG.TRAIN.LEARNING_RATE
cfg_train_continue = CONFIG.TRAIN.CONTINUE
cfg_loss_class_weight = CONFIG.LOSS.CLASS_WEIGHT
cfg_loss_offset_weight = CONFIG.LOSS.OFFSET_WEIGHT
cfg_loss_scale_weight = CONFIG.LOSS.SCALE_WEIGHT
cfg_loss_landmark_weight = CONFIG.LOSS.LANDMARK_WEIGHT

"""
dataset
"""
print('*' * 80)
dataset_root = cfg_dataset_root
train_dataset = []
for item in os.listdir(dataset_root):
    file_path = os.path.join(dataset_root, item.strip())
    if os.path.isdir(file_path):
        continue
    with open(file_path, 'rb') as fid:
        dataset_ = pickle.load(fid)
        print(len(dataset_))
        train_dataset += dataset_
print('total: ', len(train_dataset))
print('*' * 80)

train_data = DataProviderSet(train_dataset,
                             img_size=cfg_train_input_size,
                             epoch_size=cfg_train_epoch_size,
                             batch_size=cfg_train_batch_size,
                             angle_list=cfg_train_angle_list)

"""
Model
"""
batch_size = len(cfg_train_angle_list) * cfg_train_batch_size
input_image = tf.placeholder(tf.float32, (batch_size, None, None, 3), name='input_image')
input_class = tf.placeholder(tf.float32, (batch_size, None, None, 1 + 1 + 1), name='input_class')
input_scale = tf.placeholder(tf.float32, (batch_size, None, None, 1 + 3), name='input_scale')
input_offset = tf.placeholder(tf.float32, (batch_size, None, None, 1 + 2), name='input_offset')
input_landmark = tf.placeholder(tf.float32, (batch_size, None, None, 1 + 10), name='input_landmark')

map_class, map_scale, map_offset, map_landmark = net_factory.build(input_image, is_training=True)

"""
Loss
"""
loss_class_op, debug_info = loss_builder.loss_center(input_class, map_class)
loss_scale_op = loss_builder.loss_scale(input_scale, map_scale)
loss_offset_op = loss_builder.loss_offset(input_offset, map_offset)
loss_landmark_op = loss_builder.loss_landmark(input_landmark, map_landmark)
loss_l2_op = tf.add_n(slim.losses.get_regularization_losses())

loss_l2_weight = 0
# if cfg_train_continue:
#     loss_l2_weight = 1

loss_all_op = cfg_loss_class_weight * loss_class_op + \
              cfg_loss_scale_weight * loss_scale_op + \
              cfg_loss_offset_weight * loss_offset_op + \
              cfg_loss_landmark_weight * loss_landmark_op + \
              loss_l2_weight * loss_l2_op

tf.summary.scalar('loss_class', loss_class_op)
tf.summary.scalar('loss_scale', loss_scale_op)
tf.summary.scalar('loss_offset', loss_offset_op)
tf.summary.scalar('loss_landmark', loss_landmark_op)
tf.summary.scalar('loss_l2', loss_l2_op)
tf.summary.scalar('loss_all', loss_all_op)
summary_op = tf.summary.merge_all()

"""
Optimizer
"""
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = tf.train.AdamOptimizer(cfg_train_learning_rate).minimize(loss_all_op)

"""
Session
"""
saver = tf.train.Saver(max_to_keep=5)

gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.8
with tf.Session(config=gpu_config) as sess:
    summary_writer = tf.summary.FileWriter('../logs', sess.graph)
    sess.run(tf.global_variables_initializer())
    train_data.init(sess)

    global_step = 0
    new_checkpoint = None
    if cfg_train_continue:
        new_checkpoint = tf.train.latest_checkpoint('../checkpoints/checkpoints')
    if new_checkpoint:
        exclusions = ['global_step']
        net_except_logits = slim.get_variables_to_restore(exclude=exclusions)
        init_fn = slim.assign_from_checkpoint_fn(new_checkpoint, net_except_logits, ignore_missing_vars=True)
        init_fn(sess)
        print('load params from {}'.format(new_checkpoint))

    try:
        cur_epoch = 1
        while True:
            t0 = time.time()
            batch_x_img, batch_center_map, batch_scale_map, batch_offset_map, batch_landmark_map = train_data.batch(sess)
            t1 = time.time()
            debug_info_, train_loss_, loss_class_, loss_scale_, loss_offset_, loss_landmark_, loss_l2_, summary_, _ = sess.run(
                [debug_info, loss_all_op, loss_class_op, loss_scale_op, loss_offset_op, loss_landmark_op, loss_l2_op, summary_op,
                 train_op],
                feed_dict={input_image: batch_x_img,
                           input_class: batch_center_map,
                           input_scale: batch_scale_map,
                           input_offset: batch_offset_map,
                           input_landmark: batch_landmark_map})
            t2 = time.time()
            global_step += 1
            print('epoch: {: <6d} global_step: {: <6d} train_loss: {: <12.3f} net: {: <12.3f} data: {: <12.3f} debug_info: {:.3f}'.format(train_data.curEpoch(),
                                                                                                                                          global_step,
                                                                                                                                          train_loss_,
                                                                                                                                          (t2 - t1) * 1000,
                                                                                                                                          (t1 - t0) * 1000,
                                                                                                                                          debug_info_))
            summary_writer.add_summary(summary_, global_step)
            if 0 == global_step % 1000:
                saver.save(sess, '../checkpoints/checkpoints/model.ckpt-{}'.format(global_step))
    except tf.errors.OutOfRangeError:
        print('train stop.')
