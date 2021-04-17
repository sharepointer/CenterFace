from __future__ import absolute_import
from __future__ import division
import numpy as np
import cv2
import os
import traceback
import random
import copy
import pickle
import queue
from multiprocessing import Event
import threading
import matplotlib.pyplot as plt

import tensorflow as tf

import data_augment as img_augment
import data_generate_gt
import util_box


def _process_one_image(src_data, train_size, angle=0, quantize_train=False):
    src_img = cv2.imread(src_data['filepath'])
    src_boxes = src_data['bboxes']
    src_landmarks = src_data['landmarks']

    temp_boxes = np.array(src_boxes, np.float).reshape([-1, 4])
    temp_landmarks = np.array(src_landmarks, np.float).reshape([-1, 10])
    temp_labels = np.ones(len(temp_boxes), dtype=np.int32).reshape([-1, 1])
    augment_fn = img_augment.Augmentation(train_size, angle=angle)
    img_crop, boxes_crop, landmarks_crop, labels_crop = augment_fn(src_img, temp_boxes, temp_landmarks, temp_labels)

    gt_boxes = []
    gt_landmark = []
    igs_boxes = []
    igs_landmark = []
    box_num = len(boxes_crop)
    for ind in range(box_num):
        label = labels_crop[ind]
        if -1 == label:
            igs_boxes.append(boxes_crop[ind])
            igs_landmark.append(landmarks_crop[ind])
        else:
            gt_boxes.append(boxes_crop[ind])
            gt_landmark.append(landmarks_crop[ind])

    gts = gt_boxes
    igs = igs_boxes
    landmarks = []
    for temp in gt_landmark:
        landmarks.append([temp[1], temp[3], temp[5], temp[7], temp[9],
                          temp[0], temp[2], temp[4], temp[6], temp[8]])

    gts = np.array(gts, dtype=np.int32).reshape([-1, 4])
    igs = np.array(igs, dtype=np.int32).reshape([-1, 4])
    landmarks = np.array(landmarks, dtype=np.int32).reshape([-1, 10])
    img_data = {'size': train_size, 'bboxes': gts, 'landmarks': landmarks, 'ignore_bboxes': igs}

    x_img = (img_crop - 127.5) / 128
    center_map, scale_map, offset_map, landmark_map = data_generate_gt.calc_gt_center(img_data, quantize=quantize_train)

    return x_img, center_map, scale_map, offset_map, landmark_map


class DataProvider(object):
    def __init__(self, dataset, img_size=320, epoch_size=1, batch_size=8, angle=0):
        self.dataset = copy.deepcopy(dataset)
        random.shuffle(self.dataset)
        self.batch_size = batch_size
        self.net_size = img_size
        self.angle = angle

        self.cur_id = 0
        self.global_step = 0
        self.max_step = (self.size() // batch_size * batch_size) * epoch_size
        self._getDataPipeline()

    def size(self):
        return len(self.dataset)

    def batchSize(self):
        return self.batch_size

    def curEpoch(self):
        return self.global_step // (self.size() // self.batch_size * self.batch_size) + 1

    def _processOneImage(self, index):
        return _process_one_image(self.dataset[index], self.net_size, self.angle)

    def _generator(self):
        while True:
            if self.global_step > self.max_step:
                break
            if self.cur_id == self.size():
                random.shuffle(self.dataset)
                self.cur_id = 0
            x_img, center_map, scale_map, offset_map, landmark_map = self._processOneImage(self.cur_id)
            self.cur_id += 1
            self.global_step += 1
            yield x_img, center_map, scale_map, offset_map, landmark_map

    def _getDataPipeline(self):
        ds = tf.data.Dataset.from_generator(self._generator, (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))
        self.ds = ds.batch(batch_size=self.batch_size, drop_remainder=True)

    def init(self, sess):
        self.data_iter = self.ds.make_initializable_iterator()
        self.data_next_element = self.data_iter.get_next()
        sess.run(self.data_iter.initializer)

    def batch(self, sess):
        return sess.run(self.data_next_element)


class DataProviderSet(object):
    def __init__(self, dataset, img_size=320, epoch_size=1, batch_size=8, angle_list=[0]):
        self.data_provider = []
        for angle in angle_list:
            self.data_provider.append(DataProvider(dataset, img_size=img_size, epoch_size=epoch_size, batch_size=batch_size, angle=angle))

    def size(self):
        return self.data_provider[0].size()

    def batchSize(self):
        return self.data_provider[0].batchSize()

    def curEpoch(self):
        return self.data_provider[0].curEpoch()

    def init(self, sess):
        for ds in self.data_provider:
            ds.init(sess)

    def batch(self, sess):
        batch_x_img = []
        batch_center_map = []
        batch_scale_map = []
        batch_offset_map = []
        batch_landmark_map = []
        for ds in self.data_provider:
            x_img, center_map, scale_map, offset_map, landmark_map = ds.batch(sess)
            batch_x_img.append(x_img)
            batch_center_map.append(center_map)
            batch_scale_map.append(scale_map)
            batch_offset_map.append(offset_map)
            batch_landmark_map.append(landmark_map)
        batch_x_img = np.concatenate(batch_x_img, axis=0)
        batch_center_map = np.concatenate(batch_center_map, axis=0)
        batch_scale_map = np.concatenate(batch_scale_map, axis=0)
        batch_offset_map = np.concatenate(batch_offset_map, axis=0)
        batch_landmark_map = np.concatenate(batch_landmark_map, axis=0)
        return batch_x_img, batch_center_map, batch_scale_map, batch_offset_map, batch_landmark_map


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    dataset_root = '../data'

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

    img_id = 0
    img_size = 416
    train_data = DataProviderSet(train_dataset, img_size=img_size, epoch_size=1, batch_size=2, angle_list=[0])
    with tf.Session() as sess:
        train_data.init(sess)
        try:
            batch_num = 0
            while True:
                x_img, center_map, scale_map, offset_map, landmark_map = train_data.batch(sess)
                print(x_img.shape)
                # print(center_map.shape)
                # print(x_data)
                # print(y_data)
                for index in range(x_img.shape[0]):
                    detect_img = (x_img[index] * 128 + 127.5).astype(np.uint8)
                    img_id += 1
                    detect_label = center_map[index]

                    class_v = np.expand_dims(np.expand_dims(center_map[index][:, :, 2], axis=0), axis=3)
                    scale_v = np.expand_dims(scale_map[index], axis=0)
                    offset_v = np.expand_dims(offset_map[index], axis=0)
                    landmark_v = np.expand_dims(landmark_map[index], axis=0)
                    faces = util_box.parse_wider_offset([class_v, scale_v, offset_v, landmark_v], (img_size, img_size))
                    for box in faces:
                        face_score = box[4]
                        landmark = box[5:]
                        box = box[0:4]
                        box = np.array(box, np.int32)
                        landmark = np.array(landmark, np.int32).reshape([-1, 5])
                        w_scale = 1.0
                        h_scale = 1.0
                        x1, y1, x2, y2 = box
                        x1 = int(x1 * w_scale)
                        x2 = int(x2 * w_scale)
                        y1 = int(y1 * h_scale)
                        y2 = int(y2 * h_scale)
                        cv2.rectangle(detect_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                        for ind in range(5):
                            y = landmark[0][ind]
                            x = landmark[1][ind]
                            x = int(x * w_scale)
                            y = int(y * h_scale)
                            cv2.circle(detect_img, (x, y), 1, (0, 0, 255), 1)

                    # plt.ion()
                    plt.figure(figsize=(12, 6))
                    plt.subplot(1, 3, 1)
                    plt.imshow(cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB))
                    plt.subplot(1, 3, 2)
                    plt.imshow(detect_label[:, :, 0])
                    plt.subplot(1, 3, 3)
                    plt.imshow(detect_label[:, :, 2])
                    plt.show()
                    # plt.pause(1)
                    # plt.close()

                batch_num += 1
        except tf.errors.OutOfRangeError:
            print('train stop.', train_data.size(), batch_num)
