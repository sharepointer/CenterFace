import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import argparse
import glob


def _list_target_dir(root_dir):
    list_ = []
    for item in os.listdir(root_dir):
        full_path = os.path.join(root_dir, item.strip())
        if os.path.isdir(full_path):
            list_.append(full_path)
    return list_


params = argparse.ArgumentParser()
params.add_argument('--dir', default='', type=str)
params.add_argument('--save_name', default='', type=str)
params.add_argument('--show', default=0, type=int)
ARGS = params.parse_args()

assert ARGS.dir != ''

root_dir = ARGS.dir
save_name = ARGS.save_name
if save_name == '':
    save_name = 'FaceDetection_train_set_' + os.path.split(root_dir)[1]

image_data = []
dir_list = _list_target_dir(root_dir)
for dir_item in dir_list:
    img_path = os.path.join(dir_item, 'images')
    txt_list = glob.glob(os.path.join(dir_item, '*.txt'))
    # print(txt_list)
    anno_path = txt_list[0]

    image_set = {}
    cur_key = ''
    for item in open(anno_path).readlines():
        if item.startswith('#'):
            cur_key = item.strip()[1:].strip()
            image_set[cur_key] = []
        else:
            image_set[cur_key].append(item.strip())

    wider_root = img_path
    for item in image_set.keys():
        image_path = os.path.join(wider_root, item)
        print(image_path)
        img = cv2.imread(image_path)
        img_h, img_w, _ = img.shape
        boxes = image_set[item]

        gt_boxes = []
        gt_landmarks = []

        for box in boxes:
            box = box.split(' ')
            face = np.array(box[0:4], dtype=np.int32)
            landmark = np.array(box[4:], dtype=np.int32)
            gt_boxes.append(face)
            gt_landmarks.append(landmark)

            x1, y1, x2, y2 = face
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            landmark = landmark.reshape([-1, 2])
            for pt in landmark:
                x, y = pt
                cv2.circle(img, (x, y), 2, (0, 0, 255), 2)

        if ARGS.show:
            cv2.imshow('as', img)
            cv2.waitKey(1 * 1000)

        gt_boxes = np.array(gt_boxes).reshape([-1, 4])
        gt_landmarks = np.array(gt_landmarks).reshape([-1, 10])
        annotation = {}
        annotation['filepath'] = image_path
        annotation['bboxes'] = gt_boxes
        annotation['landmarks'] = gt_landmarks
        image_data.append(annotation)

# save
save_root = '../data'
if os.path.exists(save_root) is False:
    os.mkdir(save_root)
save_path = os.path.join(save_root, save_name)
print(len(image_data))
with open(save_path, 'wb') as fid:
    pickle.dump(image_data, fid, pickle.HIGHEST_PROTOCOL)
