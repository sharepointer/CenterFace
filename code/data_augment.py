from __future__ import division

import cv2
import numpy as np
import copy
from numpy import random
import data_augment_crop as random_crop
import data_augment_rotate as util_rotate


def _randint(low, high):
    return random.randint(low, high + 1)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes, landmarks, labels):
        for t in self.transforms:
            img, boxes, landmarks, labels = t(img, boxes, landmarks, labels)
        return img, boxes, landmarks, labels


class ConvertFromInts(object):
    def __call__(self, image, boxes, landmarks, labels):
        return image.astype(np.float32), boxes, landmarks, labels


class Convert2uint8(object):
    def __call__(self, image, boxes, landmarks, labels):
        return np.clip(image, 0, 255).astype(np.uint8), boxes, landmarks, labels


class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes, landmarks, labels):
        h, w, c = image.shape
        if len(boxes) > 0:
            if _randint(0, 1):
                max_size = max(w, h)
                x_move = _randint(0, max_size - w)
                y_move = _randint(0, max_size - h)
                pad_image = np.zeros([max_size, max_size, c], dtype=np.uint8)
                pad_image[:, :, :] = 127
                pad_image[y_move:(h + y_move), x_move:(w + x_move), :] = image[:, :, :]

                boxes[:, [0, 2]] += x_move
                boxes[:, [1, 3]] += y_move
                landmarks[:, [0, 2, 4, 6, 8]] += x_move
                landmarks[:, [1, 3, 5, 7, 9]] += y_move
                x_ratio = self.size / max_size
                y_ratio = self.size / max_size
                boxes[:, [0, 2]] *= x_ratio
                boxes[:, [1, 3]] *= y_ratio
                landmarks[:, [0, 2, 4, 6, 8]] *= x_ratio
                landmarks[:, [1, 3, 5, 7, 9]] *= y_ratio
                image = cv2.resize(pad_image, (self.size, self.size))
            else:
                x_ratio = self.size / w
                y_ratio = self.size / h
                boxes[:, [0, 2]] *= x_ratio
                boxes[:, [1, 3]] *= y_ratio
                landmarks[:, [0, 2, 4, 6, 8]] *= x_ratio
                landmarks[:, [1, 3, 5, 7, 9]] *= y_ratio
                img_resize = cv2.resize(image, (self.size, self.size))
                image = copy.deepcopy(img_resize)

            for ind in range(len(boxes)):
                x1, y1, x2, y2 = boxes[ind, :]
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                if cx < 0 or cx >= w or cy < 0 or cy >= h:
                    labels[ind, :] = -1
        else:
            image = cv2.resize(image, (self.size, self.size))
        return image, boxes, landmarks, labels


class RandomMirror(object):
    def __init__(self, angle=0):
        self.angle = angle

    def __call__(self, image, boxes, landmarks, labels):
        _, width, _ = image.shape
        if len(boxes) > 0:
            if _randint(0, 1):
                image = cv2.flip(image, 1)
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                landmarks[:, [0, 2, 6, 8]] = width - landmarks[:, [2, 0, 8, 6]]
                landmarks[:, [1, 3, 7, 9]] = landmarks[:, [3, 1, 9, 7]]
                landmarks[:, [4]] = width - landmarks[:, [4]]
                landmarks[:, [5]] = landmarks[:, [5]]
        return image, boxes, landmarks, labels


class RandomRotate(object):
    def __init__(self, angle=0):
        self.angle = angle

    def __call__(self, image, boxes, landmarks, labels):
        if len(boxes) > 0:
            if _randint(0, 1):
                image_in = np.copy(image)
                boxes_in = np.copy(boxes)
                landmarks_in = np.copy(landmarks).reshape([-1, 5, 2])
                image_rot, boxes_rot, landmarks_rot = util_rotate.rotate_image_box_and_landmark(image_in, boxes_in, landmarks_in, 30)
                boxes_rot = boxes_rot.reshape([-1, 4]).astype(np.float32)
                landmarks_rot = landmarks_rot.reshape([-1, 10]).astype(np.float32)
                if len(boxes_rot) == len(landmarks_rot) and len(boxes_rot) == len(labels):
                    return image_rot, boxes_rot, landmarks_rot, labels
        return image, boxes, landmarks, labels


class RandomRotateFull(object):
    def __init__(self, angle=0):
        self.angle = angle

    def __call__(self, image, boxes, landmarks, labels):
        if self.angle > 0 and len(boxes) > 0:
            image_in = np.copy(image)
            boxes_in = np.copy(boxes)
            landmarks_in = np.copy(landmarks).reshape([-1, 5, 2])
            image_rot, boxes_rot, landmarks_rot = util_rotate.rotate_image_box_and_landmark(image_in, boxes_in, landmarks_in, self.angle)
            boxes_rot = boxes_rot.reshape([-1, 4]).astype(np.float32)
            landmarks_rot = landmarks_rot.reshape([-1, 10]).astype(np.float32)
            if len(boxes_rot) == len(landmarks_rot) and len(boxes_rot) == len(labels):
                return image_rot, boxes_rot, landmarks_rot, labels
        return image, boxes, landmarks, labels


class RandomGasussNoise(object):
    def __init__(self, mean=0, var=0.001):
        self.mean = mean
        self.var = var

    def __call__(self, image, boxes, landmarks, labels):
        out = image
        if _randint(0, 1):
            mean = self.mean
            var = self.var
            image = np.array(image / 255, dtype=float)
            noise = np.random.normal(mean, var ** 0.5, image.shape)
            out = image + noise
            if out.min() < 0:
                low_clip = -1.
            else:
                low_clip = 0.
            out = np.clip(out, low_clip, 1.0) * 255
            out = np.clip(out, 0, 255).astype(np.uint8)
        return out, boxes, landmarks, labels


class RandomBrightness(object):
    def __init__(self, min=0.5, max=2.0):
        self.min = min
        self.max = max

    def __call__(self, image, boxes, landmarks, labels):
        out = image
        if _randint(0, 1):
            min = self.min
            max = self.max
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            random_br = np.random.uniform(min, max)
            mask = hsv[:, :, 2] * random_br > 255
            v_channel = np.where(mask, 255, hsv[:, :, 2] * random_br)
            hsv[:, :, 2] = v_channel
            out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return out, boxes, landmarks, labels


class RandomConvert2Gray(object):
    def __init__(self, is_random=False):
        self.is_random = is_random

    def __call__(self, image, boxes, landmarks, labels):
        is_do = 1
        if self.is_random:
            is_do = _randint(0, 1)
        if is_do:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image[:, :, 0] = img_gray[:, :]
            image[:, :, 1] = img_gray[:, :]
            image[:, :, 2] = img_gray[:, :]
        return image, boxes, landmarks, labels


class RandomGamma(object):
    def __call__(self, image, boxes, landmarks, labels):
        if 1 == image.shape[2] and _randint(0, 1):
            gamma = (5 + _randint(0, 1) * 15) / 10
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            image = cv2.LUT(image, table)
        return image, boxes, landmarks, labels


class BoxesFilter(object):
    def __call__(self, image, boxes, landmarks, labels):
        if len(boxes) > 0:
            min_size = 13
            box_num = boxes.shape[0]
            for ind in range(box_num):
                x1, y1, x2, y2 = boxes[ind, :]
                w = x2 - x1
                h = y2 - y1
                if w < min_size or h < min_size:
                    labels[ind, :] = -1
        return image, boxes, landmarks, labels


class Augmentation(object):
    def __init__(self, size=320, angle=0):
        self.size = size
        self.angle = angle
        self.augment = Compose([
            RandomMirror(self.angle),
            RandomRotateFull(self.angle),
            RandomRotate(self.angle),
            ConvertFromInts(),
            random_crop.RandomSSDCrop(),
            Resize(self.size),
            # RandomBrightness(),
            RandomGasussNoise(),
            Convert2uint8(),
            RandomConvert2Gray(is_random=True),
            RandomGamma(),
            BoxesFilter(),
        ])

    def __call__(self, img, boxes, landmarks, labels):
        return self.augment(img, boxes, landmarks, labels)
