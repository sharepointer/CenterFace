from __future__ import division

import cv2
import numpy as np
from numpy import random
import math
from PIL import Image


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2] - box_b[0]) *
              (box_b[3] - box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class RandomSSDCrop(object):
    def __init__(self):
        self.sample_options = (
            None,
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            (None, None),
        )

    def __call__(self, image, boxes, landmarks, labels):
        height, width, _ = image.shape
        while True:
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, landmarks, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image
                current_boxes = []
                current_landmarks = []
                current_labels = []

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if min(w, h) / max(w, h) < 0.6:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left + w), int(top + h)])

                if len(boxes) > 0:
                    # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                    overlap = jaccard_numpy(boxes, rect)

                    # is min and max overlap constraint satisfied? if not try again
                    if overlap.min() < min_iou and max_iou < overlap.max():
                        continue

                    # cut the crop from the image
                    current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]

                    # keep overlap with gt box IF center in sampled patch
                    centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                    # mask in all gt boxes that above and to the left of centers
                    m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                    # mask in all gt boxes that under and to the right of centers
                    m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                    # mask in that both m1 and m2 are true
                    mask = m1 * m2

                    # have any valid boxes? try again if not
                    if not mask.any():
                        continue

                    # take only matching gt boxes
                    current_boxes = boxes[mask, :].copy()

                    # should we use the box left and top corner or the crop's
                    current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])
                    # adjust to crop (by substracting crop's left,top)
                    current_boxes[:, :2] -= rect[:2]
                    current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rect[2:])
                    # adjust to crop (by substracting crop's left,top)
                    current_boxes[:, 2:] -= rect[:2]

                    current_labels = labels[mask, :].copy()
                    current_landmarks = landmarks[mask, :].copy()
                    current_landmarks[:, [0, 2, 4, 6, 8]] -= rect[0]
                    current_landmarks[:, [1, 3, 5, 7, 9]] -= rect[1]

                return current_image, current_boxes, current_landmarks, current_labels


class RandomBaiduCrop(object):
    def __init__(self, size):
        self.maxSize = 12000  # max size
        self.size = size
        if size == 320:
            self.base_anchors = [32, 64, 128, 256]
        elif size == 640:
            self.base_anchors = [16, 32, 64, 128, 256, 512]
        else:
            self.base_anchors = [16, 32, 64, 128, 256, 512]

    def __call__(self, image, boxes=None, labels=None):
        """
        :param image: 3-d array,channel last
        :param boxes: 2-d array,(num_gt,(x1,y1,x2,y2)
        :param labels: 1-d array(num_gt)
        :return:
        """
        # resize original image and transfer the gt accordingly
        height, width, _ = image.shape
        box_area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
        # print(boxes,box_area)
        rand_idx = random.randint(len(box_area))
        side_len = box_area[rand_idx] ** 0.5

        anchors = np.array(self.base_anchors)
        distances = abs(anchors - side_len)
        anchor_idx = np.argmin(distances)
        target_anchor = random.choice(anchors[0:min(anchor_idx + 1, 5) + 1])

        ratio = float(target_anchor) / side_len
        # print('ratio:', ratio)
        ratio = ratio * (2 ** random.uniform(-1, 1))
        if int(height * ratio * width * ratio) > self.maxSize * self.maxSize:
            ratio = (self.maxSize * self.maxSize / (height * width)) ** 0.5
        # print('ratio:', ratio)

        # interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC]
        interp_method = random.choice(interp_methods)
        image = cv2.resize(image, None, None, fx=ratio, fy=ratio, interpolation=interp_method)
        boxes = boxes * ratio

        # randomly select 50 crop box which covers the selected gt
        height, width, _ = image.shape
        sample_boxes = []
        gt_x1, gt_y1, gt_x2, gt_y2 = boxes[rand_idx, :]
        crop_w = crop_h = self.size

        # randomly select a crop box
        if crop_w < max(height, width):
            crop_x1 = random.uniform(gt_x2 - crop_w, gt_x1)
            crop_y1 = random.uniform(gt_y2 - crop_h, gt_y1)
        else:
            crop_x1 = random.uniform(width - crop_w, 0)
            crop_y1 = random.uniform(height - crop_h, 0)
        crop_x1 = math.floor(crop_x1)
        crop_y1 = math.floor(crop_y1)
        choice_box = np.array([int(crop_x1), int(crop_y1), int(crop_x1 + crop_w), int(crop_y1 + crop_h)])

        # perform crop, keep gts with centers lying inside the cropped box

        # pil_img = Image.fromarray(image.astype(np.uint8))
        pil_img = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))

        current_image = np.array(pil_img.crop([i for i in choice_box]))

        c_xs = (boxes[:, 0] + boxes[:, 2]) * 0.5
        c_ys = (boxes[:, 1] + boxes[:, 3]) * 0.5
        m1 = (choice_box[0] < c_xs) * (c_xs < choice_box[2])
        m2 = (choice_box[1] < c_ys) * (c_ys < choice_box[3])
        mask = m1 * m2
        current_boxes = boxes[mask, :].copy()
        current_labels = labels[mask]

        current_boxes[:, :2] = np.maximum(current_boxes[:, :2], choice_box[:2])  # make sure gt is inside the crop
        current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], choice_box[2:])
        current_boxes[:, :2] -= choice_box[:2]
        current_boxes[:, 2:] -= choice_box[:2]

        return current_image, current_boxes, current_labels
