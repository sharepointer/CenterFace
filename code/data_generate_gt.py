import numpy as np
import copy
import traceback
import util_landmark
import matplotlib.pyplot as plt


def gaussian_v0(kernel):
    sigma = ((kernel - 1) * 0.5 - 1) * 0.3 + 0.8
    s = 2 * (sigma ** 2)
    dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
    return np.reshape(dx, (-1, 1))


def gaussian_v1(kernel, gaussian_sigma_ratio=4):
    sigma = kernel / gaussian_sigma_ratio
    if sigma < 3:
        sigma = 3
    s = 2 * (sigma ** 2)
    dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
    return np.reshape(dx, (-1, 1))


def gaussian_v2(kernel, center_ratio):
    sigma = kernel / 4
    s = 2 * (sigma ** 2)
    dx = np.exp(-np.square(np.arange(kernel) - int(kernel * center_ratio)) / s)
    return np.reshape(dx, (-1, 1))


def gaussian(kernel):
    return gaussian_v1(kernel)


def calc_gt_center(img_data, r=1, down=4, quantize=False):
    size_train = [img_data['size'], img_data['size']]
    gts = np.copy(img_data['bboxes'])
    gt_landmark = np.copy(img_data['landmarks'])
    igs = np.copy(img_data['ignore_bboxes'])

    seman_map = np.zeros((int(size_train[0] / down), int(size_train[1] / down), 3))
    seman_map[:, :, 1] = 1
    scale_map = np.zeros((int(size_train[0] / down), int(size_train[1] / down), 4))
    offset_map = np.zeros((int(size_train[0] / down), int(size_train[1] / down), 3))
    landmark_map = np.zeros((int(size_train[0] / down), int(size_train[1] / down), 1 + 10))

    try:
        if len(igs) > 0:
            igs = igs / down
            for ind in range(len(igs)):
                x1, y1, x2, y2 = int(igs[ind, 0]), int(igs[ind, 1]), int(np.ceil(igs[ind, 2])), int(np.ceil(igs[ind, 3]))
                seman_map[y1:y2, x1:x2, 1] = 0

        if len(gts) > 0:
            org_gts = copy.deepcopy(gts)
            org_landmark = copy.deepcopy(gt_landmark)
            gts = gts / down
            gt_landmark = gt_landmark / down
            for ind in range(len(gts)):
                # center
                x1, y1, x2, y2 = int(np.ceil(gts[ind, 0])), int(np.ceil(gts[ind, 1])), int(gts[ind, 2]), int(gts[ind, 3])
                c_x, c_y = int((gts[ind, 0] + gts[ind, 2]) / 2), int((gts[ind, 1] + gts[ind, 3]) / 2)
                dx = gaussian(x2 - x1)
                dy = gaussian(y2 - y1)
                gau_map = np.multiply(dy, np.transpose(dx))
                seman_map[y1:y2, x1:x2, 0] = np.maximum(seman_map[y1:y2, x1:x2, 0], gau_map)
                seman_map[y1:y2, x1:x2, 1] = 1
                # seman_map[c_y, c_x, 2] = 1
                seman_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 2] = 1

                # # show map
                # dx1 = gaussian_v2(x2 - x1, (c_x - x1) / (x2 - x1))
                # dy1 = gaussian_v2(y2 - y1, (c_y - y1) / (y2 - y1))
                # gau_map1 = np.multiply(dy1, np.transpose(dx1))
                # plt.ion()
                # plt.subplot(1, 2, 1)
                # # plt.plot(gau_map)
                # plt.imshow(gau_map)
                # plt.subplot(1, 2, 2)
                # # plt.plot(gau_map1)
                # plt.imshow(gau_map1)
                # # plt.show()
                # plt.pause(1)
                # plt.close()

                # scale
                scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 0] = np.log(gts[ind, 3] - gts[ind, 1])
                scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 1] = np.log(gts[ind, 2] - gts[ind, 0])
                scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 2] = 1

                # center offset
                if quantize:
                    offset_map[c_y, c_x, 0] = (gts[ind, 1] + gts[ind, 3]) / 2 - c_y
                    offset_map[c_y, c_x, 1] = (gts[ind, 0] + gts[ind, 2]) / 2 - c_x
                    offset_map[c_y, c_x, 2] = 1
                else:
                    for y_v in range(c_y - r, c_y + r + 1):
                        for x_v in range(c_x - r, c_x + r + 1):
                            offset_map[y_v, x_v, 0] = (gts[ind, 1] + gts[ind, 3]) / 2 - y_v - 0.5
                            offset_map[y_v, x_v, 1] = (gts[ind, 0] + gts[ind, 2]) / 2 - x_v - 0.5
                            offset_map[y_v, x_v, 2] = 1
                    # offset_map[c_y, c_x, 0] = (gts[ind, 1] + gts[ind, 3]) / 2 - c_y - 0.5
                    # offset_map[c_y, c_x, 1] = (gts[ind, 0] + gts[ind, 2]) / 2 - c_x - 0.5
                    # offset_map[c_y, c_x, 2] = 1

                # landmark
                if quantize:
                    landmark_map[c_y, c_x, 0:5] = (org_landmark[ind, 0:5] - org_gts[ind, 1]) / (org_gts[ind, 3] - org_gts[ind, 1])
                    landmark_map[c_y, c_x, 5:10] = (org_landmark[ind, 5:10] - org_gts[ind, 0]) / (org_gts[ind, 2] - org_gts[ind, 0])
                    landmark_map[c_y, c_x, 10] = 1
                else:
                    for y_v in range(c_y - r, c_y + r + 1):
                        for x_v in range(c_x - r, c_x + r + 1):
                            landmark_map[y_v, x_v, 0:5] = util_landmark.log_landmark(gt_landmark[ind, 0:5] - y_v)
                            landmark_map[y_v, x_v, 5:10] = util_landmark.log_landmark(gt_landmark[ind, 5:10] - x_v)
                            landmark_map[y_v, x_v, 10] = 1
                    # landmark_map[c_y, c_x, 0:5] = util_landmark.log_landmark(gt_landmark[ind, 0:5] - c_y)
                    # landmark_map[c_y, c_x, 5:10] = util_landmark.log_landmark(gt_landmark[ind, 5:10] - c_x)
                    # landmark_map[c_y, c_x, 10] = 1

    except Exception as e:
        pass
    finally:
        return seman_map, scale_map, offset_map, landmark_map
