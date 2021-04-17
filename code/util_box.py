import numpy as np
import util_landmark
import traceback


def soft_bbox_vote(det, thre=0.35, score=0.5):
    if det.shape[0] <= 1:
        return det
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    dets = []
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these det
        merge_index = np.where(o >= thre)[0]
        det_accu = det[merge_index, :]
        det_accu_iou = o[merge_index]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            try:
                dets = np.row_stack((dets, det_accu))
            except:
                dets = det_accu
            continue
        else:
            soft_det_accu = det_accu.copy()
            soft_det_accu[:, 4] = soft_det_accu[:, 4] * (1 - det_accu_iou)
            soft_index = np.where(soft_det_accu[:, 4] >= score)[0]
            soft_det_accu = soft_det_accu[soft_index, :]

            # det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
            # max_score = np.max(det_accu[:, 4])

            # det_accu_sum = np.zeros((1, 5))
            # det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
            # det_accu_sum[:, 4] = max_score

            det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, 4:5], (1, 4))
            max_score = np.max(det_accu[:, 4])
            det_accu[:, 5:15] = det_accu[:, 5:15] * np.tile(det_accu[:, 4:5], (1, 10))

            det_accu_sum = np.zeros((1, 5 + 10))
            det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, 4])
            det_accu_sum[:, 4] = max_score
            det_accu_sum[:, 5:15] = np.sum(det_accu[:, 5:15], axis=0) / np.sum(det_accu[:, 4])

            if soft_det_accu.shape[0] > 0:
                det_accu_sum = np.row_stack((soft_det_accu, det_accu_sum))

            try:
                dets = np.row_stack((dets, det_accu_sum))
            except:
                dets = det_accu_sum

    order = dets[:, 4].ravel().argsort()[::-1]
    dets = dets[order, :]
    return dets


def parse_wider_offset(Y, image_shape=(320, 320), score=0.5, down=4, nmsthre=0.5, quantize=False):
    try:
        random_crop = image_shape

        seman = Y[0][0, :, :, 0]

        height = Y[1][0, :, :, 0]
        width = Y[1][0, :, :, 1]

        offset_y = Y[2][0, :, :, 0]
        offset_x = Y[2][0, :, :, 1]

        landmark_y = Y[3][0, :, :, 0:5]
        landmark_x = Y[3][0, :, :, 5:10]

        y_c, x_c = np.where(seman > score)
        map_h, map_w = seman.shape

        boxs = []
        if len(y_c) > 0:
            for i in range(len(y_c)):
                h = np.exp(height[y_c[i], x_c[i]]) * down
                w = np.exp(width[y_c[i], x_c[i]]) * down

                o_y = offset_y[y_c[i], x_c[i]]
                o_x = offset_x[y_c[i], x_c[i]]
                s = seman[y_c[i], x_c[i]]

                if quantize:
                    x1, y1 = max(0, (x_c[i] + o_x) * down - w / 2), max(0, (y_c[i] + o_y) * down - h / 2)
                    x1, y1 = min(x1, random_crop[1]), min(y1, random_crop[0])
                    lm_y = (landmark_y[y_c[i], x_c[i]] * h + y1)
                    lm_x = (landmark_x[y_c[i], x_c[i]] * w + x1)
                else:
                    x1, y1 = max(0, (x_c[i] + o_x + 0.5) * down - w / 2), max(0, (y_c[i] + o_y + 0.5) * down - h / 2)
                    x1, y1 = min(x1, random_crop[1]), min(y1, random_crop[0])
                    lm_y = (util_landmark.exp_landmark(landmark_y[y_c[i], x_c[i]]) + y_c[i]) * down
                    lm_x = (util_landmark.exp_landmark(landmark_x[y_c[i], x_c[i]]) + x_c[i]) * down

                boxs.append([x1, y1, min(x1 + w, random_crop[1]), min(y1 + h, random_crop[0]), s,
                             lm_y[0], lm_y[1], lm_y[2], lm_y[3], lm_y[4],
                             lm_x[0], lm_x[1], lm_x[2], lm_x[3], lm_x[4]])

            boxs = np.asarray(boxs, dtype=np.float32)
            boxs = soft_bbox_vote(boxs, thre=nmsthre, score=score)
        return boxs
    except Exception as e:
        print(e, traceback.format_exc())
