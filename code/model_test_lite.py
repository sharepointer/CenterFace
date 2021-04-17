import numpy as np
import tensorflow as tf
import cv2
import copy
import argparse
import os
import struct

import util_box


def rotate_boxes(image, gt_boxes, angle):
    rot_boxes = []

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]])

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2, image_h2]) * rot_mat_notranslate).A[0],
        (np.array([image_w2, image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([[1, 0, int(new_w * 0.5 - image_w2)],
                           [0, 1, int(new_h * 0.5 - image_h2)], [0, 0, 1]])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    img_rotate = cv2.warpAffine(image, affine_mat, (new_w, new_h), flags=cv2.INTER_LINEAR)

    # h, w, c = image.shape
    # center = (w / 2, h / 2)
    # rot_mat = cv2.getRotationMatrix2D(center, angle, 1)
    # print(rot_mat)
    # img_rotate = cv2.warpAffine(image, rot_mat, (h, w))
    rotate_mat = affine_mat
    # print(rotate_mat.shape)
    # print(rotate_mat[1, 0])

    rot_h, rot_w, rot_c = img_rotate.shape
    for gt_face in gt_boxes:
        x1 = gt_face[0]
        y1 = gt_face[1]
        x2 = gt_face[2]
        y2 = gt_face[3]
        score = gt_face[4]

        box_rot = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        box_rot = np.asarray([(rotate_mat[0, 0] * x + rotate_mat[0, 1] * y + rotate_mat[0, 2],
                               rotate_mat[1, 0] * x + rotate_mat[1, 1] * y + rotate_mat[1, 2]) for (x, y) in box_rot])
        box_rot = box_rot.astype(np.int32)

        (p1_x, p1_y) = box_rot[0]
        (p2_x, p2_y) = box_rot[1]
        (p3_x, p3_y) = box_rot[2]
        (p4_x, p4_y) = box_rot[3]

        sort_x = np.sort([p1_x, p2_x, p3_x, p4_x])
        sort_y = np.sort([p1_y, p2_y, p3_y, p4_y])

        minx = int((sort_x[0] + sort_x[1]) / 2)
        maxx = int((sort_x[3] + sort_x[2]) / 2)
        miny = int((sort_y[0] + sort_y[1]) / 2)
        maxy = int((sort_y[3] + sort_y[2]) / 2)

        minx = int((minx * 0.7 + sort_x[0] * 0.3))
        maxx = int((maxx * 0.7 + sort_x[3] * 0.3))
        miny = int((miny * 0.7 + sort_y[0] * 0.3))
        maxy = int((maxy * 0.7 + sort_y[3] * 0.3))

        if minx < 0 or miny < 0 or maxx >= rot_w or maxy >= rot_h:
            continue

        rot_boxes.append([minx, miny, maxx, maxy, score])

    return img_rotate, rot_boxes


def test_common():
    FIXED_SIZE = 320

    image_root = './images'
    image_list = []
    for item in os.listdir(image_root):
        if ('.jpg' in item or '.jpeg' in item) and '_out' not in item:
            image_list.append(os.path.join(image_root, item.strip()))

    # Load TFLite model and allocate tensors.
    interpreter = tf.contrib.lite.Interpreter(model_path="./model_export/centerface_tflite_quantize.tflite")

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print('[input_details:]')
    for in_temp in input_details:
        print(in_temp)
    print('[output_details:]')
    for out_temp in output_details:
        print(out_temp)

    # interpreter.resize_tensor_input(input_details[0]['index'], [1, 320, 320, 3])
    interpreter.allocate_tensors()

    for image_path in image_list:
        img = cv2.imread(image_path)

        # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img[:, :, 0] = img_gray[:, :]
        # img[:, :, 1] = img_gray[:, :]
        # img[:, :, 2] = img_gray[:, :]

        img_h, img_w, _ = img.shape
        max_size = max(img_w, img_h)
        img_pad = np.zeros([max_size, max_size, 3], dtype=np.uint8)
        img_pad[0:img_h, 0:img_w, :] = img[0:img_h, 0:img_w, :]
        img_pad = cv2.resize(img_pad, (FIXED_SIZE, FIXED_SIZE))

        scale_ratio = max_size / FIXED_SIZE

        src_img = copy.deepcopy(img)
        x_img = img_pad

        # set input
        interpreter.set_tensor(input_details[0]['index'], np.expand_dims(x_img, 0))

        # run model
        interpreter.invoke()

        # get outputs
        # output_data = interpreter.get_tensor(output_details[0]['index'])

        # decode
        class_v = interpreter.get_tensor(output_details[0]['index'])
        q1, q2 = output_details[0]['quantization']
        class_v = (class_v - q2) * q1

        scale_v = interpreter.get_tensor(output_details[1]['index'])
        q1, q2 = output_details[1]['quantization']
        scale_v = (scale_v - q2) * q1

        offset_v = interpreter.get_tensor(output_details[2]['index'])
        q1, q2 = output_details[2]['quantization']
        offset_v = (offset_v - q2) * q1

        landmark_v = interpreter.get_tensor(output_details[3]['index'])
        q1, q2 = output_details[3]['quantization']
        landmark_v = (landmark_v - q2) * q1

        score = 0.5
        resize_h = FIXED_SIZE
        resize_w = FIXED_SIZE

        bboxes = util_box.parse_wider_offset([class_v, scale_v, offset_v, landmark_v], image_shape=(resize_h, resize_w), score=score)
        if len(bboxes) > 0:
            keep_index = np.where(np.minimum(bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1]) >= 8)[0]
            bboxes = bboxes[keep_index, :]

        w_scale = scale_ratio
        h_scale = scale_ratio
        boxes = []
        if len(bboxes) > 0:
            for box in bboxes:
                # print(box)
                face_score = box[4]
                landmark = box[5:]

                box = box[0:4]
                box = np.array(box, np.int32)
                # print(box, score)

                landmark = np.array(landmark, np.int32).reshape([-1, 5])
                # print(landmark)

                x1, y1, x2, y2 = box
                x1 = int(x1 * w_scale)
                x2 = int(x2 * w_scale)
                y1 = int(y1 * h_scale)
                y2 = int(y2 * h_scale)

                boxes.append((x1, y1, x2, y2, face_score))

                # test_output_stream.write(
                #     '%d %d %d %d %f\n' % (x1, y1, (x2 - x1 + 1), (y2 - y1 + 1), face_score))

                cv2.rectangle(src_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                for ind in range(5):
                    y = landmark[0][ind]
                    x = landmark[1][ind]
                    x = int(x * w_scale)
                    y = int(y * h_scale)

                    cv2.circle(src_img, (x, y), 2, (0, 0, 255), 2)

        cv2.imshow('as', src_img)
        cv2.waitKey(0 * 1000)


def test_fddb(score=0.5, angle=0, show=0):
    # input
    FIXED_SIZE = 320

    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="./model_export/centerface_tflite_quantize.tflite")

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print('[input_details:]')
    for in_temp in input_details:
        print(in_temp)
    print('[output_details:]')
    for out_temp in output_details:
        print(out_temp)

    # interpreter.resize_tensor_input(input_details[0]['index'], [1, 320, 320, 3])
    interpreter.allocate_tensors()

    count = 0
    test_output_stream = open('./model_test/centerface_output_{}.txt'.format(angle), 'w')
    fddb_image_dir = '/home/work/Blue/dataset/FDDB/originalPics'
    for item in open('/home/work/Blue/dataset/FDDB/FDDB-fold-all.txt').readlines():
        im_name = item.strip()
        test_output_stream.write('%s\n' % (im_name))

        img_fullpath = os.path.join(fddb_image_dir, im_name + '.jpg')
        org_img = cv2.imread(img_fullpath)

        # org_h, org_w, _ = org_img.shape
        # gray_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
        # org_img[0:org_h, 0:org_w, 0] = gray_img[0:org_h, 0:org_w]
        # org_img[0:org_h, 0:org_w, 1] = gray_img[0:org_h, 0:org_w]
        # org_img[0:org_h, 0:org_w, 2] = gray_img[0:org_h, 0:org_w]

        boxes = []

        # src_img = copy.deepcopy(org_img)
        src_img, _ = rotate_boxes(org_img, boxes, angle)

        img_h, img_w, _ = src_img.shape
        max_size = max(img_h, img_w)
        pad_img = np.zeros([max_size, max_size, 3], np.uint8)
        pad_img[0:img_h, 0:img_w, :] = src_img[0:img_h, 0:img_w, :]

        resize_w = FIXED_SIZE
        resize_h = FIXED_SIZE

        w_scale = max_size / resize_w
        h_scale = max_size / resize_h

        detect_img = cv2.resize(pad_img, (resize_w, resize_h))

        # set input
        interpreter.set_tensor(input_details[0]['index'], np.expand_dims(detect_img, 0))

        # run model
        interpreter.invoke()

        # get outputs
        # output_data = interpreter.get_tensor(output_details[0]['index'])

        # decode
        class_v = interpreter.get_tensor(output_details[0]['index'])
        q1, q2 = output_details[0]['quantization']
        class_v = (class_v - q2) * q1

        scale_v = interpreter.get_tensor(output_details[1]['index'])
        # print(scale_v)
        q1, q2 = output_details[1]['quantization']
        scale_v = (scale_v - q2) * q1

        offset_v = interpreter.get_tensor(output_details[2]['index'])
        q1, q2 = output_details[2]['quantization']
        offset_v = (offset_v - q2) * q1

        landmark_v = interpreter.get_tensor(output_details[3]['index'])
        q1, q2 = output_details[3]['quantization']
        landmark_v = (landmark_v - q2) * q1

        bboxes = util_box.parse_wider_offset([class_v, scale_v, offset_v, landmark_v], image_shape=(resize_h, resize_w), score=score,quantize=True)
        if len(bboxes) > 0:
            keep_index = np.where(np.minimum(bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1]) >= 8)[0]
            bboxes = bboxes[keep_index, :]

        if len(bboxes) > 0:
            for box in bboxes:
                # print(box)
                face_score = box[4]
                landmark = box[5:]

                box = box[0:4]
                box = np.array(box, np.int32)
                # print(box, score)

                landmark = np.array(landmark, np.int32).reshape([-1, 5])
                # print(landmark)

                x1, y1, x2, y2 = box
                x1 = int(x1 * w_scale)
                x2 = int(x2 * w_scale)
                y1 = int(y1 * h_scale)
                y2 = int(y2 * h_scale)

                boxes.append((x1, y1, x2, y2, face_score))

                # test_output_stream.write(
                #     '%d %d %d %d %f\n' % (x1, y1, (x2 - x1 + 1), (y2 - y1 + 1), face_score))

                cv2.rectangle(src_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                for ind in range(5):
                    y = landmark[0][ind]
                    x = landmark[1][ind]
                    x = int(x * w_scale)
                    y = int(y * h_scale)

                    cv2.circle(src_img, (x, y), 2, (0, 0, 255), 2)
        else:
            pass
            # test_output_stream.write('%d\n' % 0)

        # rot_img = copy.deepcopy(src_img)
        # rot_boxes = copy.deepcopy(boxes)
        rot_img, rot_boxes = rotate_boxes(src_img, boxes, -angle)

        face_count = len(rot_boxes)
        test_output_stream.write('%d\n' % face_count)
        for face in rot_boxes:
            x1, y1, x2, y2, face_score = face

            cv2.rectangle(rot_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            test_output_stream.write(
                '%d %d %d %d %f\n' % (x1, y1, (x2 - x1 + 1), (y2 - y1 + 1), face_score))

        count = count + 1
        print('%d %s ok~' % (count, img_fullpath))

        if show:
            img_show = copy.deepcopy(src_img)
            cv2.imshow('as', img_show)
            cv2.waitKey(2 * 1000)

    test_output_stream.close()


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--score', default=0.5, type=float)
    args.add_argument('--show', default=1, type=int)
    args.add_argument('--angle', default=0, type=int)
    args.add_argument('--mode', default=0, type=int)
    ARGS = args.parse_args()

    is_show = ARGS.show
    detect_score = ARGS.score
    fddb_angle = ARGS.angle
    work_mode = ARGS.mode

    angle_list = [0, 90, 180, 270]
    if fddb_angle not in angle_list:
        fddb_angle = 0

    if 0 == work_mode:
        test_fddb(angle=fddb_angle, show=is_show)
