import tensorflow as tf
import numpy as np
import cv2
import copy
import argparse
import os
import scipy.io as sio
import matplotlib.pyplot as plt

import net_factory
import util_box
from config import _C as CONFIG


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


input_image = tf.placeholder(tf.float32, (1, None, None, 3), name='input_image')
map_class, map_scale, map_offset, map_landmark = net_factory.build(input_image, backbone=CONFIG.MODEL.BACKBONE_NAME)


def test_fddb(sess, score=0.5, angle=0, show=0):
    count = 0
    test_output_stream = open('../simulation/fddb_output_{}.txt'.format(angle), 'w')
    fddb_image_dir = '/home/work/Blue/dataset/FDDB/originalPics'
    for item in open('/home/work/Blue/FaceDetection/CenterFace/FDDB/FDDB-fold-all.txt').readlines():
        im_name = item.strip()
        test_output_stream.write('%s\n' % (im_name))

        img_fullpath = os.path.join(fddb_image_dir, im_name + '.jpg')
        org_img = cv2.imread(img_fullpath)
        org_h, org_w, _ = org_img.shape

        # gray_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
        # org_img[0:org_h, 0:org_w, 0] = gray_img[0:org_h, 0:org_w]
        # org_img[0:org_h, 0:org_w, 1] = gray_img[0:org_h, 0:org_w]
        # org_img[0:org_h, 0:org_w, 2] = gray_img[0:org_h, 0:org_w]

        boxes = []

        # src_img = copy.deepcopy(org_img)
        src_img, _ = rotate_boxes(org_img, boxes, angle)
        img_h, img_w, _ = src_img.shape

        max_size = max(img_w, img_h)
        resize_w = int(img_w * 320 / max_size) // 32 * 32
        resize_h = int(img_h * 320 / max_size) // 32 * 32
        w_scale = img_w / resize_w
        h_scale = img_h / resize_h

        detect_img = cv2.resize(src_img, (resize_w, resize_h))
        detect_img = (detect_img - 127.5) / 128
        detect_img = np.expand_dims(detect_img, axis=0)

        class_v, scale_v, offset_v, landmark_v = sess.run([map_class, map_scale, map_offset, map_landmark], feed_dict={input_image: detect_img})
        class_show = np.squeeze(class_v)

        bboxes = util_box.parse_wider_offset([class_v, scale_v, offset_v, landmark_v], image_shape=(resize_h, resize_w), score=score)
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
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
            plt.subplot(1, 2, 2)
            plt.imshow(class_show)
            plt.show()

    test_output_stream.close()


def test_fddb_for_vdsp(sess, score=0.5, show=0):
    save_val_image_index = 0
    save_val_image_dir = './z_output/for_vdsp/images'
    fs_imglist = open('./z_output/for_vdsp/img_list.txt', 'w')
    fs_imglist_map = open('./z_output/for_vdsp/img_list_map.txt', 'w')

    count = 0
    test_output_stream = open('./z_output/centerface_output.txt', 'w')
    fddb_image_dir = '/home/apuser/Blue/DataSet/FDDB/originalPics'
    for item in open('/home/apuser/Blue/DataSet/FDDB/FDDB-fold-all.txt').readlines():
        im_name = item.strip()
        test_output_stream.write('%s\n' % (im_name))

        img_fullpath = os.path.join(fddb_image_dir, im_name + '.jpg')
        src_img = cv2.imread(img_fullpath)
        gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

        img_h, img_w, _ = src_img.shape

        resize_w = 128
        resize_h = 128
        pad_w = max(img_w, img_h)
        pad_h = max(img_w, img_h)

        print("pad scale: {}".format(pad_w / pad_h))

        pad_img = np.zeros([pad_h, pad_w, 3], dtype=np.uint8)
        pad_img[0:img_h, 0:img_w, 0] = gray_img[0:img_h, 0:img_w]
        pad_img[0:img_h, 0:img_w, 1] = gray_img[0:img_h, 0:img_w]
        pad_img[0:img_h, 0:img_w, 2] = gray_img[0:img_h, 0:img_w]

        # cv2.imshow('as', pad_img)
        # cv2.waitKey(2*1000)

        w_scale = pad_w / resize_w
        h_scale = pad_h / resize_h

        detect_img = cv2.resize(pad_img, (resize_w, resize_h))

        save_val_image_index += 1
        filename_ = 'fd_val_{}.ppm'.format(save_val_image_index)
        cv2.imwrite(os.path.join(save_val_image_dir, filename_), detect_img)
        fs_imglist.write('%s\n' % filename_)
        fs_imglist_map.write('%s %s\n' % (filename_, im_name))

        detect_img = (detect_img - 127.5) / 128
        detect_img = np.expand_dims(detect_img, axis=0)

        class_v, scale_v, offset_v, landmark_v = sess.run([map_class, map_scale, map_offset, map_landmark],
                                                          feed_dict={input_image: detect_img})
        # print(class_v)
        # print(scale_v)
        # print(offset_v)

        bboxes = util_box.parse_wider_offset([class_v, scale_v, offset_v, landmark_v], image_shape=(resize_h, resize_w), score=score)
        if len(bboxes) > 0:
            keep_index = np.where(np.minimum(bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1]) >= 8)[0]
            bboxes = bboxes[keep_index, :]
        # print(bboxes)
        if len(bboxes) > 0:
            faceNum = len(bboxes)
            test_output_stream.write('%d\n' % faceNum)
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

                test_output_stream.write(
                    '%d %d %d %d %f\n' % (x1, y1, (x2 - x1 + 1), (y2 - y1 + 1), face_score))

                cv2.rectangle(src_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                for ind in range(5):
                    y = landmark[0][ind]
                    x = landmark[1][ind]
                    x = int(x * w_scale)
                    y = int(y * h_scale)

                    cv2.circle(src_img, (x, y), 2, (0, 0, 255), 2)
        else:
            test_output_stream.write('%d\n' % 0)

        count = count + 1
        print('%d %s ok~' % (count, img_fullpath))

        if show:
            cv2.imshow('as', src_img)
            cv2.waitKey(2 * 1000)

    test_output_stream.close()
    fs_imglist.close()
    fs_imglist_map.close()


def test_widerface(sess, score=0.5, show=0):
    subset = 'val'  # 'val'  # val or test
    if subset is 'val':
        wider_face = sio.loadmat('/home/apuser/Blue/DataSet/WIDERFACE/wider_face_split/wider_face_val.mat')  # Val set
    else:
        wider_face = sio.loadmat('/home/apuser/Blue/DataSet/WIDERFACE/wider_face_split/wider_face_test.mat')  # Test set
    event_list = wider_face['event_list']
    file_list = wider_face['file_list']
    del wider_face

    count = 0
    im_root = '/home/apuser/Blue/DataSet/WIDERFACE/WIDER_{}/images/'.format(subset)
    save_path = './z_output/pred/'
    for index, event in enumerate(event_list):
        # print(event)
        filelist = file_list[index][0]
        im_dir = event[0][0]
        if not os.path.exists(save_path + im_dir):
            os.makedirs(save_path + im_dir)

        for num, file in enumerate(filelist):
            im_name = file[0][0]
            image_path = im_root + im_dir + '/' + im_name + '.jpg'
            count += 1
            print(count, image_path)

            img_fullpath = image_path
            src_img = cv2.imread(img_fullpath)
            img_h, img_w, _ = src_img.shape
            resize_w = img_w // 32 * 32
            resize_h = img_h // 32 * 32

            w_scale = img_w / resize_w
            h_scale = img_h / resize_h

            detect_img = cv2.resize(src_img, (resize_w, resize_h))
            detect_img = (detect_img - 127.5) / 128
            detect_img = np.expand_dims(detect_img, axis=0)

            class_v, scale_v, offset_v, landmark_v = sess.run([map_class, map_scale, map_offset, map_landmark],
                                                              feed_dict={input_image: detect_img})
            # print(class_v)
            # print(scale_v)
            # print(offset_v)

            bboxes = util_box.parse_wider_offset([class_v, scale_v, offset_v, landmark_v], image_shape=(resize_h, resize_w), score=score)
            if len(bboxes) > 0:
                keep_index = np.where(np.minimum(bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1]) >= 5)[0]
                bboxes = bboxes[keep_index, :]
            # print(bboxes)

            fs = open(save_path + im_dir + '/' + im_name + '.txt', 'w')
            fs.write('{:s}\n'.format('%s/%s.jpg' % (im_dir, im_name)))
            fs.write('{:d}\n'.format(len(bboxes)))
            if len(bboxes) > 0:
                faceNum = len(bboxes)
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

                    fs.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.format(x1, y1, (x2 - x1 + 1), (y2 - y1 + 1), face_score))

                    cv2.rectangle(src_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    for ind in range(5):
                        y = landmark[0][ind]
                        x = landmark[1][ind]
                        x = int(x * w_scale)
                        y = int(y * h_scale)

                        cv2.circle(src_img, (x, y), 2, (0, 0, 255), 2)
            fs.close()

            if show:
                cv2.imshow('as', src_img)
                cv2.waitKey(2 * 1000)

    print('{} data set: {}'.format(subset, count))


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
    work_mode = ARGS.mode  # 0:fddb 1:widerface 2:vdsp

    angle_list = [0, 90, 180, 270]
    if fddb_angle not in angle_list:
        fddb_angle = 0

    saver = tf.train.Saver()

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.5

    with tf.Session(config=gpu_config) as sess:
        new_checkpoint = tf.train.latest_checkpoint('../checkpoints/checkpoints')
        print('{}'.format(new_checkpoint))
        saver.restore(sess, new_checkpoint)

        if 0 == work_mode:
            test_fddb(sess, score=detect_score, angle=fddb_angle, show=is_show)
        if 1 == work_mode:
            test_widerface(sess, score=detect_score, show=is_show)
        if 2 == work_mode:
            test_fddb_for_vdsp(sess, show=is_show)
