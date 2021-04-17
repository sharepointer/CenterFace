import numpy as np
import cv2


def _rotate_box(image, gt_box, angle):
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

    x1 = gt_box[0]
    y1 = gt_box[1]
    x2 = gt_box[2]
    y2 = gt_box[3]

    # h, w, c = image.shape
    # center = (w / 2, h / 2)
    # rot_mat = cv2.getRotationMatrix2D(center, angle, 1)
    # img_rotate = cv2.warpAffine(image, rot_mat, (h, w))
    rotate_mat = affine_mat

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

    rot_h, rot_w, rot_c = img_rotate.shape
    if minx < 0 or miny < 0 or maxx >= rot_w or maxy >= rot_h:
        return img_rotate, None, box_rot

    return img_rotate, (minx, miny, maxx, maxy), box_rot


def _rotate_boxes(image, gt_boxes, angle):
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

        rot_boxes.append([minx, miny, maxx, maxy])

    return img_rotate, rot_boxes


def _rotate_landmark(image, landmark, angle):
    h, w, c = image.shape
    center = (w / 2, h / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1)
    img_rotate = cv2.warpAffine(image, rot_mat, (h, w))
    rot_landmark = np.asarray([(rot_mat[0][0] * x + rot_mat[0][1] * y + rot_mat[0][2],
                                rot_mat[1][0] * x + rot_mat[1][1] * y + rot_mat[1][2]) for (x, y) in landmark])
    rot_landmark = rot_landmark.astype(np.int32)

    return img_rotate, rot_landmark


def _image_rotate(image, angle):
    h, w, c = image.shape
    center = (w / 2, h / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1)
    img_rotate = cv2.warpAffine(image, rot_mat, (h, w))
    return img_rotate


def _gen_landmark_data(gt_box, landmarks):
    x1 = gt_box[0]
    y1 = gt_box[1]
    x2 = gt_box[2]
    y2 = gt_box[3]
    gt_w = x2 - x1 + 1
    gt_h = y2 - y1 + 1

    res_landmark = []

    for pt in landmarks:
        x, y = pt
        if x <= x1 or x >= x2 or y <= y1 or y >= y2:
            return None
        res_landmark.append((x - x1) / gt_w)
        res_landmark.append((y - y1) / gt_h)

    return res_landmark


def _rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH)), M


def _rotate_image_with_landmark(image, box, landmark, angle):
    rotate_img, rotate_mat = _rotate_bound(image, angle)
    # print(rotate_mat)
    # print(landmark)

    x1, y1, x2, y2 = box

    rect = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    rotate_rect = np.asarray([(rotate_mat[0, 0] * x + rotate_mat[0, 1] * y + rotate_mat[0, 2],
                               rotate_mat[1, 0] * x + rotate_mat[1, 1] * y + rotate_mat[1, 2]) for (x, y) in rect])
    rotate_rect = rotate_rect.astype(np.int32)

    # x1, y1 = rotate_rect[0]
    # x2, y2 = rotate_rect[3]
    (p1_x, p1_y) = rotate_rect[0]
    (p2_x, p2_y) = rotate_rect[1]
    (p3_x, p3_y) = rotate_rect[2]
    (p4_x, p4_y) = rotate_rect[3]

    sort_x = np.sort([p1_x, p2_x, p3_x, p4_x])
    sort_y = np.sort([p1_y, p2_y, p3_y, p4_y])

    min_x = int((sort_x[0] + sort_x[1]) / 2)
    max_x = int((sort_x[3] + sort_x[2]) / 2)
    min_y = int((sort_y[0] + sort_y[1]) / 2)
    max_y = int((sort_y[3] + sort_y[2]) / 2)

    min_x = int((min_x * 0.7 + sort_x[0] * 0.3))
    max_x = int((max_x * 0.7 + sort_x[3] * 0.3))
    min_y = int((min_y * 0.7 + sort_y[0] * 0.3))
    max_y = int((max_y * 0.7 + sort_y[3] * 0.3))

    rotate_box = [min_x, min_y, max_x, max_y]

    rotate_landmark = np.asarray([(rotate_mat[0][0] * x + rotate_mat[0][1] * y + rotate_mat[0][2],
                                   rotate_mat[1][0] * x + rotate_mat[1][1] * y + rotate_mat[1][2]) for (x, y) in landmark])
    rotate_landmark = rotate_landmark.astype(np.int32)

    return rotate_img, rotate_box, rotate_landmark


def rotate_image_box_and_landmark(image, boxes, landmarks, angle, smooth=True):
    _rotate_img, rotate_mat = _rotate_bound(image, angle)
    rot_h, rot_w, rot_c = _rotate_img.shape
    # print(rotate_mat)
    # print(landmark)

    rotate_box = []
    rotate_landmark = []

    for box in boxes:
        x1, y1, x2, y2 = box

        rect = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        rotate_rect = np.asarray([(rotate_mat[0, 0] * x + rotate_mat[0, 1] * y + rotate_mat[0, 2],
                                   rotate_mat[1, 0] * x + rotate_mat[1, 1] * y + rotate_mat[1, 2]) for (x, y) in rect])
        rotate_rect = rotate_rect.astype(np.int32)

        (p1_x, p1_y) = rotate_rect[0]
        (p2_x, p2_y) = rotate_rect[1]
        (p3_x, p3_y) = rotate_rect[2]
        (p4_x, p4_y) = rotate_rect[3]

        sort_x = np.sort([p1_x, p2_x, p3_x, p4_x])
        sort_y = np.sort([p1_y, p2_y, p3_y, p4_y])

        if smooth:
            min_x = ((sort_x[0] + sort_x[1]) / 2)
            max_x = ((sort_x[3] + sort_x[2]) / 2)
            min_y = ((sort_y[0] + sort_y[1]) / 2)
            max_y = ((sort_y[3] + sort_y[2]) / 2)
            min_x = int((min_x * 0.7 + sort_x[0] * 0.3))
            max_x = int((max_x * 0.7 + sort_x[3] * 0.3))
            min_y = int((min_y * 0.7 + sort_y[0] * 0.3))
            max_y = int((max_y * 0.7 + sort_y[3] * 0.3))
        else:
            min_x = sort_x[0]
            max_x = sort_x[3]
            min_y = sort_y[0]
            max_y = sort_y[3]

        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(max_x, rot_w - 1)
        max_y = min(max_y, rot_h - 1)

        rotate_box.append([min_x, min_y, max_x, max_y])

    for landmark in landmarks:
        temp_landmark = np.asarray([(rotate_mat[0][0] * x + rotate_mat[0][1] * y + rotate_mat[0][2],
                                     rotate_mat[1][0] * x + rotate_mat[1][1] * y + rotate_mat[1][2]) for (x, y) in landmark])
        temp_landmark = temp_landmark.astype(np.int32)
        rotate_landmark.append(temp_landmark)

    return _rotate_img, np.array(rotate_box, dtype=np.int32), np.array(rotate_landmark, np.int32)
