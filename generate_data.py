import tensorflow as tf
import numpy as np
import cv2
from config.config import config
import math
import os
# config = config.config
from scipy import misc

from tools import euler_angles_utils

def DateSet(file_list, args, debug=False, img_dir=config.drewimgdir_all_ssd, isTest=False):
    mean_lmk = None
    if args.mean_pts is not None:
        mean_lmk = np.loadtxt(args.mean_pts)
    file_list, landmarks, attributes, euler_angles, video_list, attr4 = gen_data(file_list, img_dir, args)
    if debug:
        n = args.batch_size * 10
        file_list = file_list[:n]
        landmarks = landmarks[:n]
        attributes = attributes[:n]
        euler_angles = euler_angles[:n]
    dataset = tf.data.Dataset.from_tensor_slices((file_list, landmarks, attributes, euler_angles))

    def random_rotate_image_func(image, lmk):
        # img_draw1 = image.copy()
        # img_draw1 = img_draw1.astype(np.uint8)
        # img_draw1 = img_draw1[:,:,::-1]
        # cv2.imwrite('/lp/random-transformed-img/000.jpg',img_draw1)
        # np.savetxt('/lp/random-transformed-img/000.txt',lmk)
        assert (image.shape == (config.imgsize, config.imgsize, args.image_channels))
        # assert (lmk.shape == ())
        # 旋转角度范围
        angle = np.random.uniform(low=-30.0, high=30.0)
        # angle = 20
        rotated_img = misc.imrotate(image, angle, 'bicubic')
        # print (rotated_img.dtype)
        rotated_lmk = rotate_lmk_function(lmk, angle)
        rotated_lmk = np.reshape(rotated_lmk, [config.point_size, 2])
        key_lmk = rotated_lmk[config.mainPlaneLmkIndex]
        pitch, yaw, roll = euler_angles_utils.calculate_pitch_yaw_roll(key_lmk, mean_lmk=mean_lmk)
        # if img_count <= 10:
        #     img_draw = rotated_img.copy()
        #     img_draw = img_draw.astype(np.uint8)
        #     # np.copy()
        #     img_draw = img_draw[:, :, ::-1]
        #     # print(img_draw.shape)
        #     # for p in (rotated_lmk * config.imgsize).reshape([-1, 2]).astype(int):
        #         # cv2.circle(img_draw, tuple(p), 1, (0, 255, 0), -1)
        #         # cv2.circle(img_draw, tuple(p), 1, (0,255,0), -1)
        #     cv2.imwrite('/lp/random-transformed-img/001.jpg', img_draw)
        #     np.savetxt('/lp/random-transformed-img/001.txt',rotated_lmk)
        #     # exit()
        #     img_count += 1
        return rotated_img, rotated_lmk.reshape([-1]), np.array(
            [pitch * np.pi / 180, yaw * np.pi / 180, roll * np.pi / 180], dtype=np.float32)

    def rotate_lmk_function(lmk, angle):
        # print (lmk)
        sin_angle = np.sin(-angle * np.pi / 180)
        cos_angle = np.cos(-angle * np.pi / 180)
        Emat = np.mat(np.eye(3, dtype=np.float32))
        Rmat = np.mat(np.eye(3,dtype=np.float32))
        Rmat[0, 0] = cos_angle
        Rmat[1, 1] = cos_angle
        Rmat[0, 1] = -sin_angle
        Rmat[1, 0] = sin_angle
        Tmat = Emat.copy()
        Tmat[0:2, 2] = -0.5
        mat = Tmat.I * Rmat * Tmat
        lmk = np.reshape(lmk, [-1, 2])
        lmkMat = np.ones([len(lmk), 3], dtype=np.float32)
        lmkMat[:, 0:2] = lmk
        lmkMat = np.mat(lmkMat)
        lmkMat = lmkMat.T
        lmk_rotated = mat * lmkMat
        lmk_rotated = np.array(lmk_rotated.T)
        rotated_lmk = lmk_rotated[:, 0:2].copy()
        return rotated_lmk

    def _parse_data(filename, landmarks, attributes, euler_angles):
        # print (filename)
        # filename, landmarks, attributes = data
        file_contents = tf.read_file(filename)
        image = tf.image.decode_png(file_contents, channels=args.image_channels)
        # print (filename , image.shape)
        # print(image.get_shape())
        # image.set_shape((args.image_size, args.image_size, args.image_channels))
        image = tf.image.resize_images(image, (args.image_size, args.image_size), method=0)
        
        if not isTest and args.augument:
            image, landmarks, euler_angles = tf.py_func(random_rotate_image_func, [image, landmarks],
                                                        [tf.uint8, tf.float32, tf.float32])
            image = tf.image.random_brightness(image, 20.0/256)
        # tf.image.random_brightness(image,
        image = tf.cast(image, tf.float32)
        image = image / 256.0
        return image, landmarks, attributes, euler_angles

    dataset = dataset.map(_parse_data)
    if not isTest:
        dataset = dataset.shuffle(buffer_size=10000)
    return dataset, len(file_list)


def generateAttribute(landmark, attribute):
    '''

    :param landmark:  106 landmark
    :param attribute: 4 attribute
    :return: new attribute
    '''

    attr = np.zeros((10,), dtype=np.int32)
    landmark = np.asarray(landmark, dtype=np.float32)

    assert len(landmark) == 106 * 2, "len lmk %d " % (len(landmark))

    distToLeft = np.sqrt(np.sum(np.square(landmark[config.noseTipIndex] - landmark[config.faceLeftIndex])))
    distToRight = np.sqrt(np.sum(np.square(landmark[config.noseTipIndex] - landmark[config.faceRightIndex])))
    if distToRight == 0.0:
        distToRight = 0.001
    if distToLeft == 0.0:
        distToLeft = 0.001

    # profile
    if distToRight / distToLeft >= 4 or distToLeft / distToRight >= 4:
        attr[0] = 1
    else:
        attr[1] = 1
    # man or woman
    if attribute[0] == 1:
        attr[2] = 1
    else:
        attr[3] = 1
    # open mouth or not
    if attribute[2] == 1:
        attr[4] = 1
    else:
        attr[5] = 1
    # smile or not
    if attribute[3] == 1:
        attr[6] = 1
    else:
        attr[7] = 1
    if attribute[1] == 1:
        attr[8] = 1
    else:
        attr[9] = 1

    return attr


def gen_data(file_list, img_dir=config.drewimgdir_all_ssd, args=None, video_paths_file=None):
    print('reading file %s' % (file_list))
    with open(file_list, 'r') as f:
        lines = f.readlines()
    filenames, landmarks, attributes10, euler_angles, attributes4 = [], [], [], [], []
    for line in lines:
        line = line.strip().split()
        if not 224:
            print('wrinf line num %d : %s ' % (len(line), line))
        path = img_dir + '/' + line[0]
        landmark = line[1:106 * 2 + 1]
        attribute = line[106 * 2 + 1:106 * 2 + 1 + 4]
        box = line[106 * 2 + 1 + 4:106 * 2 + 1 + 4 + 4]
        euler_angle = line[106 * 2 + 1 + 4 + 4:]
        # print (line)
        try:
            assert len(attribute) == 4, '%s' % (len(attribute))
        except:
            print(len(line))
            input('???')
        attribute = [attribute[0], attribute[1], attribute[2], attribute[3]]
        try:
            assert len(landmark) == 106 * 2
        except:
            print('line: %s' % line)
        if args is None:
            landmark = np.asarray(landmark, dtype=np.float32)

        else:
            if args.lmk_norm:
                landmark = np.asarray(landmark, dtype=np.float32) / config.imgsize
                # print (landmark)
                # print (landmark)
            else:
                landmark = np.asarray(landmark, dtype=np.float32)
        attribute = np.asarray(attribute, dtype=np.int32)
        attributes4.append(attribute)
        attribute10 = generateAttribute(landmark, attribute)
        euler_angle = np.asarray(euler_angle, dtype=np.float32) * math.pi / 180
        filenames.append(path)
        landmarks.append(landmark)
        attributes10.append(attribute10)
        euler_angles.append(euler_angle)
        video_paths = []
    if video_paths_file is not None:
        with open(video_paths_file, 'r') as video_file:
            lines = video_file.readlines()
        for line in lines:
            video_paths.append(line.strip())
    video_paths = np.array(video_paths, dtype=str)
    if len(video_paths) == 0:
        video_paths = None

    filenames = np.asarray(filenames, dtype=np.str)
    landmarks = np.asarray(landmarks, dtype=np.float32)
    attributes10 = np.asarray(attributes10, dtype=np.int32)
    attributes4 = np.asarray(attributes4, dtype=np.float32)
    euler_angles = np.asarray(euler_angles, dtype=np.float32)
    return filenames, landmarks, attributes10, euler_angles, video_paths, attributes4


if __name__ == '__main__':
    file_list = 'data/train_data/list.txt'
    filenames, landmarks, attributes = gen_data(file_list)
    for i in range(len(filenames)):
        filename = filenames[i]
        landmark = landmarks[i]
        attribute = attributes[i]
        print(attribute)
        img = cv2.imread(filename)
        h, w, _ = img.shape
        landmark = landmark.reshape(-1, 2) * [h, w]
        for (x, y) in landmark.astype(np.int32):
            cv2.circle(img, (x, y), 1, (0, 0, 255))
        cv2.imshow('0', img)
        cv2.waitKey(0)
