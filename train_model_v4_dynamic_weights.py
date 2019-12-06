# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

from pympler import muppy

# print('pid: {}     GPU: {}'.format(os.getpid(), os.environ['CUDA_VISIBLE_DEVICES']))

import tensorflow as tf
import numpy as np
import cv2
import argparse
import sys
# import matplotlib

# matplotlib.use('Agg')
import math
import time

from generate_data import DateSet

from train.model2 import create_model as create_model_v2
from train.model2_v1 import create_model as create_model_v1
from train.model2_v4 import create_model as create_model_v4
from utils import train_model

from config import config

config = config.config
from lk.lk_config import lk_config

from gen_data import load_data


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visibale_device
    list_ops = {}

    debug = (args.debug == 'True')
    print(args)
    np.random.seed(args.seed)
    with tf.Graph().as_default():
        with tf.variable_scope('multi_gpu'):
            print('reading train dataset')
            # train_dataset, num_train_file = DateSet(args.file_list, args, debug, args.img_dir)
            assert len(args.train_file) == len(args.batch_size)
            assert len(args.train_file) == len(args.train_data_type)
            list_ops['next_batch_group'] = None
            gloabl_batch_size = 0
            global_num_data = 0

            # for i m (test_file, )
            # test_lmk_file = args.test_lmk_file
            # test_attr_file = args.test_attr_file
            # test_batch_size = args.test_batch_size
            # test_lmk_data_set, test_lmk_num = load_data(data_set_file=test_lmk_file, datatype='jd', args=args,
            #                                             imgdir=args.img_dir)
            # test_attribute_data_set, test_attribute_num = load_data(data_set_file=test_attr_file, datatype='celeba',
            #                                                         args=args, imgdir=args.img_dir)
            # test_lmk_dataset_batch = test_lmk_data_set.batch(test_batch_size).repeat()
            # test_lmk_data_set_batch_it = test_lmk_dataset_batch.make_one_shot_iterator()
            # test_attribute_data_set_batch = test_attribute_data_set.batch(test_batch_size).repeat()
            # test_attribute_data_set_batch_it = test_attribute_data_set_batch.make_one_shot_iterator()
            # test_lmk_data_set_batch_next = test_lmk_data_set_batch_it.get_next()
            # test_attribute_data_set_batch_next = test_attribute_data_set_batch_it.get_next()

            # list_ops['test_lmk_next'] = test_lmk_data_set_batch_next
            # list_ops['test_attr_next'] = test_attribute_data_set_batch_next

            for i, (train_file, batch_size, data_type) in enumerate(
                    zip(args.train_file, args.batch_size, args.train_data_type)):
                data_set, num_data = load_data(data_set_file=train_file, datatype=data_type, args=args, imgdir=args.img_dir, isTrain=True)
                if 'jd' == data_type:
                    gloabl_batch_size = batch_size
                    global_num_data = num_data
                data_set_batch = data_set.batch(batch_size).repeat()
                data_set_batch_itor = data_set_batch.make_one_shot_iterator()
                data_set_next_batch = data_set_batch_itor.get_next()
                list_ops['train_data_set_%d' % i] = data_set_next_batch
                list_ops['train_num_data_%d' % i] = num_data

            for i, (test_file, data_type) in enumerate(
                    zip(args.test_file, args.test_data_type)):
                data_set, num_data = load_data(data_set_file=test_file, datatype=data_type, args=args,
                                               imgdir=args.img_dir,isTrain=True)
                # print('init test data set ... %d' % i)

                data_set_batch = data_set.batch(np.mean(args.batch_size).astype(int)).repeat()
                data_set_batch_itor = data_set_batch.make_one_shot_iterator()
                data_set_next_batch = data_set_batch_itor.get_next()
                list_ops['test_data_set_%d' % i] = data_set_next_batch
                list_ops['test_num_data_%d' % i] = num_data
                if 'jd' == data_type:
                    gloabl_test_batch_size = gloabl_batch_size
                    global_test_num_data = num_data

                # if list_ops['next_batch_group'] is None:
                #     list_ops['next_batch_group'] = data_set_next_batch
                # else:

                # train_dataset, num_train_file = load_data(args.train_file[i])
            # print('reading test dataser')
            # test_dataset, num_test_file = DateSet(args.test_list, args, debug, args.test_img_dir)

            # train_video_fataset, num_video_images = VideoDataSet(args.video, args)
            # print('reading video dataset')

            # batch_train_dataset = train_dataset.batch(args.batch_size).repeat()
            # train_iterator = batch_train_dataset.make_one_shot_iterator()
            # train_next_element = train_iterator.get_next()

            # batch_test_dataset = test_dataset.batch(args.batch_size).repeat()
            # test_iterator = batch_test_dataset.make_one_shot_iterator()
            # test_next_element = test_iterator.get_next()

            # batch_train_video_dataset = train_video_fataset.batch(
            #     lk_config['video_count'] * lk_config['video_seq']).repeat()
            # train_video_iterator = batch_train_video_dataset.make_one_shot_iterator()
            # train_video_next_element = train_video_iterator.get_next()
            #
            # list_ops['num_train_file'] = num_train_file
            # list_ops['num_test_file'] = num_test_file

            model_dir = args.model_dir
            # model_dir = args.model
            log_dir = args.log_dir
            # if 'test' in model_dir and debug and os.path.exists(model_dir):
            #     import shutil
            #     shutil.rmtree(model_dir)
            # assert not os.path.exists(model_dir)
            # os.mkdir(model_dir)

            # print('Total number of examples: {}'.format(num_train_file))
            # print('Test number of examples: {}'.format(num_test_file))
            print('Model dir: {}'.format(model_dir))

            tf.set_random_seed(args.seed)
            global_step = tf.Variable(0, trainable=False)

            list_ops['global_step'] = global_step
            # list_ops['train_dataset'] = train_dataset
            # list_ops['test_dataset'] = test_dataset
            # list_ops['train_next_element'] = train_next_element
            # list_ops['test_next_element'] = test_next_element
            # list_ops['train_video_next_elements'] = train_video_next_element

            epoch_size = global_num_data // gloabl_batch_size
            test_epoch_size = global_test_num_data // gloabl_test_batch_size
            print('Number of batches per epoch: {}'.format(epoch_size))

            image_batch = tf.placeholder(tf.float32, shape=(None, args.image_size, args.image_size, 3), \
                                         name='image_batch')

            landmark_batch = tf.placeholder(tf.float32, shape=(None, 106 * 2), name='landmark_batch')

            euler_angles_gt_batch = tf.placeholder(tf.float32, shape=(None, 3), name='euler_angles_gt_batch')
            # attribute4_batch = tf.placeholder(tf.float32, (None, config.anno_attr_count), name='attr4_batch')

            race_gt = tf.placeholder(tf.float32, (None, 4), name='race_gt')
            race_st = tf.placeholder(tf.float32, (None, 3), name='race_st')

            # angle_by_gt = tf.placeholder(tf.float32,(None,3),name='')
            angles_by_st = tf.placeholder(tf.float32, (None, 3), name='angles_by_st')
            expressions = tf.placeholder(tf.float32, (None, 10), name='expressions')
            attr6_gt = tf.placeholder(tf.float32, (None, 5), name='attr6_gt')
            attr6_st = tf.placeholder(tf.float32, (None, 5), name='attr6_st')

            age_gt = tf.placeholder(tf.float32, (None, 1), name='age_gt')
            age_st = tf.placeholder(tf.float32, (None, 1), name='age_st')

            gender_gt = tf.placeholder(tf.float32, (None, 1), name='gender_gt')
            gender_st = tf.placeholder(tf.float32, (None, 1), name='gender_st')

            young = tf.placeholder(tf.float32, (None, 1), name='young')
            mask = tf.placeholder(tf.float32, (None, 1), name='mask')

            def __produceAttr(attr):
                n, m = attr.shape
                attr_new = np.zeros([n, m], dtype=np.int32)
                attr_new[:, :] = np.where(attr >= 0.5, 1, 0).astype(np.int32)
                # attr_new[:, 1::2] = np.where(attr >= 0.5, 0, 1).astype(np.int32)
                # attr_new[:,:] = 1
                return attr_new

            # attribute_batch = tf.placeholder(tf.int32, shape=(None, 5), name='attribute_batch')
            attribute_batch = tf.py_func(__produceAttr, [attr6_st], tf.int32,
                                         name='attribute_batch')

            print(attribute_batch.get_shape())

            '''open_eye_gt 0 1
            open_eye_st 1 1
            mouth_open_slightly 0 1
            mouth_open_widely 0 1
            sunglasses_gt 0 1
            sunglasses_st 1 1
            forty_attr 0 33'''
            open_eye_gt = tf.placeholder(tf.float32, (None, 1), name='open_eye_gt')
            open_eye_st = tf.placeholder(tf.float32, (None, 1), name='open_eye_st')

            mouth_open_slightly = tf.placeholder(tf.float32, (None, 1), name='mouth_open_slightly')
            mouth_open_widely = tf.placeholder(tf.float32, (None, 1), name='mouth_open_widely')

            sunglasses_gt = tf.placeholder(tf.float32, (None, 1), name='sunglasses_gt')
            sunglasses_st = tf.placeholder(tf.float32, (None, 1), name='sunglasses_st')

            forty_attr = tf.placeholder(tf.float32, (None, 33), name='forty_attr')

            indicates = tf.placeholder(tf.float32, (None, 21), name='indicates')

            loss_weights = tf.placeholder(tf.float32,[14],name='loss_weights')
            list_ops['loss_weights'] = loss_weights

            list_ops['image_batch'] = image_batch
            list_ops['landmark_batch'] = landmark_batch
            list_ops['attribute_batch'] = attribute_batch
            list_ops['euler_angles_gt_batch'] = euler_angles_gt_batch
            list_ops['race_gt_batch'] = race_gt
            list_ops['race_st_batch'] = race_st
            list_ops['angles_by_st_batch'] = angles_by_st
            list_ops['expressions_baatch'] = expressions
            list_ops['attr6_gt_batch'] = attr6_gt
            list_ops['attr6_st_batch'] = attr6_st
            list_ops['age_gt_batch'] = age_gt
            list_ops['age_st_batch'] = age_st

            list_ops['gender_gt_batch'] = gender_gt
            list_ops['gender_st_batch'] = gender_st

            list_ops['young_batch'] = young
            list_ops['mask_batch'] = mask

            list_ops['open_eye_gt_batch'] = open_eye_gt
            list_ops['open_eye_st_batch'] = open_eye_st

            list_ops['mouth_open_slightly_batch'] = mouth_open_slightly
            list_ops['mouth_open_widely_batch'] = mouth_open_widely

            list_ops['sunglasses_gt_batch'] = sunglasses_gt
            list_ops['sunglasses_st_batch'] = sunglasses_st

            list_ops['forty_attr_batch'] = forty_attr

            list_ops['indicates_batch'] = indicates

            phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
            list_ops['phase_train_placeholder'] = phase_train_placeholder

            print('Building training graph.')
            # total_loss, landmarks, heatmaps_loss, heatmaps= create_model(image_batch, landmark_batch,\
            #                                                                                phase_train_placeholder, args)
            create_model = None
            if args.model_version == 1:
                create_model = create_model_v1
            elif args.model_version == 2:
                create_model = create_model_v2
            elif args.model_version == 4:
                create_model = create_model_v4

            landmarks_pre, race3_pre, race4_pre, angles_gt_pre, angles_st_pre, expressions_pre, attr6_pre, age_pre, \
            gender_pre, young_pre, mask_pre, open_eye_pre, open_mouth_degrees_pre, sunglasses_pre, forty_attrs_pre, \
            euler_angles_pre = create_model(image_batch, phase_train_placeholder, args)

            list_ops['landmarks_pre'] = landmarks_pre
            list_ops['race3_pre'] = race3_pre
            list_ops['race4_pre'] = race4_pre
            list_ops['angles_gt_pre'] = angles_gt_pre
            list_ops['angles_st_pre'] = angles_st_pre
            list_ops['expressions_pre'] = expressions_pre
            list_ops['attr6_pre'] = attr6_pre
            list_ops['age_pre'] = age_pre
            list_ops['gender_pre'] = gender_pre
            list_ops['young_pre'] = young_pre
            list_ops['mask_pre'] = mask_pre
            list_ops['open_eye_pre'] = open_eye_pre
            list_ops['open_mouth_degrees_pre'] = open_mouth_degrees_pre
            list_ops['sunglasses_pre'] = sunglasses_pre
            list_ops['forty_attrs_pre'] = forty_attrs_pre
            list_ops['euler_angles_pre'] = euler_angles_pre

            landmarks_pre_lmk_loss = tf.reduce_sum(tf.square(landmarks_pre - landmark_batch), axis=1) * indicates[:,
                                                                                                        0]  # [0,1)
            race_gt_loss = tf.nn.softmax_cross_entropy_with_logits(labels=race_gt, logits=race4_pre) * indicates[:, 1]
            race_st_loss = tf.nn.softmax_cross_entropy_with_logits(labels=race_st, logits=race3_pre) * indicates[:, 2]

            angles_gt_loss = tf.reduce_sum((1 - tf.cos(euler_angles_gt_batch - angles_gt_pre)), axis=1) * indicates[:,
                                                                                                          3]  # rad real
            angles_st_loss = tf.reduce_sum((1 - tf.cos(angles_by_st - angles_st_pre)), axis=1) * indicates[:, 4]  # rad real

            # before Nov 22
            # xxx : sum = 100?
            expressions_loss = tf.nn.softmax_cross_entropy_with_logits(labels=expressions,
                                                                       logits=expressions_pre) * indicates[:,
                                                                                                 5]  # [0,1], probability

            # after Nov 22
            # expressions_loss = tf.square(expressions_pre - expressions) * indicates[:,5:6]                                  # [0, 1], probability sum=1

            attr6_gt_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=attr6_gt, logits=attr6_pre),
                                          axis=1) * indicates[:, 6]
            attr6_st_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=attr6_st, logits=attr6_pre),
                                          axis=1) * indicates[:, 7]  # [0,1]

            # way 1
            attr6_combined_loss = (attr6_gt_loss + attr6_st_loss) / (indicates[:, 6] + indicates[:, 7])

            # # way 2
            # attr6_combined_loss = (attr6_gt_loss + attr6_st_loss ) * indicates[:,6:7] * indicates[:,7:8] \
            # / (indicates[:,6:7] + indicates[:,7:8] + 1e-10)

            age_gt_loss = tf.square(age_gt - age_pre)[:,0] * indicates[:, 8]  # x/100.0
            age_st_loss = tf.square(age_st - age_pre)[:,0] * indicates[:, 9]  # x/100.0
            age_combined_loss = (age_gt_loss + age_st_loss) / (indicates[:, 8] + indicates[:, 9])

            # sum_indicates_age = indicates[:,8:9] + indicates[]
            # sigmoid
            gender_gt_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=gender_gt, logits=gender_pre)[:,0] * indicates[:,
                                                                                                            10]
            gender_st_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=gender_st, logits=gender_pre)[:,0] * indicates[:,
                                                                                                            11]
            # gender_gt_loss = tf.square(gender_gt - gender_pre) * indicates[:, 10:11]    #0 or 1
            # gender_st_loss = tf.square(gender_st - gender_pre) * indicates[:, 11:12]    # [0,1] , probability
            gender_combined_loss = (gender_gt_loss + gender_st_loss) / (indicates[:, 10] + indicates[:, 11])
            list_ops['gender_com_loss_array'] = gender_combined_loss
            list_ops['raw_gender_st_loss'] = tf.nn.sigmoid_cross_entropy_with_logits(labels=gender_st, logits=gender_pre)
            list_ops['raw_gender_gt_loss'] = tf.nn.sigmoid_cross_entropy_with_logits(labels=gender_gt, logits=gender_pre)
            list_ops['gender_gt_indicate'] = indicates[:,10]
            list_ops['gender_st_indicate'] = indicates[:,11]

            # sigmoid
            # young_loss = tf.square(young - young_pre) * indicates[:, 12:13]    #0 or 1
            young_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=young, logits=young_pre)[:,0] * indicates[:, 12]

            # sigmoid
            # mask_loss = tf.square(mask - mask_pre) * indicates[:, 13:14]       #[0,1]
            mask_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=mask, logits=mask_pre)[:,0] * indicates[:, 13]

            # sigmoid
            # open_eye_gt_loss = tf.square(open_eye_gt - open_eye_pre) * indicates[:, 14:15]  # 0 or 1 no
            # open_eye_st_loss = tf.square(open_eye_st - open_eye_pre) * indicates[:, 15:16]  # [0,1], probability

            open_eye_gt_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=open_eye_gt, logits=open_eye_pre)[:,0] * indicates[
                                                                                                                  :, 14]
            open_eye_st_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=open_eye_st, logits=open_eye_pre)[:,0] * indicates[
                                                                                                                  :, 15]

            open_eye_combined_loss = (open_eye_gt_loss + open_eye_st_loss) / (indicates[:, 14] + indicates[:, 15])

            # sigmoid
            # sunglasses_st_loss = tf.square(sunglasses_st - sunglasses_pre) * indicates[:, 18:19] # 0 or 1
            # sunglasses_gt_loss = tf.square(sunglasses_gt - sunglasses_pre) * indicates[:, 19:20] # [0,1],probability

            sunglasses_gt_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=sunglasses_gt,
                                                                         logits=sunglasses_pre)[:,0] * indicates[:, 18]
            sunglasses_st_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=sunglasses_st,
                                                                         logits=sunglasses_pre)[:,0] * indicates[:, 19]

            sunglasses_combined_loss = (sunglasses_gt_loss + sunglasses_st_loss) / (
                    indicates[:, 18] + indicates[:, 19])

            # forty_attr_loss = tf.nn.sigmoid(label forty_attr, forty_attrs_pre) * indicates[:, 20:21]
            forty_attr_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=forty_attr,
                                                                                    logits=forty_attrs_pre),
                                            axis=1) * indicates[:,
                                                      20]  # 0 or 1
            list_ops['raw_losses'] = []
            list_ops['raw_losses'].append(landmarks_pre_lmk_loss)
            list_ops['raw_losses'].append(race_gt_loss)
            list_ops['raw_losses'].append(race_st_loss)
            list_ops['raw_losses'].append(angles_gt_loss)
            list_ops['raw_losses'].append(angles_st_loss)
            list_ops['raw_losses'].append(expressions_loss)


            def __reduce_sum_indicates(indicate):
                def __indicate_reduce_sum(_indicate):
                    sum_indicate = np.sum(_indicate)
                    sum_indicate = sum_indicate if sum_indicate != 0 else 1e-20
                    sum_indicate = np.float32(sum_indicate)
                    return sum_indicate
                return tf.py_func(__indicate_reduce_sum, [indicate], tf.float32)

            list_ops['landmarks_pre_lmk_loss'] = tf.reduce_sum(
                landmarks_pre_lmk_loss)/ __reduce_sum_indicates(indicates[:,0])  # each valid = mean * (sum(batch)) / batch_jd / 212
            list_ops['race_gt_loss'] = tf.reduce_sum(race_gt_loss) / __reduce_sum_indicates(indicates[:,1])  # each valid = mean * (sum(batch)) / batch_lfw / 4
            list_ops['race_st_loss'] = tf.reduce_sum(race_st_loss) / __reduce_sum_indicates(indicates[:,2])  # each valid = mean / 3
            list_ops['angles_gt_loss'] = tf.reduce_sum(angles_gt_loss) / __reduce_sum_indicates(indicates[:, 3])
            list_ops['angles_st_loss'] = tf.reduce_sum(angles_st_loss) / __reduce_sum_indicates(indicates[:, 4])  # each valid = mean / 3
             # each valid = mean * sum(batch) / batch_jd / 3
            # list_ops['angles_gt_loss'] = angles_gt_loss
            list_ops['expressions_loss'] = tf.reduce_sum(expressions_loss) / __reduce_sum_indicates(indicates[:,5])  # each valid = mean / 10

            list_ops['attr6_combined_loss'] = tf.reduce_mean(attr6_combined_loss)  # each valid = mean / 5

            list_ops['age_gt_loss'] = tf.reduce_sum(age_gt_loss) / __reduce_sum_indicates(indicates[:,8]) # each valid = mean * sum(batch) / batch_imdb
            list_ops['age_st_loss'] = tf.reduce_sum(age_st_loss) / __reduce_sum_indicates(indicates[:,9]) # each valid = mean
            list_ops['age_combined_loss'] = tf.reduce_mean(age_combined_loss)

            list_ops['gender_gt_loss'] = tf.reduce_sum(
                gender_gt_loss) / __reduce_sum_indicates(indicates[:,10]) # each valid = mean * sum(batch) / (batch_imdb + batch_lfw + batch_celeba)
            list_ops['gender_st_loss'] = tf.reduce_sum(gender_st_loss) / __reduce_sum_indicates(indicates[:,11])
            list_ops['gender_st_loss_sum'] = tf.reduce_sum(gender_st_loss)
            list_ops['gender_st_batch_size'] = __reduce_sum_indicates(indicates[:,11])
            list_ops['gender_combined_loss'] = tf.reduce_mean(gender_combined_loss)

            list_ops['young_loss'] = tf.reduce_sum(young_loss) / __reduce_sum_indicates(indicates[:,12]) # each valid = mean * sum(batch) / batch_celeba

            list_ops['mask_loss'] = tf.reduce_sum(mask_loss) / __reduce_sum_indicates(indicates[:,13]) # each valid = mean

            list_ops['open_eye_gt_loss'] = tf.reduce_sum(open_eye_gt_loss) / __reduce_sum_indicates(indicates[:,14])  # each valid = mean * sum(batch) / batch_lfw
            list_ops['open_eye_st_loss'] = tf.reduce_sum(open_eye_st_loss) / __reduce_sum_indicates(indicates[:,15]) # each valid = mean
            list_ops['open_eye_combined_loss'] = tf.reduce_mean(open_eye_combined_loss)  # each valid = mean

             # each valid = mean
            list_ops['sunglasses_gt_loss'] = tf.reduce_sum(
                sunglasses_gt_loss) / __reduce_sum_indicates(indicates[:,18]) # each valid = mean * sum(batch) / (batch_lfw)
            list_ops['sunglasses_st_loss'] = tf.reduce_sum(sunglasses_st_loss) / __reduce_sum_indicates(indicates[:, 19])
            list_ops['sunglasses_combined_loss'] = tf.reduce_mean(sunglasses_combined_loss) # mean

            list_ops['forty_attr_loss'] = tf.reduce_sum(
                forty_attr_loss) / __reduce_sum_indicates(indicates[:,20]) # each valid = mean * sum(batch) / (batch_lfw + batch_celeba)

            attributes_w_n = tf.to_float(attribute_batch[:])
            # _num = attributes_w_n.shape[0]
            mat_ratio = tf.reduce_mean(attributes_w_n, axis=0)
            list_ops['mat_ratio'] = mat_ratio
            mat_ratio = tf.map_fn(lambda x: (tf.cond(x > 0, lambda: 1 / x, lambda: float(sum(args.batch_size)))), mat_ratio)
            list_ops['mat_ratio2'] = mat_ratio
            list_ops['attr_w_n0'] = attributes_w_n
            attributes_w_n = tf.convert_to_tensor(attributes_w_n * mat_ratio)
            list_ops['attr_w_n1'] = attributes_w_n
            attributes_w_n = tf.reduce_sum(attributes_w_n, axis=1)
            list_ops['attributes_w_n_batch'] = attributes_w_n

            L2_loss = tf.add_n(tf.losses.get_regularization_losses())
            _sum_k = tf.reduce_sum(tf.map_fn(lambda x: 1 - tf.cos(abs(x)), euler_angles_gt_batch - euler_angles_pre),
                                   axis=1) * indicates[:, 3]
            list_ops['sum_k_loss'] = tf.reduce_sum(_sum_k) / __reduce_sum_indicates(indicates[:,3])
            loss_sum = tf.reduce_sum(tf.square(landmark_batch - landmarks_pre), axis=1) * indicates[:, 0]
            list_ops['landmark_loss'] = tf.reduce_sum(loss_sum) / __reduce_sum_indicates(indicates[:,0])
            # if args.multi_task:
            #     attr_loss = tf.reduce_sum(config.multi_task_weight * attr_loss, axis=1)
            #     loss_sum += attr_loss
            # attr_loss = tf.reduce_mean(attr_loss)
            if args.loss_type == 0:  # 不加类别加L2Loss
                loss_sum = tf.reduce_mean(loss_sum * _sum_k)
                loss_sum += L2_loss
            elif args.loss_type == 1:  # 不加类别不加L2_loss
                loss_sum = tf.reduce_mean(loss_sum * _sum_k)
                # loss_sum += L2_loss
                #
            elif args.loss_type == 2:  # 加类别加L2-loss
                list_ops['splited_loss_sum'] = loss_sum
                list_ops['splited_sum_k'] = _sum_k
                list_ops['splited_attr_w_n'] = attributes_w_n
                list_ops['splited_indicates'] = indicates[:, 0]
                loss_sum = tf.reduce_sum(loss_sum * _sum_k * attributes_w_n) / __reduce_sum_indicates(indicates[:,0])
            elif args.loss_type == 3:  # 加类别，不加L2Loss
                loss_sum = tf.reduce_mean(loss_sum * _sum_k * attributes_w_n)
                # loss_sum += L2_loss

            list_ops['pfld_sum_loss'] = loss_sum
            if not args.no_addition:
                loss_sum = loss_weights[0] * loss_sum + \
                           loss_weights[1] * list_ops['race_gt_loss'] + \
                           loss_weights[2] * list_ops['race_st_loss'] + \
                           loss_weights[3] * list_ops['angles_gt_loss'] + \
                           loss_weights[4] * list_ops['angles_st_loss'] + \
                           loss_weights[5] * list_ops['expressions_loss'] + \
                           loss_weights[6] * list_ops['attr6_combined_loss'] + \
                           loss_weights[7] * list_ops['age_combined_loss'] + \
                           loss_weights[8] * list_ops['gender_combined_loss'] + \
                           loss_weights[9] * list_ops['young_loss'] + \
                           loss_weights[10] * list_ops['mask_loss'] + \
                           loss_weights[11] * list_ops['open_eye_combined_loss'] + \
                           loss_weights[12] * list_ops['sunglasses_combined_loss'] + \
                           loss_weights[13] * list_ops['forty_attr_loss'] + \
                           L2_loss
            # loss_sum  = np.array([10],dtype=np.float32)

            # loss_sum = list_ops['landmarks_pre_lmk_loss']

            # loss_sum +=indicates[0] * list_ops['landmarks_pre_lmk_loss'] + \
            #     indicates[1] * list_ops['race_gt_loss'] + \
            #     indicates[2] *
            train_op, lr_op = train_model(loss_sum, global_step, global_num_data, gloabl_batch_size, args)

            list_ops['landmarks'] = landmarks_pre
            list_ops['L2_loss'] = L2_loss
            list_ops['loss'] = loss_sum
            list_ops['train_op'] = train_op
            list_ops['lr_op'] = lr_op
            #
            # list_ops['attr_pre'] = attr_pre
            # list_ops['attr_pre_sigmoid'] = tf.nn.sigmoid(attr_pre)
            # list_ops['attr_loss'] = attr_loss

            # test_mean_error = tf.Variable(tf.constant(0.0), dtype=tf.float32, name='ME', trainable=False)
            # test_failure_rate = tf.Variable(tf.constant(0.0), dtype=tf.float32, name='FR', trainable=False)
            # test_10_loss = tf.Variable(tf.constant(0.0), dtype=tf.float32, name='TestLoss', trainable=False)
            # train_loss = tf.Variable(tf.constant(0.0), dtype=tf.float32, name='TrainLoss', trainable=False)
            # train_loss_l2 = tf.Variable(tf.constant(0.0), dtype=tf.float32, name='TrainLoss2', trainable=False)
            # train_attr_loss = tf.Variable(tf.constant(0.0), dtype=tf.float32, name='Train_attr_loss', trainable=False)
            # test_attr_loss = tf.Variable(tf.constant(0.0), dtype=tf.float32, name='Test_attr_loss', trainable=False)
            # train_landmark_loss = tf.Variable(tf.constant(0.0), dtype=tf.float32, name='train_landmark_loss',
            #                                   trainable=False)

            # loss_list[]
            loss_name_list = '''
            loss
            pfld_sum_loss
            landmark_loss
            sum_k_loss
            landmarks_pre_lmk_loss
            race_gt_loss
            race_st_loss
            angles_gt_loss
            angles_st_loss
            expressions_loss
            attr6_combined_loss
            age_gt_loss
            age_st_loss
            age_combined_loss
            gender_gt_loss
            gender_st_loss
            gender_combined_loss
            young_loss
            mask_loss
            open_eye_gt_loss
            open_eye_st_loss
            open_eye_combined_loss
            sunglasses_gt_loss
            sunglasses_st_loss
            sunglasses_combined_loss
            forty_attr_loss'''.split()
            loss_list = {}
            for key in loss_name_list:
                loss_true_placeholder = tf.placeholder(shape=(), dtype=tf.float32, name='%s_ph' % key)
                loss_true_test_place_holder = tf.placeholder(shape=(), dtype=tf.float32, name='%s_test_ph' % key)
                
                loss_place_holder = tf.Variable(tf.constant(0.0), dtype=tf.float32, name='%s_var' % key, trainable=False)
                loss_test_place_holder = tf.Variable(tf.constant(0.0), dtype=tf.float32, name='%s_test_var' % key,
                                                     trainable=False)

                tf.summary.scalar(key, loss_place_holder)
                tf.summary.scalar(key + '_test', loss_test_place_holder)

                list_ops[key+'_loss_place_holder'] = loss_true_placeholder
                list_ops[key+'_test_loss_place_holder'] = loss_true_test_place_holder

                loss_list[key+'_assgin'] = loss_place_holder.assign(loss_true_placeholder)
                loss_list[key+'_test_assign'] = loss_test_place_holder.assign(loss_true_test_place_holder)

            merged = tf.summary.merge_all()

        # loss_lmk = __test(sess, next_batch_lmk, [list_ops['landmarks_pre_lmk_loss'], list_ops['angles_gt_loss']])
        # loss_attr = __test(sess, next_batch_attr,
        #                    [list_ops['attr6_combined_loss'], list_ops['open_eye_gt_loss'],
        #                     list_ops['sunglasses_st_loss'],
        #                     list_ops['expressions_loss']])

        # test_loss_attr_list = ['attr6_combined_loss', 'open_eye_gt_loss', 'sunglasses_st_loss', 'expressions_loss']
        # test_loss_lmk_list = ['landmarks_pre_lmk_loss', 'angles_gt_loss']
        # test_loss_list = {}
        # for loss_name in test_loss_attr_list+test_loss_lmk_list:
        #     loss_place_holder = tf.Variable(tf.constant(0.0), dtype=tf.float32, name = '%s_test_var' % loss_name, trainable=False)
        #     tf.summary.scalar('Test_%s' % loss_name, loss_place_holder)
        #     loss_list[key + '_test_place_holder'] = loss_place_holder

        # eulerangle_loss = tf.Variable(tf.constant(0.0), dtype=tf.float32, name='eulerAngleLoss')
        # tf.summary.scalar('test_mean_error', test_mean_error)
        # tf.summary.scalar('test_failure_rate', test_failure_rate)
        # tf.summary.scalar('test_10_loss', test_10_loss)
        # tf.summary.scalar('train_loss', train_loss)
        # tf.summary.scalar('train_loss_l2', train_loss_l2)
        # tf.summary.scalar('train_attr_loss', train_attr_loss)
        # tf.summary.scalar('test_attr_loss', test_attr_loss)
        # tf.summary.scalar('train_landmark_loss', train_landmark_loss)
        # tf.summary.scalar('euler_angle_loss', eulerangle_loss)

        save_params = tf.trainable_variables()
        saver = tf.train.Saver(save_params, max_to_keep=None)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)

        tfConfig = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=False, log_device_placement=False)
        tfConfig.gpu_options.allow_growth = True
        sess = tf.Session(config=tfConfig)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        with sess.as_default():
            epoch_start = 0
            try:
                if args.pretrained_model:
                    pretrained_model = args.pretrained_model
                    if (not os.path.isdir(pretrained_model)):
                        print('Restoring pretrained model: {}'.format(pretrained_model))
                        saver.restore(sess, args.pretrained_model)
                    else:
                        print('Model directory: {}'.format(pretrained_model))
                        ckpt = tf.train.get_checkpoint_state(pretrained_model)
                        model_path = ckpt.model_checkpoint_path
                        assert (ckpt and model_path)
                        epoch_start = int(model_path[model_path.find('model.ckpt-') + 11:]) + 1
                        print('Checkpoint file: {}'.format(model_path))
                        saver.restore(sess, model_path)
            except:
                epoch_start = 0
                pass

            # if args.save_image_example:
            #     save_image_example(sess, list_ops, args)
            sess.graph.finalize()

            print('Running train.')


            # for j, key in enumerate(loss_name_list):
                # loss_list[key+'_assgin'] = loss_place_holder.assign(loss_true_placeholder)
                # loss_list[key+'_test_assign']
                # loss_list[key + '_place_holder'].assign(training_loss[i])
                # loss_var_list.append(loss_list[key+'_assgin'])
                # loss_var_list.append(loss_list[key+'_test_assign'])
            train_write = tf.summary.FileWriter(log_dir, sess.graph)
            # loss_weights = np.ones([14],dtype=np.float32)

            testing_loss = None
            smooth_losses = None
            loss_weights = None

            lastNodesSet = None

            all_objects = None

            for epoch in range(epoch_start, args.max_epoch):

                # gen_loss_weights(sess,)
                # if loss_weights is None and os.path.exists(args.model_dir+'/.dyn_temp_testing_loss.npy'):
                    # testing_loss = np.load(args.model_dir+'/.dyn_temp_testing_loss.npy')
                    # smooth_losses = np.load(args.model_dir+'/.dyn_temp_smooth_loss.npy')
                    # loss_weights = np.load(args.model_dir+'/.dyn_temp_loss_weights.npy')

                loss_weights, smooth_losses, weights_changed = gen_loss_weights(testing_loss,smooth_losses,loss_weights,epoch)
                # np.save(args.model_dir+'/.dyn_temp_smooth_loss.npy', smooth_losses)
                # if weights_changed:
                    # np.save(args.model_dir+'/.dyn_temp_loss_weights.npy', loss_weights)

                print(loss_weights)
                print(smooth_losses)
                start = time.time()
                # loss_weights[1:] = np.where(loss_weights[1:] > 0.001, 0.001, loss_weights[1:])
                training_loss = train(sess, epoch_size, epoch, list_ops, args,loss_weights)

                print("train time: {}".format(time.time() - start))
                checkpoint_path = os.path.join(model_dir, 'model.ckpt')
                metagraph_path = os.path.join(model_dir, 'model.meta')
                saver.save(sess, checkpoint_path, global_step=epoch, write_meta_graph=False)
                if not os.path.exists(metagraph_path):
                    saver.export_meta_graph(metagraph_path)

                start = time.time()
                testing_loss = test_v41(sess, test_epoch_size, epoch, list_ops, args,loss_weights)
                # np.save(args.model_dir+'/.dyn_temp_testing_loss.npy', testing_loss)
                print("test time: {}".format(time.time() - start))
                # print(training_loss)
                # loss_var_list = [merged]
                
                    # print('%s %s' % (key + '_place_holder', training_loss[j]))
                # for j, key in enumerate(test_loss_attr_list):

                # loss_list = [merged]
                # for key in list_ops.keys():
                #     if key[-5:] == '_loss':
                #         loss_list.append()

                # summary, _ = sess.run(
                #     [merged
                # loss_list['young_loss_place_holder'].assign(training_loss[12])
                # train_loss.assign(training_loss[0]),
                # train_loss_l2.assign(train_L2),
                # train_attr_loss.assign(trainAttrLoss),
                # train_landmark_loss.assign(training_loss[1])
                # ])
                feed_dict={}
                assign_list = []
                for key, train_loss, test_loss in zip(loss_name_list, training_loss, testing_loss):
                #      list_ops[key+'_loss_place_holder'] = loss_true_placeholder
                # list_ops[key+'_test_loss_place_holder'] = loss_true_test_place_holder
                    feed_dict[list_ops[key+'_loss_place_holder']] = train_loss
                    feed_dict[list_ops[key+'_test_loss_place_holder']] = test_loss
                #     loss_list[key+'_assgin'] = loss_place_holder.assign(loss_true_placeholder)
                # loss_list[key+'_test_assign']
                    assign_list.append(loss_list[key+'_assgin'])
                    assign_list.append(loss_list[key+'_test_assign'])

                sess.run(assign_list,feed_dict=feed_dict)
                summary = sess.run(merged)
                train_write.add_summary(summary, epoch)

def train(sess, epoch_size, epoch, list_ops, args, loss_weights):
    loss_name_list = '''
            loss
            pfld_sum_loss
            landmark_loss
            sum_k_loss
            landmarks_pre_lmk_loss
            race_gt_loss
            race_st_loss
            angles_gt_loss
            angles_st_loss
            expressions_loss
            attr6_combined_loss
            age_gt_loss
            age_st_loss
            age_combined_loss
            gender_gt_loss
            gender_st_loss
            gender_combined_loss
            young_loss
            mask_loss
            open_eye_gt_loss
            open_eye_st_loss
            open_eye_combined_loss
            sunglasses_gt_loss
            sunglasses_st_loss
            sunglasses_combined_loss
            forty_attr_loss'''.split()

    # image_batch, landmarks_batch, attribute_batch, euler_batch, attr4_batch = list_ops['train_next_element']
    next_batches = []
    for i in range(len(args.batch_size)):
        next_batches.append(list_ops['train_data_set_%d' % i])
    # for key in list_ops.keys():
    #     if 'train_data_set_' == key[:15]:

    # video_batch = None
    # if 'train_video_next_element' in list_ops:
    #     video_batch = list_ops['train_video_next_element']
    # video_batch = []
    # print(epoch_size)
    losses = []
    losses.append(list_ops['lr_op'])
    losses.append(list_ops['train_op'])
    for name in loss_name_list:
        losses.append(list_ops[name])

    # losses = \
    #     [list_ops['loss']] + \
    #     [list_ops['landmarks_pre_lmk_loss']] + \
    #     [list_ops['race_gt_loss']] + \
    #     [list_ops['race_st_loss']] + \
    #     [list_ops['angles_gt_loss']] + \
    #     [list_ops['angles_st_loss']] + \
    #     [list_ops['expressions_loss']] + \
    #     [list_ops['attr6_combined_loss']] + \
    #     [list_ops['age_gt_loss']] + \
    #     [list_ops['age_st_loss']] + \
    #     [list_ops['gender_gt_loss']] + \
    #     [list_ops['gender_st_loss']] + \
    #     [list_ops['young_loss']] + \
    #     [list_ops['mask_loss']] + \
    #     [list_ops['open_eye_gt_loss']] + \
    #     [list_ops['open_eye_st_loss']] + \
    #     [list_ops['sunglasses_gt_loss']] + \
    #     [list_ops['sunglasses_st_loss']] + \
    #     [list_ops['forty_attr_loss']] + \
    #     [list_ops['lr_op']]
    total_loss = np.array([0.0] * (len(losses) - 2), dtype=np.float32)
    total_batch = 0.0
    landmark_diffs = []
    training = True
    # total_losses = []
    for i in range(epoch_size):
        # print('training btach %d ...' % i)
        # TODO : get the w_n and euler_angles_gt_batch
        # for bi in range(2889):
        filename = []
        feeding_data = []
        # print('len next batches : %d' % len(next_batches))
        for k, next_batch in enumerate(next_batches):
            # if k == 0:
            #     filename = next_batch
            #     continue

            data_batch = sess.run(next_batch)
            # print ('len %d' % len(data_batch[0]))
            while args.batch_size[k] != len(data_batch[0]):
                data_batch = sess.run(next_batch)
                # print(data_batch.shape)
                # print('batch size : %d len next_natch : %d' % (args.batch_size[k], len(data_batch[0])))
                # training = False
                # break
            if len(feeding_data) == 0:
                for j, data_item in enumerate(data_batch):
                    if j == 0:
                        data_item = data_item.astype(np.str).reshape([-1, 1])
                    else:
                        data_item = data_item.astype(np.float32)
                    feeding_data.append(data_item)

            else:
                for j, data_item in enumerate(data_batch):
                    if j == 0:
                        data_item = data_item.astype(np.str).reshape([-1, 1])
                    else:
                        data_item = data_item.astype(np.float32)
                    # print (feeding_data[j].shape, data_item.shape)
                    # input()
                    feeding_data[j] = np.vstack((feeding_data[j], data_item))

        # index = list(range(0, len(feeding_data[0])))
        # np.random.shuffle(index)
        # for j in range(len(feeding_data)):
        #     feeding_data[j] = feeding_data[j][index]
        # print(feeding_data)
        # for fd in feeding_data:
        # print(fd.shape)
        # input()
        # continue

        # if not training:
        #     break

        '''
        calculate the w_n: return the batch [-1,1]
        c :
        #201: 表情(expression)   0->正常表情(normal expression)    1->夸张的表情(exaggerate expression)
        #202: 照度(illumination) 0->正常照明(normal illumination)  1->极端照明(extreme illumination)
        #203: 化妆(make-up)      0->无化妆(no make-up)             1->化妆(make-up)
        #204: 遮挡(occlusion)    0->无遮挡(no occlusion)           1->遮挡(occlusion)
        #205: 模糊(blur)         0->清晰(clear)                    1->模糊(blur)
        '''

        # attributes_w_n = sess.run(list_ops['attributes_w_n_batch'],
        #                           feed_dict={list_ops['attr6_st_batch']: feeding_data[8],list_ops['indicates_batch']:feeding_data[-1]})

        # for i in range(len(feeding_data)):
        # feeding_data[i][:] = 0.0

        # np.save('/lp2/img.npy',feeding_data[0])
        # np.save('/lp2/lmk.npy',feeding_data[1])
        # exit()

        # for j in range(len(feeding_data)):
        # feeding_data[j] = feeding_data[j].astype(np.float32)

        # print(feeding_data[-1][:,0:1])

        feed_dict = {
            list_ops['image_batch']: feeding_data[1],
            list_ops['landmark_batch']: feeding_data[2],
            # list_ops['attribute_batch']: feeding_data[9],
            list_ops['phase_train_placeholder']: True,
            list_ops['euler_angles_gt_batch']: feeding_data[5],
            # list_ops['attributes_w_n_batch']: attributes_w_n,
            list_ops['race_gt_batch']: feeding_data[3],
            list_ops['race_st_batch']: feeding_data[4],
            list_ops['angles_by_st_batch']: feeding_data[6],
            list_ops['expressions_baatch']: feeding_data[7],
            list_ops['attr6_gt_batch']: feeding_data[8],
            list_ops['attr6_st_batch']: feeding_data[9],
            list_ops['age_gt_batch']: feeding_data[10],
            list_ops['age_st_batch']: feeding_data[11],
            list_ops['gender_gt_batch']: feeding_data[12],
            list_ops['gender_st_batch']: feeding_data[13],
            list_ops['young_batch']: feeding_data[14],
            list_ops['mask_batch']: feeding_data[15],

            list_ops['open_eye_gt_batch']: feeding_data[16],
            list_ops['open_eye_st_batch']: feeding_data[17],

            list_ops['mouth_open_slightly_batch']: feeding_data[18],
            list_ops['mouth_open_widely_batch']: feeding_data[19],

            list_ops['sunglasses_gt_batch']: feeding_data[20],
            list_ops['sunglasses_st_batch']: feeding_data[21],

            list_ops['forty_attr_batch']: feeding_data[-2],

            list_ops['indicates_batch']: feeding_data[-1],

            list_ops['loss_weights']:loss_weights
        }

        # age_diff = np.abs(feeding_data[10] - feeding_data[11])
        # print(age_diff).reshape([-1,1])

        # print (feeding_data[9])

        # datas = []
        # datas.append(list_ops['splited_loss_sum']) 
        # datas.append(list_ops['splited_sum_k'])
        # datas.append(list_ops['splited_attr_w_n']) 
        # datas.append(list_ops['splited_indicates'])
        # datas = sess.run(datas,feed_dict = feed_dict)
        # datas = np.array(datas).transpose()
        # # print (datas)

        # aw0, aw1, mat1, mat2, a, aw,l, s, L = sess.run([list_ops['attr_w_n0'],list_ops['attr_w_n1'], list_ops['mat_ratio'], list_ops['mat_ratio2'], list_ops['attribute_batch'], list_ops['attributes_w_n_batch'], list_ops['landmark_loss'], list_ops['sum_k_loss'], list_ops['loss']],feed_dict=feed_dict)

        # # print ('aw0 ', aw0.shape, 'aw1 ',aw1.shape,'mat1', mat1.shape,'mat2', mat2.shape,'aw', aw.shape)
        # # print(a.mean(axis=0), l, s, L)

        # shapes_to_com = [list_ops['splited_loss_sum']]+ \
        #     [list_ops['splited_sum_k']] + \
        #     [list_ops['splited_attr_w_n'] ] + \
        #     [list_ops['splited_indicates']] 

        # shapes = sess.run(shapes_to_com,feed_dict = feed_dict)

        # shapes = sess.run(list_ops['raw_losses'],feed_dict = feed_dict)
        # for shape in shapes:
        #     print (shape.shape)
        # exit()

        # exit()

        '''
        list_ops['image_batch'] = image_batch
        list_ops['landmark_batch'] = landmark_batch
        list_ops['attribute_batch'] = attribute_batch
        list_ops['euler_angles_gt_batch'] = euler_angles_gt_batch
        list_ops['race_gt_batch'] = race_gt
        list_ops['race_st_batch'] = race_st
        list_ops['angles_by_st_batch'] = angles_by_st
        list_ops['expressions_baatch'] = expressions
        list_ops['attr6_gt_batch'] = attr6_gt
        list_ops['attr6_st_batch'] = attr6_st
        list_ops['age_gt_batch'] = age_gt
        list_ops['age_st_batch'] = age_st

        list_ops['gender_gt_batch'] = gender_gt
        list_ops['gender_st_batch'] = gender_st

        list_ops['young_batch'] = young
        list_ops['mask_batch'] = mask

        list_ops['open_eye_gt_batch'] = open_eye_gt
        list_ops['open_eye_st_batch'] = open_eye_st

        list_ops['mouth_open_slightly_batch'] = mouth_open_slightly
        list_ops['mouth_open_widely_batch'] = mouth_open_widely

        list_ops['sunglasses_gt_batch'] = sunglasses_gt
        list_ops['sunglasses_st_batch'] = sunglasses_st

        list_ops['forty_attr_batch'] = forty_attr

        list_ops['indicates_batch'] = indicates'''

        # losses = [list_ops['train_op']] + \

        # [list_ops['']]
        # sess.run(list_ops['train_op'], feed_dict=feed_dict)

        # fortyattr_pre = sess.run(list_ops['forty_attrs_pre'],feed_dict=feed_dict)
        # print(fortyattr_pre.shape, feeding_data[21].shape, feeding_data[22].shape)
        # # the_open_eye_gt = np.hstack((filename.reshape([-1,1]),fortyattr_pre, feeding_data[21], feeding_data[22]))
        # print(np.hstack((filename.reshape([-1,1]), feeding_data[22][:,20:21])))
        # temp_loss = np.square(fortyattr_pre - feeding_data[21]) * feeding_data[22][:,20:21]
        # print(np.mean(temp_loss))
        # print(the_open_eye_gt)

        # loss_weights[0] * loss_sum + \
        # loss_weights[1] * list_ops['race_gt_loss'] + \
        # loss_weights[2] * list_ops['race_st_loss'] + \
        # loss_weights[3] * list_ops['angles_gt_loss'] + \
        # loss_weights[4] * list_ops['angles_st_loss'] + \
        # loss_weights[5] * list_ops['expressions_loss'] + \
        # loss_weights[6] * list_ops['attr6_combined_loss'] + \
        # loss_weights[7] * list_ops['age_combined_loss'] + \
        # loss_weights[8] * list_ops['gender_combined_loss'] + \
        # loss_weights[9] * list_ops['young_loss'] + \
        # loss_weights[10] * list_ops['mask_loss'] + \
        # loss_weights[11] * list_ops['open_eye_combined_loss'] + \
        # loss_weights[12] * list_ops['sunglasses_combined_loss'] + \
        # loss_weights[13] * list_ops['forty_attr_loss']

        # loss_weights = np.ones([14],dtype=np.float32)
        # if epoch

        losses_temp = sess.run(losses, feed_dict=feed_dict)

        # list_ops['gender_st_loss_sum'] = tf.reduce_sum(gender_st)
        # list_ops['gender_st_batch_size'] = __reduce_sum_indicates(indicates[:, 11])

        # gender_st_loss_sum,gender_st_batch_size , gender_combined_loss_array = sess.run([list_ops['gender_st_loss_sum'], list_ops['gender_st_batch_size'], list_ops['gender_com_loss_array']], feed_dict=feed_dict)
        # print(gender_st_loss_sum, gender_st_batch_size, gender_combined_loss_array)
        # # list_ops['raw_gender_st_loss'] = gender_st_loss
        # # list_ops['raw_gender_gt_loss'] = gender_gt_loss
        #
        # raw_gender_st_loss, raw_gender_gt_loss = sess.run([list_ops['raw_gender_st_loss'], list_ops['raw_gender_gt_loss']], feed_dict = feed_dict)
        # print(raw_gender_st_loss.shape, raw_gender_gt_loss.shape, gender_combined_loss_array.shape)
        # # list_ops['gender_gt_indicate'] = indicates[:, 10]
        # # list_ops['gender_st_indicate'] = indicates[:, 11]
        # gender_gt_indicate, gender_st_indicate = sess.run([list_ops['gender_gt_indicate'], list_ops['gender_st_indicate']],feed_dict = feed_dict)
        #
        # print(gender_gt_indicate.shape, gender_st_indicate.shape)
        #
        # input()

        # attribute_w_n = sess.run(list_ops['attributes_w_n_batch'], feed_dict=feed_dict)
        # attribute = sess.run(list_ops['attribute_batch'], feed_dict=feed_dict)
        # print('attr w n shape:')
        # print(attribute_w_n.shape)
        # print(attribute)
        # print(np.mean(attribute_w_n))
        lmk_indicate = feeding_data[-1][:,0]
        lmk_indicate = np.where(lmk_indicate > 0 ,True, False)


        landmarks_pre = sess.run(list_ops['landmarks_pre'], feed_dict=feed_dict)
        landmarks_gt = feeding_data[2].reshape([-1, 106, 2])[lmk_indicate]
        landmarks_pre = landmarks_pre.reshape([-1, 106, 2])[lmk_indicate]
        landmark_diff = np.mean(np.sqrt(np.sum(np.square(landmarks_pre - landmarks_gt), axis=2)), axis=1)
        box = np.zeros([len(landmarks_gt), 2, 2])
        box[:, 0, :] = np.min(landmarks_gt, axis=1)
        box[:, 1, :] = np.max(landmarks_gt, axis=1)
        wh = box[:, 1] - box[:, 0]
        area = wh[:, 0] * wh[:,1]
        landmark_diff = landmark_diff / np.sqrt(area)
        landmark_diff = np.mean(landmark_diff)
        landmark_diffs.append(landmark_diff)

        if ((i + 1) % 10) == 0 or (i + 1) == epoch_size:
            # print('printing ...')
            Epoch = 'Epoch:[{:<4}][{:<4}/{:<4}]'.format(epoch, i + 1, epoch_size)

            # Loss = 'Loss {:2.3f}\tLmk_loss {:2.3f}'.format(losses_temp[0], losses_temp[1])
            Loss = '\n'
            for j, l in enumerate(losses_temp[2:]):
                Loss += '%s  %.4f\n' % (loss_name_list[j], l)
            # print('{}\t{}\t lr {:2.3}'.format(Epoch, Loss, losses_temp[-1]))

            print('%s %s' % (Epoch, Loss))
        losses_temp = np.array(losses_temp[2:], dtype=np.float32)
        total_loss += losses_temp
        total_batch += 1.0
        print('train_landmark_diff: %.5f' % landmark_diff)
        # print(losses_temp)
    print('\nwhole_train_landmark_diff %.5f' % np.mean(landmark_diffs))
    return total_loss / total_batch


def test_v41(sess, epoch_size, epoch, list_ops, args,loss_weights):
    loss_name_list = '''
            loss
            pfld_sum_loss
            landmark_loss
            sum_k_loss
            landmarks_pre_lmk_loss
            race_gt_loss
            race_st_loss
            angles_gt_loss
            angles_st_loss
            expressions_loss
            attr6_combined_loss
            age_gt_loss
            age_st_loss
            age_combined_loss
            gender_gt_loss
            gender_st_loss
            gender_combined_loss
            young_loss
            mask_loss
            open_eye_gt_loss
            open_eye_st_loss
            open_eye_combined_loss
            sunglasses_gt_loss
            sunglasses_st_loss
            sunglasses_combined_loss
            forty_attr_loss'''.split()

    # image_batch, landmarks_batch, attribute_batch, euler_batch, attr4_batch = list_ops['train_next_element']
    next_batches = []
    for key in list_ops.keys():
        # print(key)
        if 'test_data_set_' == key[:14]:
            next_batches.append(list_ops[key])
    # print(len(next_batches))
    # input()

    # video_batch = None
    # if 'train_video_next_element' in list_ops:
    #     video_batch = list_ops['train_video_next_element']
    # video_batch = []
    # print(epoch_size)
    losses = []
    for name in loss_name_list:
        losses.append(list_ops[name])
    losses.append(list_ops['lr_op'])
    # losses = \
    #     [list_ops['loss']] + \
    #     [list_ops['landmarks_pre_lmk_loss']] + \
    #     [list_ops['race_gt_loss']] + \
    #     [list_ops['race_st_loss']] + \
    #     [list_ops['angles_gt_loss']] + \
    #     [list_ops['angles_st_loss']] + \
    #     [list_ops['expressions_loss']] + \
    #     [list_ops['attr6_combined_loss']] + \
    #     [list_ops['age_gt_loss']] + \
    #     [list_ops['age_st_loss']] + \
    #     [list_ops['gender_gt_loss']] + \
    #     [list_ops['gender_st_loss']] + \
    #     [list_ops['young_loss']] + \
    #     [list_ops['mask_loss']] + \
    #     [list_ops['open_eye_gt_loss']] + \
    #     [list_ops['open_eye_st_loss']] + \
    #     [list_ops['sunglasses_gt_loss']] + \
    #     [list_ops['sunglasses_st_loss']] + \
    #     [list_ops['forty_attr_loss']] + \
    #     [list_ops['lr_op']]
    total_loss = np.array([0.0] * (len(losses) - 1), dtype=np.float32)
    total_batch = 0.0
    landmarks_diffs = []

    for i in range(epoch_size):
        # print('training btach %d ...' % i)
        # TODO : get the w_n and euler_angles_gt_batch
        # for bi in range(2889):
        filename = []
        feeding_data = []
        for k, next_batch in enumerate(next_batches):
            # if k == 0:
            #     filename = next_batch
            #     continue
            data_batch = sess.run(next_batch)
            if len(feeding_data) == 0:
                for j, data_item in enumerate(data_batch):
                    if j == 0:
                        filename = data_item
                        continue
                    feeding_data.append(data_item)

            else:
                for j, data_item in enumerate(data_batch):
                    if j == 0:
                        filename = np.hstack((filename, data_item))
                        continue
                    feeding_data[j - 1] = np.vstack((feeding_data[j - 1], data_item))
        # print(feeding_data)
        # for fd in feeding_data:
        # print(fd.shape)
        # input()
        # continue

        '''
        calculate the w_n: return the batch [-1,1]
        c :
        #201: 表情(expression)   0->正常表情(normal expression)    1->夸张的表情(exaggerate expression)
        #202: 照度(illumination) 0->正常照明(normal illumination)  1->极端照明(extreme illumination)
        #203: 化妆(make-up)      0->无化妆(no make-up)             1->化妆(make-up)
        #204: 遮挡(occlusion)    0->无遮挡(no occlusion)           1->遮挡(occlusion)
        #205: 模糊(blur)         0->清晰(clear)                    1->模糊(blur)
        '''
        # print(feeding_data)
        # input()
        # attributes_w_n = sess.run(list_ops['attributes_w_n_batch'], feed_dict={list_ops['image_batch']: feeding_data[0],
        #                                                                        list_ops['attribute_batch']:
        #                                                                            feeding_data[-2]})

        # for i in range(len(feeding_data)):
        # feeding_data[i][:] = 0.0

        # print('attr6 ')
        # print(feeding_data[8])


        feed_dict = {
            list_ops['image_batch']: feeding_data[0],
            list_ops['landmark_batch']: feeding_data[1],
            # list_ops['attribute_batch']: feeding_data[-2],
            list_ops['phase_train_placeholder']: False,
            list_ops['euler_angles_gt_batch']: feeding_data[4],
            # list_ops['attributes_w_n_batch']: attributes_w_n,
            list_ops['race_gt_batch']: feeding_data[2],
            list_ops['race_st_batch']: feeding_data[3],
            list_ops['angles_by_st_batch']: feeding_data[5],
            list_ops['expressions_baatch']: feeding_data[6],
            list_ops['attr6_gt_batch']: feeding_data[7],
            list_ops['attr6_st_batch']: feeding_data[8],
            list_ops['age_gt_batch']: feeding_data[9],
            list_ops['age_st_batch']: feeding_data[10],
            list_ops['gender_gt_batch']: feeding_data[11],
            list_ops['gender_st_batch']: feeding_data[12],
            list_ops['young_batch']: feeding_data[13],
            list_ops['mask_batch']: feeding_data[14],

            list_ops['open_eye_gt_batch']: feeding_data[15],
            list_ops['open_eye_st_batch']: feeding_data[16],

            list_ops['mouth_open_slightly_batch']: feeding_data[17],
            list_ops['mouth_open_widely_batch']: feeding_data[18],

            list_ops['sunglasses_gt_batch']: feeding_data[19],
            list_ops['sunglasses_st_batch']: feeding_data[20],

            list_ops['forty_attr_batch']: feeding_data[21],

            list_ops['indicates_batch']: feeding_data[22],
            list_ops['loss_weights']:loss_weights
            # list_ops['video_abtch']: videos
        }

        # attribute_w_n = sess.run(list_ops['attributes_w_n_batch'],feed_dict=feed_dict)
        # attribute = sess.run(list_ops['attribute_batch'],feed_dict = feed_dict)
        # print('attr w n shape:')
        # print(attribute_w_n.shape)
        # print(attribute)
        # print(np.mean(attribute_w_n))

        lmk_indicate = feeding_data[-1][:, 0]
        lmk_indicate = np.where(lmk_indicate > 0, True, False)

        landmarks_pre = sess.run(list_ops['landmarks_pre'],feed_dict=feed_dict)
        landmarks_gt = feeding_data[1].reshape([-1,106,2])[lmk_indicate]
        landmarks_pre = landmarks_pre.reshape([-1,106,2])[lmk_indicate]
        landmark_diff = np.mean(np.sqrt(np.sum(np.square(landmarks_pre - landmarks_gt),axis=2)),axis=1)
        box = np.zeros([len(landmarks_gt),2,2])
        box[:,0,:] = np.min(landmarks_gt,axis=1)
        box[:,1,:] = np.max(landmarks_gt,axis=1)
        wh = box[:,1] - box[:,0]
        area = wh[:,0] * wh[:,1]
        landmark_diff = landmark_diff / np.sqrt(area)
        landmark_diff = np.mean(landmark_diff)
        landmarks_diffs.append(landmark_diff)

        '''
        list_ops['image_batch'] = image_batch
        list_ops['landmark_batch'] = landmark_batch
        list_ops['attribute_batch'] = attribute_batch
        list_ops['euler_angles_gt_batch'] = euler_angles_gt_batch
        list_ops['race_gt_batch'] = race_gt
        list_ops['race_st_batch'] = race_st
        list_ops['angles_by_st_batch'] = angles_by_st
        list_ops['expressions_baatch'] = expressions
        list_ops['attr6_gt_batch'] = attr6_gt
        list_ops['attr6_st_batch'] = attr6_st
        list_ops['age_gt_batch'] = age_gt
        list_ops['age_st_batch'] = age_st

        list_ops['gender_gt_batch'] = gender_gt
        list_ops['gender_st_batch'] = gender_st

        list_ops['young_batch'] = young
        list_ops['mask_batch'] = mask

        list_ops['open_eye_gt_batch'] = open_eye_gt
        list_ops['open_eye_st_batch'] = open_eye_st

        list_ops['mouth_open_slightly_batch'] = mouth_open_slightly
        list_ops['mouth_open_widely_batch'] = mouth_open_widely

        list_ops['sunglasses_gt_batch'] = sunglasses_gt
        list_ops['sunglasses_st_batch'] = sunglasses_st

        list_ops['forty_attr_batch'] = forty_attr

        list_ops['indicates_batch'] = indicates'''

        # losses = [list_ops['train_op']] + \

        # [list_ops['']]

        losses_temp = sess.run(losses, feed_dict=feed_dict)
        if ((i + 1) % 10) == 0 or (i + 1) == epoch_size:
            # print('printing ...')
            Epoch = 'Epoch:[{:<4}][{:<4}/{:<4}]'.format(epoch, i + 1, epoch_size)

            # Loss = 'Loss {:2.3f}\tLmk_loss {:2.3f}'.format(losses_temp[0], losses_temp[1])
            Loss = '\n'
            for j, l in enumerate(losses_temp[:-1]):
                Loss += '%s  %.4f\n' % (loss_name_list[j], l)
            # print('{}\t{}\t lr {:2.3}'.format(Epoch, Loss, losses_temp[-1]))

            print('%s %s' % (Epoch, Loss))
        losses_temp = np.array(losses_temp[:-1], dtype=np.float32)
        total_loss += losses_temp
        total_batch += 1.0
        # print(losses_temp)

    print('test_landmark_diff %.5f' % np.mean(landmarks_diffs))
    return total_loss / total_batch


def test_v4(sess, list_ops, loss_list_names, args):
    '''
     list_ops['test_lmk_next'] = test_lmk_data_set_batch_next
        list_ops['test_attr_next'] = test_attribute_data_set_batch_next
    :param sess:
    :param list_ops:
    :param args:
    :return:
    '''
    # assert loss_lists

    loss_name_list = '''
                loss
                pfld_sum_loss
                sum_k_loss
                landmarks_pre_lmk_loss
                race_gt_loss
                race_st_loss
                angles_gt_loss
                angles_st_loss
                expressions_loss
                attr6_combined_loss
                age_gt_loss
                age_st_loss
                gender_gt_loss
                gender_st_loss
                young_loss
                mask_loss
                open_eye_gt_loss
                open_eye_st_loss
                sunglasses_gt_loss
                sunglasses_st_loss
                forty_attr_loss'''.split()

    def __test(sess, next_batch, loss_list):
        filename = []
        feeding_data = []
        data_batch = sess.run(next_batch)
        if len(feeding_data) == 0:
            for j, data_item in enumerate(data_batch):
                if j == 0:
                    filename = data_item
                    continue
                feeding_data.append(data_item)

        else:
            for j, data_item in enumerate(data_batch):
                if j == 0:
                    filename = np.hstack((filename, data_item))
                    continue
                feeding_data[j - 1] = np.vstack((feeding_data[j - 1], data_item))

        feed_dict = {
            list_ops['image_batch']: feeding_data[0],
            list_ops['landmark_batch']: feeding_data[1],
            list_ops['attribute_batch']: feeding_data[-2],
            list_ops['phase_train_placeholder']: False,
            list_ops['euler_angles_gt_batch']: feeding_data[4],
            # list_ops['attributes_w_n_batch']: attributes_w_n,
            list_ops['race_gt_batch']: feeding_data[2],
            list_ops['race_st_batch']: feeding_data[3],
            list_ops['angles_by_st_batch']: feeding_data[5],
            list_ops['expressions_baatch']: feeding_data[6],
            list_ops['attr6_gt_batch']: feeding_data[7],
            list_ops['attr6_st_batch']: feeding_data[8],
            list_ops['age_gt_batch']: feeding_data[9],
            list_ops['age_st_batch']: feeding_data[10],
            list_ops['gender_gt_batch']: feeding_data[11],
            list_ops['gender_st_batch']: feeding_data[12],
            list_ops['young_batch']: feeding_data[13],
            list_ops['mask_batch']: feeding_data[14],

            list_ops['open_eye_gt_batch']: feeding_data[15],
            list_ops['open_eye_st_batch']: feeding_data[16],

            list_ops['mouth_open_slightly_batch']: feeding_data[17],
            list_ops['mouth_open_widely_batch']: feeding_data[18],

            list_ops['sunglasses_gt_batch']: feeding_data[19],
            list_ops['sunglasses_st_batch']: feeding_data[20],

            list_ops['forty_attr_batch']: feeding_data[21],

            list_ops['indicates_batch']: feeding_data[22]
            # list_ops['video_abtch']: videos
        }
        return sess.run(loss_list, feed_dict=feed_dict)

    # image_batch, landmarks_batch, attribute_batch, euler_batch, attr4_batch = list_ops['train_next_element']
    # next_batches = []
    # for key in list_ops.keys():
    #     if '' == key[:15]:
    #         next_batches.append(list_ops[key])

    next_batch_lmk = sess.run(list_ops['test_lmk_next'])
    next_batch_attr = sess.run(list_ops['test_attr_next'])
    loss_lmk = __test(sess, next_batch_lmk, [list_ops['landmarks_pre_lmk_loss'], list_ops['angles_gt_loss']])
    loss_attr = __test(sess, next_batch_attr,
                       [list_ops['attr6_combined_loss'], list_ops['open_eye_gt_loss'], list_ops['sunglasses_st_loss'],
                        list_ops['expressions_loss']])

    return (list(loss_lmk), list(loss_attr))


def test(sess, list_ops, args):
    image_batch, landmarks_batch, attribute_batch, euler_batch, attr4_batch = list_ops['test_next_element']

    sample_path = os.path.join(args.model_dir, 'HeatMaps')
    if not os.path.exists(sample_path):
        os.mkdir(sample_path)

    loss_sum = 0
    landmark_error = 0
    landmark_01_num = 0

    epoch_size = math.ceil(list_ops['num_test_file'] * 1.0 / args.batch_size)
    # tempf = open('/lp/lmk_data.txt','w')
    for i in range(epoch_size):  # batch_num
        images, landmarks, attributes, eulers, attr4_label = sess.run(
            [image_batch, landmarks_batch, attribute_batch, euler_batch, attr4_batch])
        box = np.zeros([images.shape[0], 4], dtype=np.float32)
        print('test images shape', images.shape)
        feed_dict = {
            list_ops['image_batch']: images,
            list_ops['landmark_batch']: landmarks,
            list_ops['attribute_batch']: attributes,
            list_ops['phase_train_placeholder']: False,
            list_ops['box_batch']: box
        }
        pre_landmarks, attr_pre = sess.run([list_ops['landmarks'], list_ops['attr_pre']], feed_dict=feed_dict)
        # feed_dict_for_attr = {
        #     list_ops['image_batch']:images
        #     list_ops['phase_train_placeholder']:False
        # }
        attr_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=attr4_label, logits=attr_pre)
        # pre_attr = sess.run(list_ops['attr'],feed_dict=feed_dict_for_attr)
        # attr_label = tf.stop_gradient(attr4_label)
        # attr_pre = tf.nn.softmax(attr_pre)
        # attr_pre = tf.log(attr_pre)
        # attr_loss = -attr_label*attr_pre
        # # attr_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=attr_pre, labels=attr_label)
        attr_loss = tf.reduce_mean(tf.reduce_sum(attr_loss, axis=1)) * config.multi_task_weight
        # min_attr_loss  =tf.reduce_min(tf.reduce_min(attr_loss))
        # max_attr_loss = tf.reduce_max(tf.reduce_max(attr_loss))
        # attr_loss2 = (attr_loss - min_attr_loss) / (max_attr_loss - min_attr_loss)
        # attr_loss2 = tf.reduce_mean(tf.reduce_mean(attr_loss2))
        # landmarks_t
        # lmkShape = tf.TensorShape([images.shape[0],config.point_size,2])
        # pre_lmk_for_diff = tf.reshape(pre_landmarks,lmkShape)
        # lmk_for_diff = tf.reshape(landmarks,lmkShape)

        diff = pre_landmarks - landmarks
        # pre_landmarks_out = pre_landmarks.reshape([-1,212])
        # landmarks_out = landmarks.reshape([-1,212])
        # lmk_out = np.hstack((pre_landmarks_out,landmarks_out))
        # with open('/lp/lmk_data.txt','a') as out_file:
        # for lmk_temp in lmk_out:
        #     tempf.write(' '.join(lmk_temp.astype(str))+'\n')
        diff = diff.reshape(len(images), config.point_size, 2)
        diff = np.square(diff)
        diff = np.sqrt(np.sum(diff, axis=2))
        loss = np.mean(diff, axis=1)

        lmk_for_box = landmarks.reshape([-1, config.point_size, 2])
        box = np.zeros([len(lmk_for_box), 2, 2], dtype=np.float32)
        box[:, 0] = np.min(lmk_for_box, axis=1)
        box[:, 1] = np.max(lmk_for_box, axis=1)
        wh = box[:, 1] - box[:, 0] + 1
        edge = np.sqrt(wh[:, 1] * wh[:, 0])
        loss = loss / edge

        # loss = np.sum(diff * diff)
        loss_sum += loss.sum()

        print("test: loss: %d" % loss_sum)

        for k in range(pre_landmarks.shape[0]):
            error_all_points = 0
            for count_point in range(pre_landmarks.shape[1] // 2):  # num points
                error_diff = pre_landmarks[k][(count_point * 2):(count_point * 2 + 2)] - landmarks[k][
                                                                                         (count_point * 2):(
                                                                                                 count_point * 2 + 2)]
                error = np.sqrt(np.sum(error_diff * error_diff))
                error_all_points += error
            interocular_distance = np.sqrt(np.sum(pow((landmarks[k][2 * config.eyeIndex[0]:2 * config.eyeIndex[0] + 2] -
                                                       landmarks[k][2 * config.eyeIndex[1]:2 * config.eyeIndex[1] + 2]),
                                                      2)))
            error_norm = error_all_points / (interocular_distance * config.point_size)
            landmark_error += error_norm
            if error_norm >= 0.1:
                landmark_01_num += 1

        # if i == 0:
        #     image_save_path = os.path.join(sample_path, 'img')
        #     if not os.path.exists(image_save_path):
        #         os.mkdir(image_save_path)
        #
        #     for j in range(images.shape[0]): #batch_size
        #         image = images[j]*256
        #         image = image[:,:,::-1]
        #
        #         image_i = image.copy()
        #         pre_landmark = pre_landmarks[j]
        #         h, w, _ = image_i.shape
        #         pre_landmark = pre_landmark.reshape(-1, 2) * [w, h]
        #         for (x, y) in pre_landmark.astype(np.int32):
        #             cv2.circle(image_i, (x, y), 1, (0, 0, 255))
        #         landmark = landmarks[j].reshape(-1, 2) * [w, h]
        #         for (x, y) in landmark.astype(np.int32):
        #             cv2.circle(image_i, (x, y), 1, (255, 0, 0))
        #         image_save_name = os.path.join(image_save_path, '{}.jpg'.format(j))
        #         cv2.imwrite(image_save_name, image_i)
    # tempf.close()
    loss = loss_sum / (list_ops['num_test_file'] * 1.0)
    print('Test epochs: {}\tLoss {:2.3f}'.format(epoch_size, loss))

    print('mean error and failure rate')
    landmark_error_norm = landmark_error / (list_ops['num_test_file'] * 1.0)
    error_str = 'mean error : {:2.3f}'.format(landmark_error_norm)

    failure_rate_norm = landmark_01_num / (list_ops['num_test_file'] * 1.0)
    failure_rate_str = 'failure rate: L1 {:2.3f}'.format(failure_rate_norm)
    print(error_str + '\n' + failure_rate_str + '\n')

    return landmark_error_norm, failure_rate_norm, loss, attr_loss


def heatmap2landmark(heatmap):
    landmark = []
    h, w, c = heatmap.shape
    for i in range(c):
        m, n = divmod(np.argmax(heatmap[i]), w)
        landmark.append(n / w)
        landmark.append(m / h)
    return landmark


def save_image_example(sess, list_ops, args):
    save_nbatch = 10
    save_path = os.path.join(args.model_dir, 'image_example')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    image_batch, landmarks_batch, attribute_batch = list_ops['train_next_element']

    for b in range(save_nbatch):
        images, landmarks, attributes = sess.run([image_batch, landmarks_batch, attribute_batch])
        for i in range(images.shape[0]):
            img = images[i] * 256
            img = img.astype(np.uint8)
            if args.image_channels == 1:
                img = np.concatenate((img, img, img), axis=2)
            else:
                img = img[:, :, ::-1].copy()

            land = landmarks[i].reshape(-1, 2) * img.shape[:2]
            for x, y in land.astype(np.int32):
                cv2.circle(img, (x, y), 1, (0, 0, 255))
            save_name = os.path.join(save_path, '{}_{}.jpg'.format(b, i))
            cv2.imwrite(save_name, img)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--file_list', type=str, default=config.anno_with_eulerAngle_all)
    parser.add_argument('--train_file', nargs='+', type=str)
    parser.add_argument('--test_file', nargs='+', type=str)
    parser.add_argument('--train_data_type', type=str, nargs='+')
    parser.add_argument('--test_data_type', type=str, nargs='+')
    parser.add_argument('--test_list', type=str, default=config.anno_with_eulerAngle_eval)
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--max_epoch', type=int, default=500)
    parser.add_argument('--image_size', type=int, default=112)
    parser.add_argument('--image_channels', type=int, default=3)
    parser.add_argument('--batch_size', type=int, nargs='+', default=[64])
    parser.add_argument('--pretrained_model', type=str, default=config.model_dir)
    parser.add_argument('--model_dir', type=str, default=config.model_dir)
    parser.add_argument('--log_dir', type=str, default=config.tflog_dir)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--lr_epoch', type=str, default='10,20,200,500')
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--level', type=str, default='L5')
    parser.add_argument('--save_image_example', action='store_false')
    parser.add_argument('--debug', type=str, default='False')
    parser.add_argument('--img-dir', default=config.drewimgdir_all_ssd)
    parser.add_argument('--test-img-dir', default=config.drewimgdir_ssd)
    parser.add_argument('--cuda-visibale-device', default='2')
    parser.add_argument('--loss-type', type=int, choices=[0, 1, 2, 3], default=0,
                        help='0->no class, L2_loss; 1->Neither class nor L2; 2->both class and L2; class but no L2')
    parser.add_argument('--lmk-norm', action='store_true', default=True)
    parser.add_argument('--model-version', type=int, default=1)
    parser.add_argument('--training-data-file', default=None)
    parser.add_argument('--mean-pts', default=None)
    parser.add_argument('--to-train-prefix', nargs='+', default=[])
    parser.add_argument('--lmk_weight',type=float)
    parser.add_argument('--no_addition',action='store_true',default=False)
    # parser.add_argument('--mean-pts',default=None)
    return parser.parse_args(argv)
def gen_loss_weights(training_losses=None,last_smooth_losses=None,last_losses_weights = None,epoch = 0):
    '''     loss
            pfld_sum_loss
            landmark_loss
            sum_k_loss
            landmarks_pre_lmk_loss
            race_gt_loss
            race_st_loss
            angles_gt_loss
            angles_st_loss
            expressions_loss
            attr6_combined_loss
            age_gt_loss
            age_st_loss
            age_combined_loss
            gender_gt_loss
            gender_st_loss
            gender_combined_loss
            young_loss
            mask_loss
            open_eye_gt_loss
            open_eye_st_loss
            open_eye_combined_loss
            sunglasses_gt_loss
            sunglasses_st_loss
            sunglasses_combined_loss
            forty_attr_loss'''
    # loss_index =  \
    # ''' loss_weights[0] * loss_sum +                            1
    # loss_weights[1] * list_ops['race_gt_loss'] +                5
    # loss_weights[2] * list_ops['race_st_loss'] +                6
    # loss_weights[3] * list_ops['angles_gt_loss'] +              7
    # loss_weights[4] * list_ops['angles_st_loss'] +              8
    # loss_weights[5] * list_ops['expressions_loss'] +            9
    # loss_weights[6] * list_ops['attr6_combined_loss'] +         10
    # loss_weights[7] * list_ops['age_combined_loss'] +           13
    # loss_weights[8] * list_ops['gender_combined_loss'] +        16
    # loss_weights[9] * list_ops['young_loss'] +                  17
    # loss_weights[10] * list_ops['mask_loss'] +                  18
    # loss_weights[11] * list_ops['open_eye_combined_loss'] +     21
    # loss_weights[12] * list_ops['sunglasses_combined_loss'] +   24
    # loss_weights[13] * list_ops['forty_attr_loss']              25'''.split('\n')
    # loss_index = [line.split()[-1] for line in loss_index]

    w0 = 1e-3
    addition_losses_num = 13
    smooth_losses = None
    sum_of_addition_rate = 0.1
    alpha = 0.75
    epoch_step = 5
    beta = sum_of_addition_rate / addition_losses_num
    weight_changed = False
    if last_losses_weights is None:
        losses_weights = np.ones([addition_losses_num+1],dtype=np.float32)
        losses_weights[1:] = np.where(losses_weights[1:] > w0, w0, losses_weights[1:])
    else:
        losses_weights = last_losses_weights.copy()

    if training_losses is not None:
        assert last_losses_weights is not None
        loss_mapping = [1, 5, 6, 7, 8, 9, 10, 13, 16, 17, 18, 21, 24, 25]
        losses = training_losses[loss_mapping]
        if last_smooth_losses is None:
            smooth_losses = losses
        else:
            # alpha = 0.75
            smooth_losses = alpha * losses + last_smooth_losses * (1-alpha)

        if epoch % epoch_step == 0:

            losses_weights[1:] = smooth_losses[0] * beta / (smooth_losses[1:]+1e-12)
            losses_weights[1:] = np.where(losses_weights[1:] > w0, w0, losses_weights[1:])
            losses_weights[1:] = np.where(losses_weights[1:] > last_losses_weights[1:], last_losses_weights[1:], losses_weights[1:])
            weight_changed = True
        # else:
        #     losses_weights = last_losses_weights

    return losses_weights, smooth_losses, weight_changed

# args.train_file, args.batch_size,args.train_data_type


if __name__ == '__main__':
    print(sys.argv)
    main(parse_arguments(sys.argv[1:]))
