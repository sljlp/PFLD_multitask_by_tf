import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from collections import OrderedDict as dict
import setproctitle
import os
import os.path as osp
import glob
import json
import math
import abc

from .net_utils import average_gradients, aggregate_batch, get_optimizer, get_tower_summary_dict
from .saver import load_model, Saver
from .timer import Timer
from .logger import colorlogger
from .utils import approx_equal

import cv2
import seaborn as sns
import matplotlib.pyplot as plt

def save_mergedHeatmaps(hms, path, c=5, img_res=None):
    assert len(hms.shape) == 3, 'Dimension of heatmaps should be 3, keypoints x h x w'
    n = hms.shape[0] + (0 if img_res is None else 1)
    r = n // c + (0 if n % c == 0 else 1)

    #joint_names = ['Nose', 'LEye', 'REye', 'LEar', 'REar', 'LShoulder', 'RShoulder', 'LElbow', 'RElbow', 'LWrist', 'RWrist', 'LHip', 'RHip', 'LKnee', 'Rknee', 'LAnkle', 'RAnkle']

    plt.figure()
    plt.subplots_adjust(hspace=0.4)
    for i in range(hms.shape[0]):
        ax = plt.subplot(r, c, i + 1)
        #ax.set_title('{:d} {}'.format(i, joint_names[i]), fontsize=10)
        ax.set_title('{:d}'.format(i), fontsize=10)
        sns.heatmap(hms[i], cbar=False, cmap='viridis',xticklabels=False,yticklabels=False, ax=ax)
    
    if img_res is not None:
        ax = plt.subplot(r, c, n)
        ax.set_axis_off()
        ax.set_title('res', fontsize=10)
        ax.imshow(img_res)

    plt.savefig(path)
    plt.close()

class ModelDesc(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self):
        self._loss = None
        self._inputs = []
        self._outputs = []
        self._tower_summary = []
        self._heatmaps = []

    def set_inputs(self, *vars):
        self._inputs = vars

    def set_outputs(self, *vars):
        self._outputs = vars

    def set_heatmaps(self, *vars):
        self._heatmaps = vars

    def set_loss(self, var):
        if not isinstance(var, tf.Tensor):
            raise ValueError("Loss must be an single tensor.")
        # assert var.get_shape() == [], 'Loss tensor must be a scalar shape but got {} shape'.format(var.get_shape())
        self._loss = var

    def get_loss(self, include_wd=False):
        if self._loss is None:
            raise ValueError("Network doesn't define the final loss")

        if include_wd:
            weight_decay = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            weight_decay = tf.add_n(weight_decay)
            return self._loss + weight_decay
        else:
            return self._loss

    def get_inputs(self):
        if len(self._inputs) == 0:
            raise ValueError("Network doesn't define the inputs")
        return self._inputs

    def get_outputs(self):
        if len(self._outputs) == 0:
            raise ValueError("Network doesn't define the outputs")
        return self._outputs

    def get_heatmaps(self):
        if len(self._heatmaps) == 0:
            raise ValueError("Network doesn't define the heatmaps")
        return self._heatmaps

    def add_tower_summary(self, name, vars, reduced_method='mean'):
        assert reduced_method == 'mean' or reduced_method == 'sum', \
            "Summary tensor only supports sum- or mean- reduced method"
        if isinstance(vars, list):
            for v in vars:
                if vars.get_shape() == None:
                    print('Summary tensor {} got an unknown shape.'.format(name))
                else:
                    assert v.get_shape().as_list() == [], \
                        "Summary tensor only supports scalar but got {}".format(v.get_shape().as_list())
                tf.add_to_collection(name, v)
        else:
            if vars.get_shape() == None:
                print('Summary tensor {} got an unknown shape.'.format(name))
            else:
                assert vars.get_shape().as_list() == [], \
                    "Summary tensor only supports scalar but got {}".format(vars.get_shape().as_list())
            tf.add_to_collection(name, vars)
        self._tower_summary.append([name, reduced_method])

    @abc.abstractmethod
    def make_network(self, is_train, add_paf_loss=False):
        pass


class Base(object):
    __metaclass__ = abc.ABCMeta
    """
    build graph:
        _make_graph
            make_inputs
            make_network
                add_tower_summary
        get_summary
    
    train/test
    """

    def __init__(self, net, cfg, data_iter=None, log_name='logs.txt'):
        self._input_list = []
        self._output_list = []
        self._outputs = []
        self.graph_ops = None

        self.net = net
        self.cfg = cfg

        self.cur_epoch = 0

        self.summary_dict = {}

        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        # logger
        self.logger = colorlogger(cfg.log_dir, log_name=log_name)

        # initialize tensorflow
        tfconfig = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        tfconfig.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tfconfig)

        # build_graph
        self.build_graph()

        # get data iter
        self._data_iter = data_iter

    @abc.abstractmethod
    def _make_data(self):
        return

    @abc.abstractmethod
    def _make_graph(self):
        return

    def build_graph(self):
        # all variables should be in the same graph and stored in cpu.
        with tf.device('/device:CPU:0'):
            tf.set_random_seed(2333)
            self.graph_ops = self._make_graph()
            if not isinstance(self.graph_ops, list) and not isinstance(self.graph_ops, tuple):
                self.graph_ops = [self.graph_ops]
        self.summary_dict.update( get_tower_summary_dict(self.net._tower_summary) )

    def load_weights(self, model=None):

        load_ImageNet = True
        if model == 'last_epoch':
            sfiles = os.path.join(self.cfg.model_dump_dir, 'snapshot_*.ckpt.meta')
            sfiles = glob.glob(sfiles)
            if len(sfiles) > 0:
                sfiles.sort(key=os.path.getmtime)
                sfiles = [i[:-5] for i in sfiles if i.endswith('.meta')]
                model = sfiles[-1]
            else:
                self.logger.critical('No snapshot model exists.')
                return
            load_ImageNet = False

        if isinstance(model, int):
            model = os.path.join(self.cfg.model_dump_dir, 'snapshot_%d.ckpt' % model)
            load_ImageNet = False

        if isinstance(model, str) and (osp.exists(model + '.meta') or osp.exists(model)):
            self.logger.info('Initialized model weights from {} ...'.format(model))
            load_model(self.sess, model, load_ImageNet)
            if model.split('/')[-1].startswith('snapshot_'):
                self.cur_epoch = int(model[model.find('snapshot_')+9:model.find('.ckpt')])
                self.logger.info('Current epoch is %d.' % self.cur_epoch)
        else:
            self.logger.critical('Load nothing. There is no model in path {}.'.format(model))

    def next_feed(self):
        if self._data_iter is None:
            raise ValueError('No input data.')
        feed_dict = dict()
        for inputs in self._input_list:
            blobs = next(self._data_iter)
            for i, inp in enumerate(inputs):
                inp_shape = inp.get_shape().as_list()
                if None in inp_shape:
                    feed_dict[inp] = blobs[i]
                else:
                    feed_dict[inp] = blobs[i].reshape(*inp_shape)
        return feed_dict

class Trainer(Base):
    def __init__(self, net, cfg, data_iter=None):
        self.lr_eval = cfg.lr
        self.lr = tf.Variable(cfg.lr, trainable=False)
        self._optimizer = get_optimizer(self.lr, cfg.optimizer)

        super(Trainer, self).__init__(net, cfg, data_iter, log_name='train_logs.txt')

        # make data
        self._data_iter, self.itr_per_epoch = self._make_data()
    
    def compute_iou(self, src_roi, dst_roi):
        
        # IoU calculate with GTs
        xmin = np.maximum(dst_roi[:,0], src_roi[:,0])
        ymin = np.maximum(dst_roi[:,1], src_roi[:,1])
        xmax = np.minimum(dst_roi[:,0]+dst_roi[:,2], src_roi[:,0]+src_roi[:,2])
        ymax = np.minimum(dst_roi[:,1]+dst_roi[:,3], src_roi[:,1]+src_roi[:,3])
        
        interArea = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
        
        boxAArea = dst_roi[:,2] * dst_roi[:,3]
        boxBArea = np.tile(src_roi[:,2] * src_roi[:,3],(len(dst_roi),1))
        sumArea = boxAArea + boxBArea

        iou = interArea / (sumArea - interArea + 1e-5)

        return iou

    def _make_data(self):
        from dataset import Dataset
        from gen_batch import generate_batch

        d = Dataset()
        train_data = d.load_train_data()
        
        ## modify train_data to the result of the decoupled initial model
        with open(d.test_on_trainset_path, 'r') as f:
            test_on_trainset = json.load(f)
            for data in test_on_trainset:
                if isinstance(data['image_id'], str):
                    data['image_id'] = int(data['image_id'].split('.')[0])

        # sort list by img_id
        train_data = sorted(train_data, key=lambda k: k['image_id']) 
        test_on_trainset = sorted(test_on_trainset, key=lambda k: k['image_id'])
        
        # cluster train_data and test_on_trainset by img_id
        cur_img_id = train_data[0]['image_id']
        data_gt = []
        data_gt_per_img = []
        for i in range(len(train_data)):
            if train_data[i]['image_id'] == cur_img_id:
                data_gt_per_img.append(train_data[i])
            else:
                data_gt.append(data_gt_per_img)
                cur_img_id = train_data[i]['image_id']
                data_gt_per_img = [train_data[i]]
        if len(data_gt_per_img) > 0:
            data_gt.append(data_gt_per_img)

        cur_img_id = test_on_trainset[0]['image_id']
        data_out = []
        data_out_per_img = []
        for i in range(len(test_on_trainset)):
            if test_on_trainset[i]['image_id'] == cur_img_id:
                data_out_per_img.append(test_on_trainset[i])
            else:
                data_out.append(data_out_per_img)
                cur_img_id = test_on_trainset[i]['image_id']
                data_out_per_img = [test_on_trainset[i]]
        if len(data_out_per_img) > 0:
            data_out.append(data_out_per_img)

        # remove false positive images
        i = 0
        j = 0
        aligned_data_out = []
        while True:
            gt_img_id = data_gt[i][0]['image_id']
            out_img_id = data_out[j][0]['image_id']
            if gt_img_id > out_img_id:
                j = j + 1
            elif gt_img_id < out_img_id:
                i = i + 1
            else:
                aligned_data_out.append(data_out[j])
                i = i + 1
                j = j + 1

            if j == len(data_out) or i == len(data_gt):
                break
        data_out = aligned_data_out

        # add false negative images
        j = 0
        aligned_data_out = []
        for i in range(len(data_gt)):
            gt_img_id = data_gt[i][0]['image_id']
            out_img_id = data_out[j][0]['image_id']
            if gt_img_id == out_img_id:
                aligned_data_out.append(data_out[j])
                j = j + 1
            else:
                aligned_data_out.append([])

            if j == len(data_out):
                break
        data_out = aligned_data_out

        # they should contain annotations from all the images
        assert len(data_gt) == len(data_out)

        # for each img
        for i in range(len(data_gt)):
            
            bbox_out_per_img = np.zeros((len(data_out[i]),4))
            joint_out_per_img = np.zeros((len(data_out[i]),self.cfg.num_kps*3))
            
            # for each data_out in an img
            for j in range(len(data_out[i])):
                joint = data_out[i][j]['keypoints']

                if 'bbox' in data_out[i][j]:
                    bbox = data_out[i][j]['bbox'] #x, y, width, height
                else:
                    coords = np.array(joint).reshape(-1,3)
                    xmin = np.min(coords[:,0])
                    xmax = np.max(coords[:,0])
                    width = xmax - xmin if xmax > xmin else 20
                    center = (xmin + xmax)/2.
                    xmin = center - width/2.*1.1
                    xmax = center + width/2.*1.1

                    ymin = np.min(coords[:,1])
                    ymax = np.max(coords[:,1])
                    height = ymax - ymin if ymax > ymin else 20
                    center = (ymin + ymax)/2.
                    ymin = center - height/2.*1.1
                    ymax = center + height/2.*1.1
                    bbox = [xmin, xmax, ymin, ymax]
                
                bbox_out_per_img[j,:] = bbox
                joint_out_per_img[j,:] = joint
            
            # for each gt in an img
            for j in range(len(data_gt[i])):
                bbox_gt = np.array(data_gt[i][j]['bbox']) #x, y, width, height
                joint_gt = np.array(data_gt[i][j]['joints'])
                
                # IoU calculate with detection outputs of other methods
                iou = self.compute_iou(bbox_gt.reshape(1,4), bbox_out_per_img)
                if len(iou) == 0:
                    continue
                out_idx = np.argmax(iou)
                data_gt[i][j]['estimated_joints'] = [joint_out_per_img[out_idx,:]]

                # for swap
                num_overlap = 0
                near_joints = []
                for k in range(len(data_gt[i])):
                    bbox_gt_k = np.array(data_gt[i][k]['bbox'])
                    iou_with_gt_k = self.compute_iou(bbox_gt.reshape(1,4), bbox_gt_k.reshape(1,4))
                    if k == j or iou_with_gt_k < 0.1:
                        continue
                    num_overlap += 1
                    near_joints.append(np.array(data_gt[i][k]['joints']).reshape(self.cfg.num_kps,3))
                data_gt[i][j]['overlap'] = num_overlap
                if num_overlap > 0:
                    data_gt[i][j]['near_joints'] = near_joints
                else:
                    data_gt[i][j]['near_joints'] = [np.zeros([self.cfg.num_kps,3])]

        # flatten data_gt
        train_data = [y for x in data_gt for y in x]

        from tfflat.data_provider import DataFromList, MultiProcessMapDataZMQ, BatchData, MapData
        data_load_thread = DataFromList(train_data)
        if self.cfg.multi_thread_enable:
            data_load_thread = MultiProcessMapDataZMQ(data_load_thread, self.cfg.num_thread, generate_batch, strict=True, add_paf=self.cfg.add_paf)
        else:
            data_load_thread = MapData(data_load_thread, generate_batch, add_paf=self.cfg.add_paf)
        data_load_thread = BatchData(data_load_thread, self.cfg.batch_size)

        data_load_thread.reset_state()
        dataiter = data_load_thread.get_data()

        return dataiter, math.ceil(len(train_data)/self.cfg.batch_size/self.cfg.num_gpus)

    def _make_graph(self):
        self.logger.info("Generating training graph on {} GPUs ...".format(self.cfg.num_gpus))

        weights_initializer = slim.xavier_initializer()
        biases_initializer = tf.constant_initializer(0.)
        biases_regularizer = tf.no_regularizer
        weights_regularizer = tf.contrib.layers.l2_regularizer(self.cfg.weight_decay)

        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(self.cfg.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('tower_%d' % i) as name_scope:
                        # Force all Variables to reside on the CPU.
                        with slim.arg_scope([slim.model_variable, slim.variable], device='/device:CPU:0'):
                            with slim.arg_scope([slim.conv2d, slim.conv2d_in_plane, \
                                                 slim.conv2d_transpose, slim.separable_conv2d,
                                                 slim.fully_connected],
                                                weights_regularizer=weights_regularizer,
                                                biases_regularizer=biases_regularizer,
                                                weights_initializer=weights_initializer,
                                                biases_initializer=biases_initializer):
                                # loss over single GPU
                                self.net.make_network(is_train=True, add_paf_loss=self.cfg.add_paf)
                                if i == self.cfg.num_gpus - 1:
                                    loss = self.net.get_loss(include_wd=True)
                                else:
                                    loss = self.net.get_loss()
                                self._input_list.append( self.net.get_inputs() )

                        tf.get_variable_scope().reuse_variables()

                        if i == 0:
                            if self.cfg.num_gpus > 1 and self.cfg.bn_train is True:
                                self.logger.warning("BN is calculated only on single GPU.")
                            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope)
                            with tf.control_dependencies(extra_update_ops):
                                grads = self._optimizer.compute_gradients(loss)
                        else:
                            grads = self._optimizer.compute_gradients(loss)
                        final_grads = []
                        with tf.variable_scope('Gradient_Mult') as scope:
                            for grad, var in grads:
                                final_grads.append((grad, var))
                        tower_grads.append(final_grads)

        if len(tower_grads) > 1:
            grads = average_gradients(tower_grads)
        else:
            grads = tower_grads[0]

        apply_gradient_op = self._optimizer.apply_gradients(grads)
        train_op = tf.group(apply_gradient_op, *extra_update_ops)

        return train_op

    def train(self):
        
        # saver
        self.logger.info('Initialize saver ...')
        train_saver = Saver(self.sess, tf.global_variables(), self.cfg.model_dump_dir)

        # initialize weights
        self.logger.info('Initialize all variables ...')
        self.sess.run(tf.variables_initializer(tf.global_variables(), name='init'))
        self.load_weights('last_epoch' if self.cfg.continue_train else self.cfg.init_model)

        self.logger.info('Start training ...')
        start_itr = self.cur_epoch * self.itr_per_epoch + 1
        end_itr = self.itr_per_epoch * self.cfg.end_epoch + 1
        for itr in range(start_itr, end_itr):
            self.tot_timer.tic()

            self.cur_epoch = itr // self.itr_per_epoch
            setproctitle.setproctitle('train epoch:' + str(self.cur_epoch))

            # apply current learning policy
            cur_lr = self.cfg.get_lr(self.cur_epoch)
            if not approx_equal(cur_lr, self.lr_eval):
                print(self.lr_eval, cur_lr)
                self.sess.run(tf.assign(self.lr, cur_lr))

            # input data
            self.read_timer.tic()
            feed_dict = self.next_feed()
            self.read_timer.toc()
            #count = 0
            #for v in feed_dict.values():
            #    if v.shape == (16, 96, 72, 34):
            #        print('sum check D', np.sum(v[0,:,20,8]))
            #        np.savez('temp/{:d}.npz'.format(count), data=v)
            #    count += 1
            #    
            #    if v.shape == (16, 96, 72, 34):
            #        for i in range(v.shape[0]):
            #            for j in range(34):
            #                temp = v[i,:,:,j].copy()
            #                temp[np.abs(v[i,:,:,j]) > 1e-6] = 255
            #                cv2.imwrite('temp/{:d}_{:d}.jpg'.format(count, j), temp.astype(np.uint8))
            #            # save_mergedHeatmaps(np.transpose(v[i], (2, 1, 0)), 'temp/{:d}.jpg'.format(count))
            #            count += 1
            #    print(v.shape)
            #    
            #break
            # train one step
            self.gpu_timer.tic()
            _, self.lr_eval, *summary_res = self.sess.run(
                [self.graph_ops[0], self.lr, *self.summary_dict.values()], feed_dict=feed_dict)
            self.gpu_timer.toc()

            itr_summary = dict()
            for i, k in enumerate(self.summary_dict.keys()):
                itr_summary[k] = summary_res[i]

            screen = [
                'Epoch %d itr %d/%d:' % (self.cur_epoch, itr, self.itr_per_epoch),
                'lr: %g' % (self.lr_eval),
                'speed: %.2f(%.2fs r%.2f)s/itr' % (
                    self.tot_timer.average_time, self.gpu_timer.average_time, self.read_timer.average_time),
                '%.2fh/epoch' % (self.tot_timer.average_time / 3600. * self.itr_per_epoch),
                ' '.join(map(lambda x: '%s: %.4f' % (x[0], x[1]), itr_summary.items())),
            ]
            

            #TODO(display stall?)
            if itr % self.cfg.display == 0:
                self.logger.info(' '.join(screen))

            if itr % self.itr_per_epoch == 0:
                train_saver.save_model(self.cur_epoch)

            self.tot_timer.toc()

class Tester(Base):
    def __init__(self, net, cfg, data_iter=None):
        super(Tester, self).__init__(net, cfg, data_iter, log_name='test_logs.txt')

    def find_peaks(self, hm, threshold=0.05):
        w, h = hm.shape[1], hm.shape[0]
        peak = (hm > threshold)[1:-1,1:-1]
        peak = peak & (hm[1:-1,1:-1] > hm[:-2,  1:-1])
        peak = peak & (hm[1:-1,1:-1] > hm[:-2,  :-2])
        peak = peak & (hm[1:-1,1:-1] > hm[1:-1:,:-2])
        peak = peak & (hm[1:-1,1:-1] > hm[2:,   :-2]) 
        peak = peak & (hm[1:-1,1:-1] > hm[2:,   1:-1])
        peak = peak & (hm[1:-1,1:-1] > hm[2:,   2:])
        peak = peak & (hm[1:-1,1:-1] > hm[1:-1, 2:])
        peak = peak & (hm[1:-1,1:-1] > hm[:-2,  2:])
        peak = np.pad(peak, ((1,1),(1,1)), mode='constant', constant_values=False)

        peak_val = hm[peak]
        peak_idx = np.array(peak.nonzero()).astype(np.float32)
        
        # average 7x7 around peak as final peak
        for i in range(peak_idx.shape[1]):
            pY, pX = int(peak_idx[0,i]), int(peak_idx[1,i])
            f = 7
            sum_score, sum_x, sum_y = 0, 0, 0
            for x in range(f):
                for y in range(f):
                    tx = pX - f // 2 + x
                    ty = pY - f // 2 + y
                    if tx < 0 or tx >= w or ty <= 0 or ty >= h: continue
                    score = hm[ty,tx]
                    sum_score += score
                    sum_x += tx * score
                    sum_y += ty * score
            peak_idx[0,i] = sum_y / sum_score
            peak_idx[1,i] = sum_x / sum_score
        
        if peak_val.shape[0] > 0:
            peak_sel = []
            maxval = np.max(peak_val)
            for i in range(peak_val.shape[0]):
                if peak_val[i] > maxval * 0.5:
                    peak_sel.append(i)
            peak_idx = peak_idx[:, peak_sel]
            peak_val = peak_val[peak_sel]

        return peak_idx[::-1,:].T, peak_val

    def extract_coordinate_paf_one(self, heatmap_outs, paf_outs):
        can_idx_list, can_val_list = [], []
        for i in range(len(heatmap_outs)):
            # peak_idx: 17x2 (x, y); peak_val: 17
            peak_idx, peak_val = self.find_peaks(heatmap_outs[i])
            can_idx_list.append(peak_idx)
            can_val_list.append(peak_val)

        height, width = heatmap_outs[0].shape[0], heatmap_outs[0].shape[1]
        joint_list_size = len(heatmap_outs) + 1
        joint_list = [-1] * joint_list_size
        joint_list[-1] = 0

        line_cand = {}
        for line_idx in range(len(self.cfg.kps_lines)):
            indexA, indexB = self.cfg.kps_lines[line_idx]
            candA = can_idx_list[indexA]
            candB = can_idx_list[indexB]
            nA = candA.shape[0]
            nB = candB.shape[0]
            
            vec_cand = []
            paf = paf_outs[line_idx*2:line_idx*2+2]

            for i in range(nA):
                for j in range(nB):
                    vec = candB[j] - candA[i]
                    vecNorm = np.linalg.norm(vec)
                    vec = vec / (vecNorm + 1e-6)

                    num_inter = 10
                    mX = np.round(np.linspace(candA[i][0], candB[j][0], num_inter)).astype(np.int32)
                    mY = np.round(np.linspace(candA[i][1], candB[j][1], num_inter)).astype(np.int32)

                    p_sum = 0
                    p_count = 0
                    for lm in range(num_inter):
                        direct = paf[:, mY[lm], mX[lm]]
                        score = vec[0] * direct[0] + vec[1] * direct[1]
                        if score > 0.05:
                            p_sum += score
                            p_count += 1

                    suc_ratio = p_count * 1.0 / num_inter
                    mid_score = p_sum / p_count if p_count > 0 else 0
                    score = mid_score + can_val_list[indexA][i] * 0.1 + can_val_list[indexB][j] * 0.1
                    vec_cand.append([i, j, score])
            
            if nA == 1 and nB == 1: 
                # certain point
                if joint_list[indexA] < 0:
                    joint_list[indexA] = 0
                if joint_list[indexB] < 0:
                    joint_list[indexB] = 0
                joint_list[-1] += vec_cand[-1][-1]
            elif nA == 1 and nB == 0:
                # single point
                if joint_list[indexA] < 0:
                    joint_list[indexA] = 0
                joint_list[-1] += can_val_list[indexA][0] * 0.1
            elif nA == 0 and nB == 1:
                # single point
                if joint_list[indexB] < 0:
                    joint_list[indexB] =0
                joint_list[-1] += can_val_list[indexB][0] * 0.1
            elif len(vec_cand) > 0:
                assert len(vec_cand) > 1
                line_cand[line_idx] = vec_cand
            #else:
            #    print('Warning: not found valid line_{:d} candidate'.format(line_idx))
        
        # split candidates to unlinked clusters
        topo_list = []
        graph = {}
        for lidx in line_cand.keys():
            jA, jB = self.cfg.kps_lines[lidx]
            if jA in graph:
                graph[jA].add(jB)
            else:
                graph[jA] = {jB}
            
            if jB in graph:
                graph[jB].add(jA)
            else:
                graph[jB] = {jA}
            
        for j in graph.keys():
            topo_set = set()
            que = [j]
            while len(que) > 0:
                topo_set.add(que[0])
                for i in graph[que[0]]:
                    if i not in topo_set:
                        que.append(i)
                del que[0]
            topo_list.append({'set':topo_set, 'idx':[], 'vec':[]})

        while(len(line_cand) > 0):
            lidx, vec_cand = list(line_cand.items())[0]
            del line_cand[lidx]

            jA, jB = self.cfg.kps_lines[lidx]
            for item in topo_list:    
                if jA in item['set'] or jB in item['set']: # linked
                    item['idx'].append(lidx)
                    item['vec'].append(vec_cand)
                    break        

        def find_max_score(idx_list, vec_list, start, pose):
            if start == len(idx_list):
                return pose.copy()

            jA, jB = self.cfg.kps_lines[idx_list[start]]        
            pA, pB, ps = pose[jA], pose[jB], pose[-1]

            max_pose = None
            vec = vec_list[start]
            for i in range(len(vec)):
                ci, cj, cs = vec[i]
                if (pA >= 0 and pA != ci) or (pB >= 0 and pB != cj): continue
                assert (pA < 0 and pB < 0) or pA == ci or pB == cj, 'ci:{:d} cj:{:d} pA:{:d} pB:{:d}'.format(ci, cj, pA, pB)
                pose[jA], pose[jB] = ci, cj            
                pose[-1] = ps + cs
                curr_pose = find_max_score(idx_list, vec_list, start + 1, pose)
                max_pose = curr_pose if max_pose is None or max_pose[-1] < curr_pose[-1] else max_pose
                pose[jA], pose[jB], pose[-1] = pA, pB, ps
            
            return max_pose

        for item in topo_list:
            pose = [-1] * joint_list_size
            pose[-1] = 0
            max_pose = find_max_score(item['idx'], item['vec'], 0, pose)
            for i in range(len(max_pose)-1):
                assert max_pose[i] < 0 or (max_pose[i] >= 0 and (joint_list[i] < 0 or max_pose[i] == joint_list[i])), 'max pose:{} joint_list:{}'.format(max_pose, joint_list)
                if max_pose[i] >= 0:
                    joint_list[i] = max_pose[i]
            joint_list[-1] += max_pose[-1]

        out_pose = np.zeros([len(heatmap_outs), 2], dtype=np.float32)

        for i in range(len(can_idx_list)):
            if can_val_list[i].size == 0: continue
            idx = np.argmax(can_val_list[i])
            out_pose[i][0] = can_idx_list[i][idx][0] / width * self.cfg.input_shape[1]
            out_pose[i][1] = can_idx_list[i][idx][1] / height * self.cfg.input_shape[0]
        
        for j in range(len(can_idx_list)):
            if joint_list[j] < 0: continue
            out_pose[j][0] = can_idx_list[j][joint_list[j]][0] / width * self.cfg.input_shape[1]
            out_pose[j][1] = can_idx_list[j][joint_list[j]][1] / height * self.cfg.input_shape[0]
        
        return out_pose

    def extract_coordinate_paf(self, heatmap_outs, paf_outs):
        assert len(heatmap_outs) == len(paf_outs), 'heatmap count: {}, paf count:'.format(heatmap_outs.shape, paf_outs.shape)
        output_hm = []
        for i in range(len(heatmap_outs)):
            output_hm.append(self.extract_coordinate_paf_one(heatmap_outs[i], paf_outs[i]))
        
        return output_hm

    def next_feed(self, batch_data=None):
        if self._data_iter is None and batch_data is None:
            raise ValueError('No input data.')
        feed_dict = dict()
        if batch_data is None:
            for inputs in self._input_list:
                blobs = next(self._data_iter)
                for i, inp in enumerate(inputs):
                    inp_shape = inp.get_shape().as_list()
                    if None in inp_shape:
                        feed_dict[inp] = blobs[i]
                    else:
                        feed_dict[inp] = blobs[i].reshape(*inp_shape)
        else:
            assert isinstance(batch_data, list) or isinstance(batch_data, tuple), "Input data should be list-type."
            assert len(batch_data) == len(self._input_list[0]), "Input data is incomplete."

            batch_size = self.cfg.batch_size
            if self._input_list[0][0].get_shape().as_list()[0] is None:
                # fill batch
                for i in range(len(batch_data)):
                    batch_size = (len(batch_data[i]) + self.cfg.num_gpus - 1) // self.cfg.num_gpus
                    total_batches = batch_size * self.cfg.num_gpus
                    left_batches = total_batches - len(batch_data[i])
                    if left_batches > 0:
                        batch_data[i] = np.append(batch_data[i], np.zeros((left_batches, *batch_data[i].shape[1:])), axis=0)
                        self.logger.warning("Fill some blanks to fit batch_size which wastes %d%% computation" % (
                            left_batches * 100. / total_batches))
            else:
                assert self.cfg.batch_size * self.cfg.num_gpus == len(batch_data[0]), \
                    "Input batch doesn't fit placeholder batch."

            for j, inputs in enumerate(self._input_list):
                for i, inp in enumerate(inputs):
                    feed_dict[ inp ] = batch_data[i][j * batch_size: (j+1) * batch_size]

            #@TODO(delete)
            assert (j+1) * batch_size == len(batch_data[0]), 'check batch'
        return feed_dict, batch_size

    def _make_graph(self):
        self.logger.info("Generating testing graph on {} GPUs ...".format(self.cfg.num_gpus))

        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(self.cfg.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('tower_%d' % i) as name_scope:
                        with slim.arg_scope([slim.model_variable, slim.variable], device='/device:CPU:0'):
                            self.net.make_network(is_train=False, add_paf_loss=self.cfg.add_paf)
                            self._input_list.append(self.net.get_inputs())
                            self._output_list.append(self.net.get_outputs())

                        tf.get_variable_scope().reuse_variables()

        self._outputs = [aggregate_batch(self._output_list)]
        self._outputs.append(self.net.get_heatmaps())
        # run_meta = tf.RunMetadata()
        # opts = tf.profiler.ProfileOptionBuilder.float_operation()
        # flops = tf.profiler.profile(self.sess.graph, run_meta=run_meta, cmd='op', options=opts)
        #
        # opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
        # params = tf.profiler.profile(self.sess.graph, run_meta=run_meta, cmd='op', options=opts)

        # print("{:,} --- {:,}".format(flops.total_float_ops, params.total_parameters))
        # from IPython import embed; embed()

        return self._outputs

    def predict_one(self, data=None):
        # TODO(reduce data in limited batch)
        assert len(self.summary_dict) == 0, "still not support scalar summary in testing stage"
        setproctitle.setproctitle('test epoch:' + str(self.cur_epoch))

        self.read_timer.tic()
        feed_dict, batch_size = self.next_feed(data)
        self.read_timer.toc()

        self.gpu_timer.tic()
        res = self.sess.run([*self.graph_ops, *self.summary_dict.values()], feed_dict=feed_dict)
        self.gpu_timer.toc()

        if self.cfg.add_paf:
            heatmap_outs = res[0][0]
            paf_outs = res[0][1]

            coord_outs = self.extract_coordinate_paf(heatmap_outs, paf_outs)
            
            assert len(heatmap_outs) == len(paf_outs), 'heatmap shape {} is not equal with paf shape {}'.format(heatmap_outs.shape, paf_outs.shape)

            return coord_outs, np.concatenate([heatmap_outs, paf_outs], axis=1)

        else:
            #if data is not None and len(data[0]) < self.cfg.num_gpus * batch_size:
            return res[0][0], res[1][0]

    def test(self):
        pass

