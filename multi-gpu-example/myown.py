import tensorflow as tf
from tensorflow.contrib import slim
import os
import numpy as np
import time


def net(input, args=None):
    gpu_list = ['/gpu:%s' % i for i in range(len(args.visibale_cuda_devices.split(',')))]
    fc1_list = []
    for i , gpu in enumerate(gpu_list):
        with tf.device(gpu):
            with tf.name_scope('gpu_%s' % i):
                _input = input[i*args.batch_size:(i+1) * args.batch_size]
                fc1 = slim.fully_connected(_input, 10, tf.nn.relu,scope='fc1_%s' % i)
                fc1_list.append(fc1)
    fc1 = tf.concat(fc1_list)
    with tf.device('/gpu:0'):
        fc1_bn = slim.batch_norm(fc1,scope='fc1_bn')

    fc2_list = []
    for i, gpu in enumerate(gpu_list):
        with tf.device(gpu):
            with tf.name_scope('gpu_%s' % i):
                _fc1_bn = fc1_bn[i*args.batch_size:(i+1) * args.batch_size]
                fc2 = slim.fully_connected(_fc1_bn,5,None,scope='fc2_%s' % i)
                fc2_list.append(fc2)
    # fc2 = tf.concat(fc2_list)
    # return fc2
    return fc2_list


def loss(labels, logits, args=None):
    return tf.reduce_mean(tf.square(labels - logits))


def setModel(X, Y):
    pre = net(X)
    L = loss(Y, pre[0])
    return pre, L


def createGraph(args=None, ops_list=None):
    batch_size = args.batch_size
    with tf.device('/cpu:0'):
        X = tf.placeholder(dtype=tf.float32, shape=(None, 5), name='X')
        Y = tf.placeholder(dtype=tf.float32, shape=(None, 5), name='Y')

        ops_list['X'] = X
        ops_list['Y'] = Y

        tower_grads = []

        for i in range(len(args.visibale_cuda_devices.split(","))):
            with tf.device('/gpu:%s' % i):
                with tf.variable_scope('gpu_%s' % i):
                    _x = X[i * batch_size:(i + 1) * batch_size]
                    _y = Y[i * batch_size:(i + 1) * batch_size]

                    _pre, _L = setModel(_x, _y)


                    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
                    grads = optimizer.compute_gradients(_L)

                    # print(grads)
                    # print('------------------')

                    if i == 0:
                        print('computing the acc of the first gpu')
                        ops_list['L'] = _L
                        ops_list['pre'] = _pre
                    if grads[0] is not None:
                        tower_grads.append(grads)
        tower_grads = avgGrad(tower_grads)
        train_op = optimizer.apply_gradients(tower_grads)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        ops_list['sess'] = sess
        ops_list['train_op'] = train_op

        # return sess, train_op

def gen_data_iterator(args=None):
    X = np.random.random([10000, 5])
    Y = X * [1, 2, 3, 4, 5] + [5, 4, 3, 2, 1]
    X = np.hstack((X, Y))
    Y = X[8000:].copy()
    X = X[:8000].copy()

    X_data = tf.data.Dataset.from_tensor_slices(X)
    Y_data = tf.data.Dataset.from_tensor_slices(Y)
    X_batch = X_data.batch(args.batch_size * len(args.visibale_cuda_devices.split(','))).repeat()
    Y_batch = Y_data.batch(args.batch_size * len(args.visibale_cuda_devices.split(','))).repeat()
    X_batch_it = X_batch.make_one_shot_iterator()
    Y_batch_it = Y_batch.make_one_shot_iterator()
    X_next_batch = X_batch_it.get_next()
    Y_next_batch = Y_batch_it.get_next()
    return X_next_batch, Y_next_batch


def train(sess, train_op, data_batch, args, s_epoch, e_epoch, ops_list):
    # batch_size = args.batch_size * len(args.visibale_cuda_devices.split(','))
    time1 = time.time()
    for i in range(s_epoch, e_epoch):
        train_data = sess.run(data_batch)
        # print(train_data.shape)
        X = train_data[:,:5].copy()
        Y = train_data[:,5:].copy()
        feed_dict = {
            ops_list['X']: X,
            ops_list['Y']: Y
        }
        L, _ = sess.run([ops_list['L'], ops_list['train_op']], feed_dict=feed_dict)

        if (i+1) % 1000 == 0:
            time2 = time.time()
            print('time: %.3f' % (time2 - time1))
            time1 = time2
            print(np.mean(L))

        # print(np.mean(L))

def avgGrad(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):

        print(grad_and_vars)
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        # g_is_None = False
        for g, _ in grad_and_vars:
            print(g)
            if g is None:
                # g_is_None = True
                continue
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    print(average_grads)
    return average_grads

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.visibale_cuda_devices

    ops_list = {}

    createGraph(args,ops_list)
    sess = ops_list['sess']
    train_op = ops_list['train_op']
    train_data, test_data = gen_data_iterator(args)
    train(sess,train_op,train_data,args,0,10000000000,ops_list)

    pass


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--visibale_cuda_devices', default='2')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate',type=float,default=0.00001)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)
    pass
