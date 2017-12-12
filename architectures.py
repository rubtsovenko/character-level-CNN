import tensorflow as tf
from config import FLAGS
from utils import prob_close

slim = tf.contrib.slim
xavier_conv = slim.xavier_initializer_conv2d()
xavier = slim.xavier_initializer()
var_scale = slim.variance_scaling_initializer()
const_init = tf.constant_initializer(0.05)
trunc_normal = tf.truncated_normal_initializer(mean=0, stddev=0.05)
regularizer = slim.l2_regularizer(FLAGS.weight_decay)


def net_1(x, is_train):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=var_scale,
                        biases_initializer=const_init):
        with slim.arg_scope([slim.conv2d], stride=1, padding='VALID'):
            net = slim.conv2d(x, 64, [26,7], scope='conv1')
            net = slim.max_pool2d(net, [1,2], stride=2, scope='pool1')
            net = slim.conv2d(net, 128, [1, 6], scope='conv2')
            net = slim.max_pool2d(net, [1, 2], stride=2, scope='pool2')
            net = slim.conv2d(net, 256, [1, 4], scope='conv3')
            net = slim.conv2d(net, 256, [1, 3], scope='conv4')
            #net = slim.conv2d(net, 256, [1, 3], scope='conv5')

            net = slim.flatten(net)
            net = slim.fully_connected(net, 100, scope='fc6')
            #net = slim.dropout(net, keep_prob=prob_close(is_train, 0.5), scope='dropout6')
            net = slim.fully_connected(net, 100, scope='fc7')
            #net = slim.dropout(net, keep_prob=prob_close(is_train, 0.5), scope='dropout7')
            net = slim.fully_connected(net, 12, activation_fn=None, scope='fc8')
    return net