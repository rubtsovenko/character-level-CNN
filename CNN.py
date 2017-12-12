import tensorflow as tf
import os
import numpy as np
import logging
import datetime
import string
from tqdm import tqdm
from architectures import net_1, net_2
from config import FLAGS
ALL_FLAGS = FLAGS.__flags


class CNN(object):
    def __init__(self):
        self.logger = get_logger(FLAGS.log_dir)
        now = datetime.datetime.now()
        self.logger.info('{} Model Created '.format(now.strftime("%Y-%m-%d %H:%M")).center(150, '*'))
        self.logger.info('CONFIG FLAGS:')
        self.logger.info(ALL_FLAGS)
        self.logger.info('')

        set_random_seed()
        self.X, self.y_, self.filenames, self.batch_size, self.num_epochs, self.iterator = network_input(FLAGS.preprocess_mode)
        self.is_train = tf.placeholder(tf.bool, name='is_train')

        self.y_logits_op = build_trunk(self.X, self.is_train)
        self.loss_vector_op, self.loss_op = add_loss(self.y_logits_op, self.y_)
        with tf.name_scope('softmax'):
            self.y_preds_op = tf.nn.softmax(self.y_logits_op)
        with tf.name_scope('accuracy'):
            self.correct_preds_op = tf.equal(tf.argmax(self.y_preds_op, 1), tf.argmax(self.y_, 1))
            self.accuracy_op = tf.reduce_mean(tf.cast(self.correct_preds_op, tf.float32))

        self.optimizer_op = add_optimizer()
        self.train_op, self.global_step = add_train_op(self.loss_op, self.optimizer_op)

        with tf.name_scope('init'):
            self.init_op = tf.global_variables_initializer()

        self.saver = tf.train.Saver(max_to_keep=100)
        tf.get_default_graph().finalize()

    def load_or_init(self, sess):
        self.logger.info('INITIALIZATION:')

        if FLAGS.ckpt == 0:
            saved_ckpt = tf.train.latest_checkpoint(FLAGS.ckpt_dir)
            if saved_ckpt is None:
                self.logger.info('There is not any saved model. Random initialization is used.')
                sess.run(self.init_op)
            else:
                sess.run(self.init_op)
                FLAGS.ckpt = int(saved_ckpt.split('/')[-1][1:])
                self.logger.info('Load model from ckpt {}'.format(FLAGS.ckpt))
                self.saver.restore(sess, saved_ckpt)
        else:
            chosen_ckpt = os.path.join(FLAGS.ckpt_dir, '-' + str(FLAGS.ckpt))
            if os.path.exists(chosen_ckpt + '.index'):
                sess.run(self.init_op)
                self.logger.info('Load model from ckpt {}'.format(FLAGS.ckpt))
                self.saver.restore(sess, chosen_ckpt)
            else:
                self.logger.info('No ckpt {} exists in {}'.format(FLAGS.ckpt, FLAGS.ckpt_dir))
                raise ValueError('No ckpt {} exists in {}'.format(FLAGS.ckpt, FLAGS.ckpt_dir))

    def train(self, sess, train_fn, train_size, train_eval_fn, train_eval_size, val_eval_fn=None, val_eval_size=None):
        self.logger.info('TRAINING:')
        self.logger.info('')

        num_batches = int(np.ceil(train_size) / FLAGS.train_batch_size)

        if self.global_step.eval() == 0:
            graph_writer = tf.summary.FileWriter(os.path.join(FLAGS.summary_dir, 'graph'), sess.graph)
            print('Epoch 0:')
            self.track_performance(sess, 0, train_eval_fn, train_eval_size, val_eval_fn, val_eval_size)

        sess.run(self.iterator.initializer, {self.filenames: train_fn,
                                             self.batch_size: FLAGS.train_batch_size,
                                             self.num_epochs: FLAGS.monitor_freq})
        for epoch in range(FLAGS.ckpt + 1, FLAGS.ckpt + 1 + FLAGS.num_epochs):
            for _ in tqdm(range(num_batches), desc='Epoch {:3d}'.format(epoch)):
                sess.run(self.train_op, {self.is_train: True})
                #print(sess.run(self.accuracy_op, {self.is_train: False}))
                #print(sess.run(self.loss_op, {self.is_train: False}))

            # print(sess.run(self.accuracy_op, {self.is_train: False}))
            # print(sess.run(self.loss_op, {self.is_train: False}))

            if epoch % FLAGS.monitor_freq == 0:
                self.track_performance(sess, epoch, train_eval_fn, train_eval_size, val_eval_fn, val_eval_size)
                sess.run(self.iterator.initializer, {self.filenames: train_fn,
                                                     self.batch_size: FLAGS.train_batch_size,
                                                     self.num_epochs: FLAGS.monitor_freq})
            if epoch % FLAGS.save_freq == 0:
                self.saver.save(sess, FLAGS.ckpt_dir, global_step=epoch)

    def track_performance(self, sess, epoch, train_eval_fn, train_eval_size, val_eval_fn=None, val_eval_size=None):
        train_accuracy, train_loss = self.eval(sess, train_eval_size, FLAGS.eval_batch_size, train_eval_fn)
        print("Train Accuracy: {:.3f}".format(train_accuracy))
        print("Train Loss: {:.3f}".format(train_loss))

        if val_eval_fn is not None:
            val_accuracy, val_loss = self.eval(sess, val_eval_size, FLAGS.eval_batch_size, val_eval_fn)
            print("Val Accuracy: {:.3f}".format(val_accuracy))
            print("Val Loss: {:.3f}".format(val_loss))

            self.logger.info('Epoch {}: train acc: {:.3f}, train loss: {:.3f}, val acc: {:.3f}, val loss: {:.3f}.'
                             .format(epoch, train_accuracy, train_loss, val_accuracy, val_loss))

    def eval(self, sess, num_images, batch_size, filenames, disable_bar=True):
        sess.run(self.iterator.initializer, {self.filenames: filenames,
                                             self.batch_size: batch_size,
                                             self.num_epochs: 1})
        num_batches = int(np.ceil(num_images) / batch_size)
        num_correct_preds = 0
        loss = 0
        for _ in tqdm(range(num_batches), desc='Eval', disable=disable_bar):
            correct_preds_batch, loss_batch = sess.run([self.correct_preds_op, self.loss_vector_op],
                                                       {self.is_train: False})
            num_correct_preds += np.sum(correct_preds_batch)
            loss += np.sum(loss_batch)
        accuracy = num_correct_preds / num_images
        loss = loss / num_images

        return accuracy, loss


def get_logger(log_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(os.path.join(log_path, 'logs.log'))
    handler.setLevel(logging.INFO)

    logger.addHandler(handler)

    return logger


def set_random_seed():
    if FLAGS.random_seed_tf != 0:
        tf.set_random_seed(FLAGS.random_seed_tf)


# encodes numbers from 0 to 2^5-1=31 to their binary form
def bit_encoding(number):
    bit_str = '{0:05b}'.format(number)
    return np.array(list(bit_str), dtype='float32')


def preprocess(line_raw, length, max_line_len=416, mode='ohe'):
    def get_ohe2d_features(line, line_len):
        matrix = np.zeros((n, max_line_len, 1), dtype='float32')
        line = line.decode('utf-8')
        for i in range(line_len):
            if i >= max_line_len:
                break
            matrix[ohe_position[line[i]], i] = 1
        return matrix

    def get_bin2d_features(line, length):
        matrix = np.zeros((m, max_line_len), dtype='float32')
        line = line.decode('utf-8')
        for i in range(length):
            if i >= max_line_len:
                break
            matrix[:, i] = bit_encoding(binary_encode[line[i]])
        return matrix

    letters = list(string.ascii_lowercase)

    if mode == 'ohe':
        n = len(letters)  # it should be 26
        ohe_position = dict()
        for i, letter in enumerate(letters):
            ohe_position[letter] = i

        line = tf.py_func(func=get_ohe2d_features, inp=[line_raw, length], Tout=tf.float32)
        line.set_shape((n, max_line_len, 1))
    elif mode == 'bin':
        # for binary encoding, as only lowercase letters are used (26) then 2^5-1=31 is enough (m=5)
        binary_encode = dict()
        m = 5
        for i, letter in enumerate(letters):
            binary_encode[letter] = i + 1

        line = tf.py_func(func=get_bin2d_features, inp=[line_raw, length], Tout=tf.float32)
        line.set_shape((m, max_line_len, 1))
    else:
        raise ValueError('Unrecognized preprocessing mode')

    return line


def parse_tfrecord(serialized_example, preprocess_mode):
    features = {'length': tf.FixedLenFeature([], tf.int64),
                'label': tf.FixedLenFeature([], tf.int64),
                'line_raw': tf.FixedLenFeature([], tf.string)}
    parsed_record = tf.parse_single_example(serialized_example, features)

    line_raw = tf.cast(parsed_record['line_raw'], tf.string)
    label = tf.cast(parsed_record['label'], tf.int32)
    length = tf.cast(parsed_record['length'], tf.int32)

    # Preprocessing
    label = tf.one_hot(label, FLAGS.num_classes)
    line = preprocess(line_raw, length, max_line_len=FLAGS.max_line_len, mode=preprocess_mode)

    return line, label


def network_input(preprocess_mode='ohe'):
    with tf.name_scope('input'):
        filenames = tf.placeholder(tf.string, shape=[None], name='filenames')
        batch_size = tf.placeholder(tf.int64, name='batch_size')
        num_epochs = tf.placeholder(tf.int64, name='num_epochs')

        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(lambda serialized_ex: parse_tfrecord(serialized_ex, preprocess_mode),
                              num_parallel_calls=FLAGS.num_threads)
        dataset = dataset.shuffle(10000)
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(100)

        iterator = dataset.make_initializable_iterator()
        lines, labels = iterator.get_next()

    return lines, labels, filenames, batch_size, num_epochs, iterator


def build_trunk(X, is_train):
    if FLAGS.trunk == 'net_1':
        y_logits = net_1(X, is_train)
    elif FLAGS.trunk == 'net_2':
         y_logits = net_2(X, is_train)
    # elif FLAGS.trunk == 'net_3':
    #     y_logits = net_3(X, is_train)
    # elif FLAGS.trunk == 'net_4':
    #     y_logits = net_4(X, is_train)
    # elif FLAGS.trunk == 'resnet20':
    #     y_logits = resnet20(X, is_train)
    # elif FLAGS.trunk == 'resnet20_preact':
    #     y_logits = resnet20_preact(X, is_train)
    else:
        raise ValueError('Network architecture {} was not recognized'.format(FLAGS.trunk))

    return y_logits


def add_loss(y_logits, y_):
    with tf.name_scope('loss'):
        loss_vector = tf.nn.softmax_cross_entropy_with_logits(logits=y_logits, labels=y_)
        cross_entropy = tf.reduce_mean(loss_vector)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = cross_entropy + sum(reg_losses)

    return loss_vector, loss


def add_optimizer():
    with tf.name_scope('optimizer'):
        if FLAGS.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
        elif FLAGS.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.adam_beta1,
                                               beta2=FLAGS.adam_beta2)
        elif FLAGS.optimizer == 'momentum':
            optimizer = tf.train.MomentumOptimizer(FLAGS.learning_rate, momentum=FLAGS.momentum,
                                                   use_nesterov=FLAGS.use_nesterov)
        elif FLAGS.optimizer == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(FLAGS.learning_rate, decay=FLAGS.rmsprop_decay,
                                                  momentum=FLAGS.rmsprop_momentum)
        else:
            raise ValueError('Optimizer {} was not recognized'.format(FLAGS.optimizer))

        return optimizer


def add_train_op(loss, optimizer):
    with tf.name_scope('train_step'):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # next line is necessary for batch normalization
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op, global_step