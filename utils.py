import tensorflow as tf
slim = tf.contrib.slim


def prob_close(is_train, prob):
    with tf.name_scope('dropout_controller'):
        return 1-tf.cast(is_train, tf.float32)*prob