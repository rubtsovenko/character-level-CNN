from CNN import CNN
import tensorflow as tf
from config import FLAGS


def main():
    if FLAGS.train_mode == 'train_full':
        model = CNN()

        with tf.Session() as sess:
            model.load_or_init(sess)
            train_fn = ['data/tfrecords/seed_1/fold_1.tfrecords', 'data/tfrecords/seed_1/fold_2.tfrecords',
                        'data/tfrecords/seed_1/fold_3.tfrecords', 'data/tfrecords/seed_1/fold_4.tfrecords']
            # 5857+5856+5852+5849=23414
            train_size = 23414
            train_eval_fn = ['data/tfrecords/seed_1/fold_1.tfrecords']
            train_eval_size = 5857
            val_fn = ['data/tfrecords/seed_1/fold_5.tfrecords']
            val_size = 5847
            model.train(sess, train_fn, train_size, train_eval_fn, train_eval_size, val_fn, val_size)
    elif FLAGS.train_mode == 'overfit_100':
        FLAGS.train_batch_size = 100
        FLAGS.eval_batch_size = 100

        model = CNN()

        with tf.Session() as sess:
            model.load_or_init(sess)
            data = ['data/tfrecords/train_overfit_100.tfrecords']
            model.train(sess, data, 100, data, 100)
    elif FLAGS.train_mode == 'overfit_1000':
        FLAGS.train_batch_size = 100
        FLAGS.eval_batch_size = 500

        model = CNN()

        with tf.Session() as sess:
            model.load_or_init(sess)
            data = ['data/tfrecords/train_overfit_1000.tfrecords']
            model.train(sess, data, 1000, data, 1000)
    else:
        raise ValueError('Unrecognized train mode')


if __name__ == '__main__':
    main()