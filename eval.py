from CNN import CNN
import tensorflow as tf
from config import FLAGS


def main():
    if FLAGS.eval_mode == 'val':
        model = CNN()
        with tf.Session() as sess:
            model.load_or_init(sess)
            train_fn = ['data/tfrecords/seed_1/fold_1.tfrecords', 'data/tfrecords/seed_1/fold_2.tfrecords',
                        'data/tfrecords/seed_1/fold_3.tfrecords', 'data/tfrecords/seed_1/fold_4.tfrecords']
            # 5857+5856+5852+5849=23414
            train_size = 23414
            val_fn = ['data/tfrecords/seed_1/fold_5.tfrecords']
            val_size = 5847

            train_acc, train_loss = model.eval(sess, train_size, FLAGS.eval_batch_size, train_fn, False)
            print("Train Accuracy: {0:.3f}".format(train_acc))
            print("Train Loss: {0:.3f}".format(train_loss))

            val_acc, val_loss = model.eval(sess, val_size, FLAGS.eval_batch_size, val_fn, False)
            print("Val Accuracy: {0:.3f}".format(val_acc))
            print("Val Loss: {0:.3f}".format(val_loss))
    elif FLAGS.eval_mode == 'test':
        model = CNN()
        with tf.Session() as sess:
            model.load_or_init(sess)
            test_fn = ['data/tfrecords/test.tfrecords']
            test_size = 3252

            test_acc, test_loss = model.eval(sess, test_size, FLAGS.eval_batch_size, test_fn, False)
            print("Test Accuracy: {0:.3f}".format(test_acc))
            print("Test Loss: {0:.3f}".format(test_loss))


if __name__ == '__main__':
    main()