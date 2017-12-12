import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import argparse


def get_data():
    root_dir = os.getcwd()
    data_dir = os.path.join(root_dir, 'data')

    with open(os.path.join(data_dir, 'xtrain_obfuscated.txt'), 'r') as f:
        train_lines = f.read().splitlines()

    with open(os.path.join(data_dir, 'ytrain.txt'), 'r') as f:
        train_labels = f.read().splitlines()
        train_labels = np.array([int(label) for label in train_labels])

    with open(os.path.join(data_dir, 'xtest_obfuscated.txt'), 'r') as f:
        predict_lines = f.read().splitlines()

    return train_lines, train_labels, predict_lines


def get_small_data(X, y, size, seed):
    if isinstance(X, list):
        dataset_size = len(X)
    elif isinstance(X, np.ndarray):
        dataset_size = X.shape[0]
    else:
        raise ValueError('Input data has unsupported format')
    _, X_test, _, y_test = train_test_split(X, y, test_size=size/dataset_size, random_state=seed,
                                            shuffle=True, stratify=y)
    return X_test, y_test


# ============================================================================================================ #
# Creating tfrecords
# ============================================================================================================ #
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def maybe_create_tfrecords(seed_kfold=1, seed_overfit=1, test_size=0.1, folds=5):
    root_dir = os.getcwd()
    data_dir = os.path.join(root_dir, 'data')
    tfrecords_dir = os.path.join(data_dir, 'tfrecords')
    seed_partition_dir = os.path.join(tfrecords_dir, 'seed_{}'.format(seed_kfold))

    if not os.path.exists(tfrecords_dir):
        os.makedirs(tfrecords_dir)
        train_lines, train_labels, predict_lines = get_data()

        lines, labels = get_small_data(train_lines, train_labels, size=100, seed=seed_overfit)
        create_tfrecords(tfrecords_dir, lines, labels, 'train_overfit_100')

        lines, labels = get_small_data(train_lines, train_labels, size=1000, seed=seed_overfit)
        create_tfrecords(tfrecords_dir, lines, labels, 'train_overfit_1000')

        create_tfrecords(tfrecords_dir, predict_lines, None, 'predict')

        X, X_test, y, y_test = train_test_split(train_lines, train_labels, test_size=test_size,
                                                random_state=seed_kfold, shuffle=True, stratify=train_labels)
        create_tfrecords(tfrecords_dir, X_test, y_test, 'test')

    if not os.path.exists(seed_partition_dir):
        os.makedirs(seed_partition_dir)
        train_lines, train_labels, predict_lines = get_data()

        X, X_test, y, y_test = train_test_split(train_lines, train_labels, test_size=test_size,
                                                        random_state=seed_kfold, shuffle=True, stratify=train_labels)
        kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed_kfold)
        i = 1
        for _, val_index in kf.split(X, y):
            create_tfrecords(seed_partition_dir, [X[index] for index in val_index], y[val_index], 'fold_'+str(i))
            print('fold_{} size: {}'.format(i, val_index.shape[0]))
            i += 1


def create_tfrecords(data_path, lines, labels, name):
    num_lines = len(lines)
    if labels is None:
        labels = np.repeat([-1], num_lines)
    lines_len = [len(line) for line in lines]
    train_filename = name + '.tfrecords'

    writer = tf.python_io.TFRecordWriter(os.path.join(data_path, train_filename))

    for i in tqdm(range(num_lines)):
        example = tf.train.Example(features=tf.train.Features(feature={
            'length': _int64_feature(lines_len[i]),
            'label': _int64_feature(labels[i]),
            'line_raw': _bytes_feature(tf.compat.as_bytes(lines[i]))}))
        writer.write(example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Loader', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed_kfold', default=1, type=int, help='seed for kfold split')
    parser.add_argument('--seed_overfit', default=1, type=int, help='see for data to overfit')
    parser.add_argument('--test_size', default=0.1, type=float, help='size of test (hold-out) dataset')
    parser.add_argument('--folds', default=5, type=int, help='number of folds to split data')
    args = parser.parse_args()

    maybe_create_tfrecords(seed_kfold=args.seed_kfold, seed_overfit=args.seed_overfit, test_size=args.test_size,
                           folds=args.folds)
