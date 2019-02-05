import glob
import os
from os.path import join, basename, dirname
from multiprocessing.dummy import Pool as ThreadPool
import tensorflow as tf


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    value = tf.compat.as_bytes(value)
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def find_files(path, pattern='**/*.jpg'):
    pathname = os.path.join(path, pattern)
    return glob.glob(pathname, recursive=True)


def create_data_sets(path):
    sub_dir_names = sorted(next(os.walk(path))[1])
    sub_dir_paths = [join(path, x) for x in sub_dir_names]
    labels = dict(zip(sub_dir_names, range(len(sub_dir_names))))

    def get_label(filepath):
        return labels.get(basename(dirname(dirname(filepath))))

    def get_image_raw(filepath):
        with tf.gfile.GFile(filepath, 'rb') as f:
            return f.read()

    def make_data_set(index):
        pathname = os.path.join(path, '**/*-{}/*.jpg'.format(index))
        filepaths = glob.glob(pathname, recursive=True)
        data_set = ((
            get_image_raw(filepath),
            get_label(filepath),
            basename(filepath)
        ) for filepath in filepaths)
        return data_set
    # data-0 ~ data-5 split
    data_sets = [make_data_set(index) for index in range(6)]
    return data_sets


# TODO sharding
def convert_to(data_set, name):
    tf_record_filename = os.path.join(
        '/tmp/test', name + '.tfrecords')
    print('Writing', tf_record_filename)
    with tf.io.TFRecordWriter(tf_record_filename) as writer:
        for image_raw, label, filename in data_set:
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'image_raw': _bytes_feature(image_raw),
                        'label': _int64_feature(label),
                        'filename': _bytes_feature(filename),
                    }))
            writer.write(example.SerializeToString())


def main(path):
    data_sets = create_data_sets(path)
    names = ['train-' + str(i) for i in range(len(data_sets))]
    pool = ThreadPool(len(data_sets))
    pool.starmap(convert_to, zip(data_sets, names))
