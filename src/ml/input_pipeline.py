import os
import glob
import tensorflow as tf
import ml
from ml.preprocessing import vgg_preprocessing, inception_preprocessing

TRAIN_NUM = 4200
TEST_NUM = 800


def get_features(serialized):
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string,
                                            default_value=''),
        'image/class/label': tf.FixedLenFeature([], dtype=tf.int64,
                                                default_value=-1),
    }
    features = tf.parse_single_example(serialized, keys_to_features)
    return features


def get_map_fn(preprocess_image, classes):
    def map_fn(serialized):
        features = get_features(serialized)
        image_raw = features['image/encoded']
        image = tf.image.decode_jpeg(image_raw)
        processed_image = preprocess_image(image)
        labels = features['image/class/label']
        one_hot_labels = tf.one_hot(labels, classes)
        return processed_image, one_hot_labels
    return map_fn


def _get_dataset(data_dir, mode, map_fn, batch_size):
    pattern = '**/{}-*'.format(mode)
    pathname = os.path.join(data_dir, pattern)
    filenames = glob.glob(pathname, recursive=True)
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.repeat().shuffle(TRAIN_NUM)
    dataset = dataset.map(map_fn).batch(batch_size).prefetch(1)
    return dataset


def vgg_preprocessing_input_fn(data_dir, batch_size, classes, mode='train'):

    def preprocess_image(image):
        return vgg_preprocessing.preprocess_image(
            image, 224, 224, is_training=(mode == 'train'))

    return _get_dataset(
        data_dir=data_dir,
        mode=mode,
        map_fn=get_map_fn(preprocess_image, classes),
        batch_size=batch_size)


def inception_preprocessing_input_fn(
        data_dir,
        batch_size,
        classes,
        mode='train'):

    def preprocess_image(image):
        return inception_preprocessing.preprocess_image(
            image, 299, 299, is_training=(mode == 'train'))

    return _get_dataset(
        data_dir=data_dir,
        mode=mode,
        map_fn=get_map_fn(preprocess_image, classes),
        batch_size=batch_size)
