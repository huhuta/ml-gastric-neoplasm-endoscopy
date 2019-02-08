import tensorflow as tf


def dataset_test(dataset):
    inputs = dataset.make_one_shot_iterator().get_next()
    for i in range(10):
        with tf.Session() as sess:
            print(sess.run(inputs))
