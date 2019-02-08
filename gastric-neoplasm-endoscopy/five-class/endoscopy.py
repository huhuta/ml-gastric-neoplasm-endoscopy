import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import \
    ResNet50, InceptionResNetV2, InceptionV3, Xception
from ml.input_pipeline import \
    inception_preprocessing_input_fn, vgg_preprocessing_input_fn, \
    NUM_TRAIN, NUM_VALIDATION
from ml.models import finetune_model

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

DATA_DIR = '~/work/models/neoplasm/data'
NUM_CLASSES = 2
BATCH_SIZE = 20
NUM_EPOCH = 25


def get_data(mode):
    return inception_preprocessing_input_fn(data_dir=DATA_DIR,
                                            batch_size=BATCH_SIZE,
                                            num_classes=NUM_CLASSES,
                                            mode=mode)


def main():
    optimizer = tf.keras.optimizers.Adam(lr=0.001, decay=0.99)
    base_model = InceptionResNetV2(
        include_top=False,
        weights='imagenet',
        classes=NUM_CLASSES)
    model = finetune_model(base_model, NUM_CLASSES)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    validation_data = get_data(mode='validation')
    train_data = get_data(mode='train')
    history = model.fit(
        train_data,
        epochs=NUM_EPOCH,
        validation_data=validation_data,
        steps_per_epoch=int(np.ceil(NUM_TRAIN / BATCH_SIZE)),
        validation_steps=int(np.ceil(NUM_VALIDATION / BATCH_SIZE)))
    print((history.history['val_acc']))


if __name__ == "__main__":
    main()
