# -*- coding: utf-8 -*-
from __future__ import absolute_import
import sys
sys.path.append('../')
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.applications import VGG16
import sklearn
import argparse
import cPickle
import gzip
import json
import logging
import tensorflow as tf
from tensorflow.python.lib.io import file_io

try:
    from batch_generator import BatchGenerator, BatchSequence
except:
    from .batch_generator import BatchGenerator, BatchSequence

def load_data(path):

    # Load images

    # Load and decompress training labels
    with file_io.FileIO(path + 'data/y_train.pickle', mode='rb') as fp:
        data = gzip.GzipFile(fileobj=fp)
        y_train = cPickle.load(data)

    # Load and decompress validation labels
    with file_io.FileIO(path + 'data/y_validation.pickle', mode='rb') as fp:
        data = gzip.GzipFile(fileobj=fp)
        y_validation = cPickle.load(data)

    # with file_io.FileIO(path + 'data/binarizer.pickle', mode='rb') as fp:
    #     #data = gzip.GzipFile(fileobj=fp)
    #     binarizer = cPickle.load(fp)

#    y_train = binarizer.transform(y_train)
#    y_validation = binarizer.transform(y_validation)

    return y_train, y_validation

def preprocessing():
    return None

def create_model():
    # Init pre-trained network
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(290,290,3))

    # Adding the last two fully-connected layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x) # global average pooling (flatten)
    x = Dense(1024, activation='relu')(x) # should be rather large with 228 output labels
    y = Dense(228, activation='sigmoid')(x) # sigmoid instead of softmax to have independent probabilities

    model = Model(inputs=base_model.input, outputs=y)
    # Train only the top layer
    for layer in base_model.layers:
        layer.trainable = False

    # Use binary loss instead of categorical loss to penalize each output independently
    model.compile(optimizer='adam', loss='binary_crossentropy')

    return model


def main(train_file, test_file, job_dir, session):
    y_train, y_validation = load_data(train_file)
    y_train = np.array([j[1:] for j in y_train])
    y_validation = np.array([j[1:] for j in y_validation])

    epochs = 10
    batch_size = 64

    #input_dir=job_dir+'data/train'
    training_gen = BatchGenerator(input_dir=job_dir+'data/train',
                                  y=y_train,
                                  epochs=epochs,
                                  batch_size=batch_size,
                                  session=session)
    validation_gen = BatchSequence(input_dir=job_dir+'data/validation',
                                    y=y_validation,
                                    batch_size=batch_size,
                                    session=session)

    model = create_model()

    #model.fit_generator(generator=training_gen,
#                        steps_per_epoch=int(len(y_train)/batch_size),
#                        epochs=epochs,
#                        validation_data=validation_gen,
#                        validation_steps=int(len(y_validation)/batch_size))

    for i in range(epochs):
        for batch_x, batch_y in training_gen:
            model.fit(batch_x, batch_y)

    model.save(job_dir + 'models/vgg16.h5')

if __name__ == '__main__':
    """
    The argparser can also be extended to take --n-epochs or --batch-size arguments
    """
    parser = argparse.ArgumentParser()

    LOGGER = logging.getLogger('trainer')
    LOGGER.info('TESTING LOGGER ITSELF')

    # Input Arguments
    parser.add_argument(
      '--train-file',
      help='GCS or local paths to training data',
      required=True
    )

    parser.add_argument(
      '--test-file',
      help='GCS or local paths to test data',
      required=False
    )

    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__
    print('args: {}'.format(arguments))
# This works
    with tf.Session() as session:
        session.run(main(args.train_file, args.test_file, args.job_dir, session))
