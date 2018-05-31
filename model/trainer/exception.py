from __future__ import absolute_import
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
import sklearn
import argparse
import cPickle
import gzip  
import json
from tensorflow.python.lib.io import file_io


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
    
    return y_train, y_validation

def preprocessing():
    return None

def create_model():
    model = Sequential()
    model.add(Dense(42, activation='relu'))
    model.add((Dense(6, activation='sigmoid')))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    return model


def main(train_file, test_file, job_dir):
    y_train, y_validation = load_data(train_file)
    
    # Save model weights
    model.save('model.h5')

    # Save model on google storage
    with file_io.FileIO('model.h5', mode='r') as input_f:
        with file_io.FileIO(job_dir + '/model.h5', mode='w+') as output_f:
            output_f.write(input_f.read())
    
    print(y_train.shape,y_validation.shape)

if __name__ == '__main__':
    """
    The argparser can also be extended to take --n-epochs or --batch-size arguments
    """
    parser = argparse.ArgumentParser()
    
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

    main(args.train_file, args.test_file, args.job_dir)
