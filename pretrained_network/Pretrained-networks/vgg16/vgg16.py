# -*- coding: utf-8 -*-
import os
import sys
sys.path.append("../../data_preparation/")
import json
import pickle
import numpy as np
import pandas as pd

from keras.applications import VGG16
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model, load_model

from batch_generator import BatchGenerator, BatchSequence

# Set the paths
input_path = os.path.abspath('../../../mlipdata/')
images_path_train = os.path.join(input_path, 'files/train/')

# Load the multilabel binarizer
with open('../binarizer.pickle', 'rb') as pickle_file:
    binarizer = pickle.load(pickle_file)

# Load training data from file
train={}
with open(os.path.join(input_path, 'train.json')) as json_data:
    train= json.load(json_data)

train_img_url = train['images']
train_img_url = pd.DataFrame(train_img_url)
train_ann = train['annotations']
train_ann = pd.DataFrame(train_ann)
train = pd.merge(train_img_url, train_ann, on='imageId', how='inner')
train['imageId'] = train['imageId'].astype(np.uint32)

y_train = np.array(train.labelId)
y_train_bin = binarizer.transform(y_train)

# Load the generator
training_gen = BatchGenerator(input_dir=images_path_train, y=y_train_bin, batch_size=64)

# Init pre-trained network
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(290,290,3))

# Adding the last two fully-connected layers
x = base_model.output
x = GlobalAveragePooling2D()(x) # global average pooling (flatten)
x = Dense(1024, activation='relu')(x) # should be rather large with 228 output labels
y = Dense(228, activation='softmax')(x) # sigmoid instead of softmax to have independent probabilities

model = Model(inputs=base_model.input, outputs=y)
# Train only the top layer
for layer in base_model.layers:
    layer.trainable = False

# Use binary loss instead of categorical loss to penalize each output independently
model.compile(optimizer='adam', loss='binary_crossentropy')

# 1000 steps = 640000 random images per epoch
model.fit_generator(training_gen, steps_per_epoch=int(3000/64), epochs=10)

model.save('./vgg16_cloud_model.h5')
