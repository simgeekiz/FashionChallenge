from os.path import join

from keras.applications import ResNet50
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model, load_model
from keras.utils.np_utils import to_categorical

import pandas as pd
import csv
import os
import numpy as np
import json

from matplotlib import pyplot as plt
import sys
sys.path.append("../../data_preparation/")

from batch_generator import BatchGenerator, BatchSequence

from sklearn.metrics import recall_score, precision_score, f1_score


#datadir = os.getcwd()
input_path = os.path.abspath('../../../mlipdata/')

train={}
test={}
validation={}
with open(os.path.join(input_path, 'train.json')) as json_data:
    train= json.load(json_data)
with open(os.path.join(input_path, 'test.json')) as json_data:
    test= json.load(json_data)
with open(os.path.join(input_path, 'validation.json')) as json_data:
    validation = json.load(json_data)

print('Train No. of images: %d'%(len(train['images'])))
print('Test No. of images: %d'%(len(test['images'])))
print('Validation No. of images: %d'%(len(validation['images'])))

# JSON TO PANDAS DATAFRAME
# train data
train_img_url=train['images']
train_img_url=pd.DataFrame(train_img_url)
train_ann=train['annotations']
train_ann=pd.DataFrame(train_ann)
train=pd.merge(train_img_url, train_ann, on='imageId', how='inner')

# test data
test=pd.DataFrame(test['images'])

# Validation Data
val_img_url=validation['images']
val_img_url=pd.DataFrame(val_img_url)
val_ann=validation['annotations']
val_ann=pd.DataFrame(val_ann)
validation=pd.merge(val_img_url, val_ann, on='imageId', how='inner')

datas = {'Train': train, 'Test': test, 'Validation': validation}
for data in datas.values():
    data['imageId'] = data['imageId'].astype(np.uint32)
    
    
images_path_train = os.path.abspath('../../../mlipdata/files/train/')

from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
# loading labels
y_train = np.array(train.labelId)
y_validation = np.array(validation.labelId)

y_train1000 = mlb.fit_transform(y_train)[:1000]
y_validation500 = mlb.fit_transform(y_validation)[:500]

# load the generator
training_gen = BatchGenerator(input_dir=images_path_train, y=y_train1000, batch_size=64)

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(290,290,3))

# Adding the last two fully-connected layers
x = base_model.output
x = GlobalAveragePooling2D()(x) # global average pooling (flatten)
x = Dense(1024, activation='relu')(x) # should be rather large with 228 output labels
#x = Dropout(0.5)(x)
y = Dense(228, activation='softmax')(x) # sigmoid instead of softmax to have independent probabilities

model = Model(inputs=base_model.input, outputs=y)
# Train only the top layer
for layer in base_model.layers:
    layer.trainable = False
    
# Use binary loss instead of categorical loss to penalize each output independently
model.compile(optimizer='adam', loss='binary_crossentropy')

# 1000 steps = 640000 random images per epoch
model.fit_generator(training_gen, steps_per_epoch=100, epochs=10)

model.save('./ResNet50.h5')