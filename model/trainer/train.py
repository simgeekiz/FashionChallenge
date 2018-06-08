from __future__ import absolute_import

import numpy as np
import pandas as pd
import tensorflow as tf

import sklearn
import argparse
import cPickle
import gzip  
import json
import random
import os

from tensorflow.python.lib.io import file_io
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.applications import Xception, VGG16, VGG19, ResNet50, InceptionV3

from data_preparation.batchgenerator import BatchGenerator, BatchSequence
from exception_callbacks.callbacks import all_call_backs 

def load_data(path):
	# Load and decompress training labels
	with file_io.FileIO(path + 'data/y_train.pickle', mode='rb') as fp:
		data = gzip.GzipFile(fileobj=fp)
		y_train = cPickle.load(data)
	
	# Load and decompress validation labels
	with file_io.FileIO(path + 'data/y_validation.pickle', mode='rb') as fp:
		data = gzip.GzipFile(fileobj=fp)
		y_validation = cPickle.load(data)
	
	return y_train, y_validation	

def preprocessing(dir):
	return None

def fine_tune_model(base_model):
	# Adding the last two fully-connected layers
	x = base_model.output
	x = GlobalAveragePooling2D()(x) 		# global average pooling (flatten)
	x = Dense(1024, activation='relu')(x) 	# should be rather large with 228 output labels
	y = Dense(228, activation='softmax')(x)	# sigmoid instead of softmax to have independent probabilities

	model = Model(inputs=base_model.input, outputs=y)
	
	# Unfreeze last few layers
	for layer in base_model.layers[:-4]:
		layer.trainable = False
	for layer in base_model.layers[-4:]:
		layer.trainable = True

	# Use binary loss instead of categorical loss to penalize each output independently
	model.compile(optimizer='adam', loss='binary_crossentropy')

	return model

def get_models():
	"""Get all five pretrained models."""
	models = []
	
	xception_base = Xception(weights='imagenet', include_top=False, input_shape=(290,290,3))
	xception = fine_tune_model(xception_base)
	models.append(xception)
	
	vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(290,290,3))
	vgg16 = fine_tune_model(vgg16_base)
	models.append(vgg16)

	vgg19_base = VGG19(weights='imagenet', include_top=False, input_shape=(290,290,3))
	vgg19 = fine_tune_model(vgg19_base)
	models.append(vgg19)

	resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=(290,290,3))
	resnet = fine_tune_model(resnet_base)
	models.append(resnet)

	inception_base = InceptionV3(weights='imagenet', include_top=False, input_shape=(290,290,3))
	inception = fine_tune_model(inception_base)
	models.append(inception)

	return models

def main(train_file, test_file, job_dir, n_epochs):
	y_train, y_validation = load_data(train_file)

	images_path_train = os.path.join(train_file, 'data/train/')
	images_path_validation = os.path.join(train_file, 'data/validation/')
	
	epochs = 30
	callbacks = all_call_backs()
	batch_size = 128

	training_gen = BatchGenerator(
		input_dir=images_path_train,
		y=y_train,
		batch_size=batch_size,
		shuffle=True,
		img_size=290
	)

	validation_gen = BatchSequence(
		input_dir=images_path_validation,
		y=y_validation,
		batch_size=batch_size,
		shuffle=True,
		img_size=290
	)

	# Initialize some pretrained keras model, add more models if want to stack/ensemble them
	models = get_models()
	
	# Train all models
	for model in models:
		# Need to still define keras.utils.Sequence to use fit_generator
		model.fit_generator(
			generator=training_gen,
			callbacks=callbacks,
			steps_per_epoch=int(len(y_train)/batch_size),
			epochs=epochs,
			validation_data=validation_gen,
			validation_steps=int(len(y_validation)/batch_size)
		)
			
	print("main success")

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
	  required=True
	)

	parser.add_argument(
		'--job-dir',
		help='GCS location to write checkpoints and export models',
		required=True
	)

	parser.add_argument(
		'--n-epochs',
		help='Number of epochs to train the model for',
		required=True
	)
	args = parser.parse_args()
	arguments = args.__dict__
	print('args: {}'.format(arguments))

	main(args.train_file, args.test_file, args.job_dir, args.n_epochs)
