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
from keras.applications import VGG16, xception

try:
	from batchgenerator import BatchGenerator, BatchSequence
except:
	from .batchgenerator import BatchGenerator, BatchSequence


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


def main(train_file, test_file, job_dir, n_epochs):
	y_train, y_validation = load_data(train_file)
	images_path_train = os.path.join(train_file, 'data/train/')
	
	training_gen = BatchGenerator(
		input_dir=images_path_train,
		y=y_train,
		epochs=int(n_epochs),
		batch_size=32,
		session = tf.Session(),
		shuffle=False,
		img_size=290
	)

	# Initialize some pretrained keras model, add more models if want to stack/ensemble them
	models = []
	
	vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(290,290,3))
	vgg = fine_tune_model(vgg_base)
	models.append(vgg)
	
	# xception_base = Xception(weights='imagenet', include_top=False, input_shape=(290,290,3))
	# xception = fine_tune_model(xception_base)
	# models.append(xception)
	
	# Train all models
	for model in models:
		# Need to still define keras.utils.Sequence to use fit_generator
		#model.fit_generator(training_gen)
		
		for i in range(n_epochs):
			for batch_x, batch_y in training_gen:
				model.fit(batch_x, batch_y[:,1:])
	
	# Save model weights
	model.save('model.h5')

	# Save model on google storage
	with file_io.FileIO('model.h5', mode='r') as input_f:
		with file_io.FileIO(job_dir + '/model.h5', mode='w+') as output_f:
			output_f.write(input_f.read())
			
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
