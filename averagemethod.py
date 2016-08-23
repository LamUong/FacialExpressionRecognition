from __future__ import print_function

import cv2
import PIL
from PIL import Image
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adadelta
from keras.utils import np_utils
from keras.regularizers import l2, activity_l2
import numpy as np
import cPickle 
import numpy
import cv2
import scipy
import csv
import dataprocessing
def model_generate():
	img_rows, img_cols = 48, 48

	model = Sequential()
	model.add(Convolution2D(64, 5, 5, border_mode='valid',
							input_shape=(1, img_rows, img_cols)))
	model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
	model.add(keras.layers.convolutional.ZeroPadding2D(padding=(2, 2), dim_ordering='th'))
	model.add(MaxPooling2D(pool_size=(5, 5),strides=(2, 2)))
	  
	model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='th')) 
	model.add(Convolution2D(64, 3, 3))
	model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
	model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='th')) 
	model.add(Convolution2D(64, 3, 3))
	model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
	model.add(keras.layers.convolutional.AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))
	 
	model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='th'))
	model.add(Convolution2D(128, 3, 3))
	model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
	model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='th'))
	model.add(Convolution2D(128, 3, 3))
	model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
	 
	model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='th'))
	model.add(keras.layers.convolutional.AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))
	 
	model.add(Flatten())
	model.add(Dense(1024))
	model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
	model.add(Dropout(0.2))
	model.add(Dense(1024))
	model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
	model.add(Dropout(0.2))
	 
	model.add(Dense(7))
	model.add(Activation('softmax'))

	ada = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
	model.compile(loss='categorical_crossentropy',
				  optimizer=ada,
				  metrics=['accuracy'])
	model.summary()
	return model

def convertPercentTage(Array):
	result = []
	Sum = float(0)
	for x in range(0,7):
		Sum += float(Array[0][x])
	for i in range(0,7): 
		result.append(float(Array[0][i])/Sum)
	return result

def ConvertToArrayandReshape(List):
	numpyarray = numpy.asarray(List)
	numpyarray = numpyarray.reshape(1,48,48)
	numpyarray = numpyarray.reshape(1, 1, 48, 48)
	numpyarray = numpyarray.astype('float32')
	return numpyarray

def ConvertTo3DVolume(data):
	img_rows, img_cols = 48, 48
	test_set_x = numpy.asarray(data) 
	test_set_x = test_set_x.reshape(test_set_x.shape[0],img_rows,img_cols)
	test_set_x = test_set_x.reshape(test_set_x.shape[0], 1, img_rows, img_cols)
	test_set_x = test_set_x.astype('float32')
	return test_set_x

model = model_generate()
filepath='MPM.hdf5'
model.load_weights(filepath)

test_set_x, test_set_y = dataprocessing.load_test_data()

'''
proba = model.predict_proba(test_set_x, batch_size=3589)
print(proba[0])
proba = model.predict_proba(ConvertToArrayandReshape(prev_test_x[0]), batch_size=3589)
print(proba)
'''

test_set_x_flip = []
test_set_x_rotleft = []
test_set_x_rotright = []
test_set_x_shiftedup = []
test_set_x_shifteddown = []
test_set_x_shiftedright = []
test_set_x_shifedleft = []

for data5 in test_set_x:
	test_set_x_flip.append(dataprocessing.Flip(data5))
	test_set_x_rotleft.append(dataprocessing.Roated15Left(data5))
	test_set_x_rotright.append(dataprocessing.Roated15Right(data5))
	test_set_x_shiftedup.append(dataprocessing.shiftedUp20(data5))
	test_set_x_shifteddown.append(dataprocessing.shiftedDown20(data5))
	test_set_x_shifedleft.append(dataprocessing.shiftedLeft20(data5))
	test_set_x_shiftedright.append(dataprocessing.shiftedRight20(data5))

test_set_x_init = ConvertTo3DVolume(test_set_x)
test_set_x_flip= ConvertTo3DVolume(test_set_x_flip)
test_set_x_rotleft= ConvertTo3DVolume(test_set_x_rotleft)
test_set_x_rotright = ConvertTo3DVolume(test_set_x_rotright)
test_set_x_shiftedup = ConvertTo3DVolume(test_set_x_shiftedup)
test_set_x_shifteddown = ConvertTo3DVolume(test_set_x_shifteddown)
test_set_x_shifedleft = ConvertTo3DVolume(test_set_x_shifedleft)
test_set_x_shiftedright = ConvertTo3DVolume(test_set_x_shiftedright)


proba = model.predict_proba(test_set_x_init)
proba1 = model.predict_proba(test_set_x_flip)
proba2 = model.predict_proba(test_set_x_rotleft)
proba3 = model.predict_proba(test_set_x_rotright)
proba4 = model.predict_proba(test_set_x_shiftedup)
proba5 = model.predict_proba(test_set_x_shifteddown)
proba6 = model.predict_proba(test_set_x_shifedleft)
proba7 = model.predict_proba(test_set_x_shiftedright)
Out = []
for row in zip(proba,proba1,proba2,proba3,proba4,proba5,proba6,proba7):
	a = numpy.argmax(np.array(row).mean(axis=0))
	Out.append(a)

a = np.array(Out)
b = np.array(test_set_y)
c = np.sum(a == b)
print((float(c)/len(Out)))

