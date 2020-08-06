import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler


def unet(input_size = (256,256,3)):
	inputs = tf.keras.layers.Input(shape=input_size)
	conv1 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
	conv1 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
	conv1 = BatchNormalization()(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
	conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
	conv2 = BatchNormalization()(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
	conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
	conv3 = BatchNormalization()(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	conv4 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
	conv4 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
	conv4 = BatchNormalization()(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

	conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
	conv5 = BatchNormalization()(conv5)

	up6 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
	merge6 = concatenate([conv4,up6], axis = 3)
	conv6 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
	conv6 = BatchNormalization()(conv6)

	up7 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
	merge7 = concatenate([conv3,up7], axis = 3)
	conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
	conv7 = BatchNormalization()(conv7)

	up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
	merge8 = concatenate([conv2,up8], axis = 3)
	conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
	conv8 = BatchNormalization()(conv8)

	up9 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
	merge9 = concatenate([conv1,up9], axis = 3)
	conv9 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
	conv9 = BatchNormalization()(conv9)

	conv10 = Conv2D(2, 1, activation = 'softmax')(conv9)

	model = Model(inputs = inputs, outputs = conv10)

	return model


# =============================================================================
# def unet(input_size = (256,256,3)):
# 	inputs = tf.keras.layers.Input(shape=input_size)
# 	conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
# 	conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
# 	conv1 = BatchNormalization()(conv1)
# 	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
# 
# 	conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
# 	conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
# 	conv2 = BatchNormalization()(conv2)
# 	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
# 
# 	conv3 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
# 	conv3 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
# 	conv3 = BatchNormalization()(conv3)
# 	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
# 
# 	conv4 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
# 	conv4 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
# 	conv4 = BatchNormalization()(conv4)
# 	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
# 
# 	conv5 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
# 	conv5 = BatchNormalization()(conv5)
# 
# 	up6 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
# 	merge6 = concatenate([conv4,up6], axis = 3)
# 	conv6 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
# 	conv6 = BatchNormalization()(conv6)
# 
# 	up7 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
# 	merge7 = concatenate([conv3,up7], axis = 3)
# 	conv7 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
# 	conv7 = BatchNormalization()(conv7)
# 
# 	up8 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
# 	merge8 = concatenate([conv2,up8], axis = 3)
# 	conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
# 	conv8 = BatchNormalization()(conv8)
# 
# 	up9 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
# 	merge9 = concatenate([conv1,up9], axis = 3)
# 	conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
# 	conv9 = BatchNormalization()(conv9)
# 
# 	conv10 = Conv2D(3, 1, activation = 'softmax')(conv9)
# 
# 	model = Model(inputs = inputs, outputs = conv10)
# 
# 	return model
# =============================================================================
