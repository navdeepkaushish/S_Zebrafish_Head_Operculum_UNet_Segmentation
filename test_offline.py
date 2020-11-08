# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 10:52:15 2020

@author: Navdeep Kumar
"""

import numpy as np
import random
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from itertools import combinations
import pandas as pd
from scipy import stats
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from sklearn.metrics import mean_squared_error
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
import tensorflow.keras.backend as K
from utils import *
from loss_functions import tversky_loss, dice_coef_loss

h_model = load_model('./models/head_ce_9963.hdf5', compile=True)
model = load_model('./models/all_tversky_9959.hdf5', compile=False)
model.compile(optimizer='adam', loss=tversky_loss,
                   metrics=['accuracy'])

o_model = load_model('./models/op_ce_9989.hdf5')

image_path = glob.glob("./images/*")

def count_ratio(mask, up_size):
    h_mask = np.where(mask==2,1,mask)
    o_mask = np.where(mask==1,0,mask)
    
    h_mask = tf.image.resize(h_mask, up_size, method='bilinear')
    o_mask = tf.image.resize(o_mask, up_size, method='bilinear')
    
    h_mask = h_mask.numpy()
    o_mask = o_mask.numpy()
    h_mask = h_mask.astype(np.uint8)
    o_mask = o_mask.astype(np.uint8)
    
    h_polygon = make_polygon(h_mask)
    o_polygon = make_polygon(o_mask)
    
    ratio = o_polygon[0].area / h_polygon[0].area
    
    return ratio
ratio = {}
for i in range(len(image_path)):
    image = load_img(image_path[i])
    image = img_to_array(image)
    up_size = image.shape[:2]
    filename =  os.path.basename(image_path[i])
    mask = predict_mask(image, model)
    pred_ratio = count_ratio(mask, up_size)
    
    ratio[filename] = pred_ratio
##############################################################################    
def ratio_two_class(image,h_model, o_model):
    h_mask = predict_mask(image, h_model)
    
    cropped_image = cropped(h_mask, image)
    
    op_mask = predict_mask(cropped_image, o_model)
    
    h_size = image.shape[:2]
    op_upsize = cropped_image.shape[:2]
    
    op_mask = tf.image.resize(op_mask, op_upsize, method='bilinear')
    op_mask = op_pad_up(h_mask, op_mask, (256,256), org_size)
    h_mask = tf.image.resize(h_mask, org_size, method='bilinear')
    
    h_polygon = make_polygon(h_mask)
    op_polygon = make_polygon(op_mask)
    
    ratio = op_polygon[0].area / h_polygon[0].area
    
    return ratio
ratio = {}
for i in range(len(image_path)):
    image = load_img(image_path[i])
    image = img_to_array(image)
    org_size = image.shape[:2]
    filename =  os.path.basename(image_path[i])
    pred_ratio = ratio_two_class(image, h_model, o_model)
    
    ratio[filename] = pred_ratio
    
    
df_29 = pd.read_excel(io="test_data.xlsx", sheet_name=0)
df_30 = pd.read_excel(io="test_data.xlsx", sheet_name=1)

frames = [df_29, df_30]
df =  pd.concat(frames)
gt_list = [i for i in df['Ratio']]
#gt_list = [ '%.3f' % elem for elem in gt_list ]

pred_list = [i for i in ratio.values()]
#pred_list = [ '%.3f' % elem for elem in pred_list ]


def compute_linear_coor(gt, p):

	df =  pd.DataFrame() 
	df['gt'] = gt
	df['p'] = p

	coor = (df.min(axis=1) / df.max(axis=1)).mean()   

	return coor

val = compute_linear_coor(gt_list, pred_list)

stats.pearsonr(gt_list, pred_list)

for idx,i in enumerate(list(comb)):
    print(idx,i)
    
    
gt_comb = combinations(gt_list, 2)

pred_comb = combinations(pred_list, 2)

pred_comb_list = []
for i in list(pred_comb):
    pred_comb_list.append(i)
    
gt_comb_list = []
for i in list(gt_comb):
    gt_comb_list.append(i)
    
count = 0    
for i in range(len(gt_comb_list)):
    a = gt_comb_list[i][0] - gt_comb_list[i][1]
    b = pred_comb_list[i][0] - pred_comb_list[i][1]
    
    if np.sign(a) == np.sign(b):
        count = count + 1
        
df_29 = pd.read_excel(io="test_data.xlsx", sheet_name=0)
df_30 = pd.read_excel(io="test_data.xlsx", sheet_name=1)

frames = [df_29, df_30]
df =  pd.concat(frames)
gt_list = [i for i in df['Ratio']]
gt_list = [ '%.3f' % elem for elem in gt_list ]

pred_list = [i for i in ratio.values()]
pred_list = [ '%.3f' % elem for elem in pred_list ]


def compute_linear_coor(gt, p):

	df =  pd.DataFrame() 
	df['gt'] = gt
	df['p'] = p

	coor = (df.min(axis=1) / df.max(axis=1)).mean()   

	return coor

val = compute_linear_coor(gt_list, pred_list)

pred_list = [float(i) for i in pred_list]
gt_list = [float(i) for i in gt_list]
    
    
    
    
    
    
    
    
    
                



















filename =  os.path.basename(image_path[0])
image = load_img(image_path[2])  # for single image
image = img_to_array(image)
#image = image[::-1,:]
#image = np.ascontiguousarray(image, dtype=np.uint8)
org_size = image.shape[:2]
size = (256, 256)
# 
h_mask = predict_mask(image, h_model)
# 
cropped_image = cropped(h_mask, image)
# 
op_mask = predict_mask(cropped_image, op_model)
# 
h_upsize = org_size
op_upsize = cropped_image.shape[:2]
# 
op_mask = tf.image.resize(op_mask, op_upsize, method='bilinear')
op_mask = op_pad_up(h_mask, op_mask, size, org_size)
h_mask = tf.image.resize(h_mask, org_size, method='bilinear')
# 
h_polygon = make_polygon(h_mask)
op_polygon = make_polygon(op_mask)
op_head_ratio = op_polygon[0].area / h_polygon[0].area

draw_contours(image, h_mask, op_mask)

plt.figure(figsize=(15, 15))
title = ['Input Image', 'Head Mask', 'Operculum Mask']

plt.subplot(1, 3, 1)
plt.title(title[0])
plt.imshow(array_to_img(image))

plt.subplot(1, 3, 2)
plt.title(title[1])
plt.imshow(array_to_img(h_mask))

plt.subplot(1, 3, 3)
plt.title(title[2])
plt.imshow(array_to_img(op_mask))

plt.show()