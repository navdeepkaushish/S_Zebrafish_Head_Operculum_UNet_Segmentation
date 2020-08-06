# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 12:37:10 2020

@author: Navdeep Kumar
"""

import numpy as np
import random
import os
import glob
import math
from PIL import Image
import matplotlib.pyplot as plt
import cv2 
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
import tensorflow.keras.backend as K
from utils import *
from loss_functions import tversky_loss

import argparse
import json
import logging

from cytomine import Cytomine
from cytomine.models import Property, Annotation, AnnotationTerm, AnnotationCollection,Project, ImageInstanceCollection
###############################################################################
if __name__ == "__main__":
# =============================================================================
#     # parse arguments
#     parser = argparse.ArgumentParser(
#                         description="unet_image_segmentation using deep learning")
#     parser.add_argument('-p', type=int, required=True,
#                         help="the project id")
#     parser.add_argument('-i', type=int, required=True,
#                         help="the image id")
#     parser.add_argument('-t', type=int, required=True,
#                         help="the term id")
#     parser.add_argument('-u', type=int, required=True,
#                         help="the user id layer")
#     parser.add_argument('-wd', type=str, required=True,
#                         help="the local working directory where ROI images are downloaded")
#     
#     args = parser.parse_args()
# =============================================================================
###############################################################################

    image_path = glob.glob('./data/*.bmp') #full image path

###############################################################################
    h_model = load_model('./models/head_tversky_9963.hdf5', compile=False) #model for head segmentation
    h_model.compile(optimizer='adam', loss= tversky_loss,
              metrics=['accuracy'])
    op_model = load_model('./models/op_ce_9989.hdf5') #model for operculum
#model = load_model('./models/all_tversky_9959.hdf5') #for three class
###############################################################################
    image = load_img(image_path[10])#for single image for testing
    image = img_to_array(image)
    org_size = image.shape[:2]
    size = (256,256)
    
    h_mask = predict_mask(image, h_model)

    cropped_image = cropped(h_mask, image)

    op_mask = predict_mask(cropped_image, op_model)

    h_upsize = org_size
    op_upsize = cropped_image.shape[:2]

    op_mask = tf.image.resize(op_mask, op_upsize ,method= 'bilinear')
    op_mask = op_pad_up(h_mask, op_mask,size, org_size)
    h_mask = tf.image.resize(h_mask, h_upsize ,method= 'bilinear')
    
    h_polygon = make_polygon(h_mask)
    op_polygon = make_polygon(op_mask)
    op_head_ratio = op_polygon[0].area / h_polygon[0].area

    cred = json.load(open('credentials-jsnow.json'))

# =============================================================================
#     with Cytomine(host= cred['host'], public_key=cred['public_key'], private_key=cred['private_key']) as conn:
#         
#         annotations = Annotation(location=h_polygon.wkt, id_image= args.id, id_terms = args.t, id_project = args.p).save()
#         annotations = Annotation(location=op_polygon.wkt, id_image= args.i, id_terms = args.t, id_project = args.p).save()
# =============================================================================
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
