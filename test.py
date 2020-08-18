# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 12:37:10 2020

@author: Navdeep Kumar
"""

import numpy as np
import random
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
import cv2
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

         # parse arguments
    parser = argparse.ArgumentParser(
                        description="unet_image_segmentation using deep learning")
    parser.add_argument('-p', type=int, required=True,
                        help="the project id")
#    parser.add_argument('-i', type=str, required=True,
#                       help="the image id")
    parser.add_argument('-wd', type=str, required=True,
                        help="the local working directory where images are downloaded")
    
    args = parser.parse_args()

    ###############################################################################
    cred = json.load(open('credentials-jsnow.json'))


    ###############################################################################
    h_model = load_model('./models/head_tversky_9963.hdf5', compile=False)
    h_model.compile(optimizer='adam', loss=tversky_loss,
                    metrics=['accuracy'])
    op_model = load_model('./models/op_ce_9989.hdf5')
    # model = load_model('./models/all_tversky_9959.hdf5') #for three class
    ###############################################################################
    

    # =============================================================================
    with Cytomine(host='https://research.cytomine.be', public_key='66e2e74c-3959-4cae-94a7-13d7acd332ac', private_key='109eb813-0515-4023-8499-30e251fe15eb') as conn:
        image_path = glob.glob(os.path.join(args.wd +'data'+'/*.bmp'))
        filename =  os.path.basename(image_path[0])
        image = load_img(image_path[0])  # for single image
        image = img_to_array(image)
        image = image[::-1,:]
        image = np.ascontiguousarray(image, dtype=np.uint8)
        org_size = image.shape[:2]
        size = (256, 256)

        h_mask = predict_mask(image, h_model)

        cropped_image = cropped(h_mask, image)

        op_mask = predict_mask(cropped_image, op_model)

        h_upsize = org_size
        op_upsize = cropped_image.shape[:2]

        op_mask = tf.image.resize(op_mask, op_upsize, method='bilinear')
        op_mask = op_pad_up(h_mask, op_mask, size, org_size)
        h_mask = tf.image.resize(h_mask, h_upsize, method='bilinear')

        h_polygon = make_polygon(h_mask)
        op_polygon = make_polygon(op_mask)
        #op_head_ratio = op_polygon[0].area / h_polygon[0].area
        image_instances = ImageInstanceCollection().fetch_with_filter("project", 142037659)
        image_id = next((x.id for x in image_instances if x.originalFilename == filename), None)
        annotations = AnnotationCollection()
        annotations.append(Annotation(location=h_polygon[0].wkt, id_image=image_id, id_terms = 143971108, id_project=142037659))
        annotations.append(Annotation(location=op_polygon[0].wkt, id_image=image_id, id_term = 143971084, id_project=142037659))
        annotations.save()

         # project 142037659


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



