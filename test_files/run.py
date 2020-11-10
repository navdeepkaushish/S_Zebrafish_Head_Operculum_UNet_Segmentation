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
from cytomine.models import Property, Annotation, AnnotationTerm, AnnotationCollection, Project, ImageInstanceCollection

###############################################################################
if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(
        description="unet_image_segmentation using deep learning")
    parser.add_argument('-p', type=int, required=True,
                        help="the project id")
    parser.add_argument('-i', type=str, required=True,
                        help="the image id")
    parser.add_argument('-wd', type=str, required=True,
                        help="the local working directory where images are downloaded")

    args = parser.parse_args()

    ###############################################################################
    cred = json.load(open('credentials-jsnow.json'))

    ###############################################################################
    with tf.device('/cpu:0'):
        h_model = load_model('./models/head_tversky_9963.hdf5', compile=False)
        h_model.compile(optimizer='adam', loss=tversky_loss,
                    metrics=['accuracy'])
        op_model = load_model('./models/op_ce_9989.hdf5')
    # model = load_model('./models/all_tversky_9959.hdf5') #for three class
    ###############################################################################

    # =============================================================================
    with Cytomine(host='https://research.cytomine.be', public_key='66e2e74c-3959-4cae-94a7-13d7acd332ac',
                  private_key='109eb813-0515-4023-8499-30e251fe15eb') as conn:
        images = ImageInstanceCollection().fetch_with_filter('project', args.p)

        if args.i != 'all':  # select only given image instances = [image for image in image_instances if image.id in id_list]
            images = [_ for _ in images if _.id
                      in map(lambda x: int(x.strip()),
                             args.i.split(','))]
        images_id = [image.id for image in images]

        img_path = os.path.join(args.wd, 'images')
        if not os.path.exists(img_path):
            os.makedirs(img_path)

        for image in images:
            fname, fext = os.path.splitext(image.filename)
            if image.download(dest_pattern=os.path.join(
                    img_path, "{}{}".format(image.id, fext))) is not True:
                print('Failed to download image {}'.format(image.filename))

        # Prepare image file paths from image directory for execution
        image_paths = glob.glob(os.path.join(img_path, '*'))
        for i in range(len(image_paths)):
            img = Image.open(image_paths[i])
            img = img_to_array(img)

            filename = os.path.basename(image_paths[i])
            fname, fext = os.path.splitext(filename)
            fname  = int(fname)
            org_size = img.shape[:2]

            h_mask = predict_mask(img, h_model)
            size = h_mask.shape[:2]
            cropped_image = cropped(h_mask, img)

            op_mask = predict_mask(cropped_image, op_model)
            op_upsize = cropped_image.shape[:2]

            op_mask = tf.image.resize(op_mask, op_upsize, method='bilinear')
            op_mask = op_pad_up(h_mask, op_mask, size, org_size)
            h_mask = tf.image.resize(h_mask, org_size, method='bilinear')

            h_polygon = make_polygon(h_mask)
            op_polygon = make_polygon(op_mask)

         #   image_id = next((x.id for x in images if x.id == fname), None)
            annotations = AnnotationCollection()
            annotations.append(
                Annotation(location=h_polygon[0].wkt, id_image=fname, id_terms=143971108, id_project=args.p))
            annotations.append(
                Annotation(location=op_polygon[0].wkt, id_image=fname, id_term=143971084, id_project=args.p))
            annotations.save()

        # project 142037659

    # =============================================================================
