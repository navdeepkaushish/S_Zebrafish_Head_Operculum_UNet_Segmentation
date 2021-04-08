from __future__ import print_function

import glob
import os
import sys
import json
from shapely.geometry import Point, Polygon

from PIL import Image
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
from cytomine import CytomineJob
from cytomine.models import (
    Property,
    Annotation,
    AnnotationTerm,
    AnnotationCollection,
    Project,
    ImageInstanceCollection,
    Job)


def main(argv):
    with CytomineJob.from_cli(argv) as conn:
        conn.job.update(status=Job.RUNNING, progress=0, statusComment='Intialization...')
        base_path = "{}".format(os.getenv('HOME'))  # Mandatory for Singularity
        working_path = os.path.join(base_path, str(conn.job.id))

        # Loading models from models directory
        with tf.device('/cpu:0'):
            h_model = load_model('/models/head_dice_9975.hdf5', compile=False)  # head model
            h_model.compile(optimizer='adam', loss=dice_coef_loss,
                            metrics=['accuracy'])
            op_model = load_model('/models/op_dice_9991.hdf5', compile=False)  # operculum model
            op_model.compile(optimizer='adam', loss=dice_coef_loss,
                            metrics=['accuracy'])

        # Select images to process
        images = ImageInstanceCollection().fetch_with_filter('project', conn.parameters.cytomine_id_project)
        if conn.parameters.cytomine_id_images != 'all':  # select only given image instances = [image for image in image_instances if image.id in id_list]
            images = [_ for _ in images if _.id
                      in map(lambda x: int(x.strip()),
                             conn.parameters.cytomine_id_images.split(','))]
        images_id = [image.id for image in images]

        # Download selected images into 'working_directory'
        img_path = os.path.join(working_path, 'images')
        # if not os.path.exists(img_path):
        os.makedirs(img_path)

        for image in conn.monitor(
                images, start=2, end=50, period=0.1,
                prefix='Downloading images into working directory...'):
            fname, fext = os.path.splitext(image.filename)
            if image.download(dest_pattern=os.path.join(
                    img_path,
                    "{}{}".format(image.id, fext))) is not True:  # images are downloaded with image_ids as names
                print('Failed to download image {}'.format(image.filename))

        # Prepare image file paths from image directory for execution
        conn.job.update(progress=50,
                        statusComment="Preparing data for execution..")
        image_paths = glob.glob(os.path.join(img_path, '*'))
        std_img_size = (1032,1376)   #maximum size that the model can handle
        for i in range(len(image_paths)):
            org_img = Image.open(image_paths[9])
            

            filename = os.path.basename(image_paths[i])
            fname, fext = os.path.splitext(filename)
            fname = int(fname)
            org_img = img_to_array(org_img)
            img = org_img.copy()
            org_size = org_img.shape[:2]
            asp_ratio = org_size[0] / org_size[1]  #for cropping and upscaling to original size
            if org_size[1] > std_img_size[1]:
                img = tf.image.resize(img, (675,900), method='nearest')
                img = tf.image.resize_with_crop_or_pad(img, 1032,1376)
                h_mask = predict_mask(img, h_model)
                h_mask = crop_to_aspect(h_mask, asp_ratio)
                h_mask = tf.image.resize(h_mask, std_img_size, method='nearest')
                h_mask = tf.image.resize_with_crop_or_pad(h_mask, 675,900)
                h_up_mask = tf.image.resize(h_mask, org_size, method='nearest')
            else:
                h_mask = predict_mask(img, h_model)
                h_mask = crop_to_aspect(h_mask, asp_ratio)
                h_up_mask = tf.image.resize(h_mask, org_size, method='nearest')
        
            crop_op_img = cropped(h_up_mask, org_img)

            op_asp_ratio = crop_op_img.shape[0] / crop_op_img.shape[1]
            op_mask = predict_mask(crop_op_img, op_model)
            op_mask = crop_to_aspect(op_mask, op_asp_ratio)
            op_mask = tf.image.resize(op_mask, (crop_op_img.shape[0], crop_op_img.shape[1]), method='nearest')
            op_up_mask = tf.image.resize_with_crop_or_pad(op_mask, org_size[0], org_size[1])
        

            h_polygon = h_make_polygon(h_up_mask)
            op_polygon = o_make_polygon(op_up_mask)

            conn.job.update(
                status=Job.RUNNING, progress=95,
                statusComment="Uploading new annotations to Cytomine server..")

            annotations = AnnotationCollection()
            annotations.append(Annotation(location=h_polygon[0].wkt, id_image=fname, id_terms=143971108,
                                          id_project=conn.parameters.cytomine_id_project))
            annotations.append(Annotation(location=op_polygon[0].wkt, id_image=fname, id_term=143971084,
                                          id_project=conn.parameters.cytomine_id_project))
            annotations.save()

        conn.job.update(status=Job.TERMINATED, status_comment="Finish", progress=100)  # 524787186


if __name__ == '__main__':
    main(sys.argv[1:])
