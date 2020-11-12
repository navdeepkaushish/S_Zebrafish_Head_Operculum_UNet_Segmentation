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
            h_model = load_model('/models/head_tversky_9963.hdf5', compile=False)  # head model
            h_model.compile(optimizer='adam', loss=tversky_loss,
                            metrics=['accuracy'])
            op_model = load_model('/models/op_ce_9989.hdf5')  # operculum model

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
        for i in range(len(image_paths)):
            img = Image.open(image_paths[i])
            img = img_to_array(img)

            filename = os.path.basename(image_paths[i])
            fname, fext = os.path.splitext(filename)
            fname = int(fname)
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

            conn.job.update(
                status=Job.RUNNING, progress=95,
                statusComment="Uploading new annotations to Cytomine server..")

            annotations = AnnotationCollection()
            annotations.append(Annotation(location=h_polygon[0].wkt, id_image=fname, id_terms=conn.parameters.cytomine_id_head_term,
                                          id_project=conn.parameters.cytomine_id_project))
            annotations.append(Annotation(location=op_polygon[0].wkt, id_image=fname, id_term=conn.parameters.cytomine_id_operculum_term,
                                          id_project=conn.parameters.cytomine_id_project))
            annotations.save()

        conn.job.update(status=Job.TERMINATED, status_comment="Finish", progress=100)  # 524787186


if __name__ == '__main__':
    main(sys.argv[1:])
