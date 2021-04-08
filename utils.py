# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 11:22:47 2020

@author: Navdeep Kumar
"""

import tensorflow as tf
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from shapely.geometry import Polygon


def predict_mask(image, model, size):
    """
	Predict the mask from image
	"""
    image = tf.image.resize_with_pad(image, size, size, method='nearest')

    image = tf.cast(image, tf.float32) / 255.0  # normalize the image
    image = tf.expand_dims(image, 0)
    pred_mask = model.predict(image)
    pred_mask = tf.squeeze(pred_mask)
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask


def bb_pts(mask):
    """
	return coordinate points from predicted mask
	"""
    #    mask = mask.numpy()
    mask_arr = np.squeeze(mask, axis=2)

    coord = np.where(mask_arr == [1])
    y_min = min(coord[0])
    y_max = max(coord[0])
    x_max = max(coord[1])
    x_min = min(coord[1])

    dim_x = x_max - x_min
    dim_y = y_max - y_min
    dim = max(dim_x, dim_y)
    dim = dim + 4

    if dim_x % 2 != 0:  # odd
        rem_x = dim - dim_x
        rem_x = rem_x // 2
        x_min = x_min - rem_x
        x_max = x_max + rem_x + 1
    else:
        rem_x = dim - dim_x
        rem_x = rem_x / 2
        x_min = x_min - rem_x
        x_max = x_max + rem_x

    if dim_y % 2 != 0:
        rem_y = dim - dim_y
        rem_y = rem_y // 2
        y_min = y_min - rem_y
        y_max = y_max + rem_y + 1
    else:
        rem_y = dim - dim_y
        rem_y = rem_y / 2
        y_min = y_min - rem_y
        y_max = y_max + rem_y

    pts = [x_min, y_min, x_max, y_max]
    pts = np.asarray(pts).astype(int)

    return pts


def upscale_pts(pts, size, up_size):
    """
		upscale points to original image size to be used for cropping
		"""
    h, w = size
    h_up, w_up = up_size

    w_fact = w_up / w
    h_fact = h_up / h

    up_x1 = pts[0] * w_fact
    up_x2 = pts[2] * w_fact

    up_y1 = pts[1] * h_fact
    up_y2 = pts[3] * h_fact

    return [up_x1, up_y1, up_x2, up_y2]

def crop_to_aspect(mask, asp_ratio):
    cr_h = mask.shape[0] * asp_ratio
    cr_h = int(cr_h)
    cr_w = mask.shape[1]
    
    crop_mask = tf.image.resize_with_crop_or_pad(mask, cr_h, cr_w)
    
    return crop_mask

def cropped(mask, image):
    """
	Creating cropped image from original size image
	"""
    bbox_pts = bb_pts(mask)
    w = bbox_pts[0]
    h = bbox_pts[1]
    tr_h = bbox_pts[3] - bbox_pts[1]  # target height
    tr_w = bbox_pts[2] - bbox_pts[0]  # target width
    image = tf.image.crop_to_bounding_box(image, h, w, tr_h, tr_w)

    return image


def draw_contours(image, h_mask, op_mask):
    h_mask = np.array(h_mask)
    op_mask = np.array(op_mask)
    #    main = cv.imread(image_path,cv.IMREAD_GRAYSCALE)
    main = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    main = cv.cvtColor(main, cv.COLOR_GRAY2RGB)
    br_main = cv.convertScaleAbs(main, alpha=5, beta=40)

    ######## get contour of head and operculum ####################
    _, head_thresh = cv.threshold(h_mask, 0.001, 255, 0)
    _, op_thresh = cv.threshold(op_mask, 0.001, 255, 0)
    head_thresh = head_thresh.astype(np.uint8)
    op_thresh = op_thresh.astype(np.uint8)

    h_contours, _ = cv.findContours(head_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    o_contours, _ = cv.findContours(op_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    cv.drawContours(br_main, h_contours, -1, (0, 255, 255), 1)
    cv.drawContours(br_main, o_contours, -1, (255, 0, 255), 1)

    plt.imshow(br_main)


def h_find_components(image):
    contours, hierarchy = cv.findContours(image, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    components = []
    if len(contours) > 0:
        top_index = 0
        tops_remaining = True
        while tops_remaining:
            exterior = contours[top_index][:, 0, :].tolist()

            interiors = []
            # check if there are children and process if necessary
            if hierarchy[0][top_index][2] != -1:
                sub_index = hierarchy[0][top_index][2]
                subs_remaining = True
                while subs_remaining:
                    interiors.append(contours[sub_index][:, 0, :].tolist())

                    # check if there is another sub contour
                    if hierarchy[0][sub_index][0] != -1:
                        sub_index = hierarchy[0][sub_index][0]
                    else:
                        subs_remaining = False

            # add component tuple to components only if exterior is a polygon
            if len(exterior) > 3:
                components.append((exterior, interiors))

            # check if there is another top contour
            if hierarchy[0][top_index][0] != -1:
                top_index = hierarchy[0][top_index][0]
            else:
                tops_remaining = False
    h, w = image.shape[:2]
    min_area = int(0.02 * w * h)
    for i, component in enumerate(components):
        components[i] = Polygon(components[i][0])
    n_comp = [component for component in components if component.area > min_area]

    return n_comp
def o_find_components(image):
    contours, hierarchy = cv.findContours(image, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    components = []
    if len(contours) > 0:
        top_index = 0
        tops_remaining = True
        while tops_remaining:
            exterior = contours[top_index][:, 0, :].tolist()

            interiors = []
            # check if there are children and process if necessary
            if hierarchy[0][top_index][2] != -1:
                sub_index = hierarchy[0][top_index][2]
                subs_remaining = True
                while subs_remaining:
                    interiors.append(contours[sub_index][:, 0, :].tolist())

                    # check if there is another sub contour
                    if hierarchy[0][sub_index][0] != -1:
                        sub_index = hierarchy[0][sub_index][0]
                    else:
                        subs_remaining = False

            # add component tuple to components only if exterior is a polygon
            if len(exterior) > 3:
                components.append((exterior, interiors))

            # check if there is another top contour
            if hierarchy[0][top_index][0] != -1:
                top_index = hierarchy[0][top_index][0]
            else:
                tops_remaining = False
    h, w = image.shape[:2]
    min_area = int(0.00002 * w * h)
    for i, component in enumerate(components):
        components[i] = Polygon(components[i][0])
    n_comp = [component for component in components if component.area > min_area]

    return n_comp


def h_make_polygon(mask):
    #mask = np.array(mask)
    mask = mask[::-1, :]  # for cytomine bottom left
    mask = np.ascontiguousarray(mask, dtype=np.uint8)
    _, thresh = cv.threshold(mask, 0.001, 255, 0)
    thresh = cv.medianBlur(thresh,11)
    thresh = thresh.astype(np.uint8)
    components = h_find_components(thresh)
    #     contour = np.squeeze(contour[0])
    #     polygon = Polygon(contour)

    return components

def o_make_polygon(mask):
    #mask = np.array(mask)
    mask = mask[::-1, :]  # for cytomine bottom left
    mask = np.ascontiguousarray(mask, dtype=np.uint8)
    _, thresh = cv.threshold(mask, 0.001, 255, 0)
    thresh = cv.medianBlur(thresh,9)
    thresh = thresh.astype(np.uint8)
    components = o_find_components(thresh)
    #     contour = np.squeeze(contour[0])
    #     polygon = Polygon(contour)

    return components


def op_pad_up(h_mask, op_mask, size, upsize):
    """
	Upsize the cropped version of the predicted op_mask to original
	image size to be used to find the contours of the operculum
	"""
    h_bb = bb_pts(h_mask)
    h_bb_up = upscale_pts(h_bb, size, upsize)
    h_bb_up = np.asarray(h_bb_up)
    h_bb_up = tf.dtypes.cast(h_bb_up, tf.int32)
    pad_arr = np.zeros(upsize)
    op_mask = np.squeeze(op_mask)

    x_min, y_min, x_max, y_max = h_bb_up  # pts from head
    pad_arr[y_min:y_max, x_min:x_max] = op_mask
    pad_arr = np.expand_dims(pad_arr, axis=2)

    return pad_arr

def pad_to_patch(image, patch_size):
    """
    Padd the image to match with the integer patch count
    """
    h = im.shape[0]
    w = im.shape[1]
    h_offset = (patch_size - h % patch_size)//2
    w_offset = (patch_size - w % patch_size)//2

    re_image = cv.copyMakeBorder(im, h_offset, h_offset, w_offset, w_offset, cv.BORDER_REFLECT)
    return re_image
