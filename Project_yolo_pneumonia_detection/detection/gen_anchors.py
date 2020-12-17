import random
import argparse
import numpy as np
import pickle
import os
import pandas as pd
from tqdm import tqdm
from PIL import Image

from voc import parse_voc_annotation
import json


def IOU(ann, centroids):
    w, h = ann
    similarities = []

    for centroid in centroids:
        c_w, c_h = centroid

        if c_w >= w and c_h >= h:
            similarity = w*h/(c_w*c_h)
        elif c_w >= w and c_h <= h:
            similarity = w*c_h/(w*h + (c_w-w)*c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w*h/(w*h + c_w*(c_h-h))
        else: #means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w*c_h)/(w*h)
        similarities.append(similarity) # will become (k,) shape

    return np.array(similarities)

def avg_IOU(anns, centroids):
    n,d = anns.shape
    sum = 0.

    for i in range(anns.shape[0]):
        sum+= max(IOU(anns[i], centroids))

    return sum/n

def print_anchors(centroids):
    out_string = ''

    anchors = centroids.copy()

    widths = anchors[:, 0]
    sorted_indices = np.argsort(widths)

    r = "anchors: ["
    for i in sorted_indices:
        out_string += str(int(anchors[i,0]*416)) + ',' + str(int(anchors[i,1]*416)) + ', '

    print(out_string[:-2])

def run_kmeans(ann_dims, anchor_num):
    ann_num = ann_dims.shape[0]
    iterations = 0
    prev_assignments = np.ones(ann_num)*(-1)
    iteration = 0
    old_distances = np.zeros((ann_num, anchor_num))

    indices = [random.randrange(ann_dims.shape[0]) for i in range(anchor_num)]
    centroids = ann_dims[indices]
    anchor_dim = ann_dims.shape[1]

    while True:
        distances = []
        iteration += 1
        for i in range(ann_num):
            d = 1 - IOU(ann_dims[i], centroids)
            distances.append(d)
        distances = np.array(distances) # distances.shape = (ann_num, anchor_num)

        print("iteration {}: dists = {}".format(iteration, np.sum(np.abs(old_distances-distances))))

        # assign samples to centroids
        assignments = np.argmin(distances,axis=1)

        if (assignments == prev_assignments).all() :
            return centroids

        # calculate new centroids
        centroid_sums=np.zeros((anchor_num, anchor_dim), np.float)
        for i in range(ann_num):
            centroid_sums[assignments[i]]+=ann_dims[i]
        for j in range(anchor_num):
            centroids[j] = centroid_sums[j]/(np.sum(assignments==j) + 1e-6)

        prev_assignments = assignments.copy()
        old_distances = distances.copy()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--split', type=int, default=1, help='training split number (1-5)')
    parser.add_argument('-a', '--anchors', default=9, help='number of anchors to use')
    parser.add_argument('--png_data_dir', default='../data/stage_2_train_images_png')
    parser.add_argument('--train_file', default='../data/splits/s1_split%d_train_data_5cv.csv')
    return parser.parse_args()

def _main_(args):

    num_anchors = int(args.anchors)

    SPLIT_NUM = int(args.split)
    TRAIN_FILE_PATH = args.train_file % SPLIT_NUM

    train_data = pd.read_csv(TRAIN_FILE_PATH)
    train_data = train_data.loc[train_data['Target'] == 1]

    # run k_mean to find the anchors
    annotation_dims = []

    train_imgs = []

    for img in tqdm(np.unique(train_data['patientId'])):
        tmpdict = {}
        tmpdict['filename'] = args.png_data_dir + '/' + img + '.png'

        im = Image.open(args.png_data_dir + '/' + img + '.png')

        tmpdict['width'], tmpdict['height'] = im.size

        img_annotations = train_data.loc[train_data['patientId'] == img]

        object_list = []

        for row in range(img_annotations.shape[0]):
            the_row = img_annotations.iloc[row]
            smallerdict = {}
            smallerdict['name'] = 'Opacity'
            smallerdict['xmax'] = int(float(the_row['x']) + float(the_row['width']))
            smallerdict['xmin'] = int(float(the_row['x']))
            smallerdict['ymax'] = int(float(the_row['y']) + float(the_row['height']))
            smallerdict['ymin'] = int(float(the_row['y']))

            object_list.append(smallerdict)

        tmpdict['object'] = object_list

        train_imgs.append(tmpdict)


    for image in train_imgs:
        print(image['filename'])
        for obj in image['object']:
            relative_w = (float(obj['xmax']) - float(obj['xmin']))/image['width']
            relatice_h = (float(obj["ymax"]) - float(obj['ymin']))/image['height']
            annotation_dims.append(tuple(map(float, (relative_w,relatice_h))))

    annotation_dims = np.array(annotation_dims)
    centroids = run_kmeans(annotation_dims, num_anchors)

    # print out anchors
    print('\naverage IOU for', num_anchors, 'anchors:', '%0.2f' % avg_IOU(annotation_dims, centroids))
    print_anchors(centroids)


if __name__ == '__main__':
    args = parse_args()
    _main_(args)
