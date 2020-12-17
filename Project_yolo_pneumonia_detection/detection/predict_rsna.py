import tensorflow as tf
import keras
import os
from keras.models import load_model

import pandas as pd
import pickle
from PIL import Image
from tqdm import tqdm
import numpy as np

import cv2
from utils.utils import get_yolo_boxes, makedirs
from utils.bbox import draw_boxes
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--split', type=int, default=4, help='training split number (1-5)')
    parser.add_argument('-a', '--anchors', default=[58,46, 60,74, 71,110, 84,157, 91,77, 95,219, 106,118, 115,177, 125,257])
    parser.add_argument('--png_data_dir', default='../data/stage_2_train_images_png')
    parser.add_argument('--test_file', default='../data/splits//s1_test.csv')
    parser.add_argument('--weights_file', default='./weights/yolo_split%d.h5')
    parser.add_argument('--results_dir', default='./results')
    return parser.parse_args()

args = parse_args()

SPLIT_NUM = int(args.split)
MODEL_WEIGHT_FILE = args.weights_file % SPLIT_NUM
TEST_IMG_DIR = args.png_data_dir
RESULTS_DIR = args.results_dir

ANCHORS = args.anchors


def generate_predictionstring(image_path, obj_threshold=0.2, nms_threshold=0.45, flip = False):

    image = cv2.imread(image_path)

    if flip:
        image = np.fliplr(image).copy()

    height_, width_, _ = image.shape

    height_ = float(height_)
    width_ = float(width_)

    boxes = get_yolo_boxes(infer_model,
                           [image],
                           608,
                           608,
                           ANCHORS,
                           obj_threshold,
                           nms_threshold)[0]


    # remove the boxes which are less likely than a obj_threshold
    boxes = [box for box in boxes if box.get_score() > obj_threshold]

    xmin = [x.xmin for x in boxes]
    xmax = [x.xmax for x in boxes]
    ymin = [x.ymin for x in boxes]
    ymax = [x.ymax for x in boxes]

    widths = [x_max - x_min for x_max, x_min in zip(xmax, xmin)]
    heights = [y_max - y_min for y_max, y_min in zip(ymax, ymin)]


    scores_ = [x.get_score() for x in boxes]

    submission_string = ""


    for i in range(len(xmin)):

        if (i != 0):
            submission_string += " "

        submission_string += str(scores_[i]) + " " + str(round(xmin[i])) + " " + \
          str(round(ymin[i])) + " " + str(round(widths[i])) + " " + str(round(heights[i], 5))



    return submission_string




print("Loading model...")
infer_model = load_model(MODEL_WEIGHT_FILE)
print("Done!")

print("Generating predictions...")
print("Generating predictions on untampered test images...")
# test_filenames = os.listdir(TEST_IMG_DIR)
# test_filenames = [x.split('.')[0] for x in test_filenames]
test_filenames = pd.read_csv(args.test_file).drop_duplicates(subset=['patientId'])['patientId']
submission = pd.DataFrame({'patientId': test_filenames})

pred_strings = []

for test_img_i in tqdm(range(submission.shape[0])):

    imgpath = TEST_IMG_DIR + '/' + submission.iloc[test_img_i]['patientId'] + '.png'
    pred_strings.append(generate_predictionstring(imgpath, 0.2, 0.45))

submission['PredictionString'] = pred_strings

submission.to_csv(os.path.join(RESULTS_DIR, 'yolo_split%d_prediction_on_s1test.csv' % SPLIT_NUM), index = False)

# print("Done!")
print("Generating predictions on horizontally flipped test images...")

pred_strings = []

for test_img_i in tqdm(range(submission.shape[0])):

    imgpath = TEST_IMG_DIR + '/' + submission.iloc[test_img_i]['patientId'] + '.png'
    pred_strings.append(generate_predictionstring(imgpath, 0.2, 0.45, True))

submission['PredictionString'] = pred_strings

submission.to_csv(os.path.join(RESULTS_DIR, 'yolo_split%d_prediction_on_s1test_flip.csv' % SPLIT_NUM), index = False)
print("Done!")
