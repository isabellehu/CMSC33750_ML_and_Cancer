#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 23:26:05 2020

@author: qhu
"""
import pandas as pd
import numpy as np
import os
import json
import argparse
from utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detection_result', default='./detection/results/yolo_split%d_prediction_on_s1test.csv')
    parser.add_argument('--detection_result_flip', default='./detection/results/yolo_split%d_prediction_on_s1test_flip.csv')
    parser.add_argument('--classification_result', default='./classification/results/densenet_prediction_on_s1test_split%d.csv')
    parser.add_argument('--save_path', default='./final_prediction_s1_5cv.csv')
    return parser.parse_args()

def load_yolo_pred(file_path, file_path_flip):
    '''
    load and merge yolo predictions on from 5cv
    '''
    # load unflipped and flipped yolo predictions
    print("Loading and processing YOLO predictions..." )

    yolo = pd.read_csv(file_path)
    yolo_flip = pd.read_csv(file_path_flip)
    
    # apply cutoff threshold to filter predicted bounding boxes
    yolo['PredictionString'] = yolo['PredictionString'].apply(lambda x: cutoffer(x, 0.8))
    yolo['PredictionString'] = yolo['PredictionString'].apply(lambda x: truncate_predstring_based_on_nms(x, nms(predstring_to_list(x), 0.1)))
    yolo_flip['PredictionString'] = yolo_flip['PredictionString'].apply(lambda x: cutoffer(x, 0.8))
    yolo_flip['PredictionString'] = yolo_flip['PredictionString'].apply(lambda x: truncate_predstring_based_on_nms(x, nms(predstring_to_list(x), 0.1)))
    
    # Reflip flipped boxes
    yolo_flip['PredictionString'] = yolo_flip["PredictionString"].apply(bounding_box_lr_flip)
    
    # Merge predictions on flipped and unflipped
    yolo = yolo.merge(yolo_flip, on = "patientId")
    
    yolo.loc[pd.isnull(yolo['PredictionString_x']), 'PredictionString_x'] = ""
    yolo.loc[pd.isnull(yolo['PredictionString_y']), 'PredictionString_y'] = ""
    
    # merge two columns
    yolo['PredictionString'] = yolo['PredictionString_x'] + " " + yolo['PredictionString_y']
    yolo = yolo[['patientId', 'PredictionString']]
    
    # combine predicted bounding boxes if they intersect
    yolo['PredictionString'] = yolo['PredictionString'].apply(lambda x: get_predstring_from_box_list(combine_boxes_intersect(get_box_list_from_predstring(x))))
    yolo.loc[pd.isnull(yolo['PredictionString']), 'PredictionString'] = ""
    
    return yolo
    
def main(args):
    
    # load and merge yolo detection bounding box predictions
    DETECT_PATH = args.detection_result
    DETECT_PATH_FLIP = args.detection_result_flip
    
    for i in range(5):
        yolo = load_yolo_pred(DETECT_PATH % (i+1), DETECT_PATH_FLIP % (i+1))
        if i == 0:
            yolo_preds = yolo
        else:
            yolo_preds = yolo_preds.merge(yolo, on="patientId")
            yolo_preds["PredictionString"] = yolo_preds['PredictionString_x'] + " " + yolo_preds["PredictionString_y"]
            yolo_preds["PredictionString"] = yolo_preds['PredictionString'].apply(lambda x: x.strip())
            yolo_preds = yolo_preds[['patientId', 'PredictionString']]
    
    # sort predicted boxes by confidence score
    yolo_preds['PredictionString'] = yolo_preds['PredictionString'].apply(sort_predstring_by_confidence)
    
    # combine predicted bounding boxes if they intersect
    yolo_preds['PredictionString'] = yolo_preds['PredictionString'].apply(lambda x: get_predstring_from_box_list(combine_boxes_intersect(count_boxes_intersect(get_box_list_from_predstring(x)))))
    yolo_preds.loc[pd.isnull(yolo_preds['PredictionString']), 'PredictionString'] = ""

    # load and merge classifier predictions
    print("Loading and processing classification results...")
        
    CLASS_PATH = args.classification_result
    
    for i in range(5):
        classif = pd.read_csv(CLASS_PATH % (i+1))
        if i == 0:
            class_preds = classif[['patientId','Target','score_1']]
        else:
            class_preds = class_preds.merge(classif[['patientId','score_'+str(i+1)]], on="patientId")
    
    # average classification scores from 5cv
    class_preds['avgclassifierScore'] = class_preds.iloc[:, 2:].mean(axis = 1)
    # print("AUROC:", roc_auc_score(class_preds["Target"], class_preds["avgclassifierScore"]))

    
    # filter yolo bounding boxes using classifier prediction score
    print("Combining YOLO and classification results...")
    
    yolo_preds = yolo_preds.merge(class_preds[['patientId', 'avgclassifierScore']], on = "patientId")
    yolo_preds.loc[yolo_preds['avgclassifierScore'] < 0.2, 'PredictionString'] = np.nan
    yolo_preds = yolo_preds[['patientId', 'PredictionString']]
    
    yolo_preds.to_csv(args.save_path, index = False)
    
    print("Done")


if __name__ == '__main__':
    args = parse_args()
    main(args)