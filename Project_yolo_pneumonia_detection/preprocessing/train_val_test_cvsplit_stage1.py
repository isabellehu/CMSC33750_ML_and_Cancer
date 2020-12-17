#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 14:21:53 2020

@author: qhu
"""

import pandas as pd
import numpy as np
import os
import pydicom
import json
import argparse
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.model_selection import KFold


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_dir', default='../data/stage_2_train_images')
    parser.add_argument('--stage2_info', default='../data/stage_2_detailed_class_info.csv')
    parser.add_argument('--stage2_label', default='../data/stage_2_train_labels.csv')
    parser.add_argument('--stage1_info', default='../data/stage_1_detailed_class_info.csv')
    parser.add_argument('--splits_dir', default='../data/splits')
    return parser.parse_args()


def image_meta_extractor(img_id, raw_data_dir):

    img_dat = pydicom.read_file(os.path.join(raw_data_dir, img_id + '.dcm'))

    PatientAge = int(img_dat.PatientAge)
    PatientSex = img_dat.PatientSex
    ViewPosition = img_dat.ViewPosition

    return(pd.Series([PatientAge, PatientSex, ViewPosition],
                      index = ['PatientAge', 'PatientSex', 'ViewPosition']))

    
def main(args):
    # load stage2 training set labels and info
    stage2_train_detailed_class_info = pd.read_csv(args.stage2_info)
    stage2_train_labels = pd.read_csv(args.stage2_label)
    
    # merges the label file and the detailed class info file
    stage2_train_combined = pd.concat([stage2_train_labels, stage2_train_detailed_class_info.drop('patientId', axis = 1)], axis = 1)
    
    # get image list and obtain info from metadata
    stage2_train_images = [f for f in os.listdir(args.raw_data_dir) if f.endswith('.dcm')]
    stage2_train_images = [f.split('.')[0] for f in stage2_train_images]

    stage2_train_images_metadata = pd.DataFrame({'patientId' : stage2_train_images})

    stage2_train_images_metadata = pd.concat([stage2_train_images_metadata['patientId'],
                                              stage2_train_images_metadata['patientId'].apply(image_meta_extractor, args=(args.raw_data_dir,))], 
                                             axis = 1)

    # merge metadata into dataset
    stage2_train_full = pd.merge(stage2_train_combined, stage2_train_images_metadata, how = "inner", on = 'patientId')
    
    # separate the stage2 train set with full info into stage1 train&test set and save them
    stage1_train_patientId = pd.read_csv(args.stage1_info)['patientId']
    stage1_train_full = stage2_train_full[stage2_train_full['patientId'].isin(stage1_train_patientId) == True]
    stage1_test_full = stage2_train_full[stage2_train_full['patientId'].isin(stage1_train_patientId) == False]
    
    # save development set and test set
    stage1_train_full.to_csv(os.path.join(args.splits_dir, 's1_trainval.csv'), index = False)
    stage1_test_full.to_csv(os.path.join(args.splits_dir, 's1_test.csv'), index = False)
        
    # split development set for 5-fold cross validation
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=8)
    stage1_train_uniqueId = stage1_train_full.drop_duplicates(['patientId'])[['patientId', 'Target']]
    patientids_ = stage1_train_uniqueId['patientId'].values
    targets_ = stage1_train_uniqueId['Target'].values

    i=1
    for train_index, test_index in kf.split(patientids_, targets_):
        train_users = patientids_[train_index]
        valid_users = patientids_[test_index]

        assert(np.intersect1d(train_users, valid_users).shape[0] == 0)

        df_train = stage1_train_full[stage1_train_full['patientId'].isin(train_users)]
        df_val = stage1_train_full[stage1_train_full['patientId'].isin(valid_users)]

        # save the splits
        df_train.to_csv(os.path.join(args.splits_dir, "s1_split%d_train_data_5cv.csv" % i), index = False)
        df_val.to_csv(os.path.join(args.splits_dir, "s1_split%d_valid_data_5cv.csv" % i), index = False)
        
        i+=1
    
if __name__ == "__main__":
    args = parse_args()
    main(args)
