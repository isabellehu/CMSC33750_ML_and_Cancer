# -*- coding: utf-8 -*-
from keras.applications.densenet import DenseNet121, preprocess_input
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import (
    Input, Dense, Dropout
)
from keras.layers import LeakyReLU
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from scipy.ndimage import rotate, zoom
from skimage import exposure
from skimage.transform import resize
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import cv2
import keras
import numpy as np
import os
import pandas as pd
import json
import argparse
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--split', type=int, default=1, help='training split number (1-5)')
    parser.add_argument('--png_data_dir', default='../data/stage_2_train_images_png')
    parser.add_argument('--train_file', default='../data/splits/s1_split%d_train_data_5cv.csv')
    parser.add_argument('--val_file', default='../data/splits/s1_split%d_valid_data_5cv.csv')
    parser.add_argument('--test_file', default='../data/splits//s1_test.csv') # ../data/splits//s1_test.csv
    parser.add_argument('--weights_file', default='./weights/densenet_s1_split%d.h5')
    return parser.parse_args()


def roc_auc(y_true, y_pred):
    # tf = keras.backend.tf
    value, update_op = tf.metrics.auc(y_true, y_pred)

    metric_vars = [i for i in tf.local_variables() if 'roc_auc' in i.name.split('/')[1]]

    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value


def create_model(input_shape):
    '''
    Define network
    '''
    inputs = Input(shape=input_shape)

    chexNet = DenseNet121(
          include_top=True
        , input_tensor=inputs
        , weights="ChexNet_weight.h5"
        , classes=14
    )

    chexNet = Model(
          inputs=inputs
        , outputs=chexNet.layers[-2].output
        , name="ChexNet"
    )

    model = Sequential()

    model.add(chexNet)

    model.add(Dropout(0.5, name="drop_0"))
    model.add(Dense(512, activation=None, name="dense_0"))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.5, name="drop_1"))
    model.add(Dense(1, activation="sigmoid", name="out"))

    model.summary()

    return model


def main(args):
    '''
    Main Method
    '''
    # params
    
    IMG_DIM = [256, 256, 3]
    BATCH_SIZE = 32

    SPLIT_NUM = args.split
    TRAIN_FILE_PATH = args.train_file % SPLIT_NUM
    VALID_FILE_PATH = args.val_file % SPLIT_NUM
    TEST_FILE_PATH = args.test_file
    IMG_DIR = args.png_data_dir
    WEIGHT_FILE_PATH = args.weights_file % SPLIT_NUM
    
    df_train = pd.read_csv(TRAIN_FILE_PATH).drop_duplicates(subset=['patientId'])
    df_train['imageName'] = df_train['patientId']+'.png'
    df_valid = pd.read_csv(VALID_FILE_PATH).drop_duplicates(subset=['patientId'])
    df_valid['imageName'] = df_valid['patientId']+'.png'


    flow_args = dict(
        directory = IMG_DIR
        , x_col = "imageName"
        , y_col = "Target"
        , class_mode="raw"
        , batch_size=BATCH_SIZE
        , color_mode='rgb'
        , shuffle=True
        , target_size=IMG_DIM[:-1]
    )

    train_gen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   horizontal_flip=True,
                                   rotation_range=10,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1).flow_from_dataframe(df_train, **flow_args)
    valid_gen = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_dataframe(df_valid, **flow_args)
    
    ###### Training ######
    model = create_model(IMG_DIM)

    checkpoint = ModelCheckpoint(
          WEIGHT_FILE_PATH
        , monitor="val_loss"
        , verbose=1
        , save_best_only=True
        , mode="min"
        , save_weights_only=True
    )

    reduceLROnPlat = ReduceLROnPlateau(
          monitor="val_loss"
        , factor=0.1
        , patience=5
        , verbose=1
        , mode="auto"
        , min_delta=0.0001
        , cooldown=0
        , min_lr=0.0001
    )

    callbacks_list = [checkpoint, reduceLROnPlat]

    model.compile(
          optimizer=Adam(lr=1e-4)
        , loss="binary_crossentropy"
        , metrics=[ "acc", roc_auc ]
    )
    
    # compute class weight
    class_weight = dict(enumerate(compute_class_weight('balanced', df_train['Target'].unique(), df_train['Target'])))

    model.fit_generator(
          train_gen
        , validation_data=valid_gen
        , epochs=16
        , callbacks=callbacks_list
        , steps_per_epoch=len(df_train) // BATCH_SIZE + 1
        , validation_steps=len(df_valid) // BATCH_SIZE + 1
        , class_weight=class_weight
    )

    ###### Evaluation ######
    assert WEIGHT_FILE_PATH
    model.load_weights(WEIGHT_FILE_PATH)
    
    df_test = pd.read_csv(TEST_FILE_PATH).drop_duplicates(subset=['patientId'])
    df_test['imageName'] = df_test['patientId']+'.png'
    
    flow_args = dict(
        directory = IMG_DIR #'../data/stage_2_test_images_png' 
        , x_col = "imageName"
        , y_col = "Target" #"imageName"
        , class_mode="raw"
        , batch_size=1
        , color_mode='rgb'
        , shuffle=False
        , target_size=IMG_DIM[:-1]
    )

    test_gen = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_dataframe(df_test, **flow_args)
    
    df_test["score_"+str(SPLIT_NUM)] = model.predict_generator(test_gen, steps=len(df_test))
    df_test.to_csv('./results/densenet_prediction_on_s1test_split%d.csv' % SPLIT_NUM, index=False)

    print("AUROC:", roc_auc_score(df_test["Target"], df_test["score_"+str(SPLIT_NUM)]))


if __name__ == '__main__':
    args = parse_args()
    main(args)
