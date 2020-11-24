#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 16:39:47 2020

@author: qhu
"""
import pandas as pd 
import numpy as np

from keras.layers import Dense, Dropout, Activation, Conv1D, MaxPooling1D, Flatten
from keras.optimizers import SGD
from keras.models import Sequential, model_from_json
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MaxAbsScaler
import matplotlib.pyplot as plt

import tensorflow as tf


# define constants
FIT = False
EPOCH = 400
BATCH = 20
CLASSES = 37 # 0=normal, 1-36=cancer types
P     = 60483   # 60483
DR    = 0.1     # Dropout rate


# load the train/test sets for the normal/tumor dataset
df_nt_train = pd.read_csv('../normal-tumor/nt_train2.csv', header=None)
df_nt_test = pd.read_csv('../normal-tumor/nt_test2.csv', header=None)
df_nt = pd.concat([df_nt_train, df_nt_test])

# put features in an array
ar_nt = df_nt.iloc[:,1:].values.astype('float32')
# log transform NT data to match the scale of TC data, replacing -Inf with 0
ar_nt = np.where(np.log(ar_nt)==-np.inf, 0, np.log(ar_nt))
# normalize NT data the same way as the TC data when training the type classifier
ar_nt = MaxAbsScaler().fit_transform(ar_nt)
# reshape the feature matrix for CNN
ar_nt = np.expand_dims(ar_nt, axis=2)

# load predicted cancer type label for tumor subset from Q1
pred_label = np.squeeze(pd.read_csv('pred_label_tumor.csv', header=None).values.astype('int'))

# assign predicted cancer type to tumors, keep the normal type as 0
label = df_nt.iloc[:,0].copy().values
label[label==1]=pred_label

# # split this new dataset into 80% training 20% testing randomly
# X_train_add, X_test_add, y_train_add, y_test_add = train_test_split(ar_tumor, pred_label, test_size=0.2, random_state=8, shuffle=True)

# Actually, use the tumor subset in the training set for the nt model for training, 
# and the tumor subset in the test set for nt model for testing
train_add_len = len(df_nt_train)
X_train_add = ar_nt[:train_add_len]
X_test_add = ar_nt[train_add_len:]
y_train_add = label[:train_add_len]
y_test_add = label[train_add_len:]


# load the previous cancer type training and testing datasets
train_path = 'type_18_300_train.csv'
test_path = 'type_18_300_test.csv'

df_train_org = pd.read_csv(train_path, header=None)
df_test_org = pd.read_csv(test_path, header=None)

X_train_org = df_train_org.values[:, 1:].astype('float32')
X_test_org = df_test_org.values[:, 1:].astype('float32')

# normalize
mat = MaxAbsScaler().fit_transform(np.concatenate((X_train_org, X_test_org), axis=0))
X_train_org = mat[:X_train_org.shape[0],:]
X_test_org = mat[X_train_org.shape[0]:,:]

# reshape features for the Conv1D to work
X_train_org = np.expand_dims(X_train_org, axis=2)
X_test_org = np.expand_dims(X_test_org, axis=2)

# concatenate relabeled tumor dataset with cancer type dataset
X_train = np.concatenate([X_train_org, X_train_add], axis=0)
X_test = np.concatenate([X_test_org, X_test_add], axis=0)
y_train = np.concatenate([df_train_org.values[:,0].astype('int'), y_train_add], axis=0)
y_test = np.concatenate([df_test_org.values[:,0].astype('int'), y_test_add], axis=0)

# save the new training and test sets for future reference
np.savetxt("q2_train.csv", np.hstack((np.expand_dims(y_train, axis=1), np.squeeze(X_train))), delimiter=",")
np.savetxt("q2_test.csv", np.hstack((np.expand_dims(y_test, axis=1), np.squeeze(X_test))), delimiter=",")

# convert label into one-hot encoding
Y_train = np_utils.to_categorical(y_train,CLASSES)
Y_test = np_utils.to_categorical(y_test,CLASSES)

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('Y_train shape:', Y_train.shape)
print('Y_test shape:', Y_test.shape)


# create, train, and save model
if FIT:
    model = Sequential()
    model.add(Conv1D(filters=128, kernel_size=20, strides=1, padding='valid', input_shape=(P, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Conv1D(filters=128, kernel_size=10, strides=1, padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=10))
    model.add(Flatten())
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(20))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(CLASSES))
    model.add(Activation('softmax'))
    
    model.summary()
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(),
                  metrics=['accuracy'])
    
    # set up a bunch of callbacks to do work during model training..
    checkpointer = ModelCheckpoint(filepath='tc1.autosave.model_2.h5', verbose=0, save_weights_only=False, save_best_only=True)
    csv_logger = CSVLogger('tc1.training_2.log')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
    early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1, restore_best_weights=True)

    history = model.fit(X_train, Y_train,
                        batch_size=BATCH, 
                        epochs=EPOCH,
                        verbose=1, 
                        validation_data=(X_test, Y_test),
                        callbacks = [checkpointer, csv_logger, reduce_lr, early_stopping])
    
    score = model.evaluate(X_test, Y_test, verbose=0)
    
    # serialize model to JSON
    model_json = model.to_json()
    with open("tc1.model_2.json", "w") as json_file:
            json_file.write(model_json)
    
    # serialize weights to HDF5
    model.save_weights("tc1.model_2.h5")
    print("Saved model to disk")


# evaluation
METRICS = [
  tf.keras.metrics.CategoricalAccuracy(name='acc'),
  tf.keras.metrics.Precision(name='precision'),
  tf.keras.metrics.Recall(name='recall'),
  tf.keras.metrics.AUC(name='auc'),
]

# load json and create model
json_file = open('tc1.model_2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model_json = model_from_json(loaded_model_json)

# load weights into new model
loaded_model_json.load_weights("tc1.model_2.h5")
print("Loaded json model from disk")

# evaluate json loaded model on test data
loaded_model_json.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=METRICS)
score_json = loaded_model_json.evaluate(X_test, Y_test, verbose=0)

print('Test score:', score_json[0])
print('Test categorical accuracy:', score_json[1])
print('Test percission:', score_json[2])
print('Test recall:', score_json[3])
print('Test AUC:', score_json[4])
# print("json %s: %.2f%%" % (loaded_model_json.metrics_names[1], score_json[1]*100))

# confusion matrix
Y_pred = loaded_model_json.predict(X_test)
cm = confusion_matrix(Y_test.argmax(axis=1), Y_pred.argmax(axis=1))
print("Confusion matrix:")
print(cm)

# plot confusion matrix for better format
def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(9,7))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()

    width, height = cm.shape

    for x in range(width):
        for y in range(height):
            plt.annotate(str(cm[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('cm_q2.png')

plot_confusion_matrix(cm, np.unique(Y_test.argmax(axis=1)))


