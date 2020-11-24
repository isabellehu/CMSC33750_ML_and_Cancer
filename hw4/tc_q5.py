#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 14:52:48 2020

@author: qhu
"""
import pandas as pd 
import numpy as np

from keras.layers import  Dense, Dropout, Activation, Conv1D, MaxPooling1D, Flatten
from keras.optimizers import SGD
from keras.models import Sequential, model_from_json
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf


# define constants
FIT = True
EPOCH = 400
BATCH = 20
P     = 60483   # 60483
DR    = 0.1     # Dropout rate


# loaded saved dataset
df_nt_matched = pd.read_csv("q4_nt_matched.csv")
# put features in an array
ar_nt = df_nt_matched.iloc[:,1:].values.astype('float32')
# log transform NT data to match the scale of TC data, replacing -Inf with 0
ar_nt = np.where(np.log(ar_nt)==-np.inf, 0, np.log(ar_nt))
# normalize NT data the same way as the TC data when training the type classifier
ar_nt = MaxAbsScaler().fit_transform(ar_nt)

# t-SNE clustering on features
X_embedded = TSNE(n_components=2).fit_transform(ar_nt)
plt.figure(figsize=(10,10))
sns.scatterplot(
    x=X_embedded[:,0], y=X_embedded[:,1],
    hue=df_nt_matched['Type'],
    palette=sns.color_palette("hls", 18),
    legend="full")

# assign clusters on the embedded data
kmeans = KMeans(n_clusters=df_nt_matched['Type'].nunique()).fit(X_embedded)
cluster_label = kmeans.labels_

# label the kmeans clusters on the t-SNE scatter plot
for label in cluster_label:
    plt.annotate(label, X_embedded[cluster_label==label].mean(axis=0),
                 horizontalalignment='center', verticalalignment='center',
                 size=20, weight='bold') 

# reshape the feature matrix for CNN
ar_nt = np.expand_dims(ar_nt, axis=2)

# one-hot encode labels
enc = OneHotEncoder()
Y_nt = enc.fit_transform(np.expand_dims(cluster_label,axis=1)).toarray()

# split this new dataset into 80% training 20% testing, stratify based on class label
X_train, X_test, Y_train, Y_test = train_test_split(ar_nt, Y_nt, test_size=0.2, random_state=8, shuffle=True, stratify=cluster_label)

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
    model.add(Dense(Y_nt.shape[1]))
    model.add(Activation('softmax'))
    
    model.summary()
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(),
                  metrics=['accuracy'])
    
    # set up a bunch of callbacks to do work during model training..
    checkpointer = ModelCheckpoint(filepath='tc1.autosave.model_5.h5', verbose=0, save_weights_only=False, save_best_only=True)
    csv_logger = CSVLogger('tc1.training_5.log')
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
    with open("tc1.model_5.json", "w") as json_file:
            json_file.write(model_json)
    
    # serialize weights to HDF5
    model.save_weights("tc1.model_5.h5")
    print("Saved model to disk")


# evaluation
METRICS = [
  tf.keras.metrics.CategoricalAccuracy(name='acc'),
  tf.keras.metrics.Precision(name='precision'),
  tf.keras.metrics.Recall(name='recall'),
  tf.keras.metrics.AUC(name='auc'),
]

# load json and create model
json_file = open('tc1.model_5.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model_json = model_from_json(loaded_model_json)

# load weights into new model
loaded_model_json.load_weights("tc1.model_5.h5")
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
    plt.xticks(tick_marks, target_names, rotation=90)
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
    plt.savefig('cm_q5.png')

plot_confusion_matrix(cm, enc.categories_[0])

