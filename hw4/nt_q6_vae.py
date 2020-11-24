#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 13:52:29 2020

@author: qhu
"""
import pandas as pd 
import numpy as np

import tensorflow as tf

from keras.layers import Input, Dense, Conv1D, Flatten, Reshape, Lambda, Conv2DTranspose, UpSampling1D
from keras.optimizers import Adam
from keras.models import Model, model_from_json
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
import keras.backend as K

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MaxAbsScaler
import matplotlib.pyplot as plt


# define constants
FIT = False
EPOCH = 1000
BATCH = 20
CLASSES = 2

P     = 60483   # 60483
DR    = 0.1     # Dropout rate


# load the train/test sets for the normal/tumor dataset
df_nt_train = pd.read_csv('nt_train2.csv', header=None)
df_nt_test = pd.read_csv('nt_test2.csv', header=None)
df_nt = pd.concat([df_nt_train, df_nt_test])

# select the tumor subset
ar_nt = df_nt.iloc[:,1:].values.astype('float32')
# normalize it the same way as when training the cancer type classifier
ar_nt = MaxAbsScaler().fit_transform(ar_nt)
# reshape the feature matrix for CNN
ar_nt = np.expand_dims(ar_nt, axis=2)

## VAE
def create_vae(input_shape, n_hidden1, n_hidden2, n_hidden3, # latent dim
               n_hidden4, n_hidden5, is_vae=True, binary_input=False):
    # encoder:
    input_encoder = Input(shape=input_shape)
    x = Flatten()(input_encoder) if len(input_shape) > 1 else input_encoder
    x = Dense(n_hidden1, activation='elu')(x)
    x = Dense(n_hidden2, activation='elu')(x)

    def sampling(args):
        z_mu, z_gamma = args
        epsilon = K.random_normal(shape=K.shape(z_mu))
        return z_mu + K.exp(z_gamma/2) * epsilon
    
    if is_vae:
        z_mu = Dense(n_hidden3, activation=None)(x)
        # using gamma=ln(sigma^2) instead of sigma
        z_gamma = Dense(n_hidden3, activation=None)(x)  
        z = Lambda(sampling, output_shape=(n_hidden3,))([z_mu, z_gamma])
    else:
        z = Dense(n_hidden3, activation='elu')(x)
        
    # decoder
    input_decoder = Input(shape=(n_hidden3,))
    x = Dense(n_hidden4, activation='elu')(input_decoder)
    x = Dense(n_hidden5, activation='elu')(x)
    outputs_flattened = Dense(np.prod(input_shape), 
                              activation='sigmoid' if binary_input else None)(x)
    outputs = Reshape(input_shape)(outputs_flattened)

    # models
    m_encoder = Model(input_encoder, z, name='encoder') 
    m_decoder = Model(input_decoder, outputs, name='decoder')
    m_ae = Model(input_encoder, m_decoder(m_encoder(input_encoder)), name='ae')
    
    # losses:
    def reconstruct_loss(x_original, x_restored):
        # sum squared errors over all axes but the first (minibatch) one
        return 0.5 * K.sum(K.square(x_original- x_restored), 
                           axis=np.arange(1, len(K.int_shape(x_original))))
    
    def reconstruct_loss_binary(x_original, x_restored):
        # sum binary crossentropies over all axes but the first (minibatch) one
        return K.sum(K.binary_crossentropy(x_original, x_restored), 
                     axis=np.arange(1, len(K.int_shape(x_original))))
    
    if binary_input:
        reconstruct_loss = reconstruct_loss_binary
    
    def latent_loss(x_original, x_restored):
        return 0.5 * K.sum(K.square(z_mu) + K.exp(z_gamma) - z_gamma - 1, axis=-1)
    
    def vae_loss(x_original, x_restored):
        return reconstruct_loss(x_original, x_restored) + latent_loss(x_original, x_restored)
    
    if not is_vae:
        latent_loss = None
        vae_loss = None

    return m_encoder, m_decoder, m_ae, reconstruct_loss, latent_loss, vae_loss

# VAE parameters
input_shape = ar_nt.shape[1:]
n_hidden1 = 128
n_hidden2 = 256
n_hidden3 = 2  # latent dimension
n_hidden4 = n_hidden2
n_hidden5 = n_hidden1

# create VAE
vae_encoder, vae_decoder, vae, reconstruct_loss, latent_loss, vae_loss = \
    create_vae(input_shape, n_hidden1, n_hidden2, n_hidden3, n_hidden4, n_hidden5, is_vae=True, binary_input=True)
vae.summary()
vae_encoder.summary()
vae_decoder.summary()

# training
checkpointer = ModelCheckpoint(filepath='vae.autosave.model.h5', monitor='loss', verbose=0, save_weights_only=False, save_best_only=True)
csv_logger = CSVLogger('vae_training.log')
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
early_stopping = EarlyStopping(monitor='loss', patience=30, verbose=1, restore_best_weights=True)

vae.compile(optimizer=Adam(0.005), loss=vae_loss, metrics=[reconstruct_loss, latent_loss])
vae.fit(ar_nt, ar_nt, epochs=EPOCH, batch_size=64, callbacks = [checkpointer, csv_logger, reduce_lr, early_stopping])

# get the VAE reconstruction of the original dataset
# this is the generated dataset that we will use to train and evaluate the normal/tumor model
X_test = vae.predict(ar_nt)
y_test = df_nt.iloc[:,0].values
Y_test = np_utils.to_categorical(y_test, CLASSES)

# save the generated dataset
df_nt_recon = df_nt.copy()
df_nt_recon.iloc[:,1:] = np.squeeze(X_test)
df_nt_recon.to_csv("q6_nt_generated.csv", header=False, index=False)


# evaluation metrics
METRICS = [
  tf.keras.metrics.CategoricalAccuracy(name='acc'),
  tf.keras.metrics.Precision(name='precision'),
  tf.keras.metrics.Recall(name='recall'),
  tf.keras.metrics.AUC(name='auc'),
]

# load json and create model
json_file = open('nt3.model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model_json = model_from_json(loaded_model_json)

# load weights into new model
loaded_model_json.load_weights("nt3.model.h5")
print("Loaded json model from disk")

# evaluate json loaded model on test data
loaded_model_json.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=METRICS)
score = loaded_model_json.evaluate(X_test, Y_test, verbose=0)

print('Test score:', score[0])
print('Test categorical accuracy:', score[1])
print('Test percission:', score[2])
print('Test recall:', score[3])
print('Test AUC:', score[4])
# print("json %s: %.2f%%" % (loaded_model_json.metrics_names[1], score_json[1]*100))

# confusion matrix
y_pred = loaded_model_json.predict(X_test)
cm = confusion_matrix(y_test, y_pred.argmax(axis=1))
print("Confusion matrix:")
print(cm)

# plot confusion matrix for better format
def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(6,5))
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
    plt.savefig('cm_q6.png')

plot_confusion_matrix(cm, ["normal","tumor"])
