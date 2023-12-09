#!/usr/bin/env python
# coding: utf-8

# # Model utilities
# It includes self defined functions for used models

# ### Imports

# In[ ]:


import numpy as np
from time import time
import random

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input, Add, BatchNormalization, ReLU, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler

# for reproduciblility, fix the randomly generated numbers
SEED = 100
tf.keras.utils.set_random_seed(SEED)


# ## Baseline model
# AdaBoost with default hyper-parameters
# 
# > **Parameters**
# > - X_train (numpy array): input features to train the model
# > - y_train (numpy array): output labels for supervised learning of the model
# > - X_test (numpy array): input features to test the model
# > - y_test: dummy, not used, default=None
# 
# > **Returns**
# > - results (dict): dictionary including:
# >    - pred_train (numpy array): predictions for training dataset
# >    - pred_test (numpy array): predictions for test dataset
# >    - model (object): trained model

# In[ ]:


def model_baseline(X_train, y_train, X_test, y_test=None):
    
    '''
    Model
    '''
    model = AdaBoostRegressor(random_state = SEED)
    
    '''
    Training
    '''
    time_start = time()
    model.fit(X_train, y_train)
    time_end = time()
    print(f"Time for training: {time_end - time_start}")
    
    '''
    Testing
    '''
    results = {}
    results['pred_train'] = model.predict(X_train)
    results['pred_test']  = model.predict(X_test)
    results['model']      = model
    
    return results


# ## Optimized AdaBoost
# AdaBoost regressor with optimized hyper-parameters for its top mutation matrix GIAG010101
# 
# > **Parameters**
# > - X_train (numpy array): input features to train the model
# > - y_train (numpy array): output labels for supervised learning of the model
# > - X_test (numpy array): input features to test the model
# > - y_test: dummy, not used, default=None
# 
# > **Returns**
# > - results (dict): dictionary including:
# >    - pred_train (numpy array): predictions for training dataset
# >    - pred_test (numpy array): predictions for test dataset
# >    - model (object): trained model

# In[ ]:


def model_AdaBoost(X_train, y_train, X_test, y_test=None):
    
    '''
    Model
    '''
    model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=1860, max_features=0.393686389369039),
                              n_estimators=230,
                              learning_rate=1.39248292746222,
                              random_state=SEED)
    
    '''
    Training
    '''
    time_start = time()
    model.fit(X_train, y_train)
    time_end = time()
    print(f"Time for training: {time_end - time_start}")
    
    '''
    Testing
    '''
    results = {}
    results['pred_train'] = model.predict(X_train)
    results['pred_test']  = model.predict(X_test)
    results['model']      = model
    
    return results


# ## Optimized AdaBoost for binary
# AdaBoost regressor with optimized hyper-parameters for binary encoding. This is used for NextFlu matched parameters simulation.
# 
# > **Parameters**
# > - X_train (numpy array): input features to train the model
# > - y_train (numpy array): output labels for supervised learning of the model
# > - X_test (numpy array): input features to test the model
# > - y_test: dummy, not used, default=None
# 
# > **Returns**
# > - results (dict): dictionary including:
# >    - pred_train (numpy array): predictions for training dataset
# >    - pred_test (numpy array): predictions for test dataset
# >    - model (object): trained model

# In[ ]:


def model_AdaBoost_binary(X_train, y_train, X_test, y_test=None):
    
    '''
    Model
    '''
    model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=7040, max_features=0.419171992638116),
                              n_estimators=410,
                              learning_rate=1.26852534318595,
                              random_state=SEED)
    
    '''
    Training
    '''
    time_start = time()
    model.fit(X_train, y_train)
    time_end = time()
    print(f"Time for training: {time_end - time_start}")
    
    '''
    Testing
    '''
    results = {}
    results['pred_train'] = model.predict(X_train)
    results['pred_test']  = model.predict(X_test)
    results['model']      = model
    
    return results


# ## Optimized RF model
# RF model with optimized hyper-parameters for its top mutation matrix AZAE970101
# 
# > **Parameters**
# > - X_train (numpy array): input features to train the model
# > - y_train (numpy array): output labels for supervised learning of the model
# > - X_test (numpy array): input features to test the model
# > - y_test: dummy, not used, default=None
# 
# > **Returns**
# > - results (dict): dictionary including:
# >    - pred_train (numpy array): predictions for training dataset
# >    - pred_test (numpy array): predictions for test dataset
# >    - model (object): trained model

# In[ ]:


def model_RF(X_train, y_train, X_test, y_test=None):
    
    '''
    Model
    '''
    model = RandomForestRegressor(n_estimators = 125,
                                  min_samples_split = 10,
                                  min_samples_leaf = 1,
                                  max_features = 0.375553860442328,
                                  max_depth = 200,
                                  bootstrap = True,
                                  random_state = SEED,
                                  n_jobs = -1)
    
    '''
    Training
    '''
    time_start = time()
    model.fit(X_train, y_train)
    time_end = time()
    print(f"Time for training: {time_end - time_start}")
    
    '''
    Testing
    '''
    results = {}
    results['pred_train'] = model.predict(X_train)
    results['pred_test']  = model.predict(X_test)
    results['model']      = model
    
    return results


# ## eXtreme Gradient Boosting (XGBoost)
# XGBoost regressor with optimized hyper-parameters for its top mutation matrix GIAG010101
# 
# > **Parameters**
# > - X_train (numpy array): input features to train the model
# > - y_train (numpy array): output labels for supervised learning of the model
# > - X_test (numpy array): input features to test the model
# > - y_test: dummy, not used, default=None
# 
# > **Returns**
# > - results (dict): dictionary including:
# >    - pred_train (numpy array): predictions for training dataset
# >    - pred_test (numpy array): predictions for test dataset
# >    - model (object): trained model

# In[ ]:


def model_XGBoost(X_train, y_train, X_test, y_test=None):
    
    '''
    Model
    '''
    model = XGBRegressor(booster='gbtree',
                         n_estimators=343, max_depth=23,
                         learning_rate=0.0586498853490469, subsample=0.790391730792872,
                         colsample_bytree=0.829414276718852, colsample_bylevel=0.360570017142831,
                         n_jobs=-1, random_state = SEED)
    
    
    '''
    Training
    '''
    time_start = time()
    model.fit(X_train, y_train,
              verbose=False)
    time_end = time()
    print(f"Time for training: {time_end - time_start}")
    
    '''
    Testing
    '''
    results = {}
    results['pred_train'] = model.predict(X_train)
    results['model']      = model
    results['pred_test']  = model.predict(X_test)
    
    return results


# ## Multi-layer Perceptron
# Multi-layer Perceptron with optimized hyperparameters for mutation matrix WEIL970102
# 
# > **Parameters**
# > - X_train (numpy array): input features to train the model
# > - y_train (numpy array): output labels for supervised learning of the model
# > - X_test (numpy array): input features to test the model
# > - y_test: dummy, not used, default=None
# 
# > **Returns**
# > - results (dict): dictionary including:
# >    - pred_train (numpy array): predictions for training dataset
# >    - pred_test (numpy array): predictions for test dataset
# >    - model (object): trained model

# In[ ]:


def model_MLP(X_train, y_train, X_test, y_test=None):
    
    '''
    Hyperparameters
    '''
    learning_rate = 0.0000168309492546526
    epochs        = 160
    n_layers      = 2
    layer_params  = {'n_units_l1': 5000,
                     'dropout_l1': 0.4,
                     'n_units_l2': 3100,
                     'dropout_l2': 0.5}
    
    '''
    Normalization
    '''
    # Input normalization
    normalizer = MinMaxScaler()
    X_train    = normalizer.fit_transform(X_train)
    X_test     = normalizer.transform(X_test)

    # target reshaping
    y_train = y_train.reshape(-1, 1)
    
    
    '''
    Model
    '''
    input1 = Input(shape=(X_train.shape[1],))
    
    # hidden layers
    for layer in range(1, n_layers+1):
        if layer == 1:
            # first hidden layer uses input1
            x1 = Dense(layer_params[f"n_units_l{layer}"])(input1)
        else:
            x1 = Dense(layer_params[f"n_units_l{layer}"])(x1)
        x1 = LeakyReLU()(x1)
        x1 = Dropout(layer_params[f"dropout_l{layer}"])(x1)
    
    # output layer
    x1 = Dense(1)(x1)
    
    model = tf.keras.models.Model(inputs=input1, outputs = x1)
    model.compile(loss = tf.keras.losses.MeanSquaredError(),
                  optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
                 )
    
    '''
    Training
    '''
    time_start = time()
    model.fit(X_train, y_train,
              epochs = epochs,
              batch_size = 1024,
              shuffle = True,
              verbose=0
             )
    time_end = time()
    print(f"Time for training: {time_end - time_start}")
    
    '''
    Testing
    '''
    results = {}
    results['pred_train'] = model.predict(X_train, verbose=0).squeeze()
    results['pred_test']  = model.predict(X_test, verbose=0).squeeze()
    results['model']      = model
    
    return results


# ## ResNet
# Residual neural network with optimized hyperparameters for mutation matrix MUET010101
# 
# > **Parameters**
# > - X_train (numpy array): input features to train the model
# > - y_train (numpy array): output labels for supervised learning of the model
# > - X_test (numpy array): input features to test the model
# > - y_test: dummy, not used, default=None
# 
# > **Returns**
# > - results (dict): dictionary including:
# >    - pred_train (numpy array): predictions for training dataset
# >    - pred_test (numpy array): predictions for test dataset
# >    - model (object): trained model

# In[ ]:


def model_ResNet(X_train, y_train, X_test, y_test=None):
    
    '''
    Hyperparameters
    '''
    lr                = 0.003494896818018
    epochs            = 140
    n_units_linear    = 3200
    n_layers          = 1
    layer_params      = {'n_units_rnb_1': 1500,
                         'dropout_rnb_1': 0.4,
                         'res_dropout_rnb_1': 0}
    
    '''
    Normalization
    '''
    # Input normalization
    normalizer = MinMaxScaler()
    X_train    = normalizer.fit_transform(X_train)
    X_test     = normalizer.transform(X_test)

    # target reshaping
    y_train = y_train.reshape(-1, 1)

    
    '''
    Model
    '''
    input1 = Input(shape=(X_train.shape[1],))
    
    # initial Linear layer
    x1 = Dense(n_units_linear)(input1)
    
    # ResNetBlock
    for resnet_block in range(1, n_layers+1):
        x1 = BatchNormalization()(x1)
        x1 = Dense(layer_params[f"n_units_rnb_{resnet_block}"], activation='relu')(x1)
        x1 = Dropout(layer_params[f"dropout_rnb_{resnet_block}"])(x1)
        x1 = Dense(X_train.shape[1])(x1)
        x1 = Dropout(layer_params[f"res_dropout_rnb_{resnet_block}"])(x1)
        x1 = Add()([x1, input1])
    
    # Prediction block
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)
    x1 = Dense(1)(x1)

    model = tf.keras.models.Model(inputs=input1, outputs = x1)

    model.compile(loss = tf.keras.losses.MeanSquaredError(),
                  optimizer = tf.optimizers.Adam(learning_rate=lr),
                  metrics = [tf.metrics.MeanAbsoluteError()])
    
    '''
    Training
    '''
    time_start = time()
    model.fit(X_train, y_train,
              epochs = epochs,
              batch_size = 1024,
              shuffle = True,
              verbose=0)
    time_end = time()
    print(f"Time for training: {time_end - time_start}")
    
    '''
    Testing
    '''
    results = {}
    results['pred_train'] = model.predict(X_train, verbose=0).squeeze()
    results['pred_test']  = model.predict(X_test, verbose=0).squeeze()
    results['model']      = model
    
    return results

