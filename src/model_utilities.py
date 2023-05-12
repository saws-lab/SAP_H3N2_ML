#!/usr/bin/env python
# coding: utf-8

# # Model utilities
# It includes self defined functions for used models

# ### Imports

# In[ ]:


import numpy as np
from time import time
from sklearn.ensemble import RandomForestRegressor

# (for reproduciblility) fix the randomly generated numbers
SEED = 100
np.random.seed(SEED)


# ## Baseline model
# RF with default hyper-parameters
# 
# > **Parameters**
# > - X_train (numpy array): Input features to train the model
# > - y_train (numpy array): output labels for supervised learning of the model
# > - X_test (numpy array): Input features to test the model
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
    model = RandomForestRegressor(random_state = SEED, n_jobs = -1)
    
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
# RF model with optimized hyper-parameters
# 
# > **Parameters**
# > - X_train (numpy array): Input features to train the model
# > - y_train (numpy array): output labels for supervised learning of the model
# > - X_test (numpy array): Input features to test the model
# > - y_test: dummy, not used, default=None
# 
# > **Returns**
# > - results (dict): dictionary including:
# >    - pred_train (numpy array): predictions for training dataset
# >    - pred_test (numpy array): predictions for test dataset
# >    - model (object): trained model

# In[ ]:


def model_optimized_RF(X_train, y_train, X_test, y_test=None):
    
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

