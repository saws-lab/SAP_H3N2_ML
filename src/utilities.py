#!/usr/bin/env python
# coding: utf-8

# # Utilities
# It contains a number of self defined helper functions.

# ### Imports

# In[ ]:


import os
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


# # Save dictionary in a CSV file
# 
# > **Parameters**
# > - output (dict): dictionary to be saved
# > - fn (str): CSV filename

# In[ ]:


def saveDict2CSV(output, fn):

    output_df   = pd.DataFrame.from_dict(output) 
    
    # if csv file already exists then append the results in a row
    if os.path.isfile(fn):
        # save without adding header
        output_df.to_csv(fn, index=False, mode='a', header=False)
    else:
        output_df.to_csv(fn, index=False)


# ## Data distribution
# Compute data distribution and save it in a CSV file
# 
# > **Parameters**
# > - nhts (array like): nht values
# > - fn (str): CSV filename to save the data distribution, default=None
# > - col (list): extra column(s) in a CSV file
# > - col_val (list): value(s) of extra column(s)

# In[ ]:


def data_distribution(nhts, fn=None, col=[], col_val=[]):

    total         = len(nhts)
    variant       = sum(nhts > 2)
    similar       = total - variant
    variant_ratio = variant/total * 100 if total else 0      # exception for division with zero
    
    # dictionary for data distribution
    dist = {'total':   total,
            'variant': variant,
            'similar': similar,
            'variant_ratio':   variant_ratio}
    
    if fn:
        output = {}

        # if 'col' is provided
        if len(col) > 0:
            # loop through col and save it
            for column, column_value in zip(col, col_val):
                output[column] = column_value

        output.update(dist)

        saveDict2CSV([output], fn)
    else:
        return dist


# ## Compute scores
# Compute scores and save in a CSV file
#     
# > **Parameters**
# > - y (numpy array): actual NHTs
# > - y_pred (numpy array): predicted NHTs
# > - filename (string): filename to save the computed scores
# > - thresh (int): threshold value for conversion to labels (default = 2)
# > - col (list): list denoting extra column(s) in a scores CSV file
# > - col_val (list): value(s) of extra column(s)
# > - y_label (numpy array): labels for actual NHTs, (default='')
# > - y_pred_label (numpy array): labels for predicted NHTs, (default='')

# In[ ]:


def compute_scores(y, y_pred, filename, thresh=2, col=[], col_val=[], y_label='', y_pred_label=''):
    
    # make sure there is data in the input
    if y.size:
        # change to 1D array
        y      = y.squeeze()
        y_pred = y_pred.squeeze()
        
        # if labels are not provided
        if (y_label=='') & (y_pred_label==''):
            # compute labels (actual threshold = 2)
            y_label      = (y > 2) * 1
            y_pred_label = (y_pred > thresh) * 1
            
    else:
        print('Error: empty input, no computation done.')
        return
    
    
    # scores
    tn, fp, fn, tp = metrics.confusion_matrix(y_label, y_pred_label, labels=[0,1]).ravel()
    
    
    # regression metrics
    mae = metrics.mean_absolute_error(y, y_pred)
    r2  = metrics.r2_score(y, y_pred)
    
    # classification metrics
    acc    = metrics.accuracy_score(y_label, y_pred_label)
    recall = metrics.recall_score(y_label, y_pred_label)
    spec   = tn/(tn+fp)
    prec   = metrics.precision_score(y_label, y_pred_label)
    npv    = tn/(tn+fn)
    mcc    = metrics.matthews_corrcoef(y_label, y_pred_label)
    try:
        auroc  = metrics.roc_auc_score(y_label, y_pred)
    except ValueError:
        auroc  = np.nan
    try:
        auprc  = metrics.average_precision_score(y_label, y_pred)
    except ValueError:
        auprc  = np.nan
        
    
    # dictionary of results
    scores = {}
    
    # if col is provided
    if len(col) > 0:
        # loop through col and save it
        for column, column_value in zip(col, col_val):
            scores[column] = column_value
    
    scores['MAE']         = mae
    scores['R2']          = r2
    scores['Accuracy']    = acc
    scores['Sensitivity'] = recall
    scores['Specificity'] = spec
    scores['MCC']         = mcc
    scores['AUROC']       = auroc
    scores['Precision']   = prec
    scores['NPV']         = npv
    scores['AUPRC']       = auprc
        
    
    # save scores to a CSV file
    saveDict2CSV([scores], filename)


# ## Dates as per the influenza season
# Provide start and end dates of circulating isolates as per the provided season
# 
# - for NH season: circulating isolates range from "season_year-1"-Sep-01 to "season_year"-Jan-31
# - for SH season: circulating isolates range from "season_year"-Feb-01 to "season_year"-Aug-31
# 
# > **Parameters**
# > - season (str): identifier for the Northern of Southern Hemisphere season such as "2015NH"
# 
# > **Returns**
# > - circ_start (string): start date of circulating isolates as per 'season'.
#     circ_end (string): end date of circulating isolates as per 'season'.

# In[ ]:


def circulating_dates(season):
    
    ###########
    # NH season
    ###########
    if season[-2] == "N":
        # get season year as int
        season_year = int(season[:-2])
        
        # circulating isolates
        # time limits according to NH season
        circ_start = f"{season_year-1}-09-01"    # Sep. start of previous year
        circ_end   = f"{season_year}-01-31"      # Jan. end
    
    ###########
    # SH season
    ###########
    elif season[-2] == "S":
        season_year = int(season[:-2])

        # time limits according to SH meeting
        circ_start = f"{season_year}-02-01"    # Feb. start
        circ_end   = f"{season_year}-08-31"    # Aug. end

   
    else:
        print("Wrong input for season: ", season)
        exit(1)
    
    
    return circ_start, circ_end


# ## Seasonal split
# Split the data into training and test datasets as per influenza seasonsal vaccine composition meetings
# 
# - Training pairs: past virus isolates paired with past antisera
# - Test pairs: circulating virus isolates paired with past antisera
# 
# 
# - for NH season
#     - past isolates before "meeting_year-1"-Sep-01
#     - circulating isolates between "meeting_year-1"-Sep-01 to "meeting_year"-Jan-31
# - for SH meeting
#     - past isolates before "meeting_year"-Feb-01
#     - circulating isolates between "meeting_year"-Feb-01 to "meeting_year"-Aug-31
# 
# 
# > **Parameters**
# > - data (DataFrame): Whole dataset to be split.
# > - test_season (string): Identifier for Northern or Southern Hemisphere season such as "2022NH" or "2022SH".
# 
# > **Returns**
# > - train_ind (numpy array): indices of training dataset as per "test_season". Dimension = (data.shape[0], 1)
# > - test_ind (numpy array): indices of test dataset as per "test_season". Dimension = (data.shape[0], 1)

# In[ ]:


def seasonal_trainTestSplit(data, test_season):
    # get start and end dates for circulating isolates
    circ_start, circ_end = circulating_dates(test_season)
    
    # indices of circulating virus
    ind_circ_virus = data.virusDate.ge(circ_start) & data.virusDate.le(circ_end)
    
    # indices of past virus and antiserum
    ind_past_virus = data.virusDate.lt(circ_start)
    ind_past_serum = data.serumDate.lt(circ_start)
    
    '''
    Training dataset
    '''
    # past virus isolates paired with past antisera
    train_ind = ind_past_virus & ind_past_serum
    
    '''
    Test dataset
    '''
    # circulating virus isolates paired with past antisera
    test_ind  = ind_circ_virus & ind_past_serum
    
    
    return train_ind.to_numpy(), test_ind.to_numpy()


# ## Seasonal split, randomly samples fraction of HI titers per season
# Split the data into training and test datasets as per influenza seasonsal vaccine composition meetings. This function corresponds to the first scenario of robustness tests (Supp. Fig. 4a), which randomly samples a fraction of HI titers in each season from training dataset.
# 
# > **Parameters**
# > - data (DataFrame): Whole dataset to be split.
# > - test_season (str): Either northern or southern hemisphere season identifier such as "2022NH" or "2022SH"
# > - titers_train (int): percentage of titers from each season in training dataset (default = 100)
# > - random_state (int): random seed for random selection of titers_train (default = None)
#     
# > **Returns**
# > - train_ind (numpy array): indices of training dataset as per "test_season". Dimension = (data.shape[0], 1)
# > - test_ind (numpy array): indices of test dataset as per "test_season". Dimension = (data.shape[0], 1)

# In[ ]:


def rndTitersTrainSeason_seasonal_trainTestSplit(data, test_season, titers_train=100, random_state=None):
    
    '''
    Test dataset
    '''
    # get start and end dates for circulating isolates
    circ_start, circ_end = circulating_dates(test_season)
    
    # indices of circulating virus
    ind_circ_virus = data.virusDate.ge(circ_start) & data.virusDate.le(circ_end)
    
    # indices of past antisera before current season
    ind_past_serum = data.serumDate.lt(circ_start)
    
    # circulating virus isolates paired with past antisera
    test_ind = ind_circ_virus & ind_past_serum
    
    
    '''
    Training dataset
    '''
    # if 100% data per meeting in training dataset is required
    if titers_train == 100:
        # then there is no need to filter data to check robustness
        # and we can use all the past data       
        
        # indices of all the virus isolates before the current meeting
        ind_past_virus = data.virusDate.lt(circ_start)
        
        # training dataset
        # past virus isolates paired with past sera
        train_ind = ind_past_virus & ind_past_serum
        
    # if filtering of data per season is required to check robustness
    else:
        titers_train = titers_train/100     # convertion due to percentage
        ind_train = np.empty(0, dtype='object')
        
        Seasons = [str(year)+s for year in range (2003, 2021) for s in ["NH", "SH"]]   # seasons from 2003NH to 2020SH
                
        # for each season, randomly pick data_train percent of data
        for season in Seasons:
            # if season is before the test season
            if season < test_season:
                # get start and end dates as per season
                season_start, season_end = circulating_dates(season)
                
                # virus isolates in season
                ind_season_virus = data.virusDate.ge(season_start) & data.virusDate.le(season_end)
                
                # virus-antiserum pairs in season
                ind_titers_season = ind_season_virus & ind_past_serum
                
                # indices of true values in ind_titers_season
                index_titers_season = ind_titers_season[ind_titers_season].index
                N_titers_season     = len(index_titers_season)
                
                # random selection
                np.random.seed(random_state)
                rnd_titers_indices = np.random.choice(N_titers_season, int(titers_train*N_titers_season), replace=False)
                
                # indices of randomly selected virus-antiserum pairs in a season
                ind_train_season = index_titers_season[rnd_titers_indices]
                
                # store selected virus isolates of this season
                ind_train = np.concatenate((ind_train, ind_train_season))
                
            else:
                break
                
        # get unique indices for training virus-antiserum pairs
        ind_train = list(set(ind_train))
        
        # indices of virus-antiserum pairs in training dataset
        # in the form of boolean series of size equal to dataset
        train_ind = pd.Series(False, range(data.shape[0]))
        train_ind[ind_train] = True

    
    return train_ind.to_numpy(), test_ind.to_numpy()


# ## Seasonal split, randomly samples fraction of isolates per season
# Split the data into training and test datasets as per influenza seasonsal vaccine composition meetings. This function corresponds to the second scenario of robustness tests (Supp. Fig. 4b), which randomly samples a fraction of virus isolates in each season from training dataset.
# 
# > **Parameters**
# > - data (DataFrame): Whole dataset to be split.
# > - test_season (str): Either northern or southern hemisphere season identifier such as "2022NH" or "2022SH"
# > - isolates_train (int): percentage of virus isolates from each season in training dataset (default = 100)
# > - random_state (int): random seed for random selection of isolates_train (default = None)
#     
# > **Returns**
# > - train_ind (numpy array): indices of training dataset as per "test_season". Dimension = (data.shape[0], 1)
# > - test_ind (numpy array): indices of test dataset as per "test_season". Dimension = (data.shape[0], 1)

# In[ ]:


def rndIsolatesTrainSeason_seasonal_trainTestSplit(data, test_season, isolates_train=100, random_state=None):
    
    '''
    Test dataset
    '''
    # get start and end dates for circulating isolates
    circ_start, circ_end = circulating_dates(test_season)
    
    # indices of circulating virus
    ind_circ_virus = data.virusDate.ge(circ_start) & data.virusDate.le(circ_end)
    
    # indices of past sera before current season
    ind_past_serum = data.serumDate.lt(circ_start)
    
    # test pairs
    test_ind = ind_circ_virus & ind_past_serum
    
    
    '''
    Training dataset
    '''
    
    # if 100% isolates per season in training dataset are required
    if isolates_train == 100:
        # then there is no need to filter isolates to check robustness
        # and we can use all the past virus isolates        
        
        # indices of all the virus isolates before the current season
        ind_past_virus = data.virusDate.lt(circ_start)
        
        # training dataset
        # past virus isolates paired with past antisera
        train_ind = ind_past_virus & ind_past_serum
        
    # if filtering of isolates per season is required to check robustness
    else:
        isolates_train = isolates_train/100     # convertion due to percentage
        train_virus = np.empty(0, dtype='object')
        
        Seasons = [str(year)+s for year in range (2003, 2021) for s in ["NH", "SH"]]   # seasons from 2003NH to 2020SH
                
        # for each season, randomly pick isolates_train percent of isolates
        for season in Seasons:
            # if season is before the test season
            if season < test_season:
                # get start and end dates as per season
                season_start, season_end = circulating_dates(season)
                
                # virus isolates in season
                ind_season_virus = data.virusDate.ge(season_start) & data.virusDate.le(season_end)
                season_virus     = data.virus[ind_season_virus].unique()
                N_season_virus   = season_virus.shape[0]
                
                # random selection
                np.random.seed(random_state)
                rnd_season_virus = np.random.choice(N_season_virus, int(isolates_train*N_season_virus), replace=False)
                season_virus     = season_virus[rnd_season_virus]
                
                # store selected virus isolates of this season
                train_virus = np.concatenate((train_virus, season_virus))
                
            else:
                break
        
        # indices of selected virus isolates for all season in training dataset
        ind_train_virus = np.in1d(data.virus.to_numpy(), train_virus)
        
        # indices of virus-antiserum pairs in training dataset
        # selected virus isolates paired with past sera
        train_ind = ind_train_virus & ind_past_serum
    
    return train_ind.to_numpy(), test_ind.to_numpy()


# ## Seasonal split, with 1 missed season from training dataset
# Split the data into training and test datasets as per influenza seasonsal vaccine composition meetings such that the training dataset will consists of all the past data except data from 1 season. This function corresponds to Supp. Fig. 4c.
# 
# > **Parameters**
# > - data (DataFrame): Whole dataset to be split.
# > - test_season (str): Either northern or southern hemisphere season identifier such as "2022NH" or "2022SH"
# > - missed_season (str): identifier for the missed season from training dataset
#     
# > **Returns**
# > - train_ind (numpy array): indices of training dataset as per "test_season". Dimension = (data.shape[0], 1)
# > - test_ind (numpy array): indices of test dataset as per "test_season". Dimension = (data.shape[0], 1)

# In[ ]:


def miss1TrainSeason_seasonal_trainTestSplit(data, test_season, missed_season):
    
    '''
    Test dataset
    '''
    # get start and end dates for circulating isolates
    circ_start, circ_end = circulating_dates(test_season)
    
    # indices of circulating virus
    ind_circ_virus = data.virusDate.ge(circ_start) & data.virusDate.le(circ_end)
    
    # indices of past sera before current season
    ind_past_serum = data.serumDate.lt(circ_start)
    
    # test pairs
    test_ind = ind_circ_virus & ind_past_serum
    
    
    '''
    Training dataset
    '''
    # get start and end dates of missed season
    missed_start, missed_end = circulating_dates(missed_season)

    # indices of virus isolates before missed season and before start of circulating isolates
    ind_virus_before_miss = data.virusDate.lt(missed_start) & data.virusDate.lt(circ_start)
    # indices of virus isolates after missed season and before start of circulating isolates
    ind_virus_after_miss  = data.virusDate.gt(missed_end) & data.virusDate.lt(circ_start)

    # indices of virus isolates for training
    ind_train_virus = ind_virus_before_miss | ind_virus_after_miss

    # training dataset
    # indices of virus-antiserum pairs in training dataset
    train_ind = ind_train_virus & ind_past_serum
    
    
    return train_ind.to_numpy(), test_ind.to_numpy()


# ## Seasonal split, training data from recent seasons only
# Split the data into training and test datasets as per influenza seasonsal vaccine composition meetings, where the training dataset include data from a few recent seasons before the current season.
# 
# > **Parameters**
# > - data (DataFrame): Whole dataset to be split.
# > - test_season (str): Either northern or southern hemisphere season identifier such as "2022NH" or "2022SH"
# > - train_seasons ('all' or int): number of recent seasons from training data, (default='all', use all training seasons)
#     
# > **Returns**
# > - train_ind (numpy array): indices of training dataset as per "test_season". Dimension = (data.shape[0], 1)
# > - test_ind (numpy array): indices of test dataset as per "test_season". Dimension = (data.shape[0], 1)

# In[ ]:


def recentTrainSeasons_seasonal_trainTestSplit(data, test_season, train_seasons='all'):
    
    '''
    Test dataset
    '''
    # get start and end dates for circulating isolates
    circ_start, circ_end = circulating_dates(test_season)
    
    # indices of circulating virus
    ind_circ_virus = data.virusDate.ge(circ_start) & data.virusDate.le(circ_end)
    
    # indices of past sera before current season
    ind_past_serum = data.serumDate.lt(circ_start)
    
    
    # test pairs
    test_ind = ind_circ_virus & ind_past_serum
    
    
    '''
    Training dataset
    '''
    # identify the past virus start date
    ###############
    # for NH season
    ###############
    if test_season[-2] == "N":
        # get year as int
        season_year = int(test_season[:-2])
        
        # past virus start
        # all seasons, H3N2 starts from 1968 for any data source
        if train_seasons == 'all':
            past_virus_start = "1968"
        # even number of seasons
        # means same NH season
        elif train_seasons%2 == 0:
            past_virus_start = f"{season_year-int(train_seasons/2)-1}-09-01"
        # odd number of seasons
        # means SH season
        elif train_seasons%2 == 1:
            past_virus_start = f"{season_year-int(train_seasons/2)-1}-02-01"
            
    ###############
    # for SH season
    ###############
    else:
        season_year = int(test_season[:-2])
        
        # past virus start
        # all seasons, H3N2 starts from 1968 for any data source
        if train_seasons == 'all':
            past_virus_start = "1968"
        # even number of seasons
        # means same SH season
        elif train_seasons%2 == 0:
            past_virus_start = f"{season_year-int(train_seasons/2)}-02-01"
        # odd number of seasons
        # means NH season
        elif train_seasons%2 == 1:
            past_virus_start = f"{season_year-int(train_seasons/2)-1}-09-01"
    
    
    # indices of past virus isolates from a few recent seasons
    ind_past_virus = data.virusDate.ge(past_virus_start) & data.virusDate.lt(circ_start)
    
    # training pairs
    # past virus isolates paired with past sera
    train_ind = ind_past_virus & ind_past_serum
    
    
    return train_ind.to_numpy(), test_ind.to_numpy()


# ## Seasonal split, circulating isolates in training dataset
# Split the data into training and test datasets as per influenza seasonsal vaccine composition meetings. Also, include the data corresponding to a few percent of circulating isolates in training dataset. This function corresponds to Supp. Fig. 7.
# 
# > **Parameters**
# > - data (DataFrame): Whole dataset to be split.
# > - test_season (str): Either northern or southern hemisphere season identifier such as "2022NH" or "2022SH"
# > - circ_train (int): percentage of circulating virus isolates in training dataset (default=0)
# > - random_state (int): random seed for random selection of circ_train (default = None)
#     
# > **Returns**
# > - train_ind (numpy array): indices of training dataset as per "test_season". Dimension = (data.shape[0], 1)
# > - test_ind (numpy array): indices of test dataset as per "test_season". Dimension = (data.shape[0], 1)

# In[ ]:


def circIsolatesTrain_seasonal_trainTestSplit(data, test_season, circ_train=0, random_state=None):
    
    # get start and end dates for circulating isolates
    circ_start, circ_end = circulating_dates(test_season)
    
    # indices of past virus and antiserum
    ind_past_virus = data.virusDate.lt(circ_start)
    ind_past_serum = data.serumDate.lt(circ_start)
    
    
    # circulating virus
    ind_circ_virus = data.virusDate.ge(circ_start) & data.virusDate.le(circ_end)
    circ_virus     = data.virus[ind_circ_virus].unique()
    N_circ_virus   = circ_virus.shape[0]

    
    # randomly select 'circ_train' percent of circulating virus isolates
    # for training
    np.random.seed(random_state)
    circ_train = circ_train/100     # convertion due to percentage
    
    rnd_train_circ_virus = np.random.choice(N_circ_virus, int(circ_train*N_circ_virus), replace=False)
    train_circ_virus     = circ_virus[rnd_train_circ_virus]


    # indices of train circulating virus isolates
    ind_train_circ_virus = np.in1d(data.virus.to_numpy(), train_circ_virus)
    
    # indices of train circulating serum
    ind_train_circ_serum = np.in1d(data.serum.to_numpy(), train_circ_virus)
    

    # Remaining circulating isolates for testing
    test_circ_virus = np.setdiff1d(circ_virus, train_circ_virus)
    
    # indices of remaining circulating isolates
    ind_test_circ_virus = np.in1d(data.virus.to_numpy(), test_circ_virus)
    
    
    '''
    Training dataset
    '''
    # pairs of past virus isolates paired with past antisera
    pairs_upper_left                     = ind_past_virus & ind_past_serum
    # pairs of past virus isolates paired with circulating antisera randomly selected for training
    pairs_upper_partial_right            = ind_past_virus & ind_train_circ_serum
    # pairs of circulating virus isolates selected for training paired with past sera
    pairs_train_circ_lower_left          = ind_train_circ_virus & ind_past_serum
    # pairs of circulating virus-circulating antisera, both selected for training
    pairs_train_circ_lower_partial_right = ind_train_circ_virus & ind_train_circ_serum
    
    train_ind = pairs_upper_left | pairs_upper_partial_right | pairs_train_circ_lower_left | pairs_train_circ_lower_partial_right

    
    '''
    Test dataset
    '''
    # pairs of circulating isolates in test dataset paired with past antisera
    pairs_test_circ_lower_left          = ind_test_circ_virus & ind_past_serum
    # pairs of circulating isolates in test dataset paired with circulating antisera selected for training
    pairs_test_circ_lower_partial_right = ind_test_circ_virus & ind_train_circ_serum
    
    test_ind = pairs_test_circ_lower_left | pairs_test_circ_lower_partial_right
    
    
    return train_ind.to_numpy(), test_ind.to_numpy()


# ## Threshold optimization using Youden’s J statistic
# J = Sensitivity + Specificity – 1, or
# J = Sensitivity + (1 – FalsePositiveRate) – 1, or
# J = TruePositiveRate – FalsePositiveRate
# 
# > **Parameters**
# > - y_label (numpy array): antigenic labels based on NHT values
# > - pred (numpy array): predicted NHT values by a model
# > - fig_fn (str): path for threshold curve figure, threshold vs. Youden’s J statistic (YJS)
# 
# > **Returns**
# > - threshold (float): threshold value for classifying virus-antiserum pairs based on NHTs and YJS

# In[ ]:


def youden_threshold(y_label, pred, fig_fn=None):

    fpr, tpr, thresholds = metrics.roc_curve(y_label, pred)
    J = tpr - fpr
    
    if fig_fn:
        fig, ax = plt.subplots(figsize=(5,4))
        ax.plot(thresholds, J, 'x-')
    
        ax.set_xlabel("Threshold")
        ax.set_ylabel("YJS")
        sns.despine()
        
        fig.tight_layout()
        fig.savefig(fig_fn, format='svg', bbox_inches='tight')
    
    ix = np.argmax(J)
    threshold = thresholds[ix]
    
    return threshold


# ## Change width of seaborn based bars
# > **Parameters**
# > - ax: figure axes handle
# > - new_width (int): width of the bars

# In[ ]:


def change_seaborn_width(ax, new_width):
    
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_width

        # we change the bar width
        patch.set_width(new_width)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)


# ## Epitope sites
# > **Parameters**
# > - epDef (str): epitope definition of either Bush or Shih, default='Bush'
# 
# > **Returns**
# > - ep_sites (dataframe): HA1 site number and corresponding epitope label

# In[ ]:


def get_ep_sites(epDef='Bush'):
    if epDef == 'Shih' :
        '''
        Epitope sites defined by Shih et. al. (2007)
        Total 62 = A-12, B-16, C-8, D-13, E-13
        '''
        epA = [122,124,126,131,133,135,137,142,143,144,145,146]
        epB = [155,156,157,158,159,160,163,164,186,188,189,190,192,193,196,197]
        epC = [50,53,54,275,276,278,299,307]
        epD = [121,172,173,174,201,207,213,217,226,227,242,244,248]
        epE = [57,62,63,67,75,78,81,82,83,92,94,260,262]
        
    
    elif epDef == 'Bush':
        '''
        Epitope sites defined by Bush et. al. (1999)
        Total 131 = A-19, B-22, C-27, D-41, E-22
        '''
        epA = [122,124,126,130,131,132,133,135,137,138,140,142,143,144,145,146,150,152,168]
        epB = [128,129,155,156,157,158,159,160,163,164,165,186,187,188,189,190,
               192,193,194,196,197,198]
        epC = [44,45,46,47,48,50,51,53,54,273,275,276,278,279,280,294,297,299,
               300,304,305,307,308,309,310,311,312]
        epD = [96,102,103,117,121,167,170,171,172,173,174,175,176,177,179,182,
               201,203,207,208,209,212,213,214,215,216,217,218,219,226,227,228,
               229,230,238,240,242,244,246,247,248]
        epE = [57,59,62,63,67,75,78,80,81,82,83,86,87,88,91,92,94,109,260,261,262,265]

    
    ep_sites = pd.DataFrame(index=[epA+epB+epC+epD+epE])
    ep_sites.loc[epA, "epitope"] = "A"
    ep_sites.loc[epB, "epitope"] = "B"
    ep_sites.loc[epC, "epitope"] = "C"
    ep_sites.loc[epD, "epitope"] = "D"
    ep_sites.loc[epE, "epitope"] = "E"
    ep_sites.reset_index(inplace=True)
    ep_sites.rename(columns={"level_0":"site"}, inplace=True)
    ep_sites["site"] = "HA1_" + ep_sites.site.astype("str")
    
    return ep_sites

