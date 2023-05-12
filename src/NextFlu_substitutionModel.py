#!/usr/bin/env python
# coding: utf-8

# # NextFlu substitution model
# Neher et. al., "Prediction, dynamics, and visualization of antigenic phenotypes of seasonal influenza viruses". Proc. Natl. Acad. Sci. USA 113, E1701–E1709 (2016)
# 
# **Note**: This is our own implementation adopted from original codes at NextStrain (https://github.com/nextstrain/augur/blob/master/augur/titer_model.py).

# ### Imports

# In[ ]:


import numpy as np
import pandas as pd
from collections import defaultdict


# ## Unique isolates
# Get unique isolates as per name
# 
# > **Parameters**
# > - df (dataframe): NHT data
#     
# > **Returns**
# > - isolates (dataframe): unique isolates with information of name and sequence
# > - viruses (dataframe): unique viruses with information of name and sequence
# > - sera (dataframe): unique isolates corresponding to sera with information of name and sequence

# In[ ]:


def unique_isolates(df):

    '''
    unique virus
    '''
    viruses = df[['virusName', 'virusSeq']].copy()
    viruses = viruses.drop_duplicates(['virusName'], keep='first', ignore_index=True)
    viruses.rename(columns={'virusName': 'isolateName',
                          'virusSeq': 'sequence'}, inplace=True)
    viruses.sort_values(['isolateName'], inplace=True, ignore_index=True)
    
    
    '''
    unique antisera
    '''
    sera = df[['serumName', 'serumSeq']].copy()
    sera = sera.drop_duplicates(['serumName'], keep='first', ignore_index=True)
    sera.rename(columns={'serumName': 'isolateName',
                         'serumSeq': 'sequence'}, inplace=True)
    sera.sort_values(['isolateName'], inplace=True, ignore_index=True)
    
    
    '''
    unique isolates
    '''
    isolates = pd.concat((viruses, sera), ignore_index=True)
    isolates = isolates.drop_duplicates(['isolateName'], keep='first', ignore_index=True)
    isolates.sort_values(['isolateName'], inplace=True, ignore_index=True)
    
    
    return isolates, viruses, sera


# ## Expand sequences
# Return expanded sequences in the form of dataframe with sites as columns
# 
# > **Parameters**
# > - isolates (dataframe): sequences of isolates
# 
# > **Returns**
# > - sequences_expand (dataframe): expanded sequences with sites as columns

# In[ ]:


def expand_sequences(isolates):

    sequence_list    = [list(seq) for seq in isolates.sequence]
    sequences_expand = pd.DataFrame(sequence_list, index=isolates.index)
    
    return sequences_expand


# ## Frequency of amino acids per site
# 
# > **Parameters**
# > - isolates (dataframe): isolates with sequences
# 
# > **Returns**
# > - aa_freq (defaultdict): Includes frequency of amino acid corresponding to a tuple of amino acid and its position identifiers 

# In[ ]:


def frequency_aa(isolates):

    sequences_expand = expand_sequences(isolates)
    
    aa_freq = defaultdict(int)
    
    # loop through the sites
    for pos in sequences_expand.columns:
        # aa count at each position
        aa_count = sequences_expand[pos].value_counts()
        
        # aa frequency
        for aa in aa_count.keys():
            aa_freq[(aa, pos)] = 1.0*aa_count[aa]/len(isolates)
    
    
    return aa_freq


# ## Get mutations
# Get amino acid mutations between sequences of virus-antiserum pair in the form "F159S". Note that, the position will be as per HA1 numbering. To get python based numbering subtract 1.
# 
# > **Parameters**
# > - seq_serum (str): sequence of isolate corresponding to antiserum
# > - seq_virus (str): sequence of virus isolate
# 
# > **Returns**
# > - muts (list): amino acid mutations with its position such as "F159S"

# In[ ]:


def get_mutations(seq_serum, seq_virus):

    muts = []
    muts.extend([aa1+str(pos+1)+aa2 for pos, (aa1, aa2) in enumerate(zip(seq_serum, seq_virus)) if aa1!=aa2])
    
    return muts


# ## Get relevant mutations from the training dataset
# A mutation in the training dataset is considered as relevant if it appears at least 10 times, the first amino acid as well as the mutated amino acid appears at that position more than the minimum frequency threshold based on a count of 10.
# 
# > **Parameters**
# > - data_train (dataframe): training dataset
# > - isolates (dataframe): isolates with information of name and sequences
# 
# > **Returns**
# > - relevant_muts (list): relevant mutations

# In[ ]:


def relevant_mutations(data_train, isolates):

    # count how often each mutation separates virus and serum
    mutation_counter = defaultdict(int)
    for ind, row in data_train.iterrows():
        muts = get_mutations(row.serumSeq, row.virusSeq)
        
        if len(muts)==0:
            continue
        for mut in muts:
            mutation_counter[mut]+=1
    
    
    # make a list of mutations deemed relevant via frequency thresholds
    aa_frequency = frequency_aa(isolates)
    relevant_muts = []
    min_count     = 10
    min_freq      = 1.0*min_count/len(isolates)
    for mut, count in mutation_counter.items():
        pos = int(mut[1:-1])-1
        aa1, aa2 = mut[0],mut[-1]
        if count>min_count and             aa_frequency[(aa1, pos)]>min_freq and             aa_frequency[(aa2, pos)]>min_freq:
                relevant_muts.append(mut)
    
    relevant_muts.sort(key = lambda x:int(x[1:-1]))
    
    return relevant_muts


# ## Find colinear columns of the design matrix, collapse them into clusters
# This function corresponds to the line in *Neher et. al. 2016* that reads as "Sets of substitutions that always occur together are merged and treated as one compound substitution".
# 
# > **Parameters**
# > - seq_graph (numpy array): binary identifier representing matrix 'A' in eq. 6 of *Neher et. al. 2016*
# > - relevant_muts (list): relevant mutations
# > - colin_thres (float): threshold for merging the mutations
# 
# > **Returns**
# > - seq_graph (numpy array): updated binary identifier
# > - relevant_muts (list): updated relevant mutations
# > - mutation_clusters (list): information about clustered mutations

# In[ ]:


def collapse_colinear_mutations(seq_graph, relevant_muts, colin_thres):

    n_genetic = len(relevant_muts)
    TT = seq_graph[:,:n_genetic].T
    mutation_clusters = [] 
    n_measurements = seq_graph.shape[0]
    
    # a greedy algorithm: if column is similar to existing cluster -> merge with cluster, else -> new cluster
    for col, mut in zip(TT, relevant_muts):
        col_found = False
        for cluster in mutation_clusters:
            # similarity is defined as number of measurements at which the cluster and column differ
            if np.sum(col==cluster[0])>=n_measurements-colin_thres:
                cluster[1].append(mut)
                col_found=True
                print("adding",mut,"to cluster ",cluster[1]) 
                break
        if not col_found:
            mutation_clusters.append([col, [mut]])
                
    print("dimensions of old design matrix",seq_graph.shape)
    seq_graph = np.hstack((np.array([c[0] for c in mutation_clusters]).T, seq_graph[:,n_genetic:]))
    n_genetic = len(mutation_clusters)
    # use the first mutation of a cluster to index the effect
    # make a dictionary that maps this effect to the cluster
    mutation_clusters = {c[1][0]:c[1] for c in mutation_clusters}
    relevant_muts = [c[1][0] for c in mutation_clusters]
    print("dimensions of new design matrix",seq_graph.shape)
    
    return seq_graph, relevant_muts, mutation_clusters


# ## NextFlu Training
# Train NextFlu substitution model
# - Find unique isolates and relevant mutations in training dataset
# - Prepare variables in eq. 3 of Neher *et. al.* 2016 and solve using cvxopt
#     - Prepare 'A' as in eq. 6 of Neher *et. al.* 2016, referred here as seq_graph
#     - Prepare 'H' as in eqs. 9 and 11 of Neher *et. al.* 2016, referred here as HI_dist
#     - Find the weights 'x' as in eq. 4 of Neher *et. al.* 2016, referred here as params
# - Find mutations effects, virus avidity, and antiserum potency
# 
# > **Parameters**
# > - data_train (dataframe): training dataset with features as defined in notebook "Fig2a_data_distribution"
# > - lam_HI (float): l1 regularization parameter for titer drops, default=1
# > - lam_pot (float): l2 regularization parameter for antiserum potency, default=0.2
# > - lam_avi (float): l2 regularization parameter for virus avidity, default=2
# > - rel_mut (float): for a value other than zero the function returns relevant mutations, default=0
# > - colin_thres (float or str): if None, the collinear mutations are not merged, default=None
# 
# > **Returns**
# > - mutation_effects (dictionary): weights associated with mutations, indexed with mutations
# > - serum_potency (dictionary): potencies of antisera, indexed with antisera names
# > - virus_avidity (dictionary): avidities of virus isolates, indexed with virus isolate names
# > - relevant_muts (list): relevant mutations

# In[ ]:


def nextflu_train(data_train, lam_HI=1, lam_pot=0.2, lam_avi=2, rel_mut=0, colin_thres=None):

    seq_graph = []
    HI_dist   = []
    
    isolates, viruses, sera = unique_isolates(data_train)
    relevant_muts = relevant_mutations(data_train, isolates)
    
    
    # parameters of the model
    n_genetic = len(relevant_muts)
    n_sera    = len(sera)
    n_v       = len(viruses)
    n_params  = n_genetic + n_sera + n_v
    
    
    # loop over all measurements and encode the HI model as [0,1,0,1,0,0..] vector:
    # 1-> mutation present, 0 not present, same for serum and virus effects
    for ind, row in data_train.iterrows():
        if not np.isnan(row.nht):
            muts = get_mutations(row.serumSeq, row.virusSeq)
            if len(muts)==0:
                continue
            tmp = np.zeros(n_params) # zero vector, ones will be filled in
            
            # determine branch indices on path
            mutation_indices = np.unique([relevant_muts.index(mut) for mut in muts if mut in relevant_muts])
            if len(mutation_indices): tmp[mutation_indices] = 1
            
            # add serum effect
            tmp[n_genetic+sera.index[sera.isolateName==row.serumName][0]] = 1
            
            # add virus effect
            tmp[n_genetic+n_sera+viruses.index[viruses.isolateName==row.virusName][0]] = 1
            
            # append model and nht value to lists seq_graph and HI_dist, respectively
            seq_graph.append(tmp)
            HI_dist.append(row.nht)
    
    
    # convert to numpy arrays
    HI_dist   = np.array(HI_dist)
    seq_graph = np.array(seq_graph)
    
    # collapse colinear mutations
    if colin_thres is not None:
        seq_graph, relevant_muts, mutation_clusters = collapse_colinear_mutations(seq_graph, relevant_muts, colin_thres)
    
    n_genetic = len(relevant_muts)
    n_params  = seq_graph.shape[1]
    
    # save product of tree graph with its transpose for future use
    TgT = np.dot(seq_graph.T, seq_graph)
    
    
    '''
    non-negative fit, branch terms L1 regularized, avidity terms L2 regularized
    '''
    from cvxopt import matrix, solvers
    
    # set up the quadratic matrix containing the deviation term (linear xterm below)
    # and the l2-regulatization of the avidities and potencies
    P1 = np.zeros((n_params,n_params))
    P1[:n_params, :n_params] = TgT
    for ii in range(n_genetic, n_genetic+n_sera):
        P1[ii,ii] += lam_pot
    for ii in range(n_genetic+n_sera, n_params):
        P1[ii,ii] += lam_avi
    P = matrix(P1)
    
    # set up cost for auxillary parameter and the linear cross-term
    q1 = np.zeros(n_params)
    q1[:n_params] = -np.dot(HI_dist, seq_graph)
    q1[:n_genetic] += lam_HI
    q = matrix(q1)
    
    # set up linear constraint matrix to enforce positivity of the
    # dHIs and bounding of dHI by the auxillary parameter
    h = matrix(np.zeros(n_genetic))   # Gw <=h
    G1 = np.zeros((n_genetic,n_params))
    G1[:n_genetic, :n_genetic] = -np.eye(n_genetic)
    G = matrix(G1)
    
    W = solvers.qp(P,q,G,h)
    
    params = np.array([x for x in W['x']])[:n_params]
    
    
    '''
    map substitution effects, serum potency and virus avidity
    '''
    mutation_effects={}
    for mi, mut in enumerate(relevant_muts):
        mutation_effects[mut] = params[mi]
    
    serum_potency = {serum:params[n_genetic+ii] for ii, serum in enumerate(sera.isolateName)}
    
    virus_avidity = {strain:params[n_genetic+n_sera+ii] for ii, strain in enumerate(viruses.isolateName)}
    
    if rel_mut==0:
        return mutation_effects, serum_potency, virus_avidity
    else:
        return mutation_effects, serum_potency, virus_avidity, relevant_muts


# ## NextFlu Predictions
# Predict the HI titer values for the provided test dataset using the provided mutations effects, virus avidities, and antiserum potencies.
# 
# > **Parameters**
# > - data_test (dataframe): test dataset with features as defined in notebook "Fig2a_data_distribution"
# > - mutation_effects (dictionary): weights associated with mutations, indexed with mutations
# > - serum_potency (dictionary): potencies of antisera, indexed with antisera names, default={}
# > - virus_avidity (dictionary): avidities of virus isolates, indexed with virus isolate name, default={}
# 
# > **Returns**
# > - pred_HI (numpy array): nht values predicted by NextFlu substitution model for the data_test

# In[ ]:


def nextflu_predict(data_test, mutation_effects, serum_potency={}, virus_avidity={}):
    pred_HI = []
    for ind, row in data_test.iterrows():
        muts = get_mutations(row.serumSeq, row.virusSeq)
        if len(muts) or len(serum_potency) or len(virus_avidity):
            pred = 0
            pred += serum_potency[row.serumName] if row.serumName in serum_potency.keys() else 0
            pred += virus_avidity[row.virusName] if row.virusName in virus_avidity.keys() else 0
            pred += np.sum([mutation_effects[mut] for mut in muts
                            if (mut in mutation_effects and mutation_effects[mut]>0.0)])
        else:
            pred = 0
        
        pred_HI.append(pred)
    
    return np.array(pred_HI)

