#ensemble_agg.py

"""Functions to train and evaluate ensembles and aggregate predictions across
different model instances applied to the same data set"""

import copy
import numpy as np
import pandas as pd
import sklearn.metrics

from . import mlp
from . import regression
from . import reformat_output

def train_and_eval_ensemble(modeling_approach, model_args, num_ensemble, fold_num):
    """Train and evaluate an ensemble consisting of <num_ensemble> MLPs,
    each trained according to <model_args>."""
    assert num_ensemble >= 1
    assert isinstance(num_ensemble, int)
    #Train
    ensemble_lst = train_ensemble(modeling_approach, model_args, num_ensemble, 'grid_search')
    #Evaluate
    fold_test_out = create_fold_test_out(ensemble_lst, model_args['decision_threshold'], 'grid_search')
    fold_eval_dfs_dict = create_fold_eval_dfs_dict(fold_test_out, fold_num)
    #note that fold_test_out contains the data and the predictions, while
    #fold_eval_dfs_dict contains the performance
    return fold_test_out, fold_eval_dfs_dict

def train_ensemble(modeling_approach, model_args, num_ensemble, what_to_run):
     """This function trains and tests MLPs for the ensemble.
        <split>: the split object to specify training and testing data
        <num_ensemble>: the number of MLPs in the ensemble
        When <num_ensemble>==1 then this is the case where only one MLP is used"""
     ensemble_lst = []
     if modeling_approach == 'MLP':
        model_class = mlp.MLP
     elif modeling_approach == 'LR':
        model_class = regression.LogisticRegression
     for i in range(num_ensemble):
         print('\tTraining and testing model',i+1,'out of',num_ensemble,'for ensemble')
         model_args['seed'] = i+1 #different random seed for each member of the ensemble
         m = model_class(**model_args)
         if what_to_run == 'grid_search':
             m.run_all_train_test() #train and test
         elif what_to_run == 'mysteryAA_pred':
             m.run_all_mysteryAA_preds()
         ensemble_lst.append(m)
     return ensemble_lst

#Part of evaluating the ensemble
def create_fold_test_out(ensemble_lst, decision_threshold, what_to_run):
    """For ensemble models in <ensemble_lst>, aggregate their predictions.
    Returns a dictionary of dataframes called 'fold_test_out':
    The dictionary keys are 'epoch_1', 'epoch_2',... and so on.
    The values are dataframes with columns ['Consensus_etc','Change_etc','Position',
    'Conservation','SigNoise','Pred_Prob','Pred_Label','True_Label']
    The data has NOT been put in human-readable format, so for example the
    Consensus is many columns (one-hot encoding) and the Position is
    normalized (not integer positions.)"""
    if what_to_run == 'grid_search':
        epochs_to_consider = list(ensemble_lst[0].test_out.keys()) #e.g. ['epoch_0','epoch_1']
    elif what_to_run == 'mysteryAA_pred':
        epochs_to_consider = list(ensemble_lst[0].mysteryAAs_out.keys())
    
    #test out is a dictionary with keys that are epochs (e.g. 'epoch_0') and
    #values that are pandas dataframes
    test_out_collection = []
    for model in ensemble_lst:
        if what_to_run == 'grid_search':
            test_out_collection.append(model.test_out)
        elif what_to_run == 'mysteryAA_pred':
            test_out_collection.append(model.mysteryAAs_out)
    
    for idx in range(len(test_out_collection)):
        test_out = test_out_collection[idx]
        if idx == 0:
            fold_test_out = test_out
        else:
            for epochstr in epochs_to_consider:
                df = test_out[epochstr] #this df
                fold_df = fold_test_out[epochstr] #the aggregated df
                
                #Sanity check: ensure that the different members of the ensemble
                #were applied to the exact same data. Data columns Consensus_etc,
                #Change_etc, Position, Conservation, SigNoise, and True_Label
                #should be identical across all models in the ensemble.
                all_cols = df.columns.values.tolist()
                samecols = [x for x in all_cols if 'Consensus' in x]+[x for x in all_cols if 'Change' in x]+['True_Label']
                assert np.equal(df[samecols].values, fold_df[samecols].values).all()
                closecols = ['Position','Conservation','SigNoise']
                assert np.isclose(df[closecols].values, fold_df[closecols].values, rtol=1e-4).all()
                
                #Now sum up the Pred_Prob column:
                fold_test_out[epochstr].loc[:,'Pred_Prob'] += df['Pred_Prob']
    
    #Since we want to store the average Pred_Prob across all members of
    #the ensemble, divide the summed Pred_Prob by the number of models
    #in the ensemble, and then determine the Pred_Label:
    for epochstr in epochs_to_consider:
        fold_test_out[epochstr].loc[:,'Pred_Prob'] = fold_test_out[epochstr].loc[:,'Pred_Prob'].div(len(ensemble_lst))
        fold_test_out[epochstr].loc[:,'Pred_Label'] = (fold_test_out[epochstr].loc[:,'Pred_Prob'].values > decision_threshold).astype('int')
    return fold_test_out

#Part of evaluating the ensemble
def create_fold_eval_dfs_dict(fold_test_out, fold_num):
    """Return the performance of the ensemble models for all epochs
    based on the predictions and ground truth in <fold_test_out>.
    Returns fold_eval_dfs_dict which has keys 'accuracy', 'auroc', and
    'avg_precision' and values that are dataframes with an index of <fold_num>
    and columns of epoch number."""
    #Initialize empty fold_eval_dfs_dict
    epochs_to_consider = list(fold_test_out.keys())
    idx = 'fold_num'+str(fold_num)
    result_df = pd.DataFrame(data=np.zeros((1, len(epochs_to_consider))),
                            index = [idx],
                            columns = epochs_to_consider)
    fold_eval_dfs_dict = {'accuracy':copy.deepcopy(result_df),
        'auroc':copy.deepcopy(result_df),
        'avg_precision':copy.deepcopy(result_df)}
    
    #Calculate performance
    for epochstr in epochs_to_consider:
        true_label = fold_test_out[epochstr]['True_Label'].values
        pred_label = fold_test_out[epochstr]['Pred_Label'].values
        pred_prob = fold_test_out[epochstr]['Pred_Prob'].values
        fold_eval_dfs_dict['accuracy'].at[idx,epochstr] = sklearn.metrics.accuracy_score(true_label, pred_label)
        fold_eval_dfs_dict['auroc'].at[idx,epochstr] = sklearn.metrics.roc_auc_score(true_label, pred_prob)
        fold_eval_dfs_dict['avg_precision'].at[idx,epochstr] = sklearn.metrics.average_precision_score(true_label, pred_prob)
    return fold_eval_dfs_dict
