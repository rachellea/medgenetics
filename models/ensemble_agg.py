#ensemble_agg.py
#Functions to train and evaluate ensembles and aggregate predictions across
#different models applied to the same data

import numpy as np
import pandas as pd

from . import mlp
from . import reformat_output

#############
# Ensembles #-------------------------------------------------------------------
#############
def train_and_eval_ensemble(mlp_args, num_ensemble):
    """Train and evaluate an ensemble consisting of <num_ensemble> MLPs,
    each trained according to <mlp_args>.
    When <num_ensemble>==1 then this is the case where only one MLP is trained"""
    assert num_ensemble >= 1
    assert isinstance(num_ensemble, int)
    #Train
    ensemble_lst = train_ensemble(mlp_args, num_ensemble)
    #Evaluate
    fold_test_out = create_fold_test_out(ensemble_lst, mlp_args['num_epochs'], mlp_args['decision_threshold'])
    fold_eval_dfs_dict = create_fold_eval_dfs_dict(fold_test_out, mlp_args['num_epochs'])
    #note that fold_test_out contains the data and the predictions, while
    #fold_eval_dfs_dict contains the performance
    return fold_test_out, fold_eval_dfs_dict

def train_ensemble(mlp_args, num_ensemble):
     """This function initializes mlps for the ensemble.
        Variables:
            <split>: the split object to specify training and testing data
            <num_ensemble>: the number of mlps in the ensemble"""
     # define a list to store mlps for our ensemble
     ensemble_lst = []

     # initialize ensembles and store in the list
     for i in range(num_ensemble):
         print("Initializing mlp number", i+1, " out of", num_ensemble, "for ensemble")
         # initialize mlp
         m = mlp.MLP(**mlp_args)
         m.run_all() #open session, train and test, close session
         ensemble_lst.append(m)
     return ensemble_lst

#Part of evaluating the ensemble
def create_fold_test_out(ensemble_lst, num_epochs, decision_threshold):
    """For ensemble models in <ensemble_lst>, aggregate their predictions.
    Returns a dictionary of dataframes.
    The dictionary keys are 'epoch_1', 'epoch_2',... and so on.
    The values are dataframes with columns ['Consensus_etc','Change_etc','Position',
    'Conservation','SigNoise','Pred_Prob','Pred_Label','True_Label']
    The data has NOT been put in human-readable format yet, so for example the
    Consensus is many columns (one-hot encoding) and the Position is
    normalized (not integer positions.)"""
    #test_out is produced by the MLP class. test_out is a pandas dataframe
    #with the data itself, the pred probs, the pred labels, and the true labels
    #test_out has columns self.train_set.data_meanings+['Pred_Prob','Pred_Label','True_Label']
    #Gather all of the test_outs for all the models in the ensemble.
    test_out_collection = []
    for model in ensemble_lst:
        test_out_collection.append(model.test_out)
    
    #test_out columns Consensus_etc, Change_etc, Position, Conservation,
    #SigNoise, and True_Label should be identical across all models in the ensemble.
    #Rows are sorted by position. The Pred_Prob column should be summed.
    for idx in range(len(test_out_collection)):
        test_out = test_out_collection[idx]
        if idx == 0:
            fold_test_out = test_out
        else:
            for epoch in range(1,num_epochs+1):
                df = test_out['epoch_'+str(epoch)] #this df
                fold_df = fold_test_out['epoch_'+str(epoch)] #the aggregated df
                
                #Sanity check: ensure that the different members of the ensemble were
                #applied to the exact same data:
                all_cols = df.columns.values.tolist()
                samecols = [x for x in all_cols if 'Consensus' in x]+[x for x in all_cols if 'Change' in x]+['True_Label']
                assert np.equal(df[samecols].values, fold_df[samecols].values).all()
                closecols = ['Position','Conservation','SigNoise']
                assert np.isclose(df[closecols].values, fold_df[closecols].values, rtol=1e-4).all()
                
                #Now sum up the Pred_Prob column:
                fold_test_out['epoch_'+str(epoch)].loc[:,'Pred_Prob'] += df['Pred_Prob']
    
    #Since we want to store the average Pred_Prob across all members of
    #the ensemble, divide the summed Pred_Prob by the number of models
    #in the ensemble, and then determine the Pred_Label:
    for epoch in range(1,num_epochs+1):
        fold_test_out['epoch_'+str(epoch)].loc[:,'Pred_Prob'].div(len(self.ensemble_lst))
        fold_test_out['epoch_'+str(epoch)].loc[:,'Pred_Label'] = (fold_test_out['epoch_'+str(epoch)].loc[:,'Pred_Prob'].values > decision_threshold).astype('int')
    return fold_test_out

#Part of evaluating the ensemble
def create_fold_eval_dfs_dict(fold_test_out, num_epochs):
    """Return the performance of the ensemble models for all epochs
    based on the predictions and ground truth in <fold_test_out>."""
    #Initialize empty fold_eval_dfs_dict
    result_df = pd.DataFrame(data=np.zeros((1, num_epochs)),
                            index = ['Label'],
                            columns = ['epoch_'+str(n) for n in range(1,num_epochs+1)])
    fold_eval_dfs_dict = {'accuracy':copy.deepcopy(result_df),
        'auroc':copy.deepcopy(result_df),
        'avg_precision':copy.deepcopy(result_df)}
    
    #Calculate performance
    for epoch in range(1,num_epochs+1):
        true_label = fold_test_out['epoch_'+str(epoch)]['True_Label'].values
        pred_label = fold_test_out['epoch_'+str(epoch)]['Pred_Label'].values
        pred_prob = fold_test_out['epoch_'+str(epoch)]['Pred_Prob'].values
        fold_eval_dfs_dict['accuracy'].at['Label','epoch_'+str(epoch)] = metrics.accuracy_score(true_label, pred_label)
        fold_eval_dfs_dict['auroc'].at['Label','epoch_'+str(epoch)] = metrics.roc_auc_score(true_label, pred_prob)
        fold_eval_dfs_dict['avg_precision'].at['Label','epoch_'+str(epoch)] = metrics.average_precision_score(true_label, pred_prob)
    return fold_eval_dfs_dict
