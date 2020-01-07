#mlp_loops.py
#Functions to train and evaluate ensembles

import numpy as np
import pandas as pd

import mlp
import reformat_output

#############
# Ensembles #-------------------------------------------------------------------
#############
def train_and_eval_ensemble(mlp_args, num_ensemble):
    ensemble_lst = train_ensemble(mlp_args, num_ensemble)
    fold_eval_dfs_dict, fold_test_data_and_preds = evaluate_ensemble(ensemble_lst, mlp_args['num_epochs'], mlp_args['decision_threshold'])
    return fold_eval_dfs_dict, fold_test_data_and_preds

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
         m.set_up_graph_and_session()
         m.train()

         # store to list
         ensemble_lst.append(m)
     return ensemble_lst

def evaluate_ensemble(ensemble_lst, num_epochs, decision_threshold):
    """This function evaluates the test set for the ensemble of mlps
        output: accuracy, auroc, and average precision of the ensemble"""
    print("Evaluating ensemble")
    fold_test_data_and_preds = create_fold_test_data_and_preds(ensemble_lst, decision_threshold)
    fold_eval_dfs_dict = create_fold_eval_dfs_dict(fold_test_data_and_preds, num_epochs)
    return fold_eval_dfs_dict, fold_test_data_and_preds
    
#Helper for evaluate_ensemble()
def create_fold_test_data_and_preds(ensemble_lst, decision_threshold)
    """For ensemble models in <ensemble_lst>, aggregate their predictions
    into a table along with the data itself, and represent the data in
    human-readable form.
    Returns a dataframe with columns ['Consensus','Change','Position',
    'Conservation','SigNoise','Pred_Prob','Pred_Label','True_Label']"""
    # get the list of fold_test_data_and_preds dfs, one for each model in the
    #ensemble.
    clean_df_list = []
    for model in ensemble_lst:
        clean_df_list.append(reformat_output.make_output_human_readable(model.test_out, model.split.scaler))
    
    #Columns Consensus, Change, Position, Conservation, SigNoise, and True_Label
    #should be identical across all models in the ensemble. They are sorted
    #by position
    #The Pred_Prob column should be summed.
    for idx in range(len(clean_df_list)):
        df = clean_df_list[idx]
        if idx == 0:
            fold_test_data_and_preds = df
        else:
            #First check that the different members of the ensemble were
            #applied to the exact same data:
            samecols = ['Consensus','Change','Position','True_Label']
            assert np.equal(df[samecols].values, fold_test_data_and_preds[samecols].values).all()
            closecols = ['Conservation','SigNoise']
            assert np.isclose(df[closecols].values, fold_test_data_and_preds[closecols].values, rtol=1e-4).all()
            #Now sum up the Pred_Prob column:
            fold_test_data_and_preds['Pred_Prob'] += df['Pred_Prob']
    
    #Since we want to store the average Pred_Prob across all members of
    #the ensemble, divide the summed Pred_Prob by the number of models
    #in the ensemble:
    fold_test_data_and_preds['Pred_Prob'].div(len(self.ensemble_lst))
    
    #Now, based on the average Pred_Prob, determine the Pred_Label:
    fold_test_data_and_preds['Pred_Label'] = (fold_test_data_and_preds['Pred_Prob'].values > decision_threshold).astype('int')
    return fold_test_data_and_preds

#Helper for evaluate_ensemble()
def create_fold_eval_dfs_dict(fold_test_data_and_preds, num_epochs):
    """Return the performance of the ensemble models based on the predictions
    and ground truth in <fold_test_data_and_preds>. Note that this is implicitly
    for the max number of epochs only (i.e. the number of epochs specified by
    <num_epochs>)."""
    #Initialize empty fold_eval_dfs_dict
    result_df = pd.DataFrame(data=np.zeros((1, 1)),
                        index = ['Label'], columns = ['epoch_'+str(num_epochs)])
    fold_eval_dfs_dict = {'accuracy':copy.deepcopy(result_df),
        'auroc':copy.deepcopy(result_df),
        'avg_precision':copy.deepcopy(result_df)}
    
    #Calculate performance
    true_label = fold_test_data_and_preds['True_Label'].values
    pred_label = fold_test_data_and_preds['Pred_Label'].values
    pred_prob = fold_test_data_and_preds['Pred_Prob'].values
    fold_eval_dfs_dict['accuracy'].at['Label','epoch_'+str(num_epochs)] = metrics.accuracy_score(true_label, pred_label)
    fold_eval_dfs_dict['auroc'].at['Label','epoch_'+str(num_epochs)] = metrics.roc_auc_score(true_label, pred_prob)
    fold_eval_dfs_dict['avg_precision'].at['Label','epoch_'+str(num_epochs)] = metrics.average_precision_score(true_label, pred_prob)
    return fold_eval_dfs_dict #for only the max number of epochs

##############
# Single MLP #------------------------------------------------------------------
##############
def train_and_eval_one_mlp(mlp_args):
    #redefine mlp object with the new split and train it
    m = mlp.MLP(**mlp_args)
    m.set_up_graph_and_session()
    m.train()
    fold_test_data_and_preds = reformat_output.make_output_human_readable(m.test_out, split.scaler)
    fold_eval_dfs_dict = m.fold_eval_dfs_dict #for all epochs!
    return fold_eval_dfs_dict, fold_test_data_and_preds
