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
    accuracy, auroc, avg_precision = evaluate_ensemble(ensemble_lst, mlp_args['num_epochs'])
    return accuracy, auroc, avg_precision

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

def evaluate_ensemble(ensemble_lst, num_epochs):
    """This function evaluates the test set for the ensemble of mlps
        output: accuracy, auroc, and average precision of the ensemble"""
    print("Evaluating ensemble")
    #Initialize eval_results_test
    result_df = pd.DataFrame(data=np.zeros((1, 1)),
                        index = ['Label'], columns = ['epoch_'+str(num_epochs)])
    eval_results_test = {'accuracy':copy.deepcopy(result_df),
        'auroc':copy.deepcopy(result_df),
        'avg_precision':copy.deepcopy(result_df)}
    
    # true label for calibration
    self.kfold_true.append(ensemble_lst[0].selected_labels_true)

    # get the true label
    true_label = ensemble_lst[0].selected_labels_true
    pred_label_lst = []
    pred_prob_lst = []
    for i in range(len(true_label)):
        pred_label = []
        pred_prob = 0
        # for each mlp, get the predicted label and predicted proba
        for j in range(len(ensemble_lst)):
            m = ensemble_lst[j]
            pred_label.append(m.selected_pred_labels[i])
            #print("Adding the predicted probability: ", m.selected_pred_probs.shape)
            pred_prob += m.selected_pred_probs[i]
        # for predicted labels, get the most frequent predicted label
        if pred_label.count(0) > pred_label.count(1):
            pred_label_lst.append(0)
        else:
            pred_label_lst.append(1)
        # for predicted probability, get the average predicted probability
        pred_prob_lst.append(pred_prob/len(ensemble_lst))

    # store accuracy, auroc, and average precision
    eval_results_test['accuracy'].at['Label','epoch_'+str(num_epochs)] = metrics.accuracy_score(true_label, pred_label_lst)
    eval_results_test['auroc'].at['Label','epoch_'+str(num_epochs)] = metrics.roc_auc_score(true_label, pred_prob_lst)
    eval_results_test['avg_precision'].at['Label','epoch_'+str(num_epochs)] = metrics.average_precision_score(true_label, pred_prob_lst)
    
    # update list for calibration
    self.kfold_prob.append(pred_prob_lst)

    # if we are saving test output
    if self.save_test_out:
        # get the list of cleanedup df
        clean_df = []
        for model in ensemble_lst:
            clean_df.append(reformat_output.make_output_human_readable(model.test_out, model.split.scaler))
        # merge the dataframes
        curr_df = ""
        for df in clean_df:
            if len(curr_df) == 0:
                curr_df = df
            else:
                curr_df = pd.merge(curr_df, df, on=['Consensus', 'Position', 'Change'])
                curr_df['Pred_Prob'] = curr_df['Pred_Prob_x'] + curr_df['Pred_Prob_y']
                curr_df = curr_df.drop(['Pred_Prob_x', 'Pred_Prob_y'], axis=1)
        # divide pred prob by number of mlps in this ensemble
        curr_df['Pred_Prob'] = curr_df['Pred_Prob'].div(len(self.ensemble_lst)).round(3)
        # for debugging purposes only
        print("The resulting df has", curr_df.isnull().sum(), " null values")

        self.fold_df = curr_df
    return eval_results_test

##############
# Single MLP #------------------------------------------------------------------
##############
def train_and_eval_one_mlp(mlp_args, kfold_prob, kfold_true):
    #redefine mlp object with the new split and train it
    m = mlp.MLP(**mlp_args)
    m.set_up_graph_and_session()
    m.train()
    
    # update lists for calibration
    kfold_prob.append(m.selected_pred_probs)
    kfold_true.append(m.selected_labels_true)
    
    # get the resulting dataframe for this fold
    if self.save_test_out:
        self.fold_df = reformat_output.make_output_human_readable(m.test_out, split.scaler)
    
    #return the eval_dfs_dict for the test set:
    return m.eval_results_test, kfold_prob, kfold_true
