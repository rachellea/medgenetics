#ensembling.py
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
    accuracy, auroc, avg_precision = evaluate_ensemble(ensemble_lst) #TODO fill in the arguments!!!
    return accuracy, auroc, avg_precision

def train_ensemble(mlp_args, num_ensemble):
     """This function initializes mlps for the ensemble.
        Inputs:
            split, the split object to specify training and testing data
            num_ensemble, the number of mlps in the ensemble"""
     # define a list to store mlps for our ensemble
     ensemble_lst = []

     # initialize ensembles and store in the list
     for i in range(num_ensemble):
         print("In CV fold number", self.cv_num, " out of", self.cv_fold_mlp)
         print("Initializing mlp number", i+1, " out of", num_ensemble, "for ensemble")
         # initialize mlp
         m = mlp.MLP(**mlp_args)
         m.set_up_graph_and_session()
         m.train()

         # store to list
         ensemble_lst.append(m)
     return ensemble_lst

def evaluate_ensemble(ensemble_lst):
    """This function evaluates the test set for the ensemble of mlps
        output: accuracy, auroc, and average precision of the ensemble"""
    print("Evaluating ensemble")
    # true label for calibration
    self.kfold_true_label.append(ensemble_lst[0].selected_labels_true)

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

    # calculate accuracy, auroc, and average precision
    accuracy = metrics.accuracy_score(true_label, pred_label_lst)
    auroc = metrics.roc_auc_score(true_label, pred_prob_lst)
    avg_prec = metrics.average_precision_score(true_label, pred_prob_lst)

    # update list for calibration
    self.mlp_kfold_probability.append(pred_prob_lst)

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
    return accuracy, auroc, avg_prec

##############
# Single MLP #------------------------------------------------------------------
##############
def train_and_eval_one_mlp(mlp_args):
     #Initialize dictionary to store epoch
     epoch_perf = {}
     
    #redefine mlp object with the new split and train it
    m = mlp.MLP(**mlp_args)
    m.set_up_graph_and_session()
    m.train()
    
    # if we are finding best mlp, then we have a dictionary of all the epochs to evaluate
    if self.find_best_mlp:
        for epoch in range(1, self.max_epochs+1):
            pred_prob = m.pred_prob_dict[epoch]
            true_label = m.true_label_dict[epoch]
            pred_label = m.pred_label_dict[epoch]
            acc = metrics.accuracy_score(true_label, pred_label)
            auc = metrics.roc_auc_score(true_label, pred_label)
            avg_prec = metrics.average_precision_score(true_label, pred_label)
            if epoch not in epoch_perf:
                epoch_perf[epoch] = [acc, auc, avg_prec]
            else: 
                epoch_perf[epoch][0] = epoch_perf[epoch][0]+acc
                epoch_perf[epoch][1] = epoch_perf[epoch][1]+auc
                epoch_perf[epoch][2] = epoch_perf[epoch][2]+avg_prec
     
    # update lists for calibration
    self.mlp_kfold_probability.append(m.selected_pred_probs)
    self.kfold_true_label.append(m.selected_labels_true)

    # get the resulting dataframe for this fold
    if self.save_test_out:
        self.fold_df = reformat_output.make_output_human_readable(m.test_out, split.scaler)
    
    return epoch_perf
