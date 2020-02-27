#cv_agg.py
#Rachel Ballantyne Draelos
#Functions to aggregate predictions across different folds of cross validation

#Imports
import numpy as np
import pandas as pd
import sklearn.metrics

###########################################################
# Both Ways: Averaged Performance and General Performance #---------------------
###########################################################
def update_and_save_cv_perf_df(perf_all_models, all_eval_dfs_dict, all_test_out,
            cv_fold_mlp, mlp_args_specific, save_path):
    #First Way: Averaged Peformance
    best_epoch_avg, best_acc_avg, best_auroc_avg, best_avg_prec_avg = calculate_avg_perf(
        all_eval_dfs_dict, cv_fold_mlp, mlp_args_specific['num_epochs'])
    
    #Second Way: General Performance
    best_epoch_gen, best_acc_gen, best_auroc_gen, best_avg_prec_gen = calculate_gen_perf(
        all_test_out, mlp_args_specific['num_epochs'])
    
    #Save to the perf_all_models dataframe:
    model_summary = {'MLP_Layer':str(mlp_args_specific['mlp_layers']),
                     'Learning_Rate':mlp_args_specific['learningrate'],
                     'Dropout_Rate':mlp_args_specific['dropout'],
                     'Ensemble_Size':num_ensemble,
                     
                     'Mean_Best_Epoch':best_epoch_avg,
                     'Mean_Accuracy':round(best_acc_avg,4),
                     'Mean_AUROC':round(best_auroc_avg,4),
                     'Mean_Avg_Precision':round(best_avg_prec_avg,4),
                     
                     'Gen_Best_Epoch':best_epoch_gen,
                     'Gen_Accuracy':round(best_acc_gen,4),
                     'Gen_AUROC':round(best_auroc_gen,4),
                     'Gen_Avg_Precision':round(best_avg_prec_gen,4),
                     }
    perf_all_models_avg = perf_all_models_avg.append(model_summary)
    
    #Print
    print('The mean accuracy is', round(best_acc,4))
    print('The mean AUROC is',round(best_auroc,4))
    print('The mean Average Precision is',round(best_avg_prec))
    
    # save output to csv
    perf_all_models_avg.to_csv(save_path, index=False)
    return perf_all_models_avg

###################################
# First Way: Averaged Performance #---------------------------------------------
###################################
#Now we are done with the folds of cross-validation and we need to calculate
#performance. We will do this in two ways:
#FIRST WAY: 'Averaged Perf':averaging the performance of each fold.
#This will help us see if there is a lot of variability in performance
#between folds. Technically if we're using average precision we also
#would need to account for the number of positives in each fold to do
#a proper weighting. But due to computational cost we are not going to
#do that.
#These functions enable averaging the performance metrics for each fold of
#cross-validation
def sum_eval_dfs_dicts(fold_eval_dfs_dict, all_eval_dfs_dict):    
    """Return <fold_eval_dfs_dict> + <all_eval_dfs_dict>"""
    #Check that columns and index are equivalent
    for key in ['accuracy','auroc','avg_precision']:
        assert all_eval_dfs_dict[key].columns.values.tolist()==fold_eval_dfs_dict[key].columns.values.tolist()
        assert all_eval_dfs_dict[key].index.values.tolist()==fold_eval_dfs_dict[key].index.values.tolist()
    
    #Sum together
    combined_dfs_dict={}
    combined_dfs_dict['accuracy'] = all_eval_dfs_dict['accuracy'].values + fold_eval_dfs_dict['accuracy'].values
    combined_dfs_dict['auroc'] = all_eval_dfs_dict['auroc'].values + fold_eval_dfs_dict['auroc'].values
    combined_dfs_dict['avg_precision'] = all_eval_dfs_dict['avg_precision'].values + fold_eval_dfs_dict['avg_precision'].values
    return combined_dfs_dict

def calculate_avg_perf(all_eval_dfs_dict, cv_fold_mlp, num_epochs):
    """Used in cross-validation. Save the performance of this particular model
    to <perf_all_models_avg>. Performance is calculated as the average of the
    performance of each fold.
    Note: I know I could do something fancy with the number of true positives
    in each fold affecting the average precision averaging but because I
    have 'Second Way' below I am not doing that here."""
    #Divide by the number of folds to get the average performance across all folds
    for key in ['accuracy','auroc','avg_precision']:
        all_eval_dfs_dict[key].loc[:,:] = all_eval_dfs_dict[key].values / cv_fold_mlp
    
    #Calculate performance
    best_epoch = None; best_acc = 0; best_auroc = 0; best_avg_prec = 0
    for epoch in range(1, num_epochs+1):
        avg_prec_epoch = all_eval_dfs_dict['avg_precision'].at['Label','epoch_'+str(epoch)]
        if avg_prec_epoch > best_avg_prec:
            best_epoch = epoch
            best_acc = all_eval_dfs_dict['accuracy'].at['Label','epoch_'+str(epoch)]
            best_auroc = all_eval_dfs_dict['auroc'].at['Label','epoch_'+str(epoch)]
            best_avg_prec = avg_prec_epoch
    return best_epoch, best_acc, best_auroc, best_avg_prec
 
###################################
# Second Way: General Performance #---------------------------------------------
###################################
#SECOND WAY: 'General Performance': we concatenate the actual predictions
#for each example in the test set of each fold so that we get a
#predicted probability for every example in the data set. Then we
#calculate the performance metrics on all this data at the same time.
#This implicitly does all of the weighting correctly based on number of
#true positives, number of false positives, and so on. But it hides
#any 'variability between folds' that may exist with this model setup.
#These functions enable calculating the performance metrics based on the
#aggregated predictions for all examples across all the folds of cross-validation
def concat_test_outs(fold_test_out, all_test_out, num_epochs):
    """<all_test_out> contains the data and predictions for many of the
    examples in the dataset but not all of them. <fold_test_out> contains some
    of the data and predictions that aren't in <all_test_out> yet. Each of
    these is a dictionary of dataframes where the keys are epochs. For each
    epoch concatenate <fold_test_out> into <all_test_out> to make <all_test_out>
    more complete."""
    combined_test_out = {}
    for epoch in range(1,num_epochs+1):
        key = 'epoch_'+str(epoch)
        assert fold_test_out[key].columns.values.tolist()==all_test_out[key].columns.values.tolist()
        combined_test_out[key] = pd.concat([fold_test_out[key], all_test_out[key]], ignore_index=True)
    return combined_test_out

def update_and_save_cv_gen_perf_df(all_test_out, num_epochs):
    """Used in cross-validation. Save the performance of this particular model
    to <perf_all_models_gen>. Performance is calculated by using the aggregated
    predictions for each example to freshly calculate all performance metrics
    across all examples in the data set simultaneously."""
    all_epochs_perf = pd.DataFrame(np.zeros(num_epochs,3),
                                   index = [x for x in range(1,num_epochs+1)],
                                   columns=['accuracy','auroc','avg_precision'])
    
    #Calculate performance at each epoch
    for epoch in range(1,num_epochs+1):
        true_label = all_test_out['epoch_'+str(epoch)]['True_Label'].values
        pred_label = all_test_out['epoch_'+str(epoch)]['Pred_Label'].values
        pred_prob = all_test_out['epoch_'+str(epoch)]['Pred_Prob'].values
        all_epochs_perf.at[epoch,'accuracy'] = metrics.accuracy_score(true_label, pred_label)
        all_epochs_perf.at[epoch,'auroc'] = metrics.roc_auc_score(true_label, pred_prob)
        all_epochs_perf.at[epoch,'avg_precision'] = metrics.average_precision_score(true_label, pred_prob)
    
    #Pick out the highest avg precision
    best = all_epochs_perf.nlargest(n=1,columns=['avg_precision'])
    best_epoch = best.index.values.tolist()[0]
    best_acc = best.at[:,'accuracy'].values.tolist()[0]
    best_auroc = best.at[:,'auroc'].values.tolist()[0]
    best_avg_prec = best.at[:,'avg_precision'].values.tolist()[0]
    return best_epoch, best_acc, best_auroc, best_avg_prec
    
