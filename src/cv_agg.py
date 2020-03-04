#cv_agg.py
#Rachel Ballantyne Draelos
#Functions to aggregate predictions across different folds of cross validation

#Imports
import copy
import numpy as np
import pandas as pd
from scipy import stats
import sklearn.metrics

from . import calibr

###########################################################
# Both Ways: Averaged Performance and General Performance #---------------------
###########################################################
def update_and_save_cv_perf_df(modeling_approach, perf_all_models, all_eval_dfs_dict, all_test_out,
            number_of_cv_folds, num_ensemble, model_args_specific, save_path):
    #First Way: Averaged Peformance
    avg_perf_dict = determine_best_epoch_by_firstway(all_eval_dfs_dict, number_of_cv_folds)
    
    #Second Way: General Performance
    gen_perf_dict = determine_best_epoch_by_secondway(all_test_out)
    
    #Save model details and performance to the perf_all_models dataframe:
    if modeling_approach == 'MLP':
        model_description = {'MLP_Layer':str(model_args_specific['mlp_layers']),
                     'Learning_Rate':model_args_specific['learningrate'],
                     'Dropout_Rate':model_args_specific['dropout'],
                     'Ensemble_Size':num_ensemble}
    elif modeling_approach == 'LR':
        model_description = {'Penalty':model_args_specific['logreg_penalty'],
                             'C':model_args_specific['C'],
                             'Ensemble_Size':num_ensemble}    
    model_summary = {}
    for dictionary in [model_description, avg_perf_dict, gen_perf_dict]:
        for key in list(dictionary.keys()):
            model_summary[key] = dictionary[key]
    perf_all_models_out = perf_all_models.append(model_summary, ignore_index=True)
    
    #Print
    print('\n****** Summary of CV folds for this model setup******')
    print('\taccuracy: folds avg:',model_summary['Mean_Accuracy'],'+/-',model_summary['Std_Accuracy'],'gen:',model_summary['Gen_Accuracy'])
    print('\tAUROC: folds avg:',model_summary['Mean_AUROC'],'+/-',model_summary['Std_AUROC'],'gen:',model_summary['Gen_AUROC'])
    print('\tAverage Precision: folds avg:',model_summary['Mean_Avg_Precision'],'+/-',model_summary['Std_Avg_Precision'],'gen:',model_summary['Gen_Avg_Precision'])
    
    #Save output to csv
    perf_all_models_out.to_csv(save_path, index=False)
    return perf_all_models_out

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
def concat_eval_dfs_dicts(fold_eval_dfs_dict, all_eval_dfs_dict):    
    """Return <fold_eval_dfs_dict> concatenated to the end of <all_eval_dfs_dict>
    These dicts have keys 'accuracy', 'auroc', and 'avg_precision' and values
    that are dataframes with an index of <fold_num> and columns of epoch number
    
    For example:
    >>> fold_eval_dfs_dict['accuracy']
                    epoch_0  epoch_1
        fold_num1  0.892523  0.99823
        fold_num2  0.892019  0.87237
    """
    #Check that columns are equivalent
    for key in ['accuracy','auroc','avg_precision']:
        assert all_eval_dfs_dict[key].columns.values.tolist()==fold_eval_dfs_dict[key].columns.values.tolist()
    #Sum together (we will divide by the number of folds later, in the function
    #determine_best_epoch_by_firstway()
    combined_dfs_dict={}
    combined_dfs_dict['accuracy'] = all_eval_dfs_dict['accuracy'].append(fold_eval_dfs_dict['accuracy'])
    combined_dfs_dict['auroc'] = all_eval_dfs_dict['auroc'].append(fold_eval_dfs_dict['auroc'])
    combined_dfs_dict['avg_precision'] = all_eval_dfs_dict['avg_precision'].append(fold_eval_dfs_dict['avg_precision'])
    return combined_dfs_dict

def determine_best_epoch_by_firstway(all_eval_dfs_dict, number_of_cv_folds):
    """Return the best epoch, and the accuracy, AUROC, and average precision at
    the best epoch. 'Best epoch' means the epoch with highest average precision.
    Performance is calculated as the average of the performance of each fold.
    Note: I know I could do something fancy with the number of true positives
    in each fold affecting the average precision averaging but because I
    have 'Second Way' below I am not doing that here."""
    num_epochs = all_eval_dfs_dict['accuracy'].shape[1]
    
    #eval_dfs_dict has keys for performance metrics and values that are dataframes
    #that have folds as index and epochs as columns.
    #make a different dict where keys are performance metrics and the values
    #are dataframes that have index of epochs and columns Mean and StDev
    summary_df = pd.DataFrame(np.empty((num_epochs,2),dtype='float'),
                              columns=['Mean','StDev'],index=['epoch_'+str(epoch) for epoch in range(num_epochs)])
    summary_dict = {'accuracy':copy.deepcopy(summary_df),
                    'auroc':copy.deepcopy(summary_df),
                    'avg_precision':copy.deepcopy(summary_df)}
    for key in ['accuracy','auroc','avg_precision']:
        eval_df = all_eval_dfs_dict[key]
        for epoch in summary_df.index.values.tolist(): #e.g. epoch_23
            epoch_data = eval_df.loc[:,epoch].values
            summary_dict[key].at[epoch,'Mean'] = np.mean(epoch_data)
            summary_dict[key].at[epoch,'StDev'] = np.std(epoch_data)
    
    #Get best performance
    best_epoch = summary_dict['avg_precision'].nlargest(n=1,columns=['Mean']).index.values.tolist()[0]
    return {'Mean_Best_Epoch':best_epoch,
            'Mean_Accuracy':round(summary_dict['accuracy'].at[best_epoch,'Mean'],4),
            'Std_Accuracy':round(summary_dict['accuracy'].at[best_epoch,'StDev'],4),
            'Mean_AUROC':round(summary_dict['auroc'].at[best_epoch,'Mean'],4),
            'Std_AUROC':round(summary_dict['auroc'].at[best_epoch,'StDev'],4),
            'Mean_Avg_Precision':round(summary_dict['avg_precision'].at[best_epoch,'Mean'],4),
            'Std_Avg_Precision':round(summary_dict['avg_precision'].at[best_epoch,'StDev'],4)}
    
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
def concat_test_outs(fold_test_out, all_test_out):
    """<all_test_out> contains the data and predictions for many of the
    examples in the dataset but not all of them. <fold_test_out> contains some
    of the data and predictions that aren't in <all_test_out> yet. Each of
    these is a dictionary where the keys are epochs and the values are
    dataframes. The value dataframes have columns corresponding to the data
    features and the predictions, and rows for different mutations in the dataset.
    For each epoch concatenate <fold_test_out> into <all_test_out> to make
    <all_test_out> more complete."""
    epochs_to_consider = list(fold_test_out.keys())
    combined_test_out = {}
    for epochstr in epochs_to_consider:
        assert fold_test_out[epochstr].columns.values.tolist()==all_test_out[epochstr].columns.values.tolist()
        combined_test_out[epochstr] = pd.concat([fold_test_out[epochstr], all_test_out[epochstr]], ignore_index=True)
    return combined_test_out

def add_fold_column(fold_test_out, fold_num):
    """Add a column to each df in <fold_test_out> specifying which fold the
    prediction was made in"""
    epochs_to_consider = list(fold_test_out.keys())
    for epochstr in epochs_to_consider:
        fold_test_out[epochstr]['FoldNum'] = 'fold_'+str(fold_num)
    return fold_test_out
    
def determine_best_epoch_by_secondway(all_test_out):
    """Return the best epoch, and the accuracy, AUROC, and average precision at
    the best epoch. 'Best epoch' means the epoch with highest average precision.
    Performance is calculated by using the aggregated predictions for each
    example to freshly calculate all performance metrics across all examples
    in the data set simultaneously."""
    epochs_to_consider = list(all_test_out.keys())
    all_epochs_perf = pd.DataFrame(np.zeros((len(epochs_to_consider),4)),
                                   index = epochs_to_consider,
                                   columns=['accuracy','auroc','avg_precision',
                                            'calibration_slope'])
    
    #Calculate performance at each epoch
    for epochstr in epochs_to_consider:
        true_label = all_test_out[epochstr]['True_Label'].values
        pred_label = all_test_out[epochstr]['Pred_Label'].values
        pred_prob = all_test_out[epochstr]['Pred_Prob'].values
        all_epochs_perf.at[epochstr,'accuracy'] = sklearn.metrics.accuracy_score(true_label, pred_label)
        all_epochs_perf.at[epochstr,'auroc'] = sklearn.metrics.roc_auc_score(true_label, pred_prob)
        all_epochs_perf.at[epochstr,'avg_precision'] = sklearn.metrics.average_precision_score(true_label, pred_prob)
        #Calibration
        fraction_of_positives, mean_predicted_prob = calibr.calibration_curve_new(true_label, pred_prob,n_bins=20,strategy='quantile')
        slope, _, _, _, _ = stats.linregress(mean_predicted_prob,fraction_of_positives)
        all_epochs_perf.at[epochstr,'calibration_slope'] = slope
    
    #Pick out the highest avg precision
    best = all_epochs_perf.nlargest(n=1,columns=['avg_precision'])
    return {'Gen_Best_Epoch':int(best.index.values.tolist()[0].replace('epoch_','')),
            'Gen_Accuracy':round(best.loc[:,'accuracy'].values.tolist()[0],4),
            'Gen_AUROC':round(best.loc[:,'auroc'].values.tolist()[0],4),
            'Gen_Avg_Precision':round(best.loc[:,'avg_precision'].values.tolist()[0],4),
            'Calibration_Slope':round(best.loc[:,'calibration_slope'].values.tolist()[0],4)}
