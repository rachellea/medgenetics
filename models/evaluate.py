#evaluate.py
#Rachel Ballantyne Draelos

#Imports
import copy
import time
import bisect
import operator
import itertools
import numpy as np
import pandas as pd
import sklearn.metrics
import matplotlib
matplotlib.use('agg') #so that it does not attempt to display via SSH
import matplotlib.pyplot as plt
plt.ioff() #turn interactive plotting off
import seaborn

################################
# Main Function used in mlp.py #------------------------------------------------
################################
def evaluate_all(eval_dfs_dict, epoch, which_label,
                 true_labels, pred_labels, pred_probs, filename_prefix,
                 total_epochs):
    """Fill out the pandas dataframes in the dictionary <eval_dfs_dict>
    which is created in cnn.py. <epoch> and <which_label> are used to index into
    the dataframe for the metric. Metrics calculated for the provided vectors
    are: accuracy, AUC, partial AUC (threshold 0.2), and average precision.
    If <subjective> is set to True, additional metrics will be calculated
    (confusion matrix, sensitivity, specificity, PPV, NPV.)
    
    Variables:
    <all_eval_results> is a dictionary of pandas dataframes created in cnn.py
    <epoch> is an integer indicating which epoch it is, starting from epoch 1
    <which_label> is a string indicating the overall label being considered e.g. 'MI'
    <true_labels> is a vector of the true labels for all the patients for the
        which_label being considered
    <pred_labels> is a vector of the predicted labels for all the patients
        for the which_label being considered
    <pred_probs> is a vector of the probabilities of the which_label (floats)
        for all the patients for the which_label being considered."""
    #Both accuracy and the confusion matrix depend on the arbitrary cutoff set in
    #when creating the labels_pred vector
    (eval_dfs_dict['accuracy']).loc[which_label, 'epoch_'+str(epoch)] = compute_accuracy(true_labels, pred_labels)
    
    #not using this:
    #confusion_matrix, sensitivity, specificity, ppv, npv = compute_confusion_matrix(true_labels, pred_labels)
    
    #the AUROC, partial AUROC, and average precision only depend on the model
    #because they are using the model's outputted probabilities directly
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true = true_labels,
                                     y_score = pred_probs,
                                     pos_label = 1) #admitted is 1; not admitted is 0
    (eval_dfs_dict['auroc']).loc[which_label, 'epoch_'+str(epoch)] = sklearn.metrics.auc(fpr, tpr)
    (eval_dfs_dict['avg_precision']).loc[which_label, 'epoch_'+str(epoch)] = sklearn.metrics.average_precision_score(true_labels, pred_probs)
    
    if 'Test' in filename_prefix:
        pass
        #plot_precision_recall_curve(true_labels, pred_probs, epoch, filename_prefix, which_label)
        #plot_roc_curve(fpr, tpr, epoch, filename_prefix, which_label)
    return eval_dfs_dict

#######################
# Reporting Functions #---------------------------------------------------------
#######################
def initialize_evaluation_dfs(all_labels, num_epochs):
    """Create empty "eval_dfs_dict"
    Variables
    <all_labels>: a list of strings describing the labels in order
    <num_epochs>: int for total number of epochs"""
    if len(all_labels)==2:
        index = [all_labels[1]]
        numrows = 1
    else:
        index = all_labels
        numrows = len(all_labels)
    #Initialize empty pandas dataframe to store evaluation results across epochs
    result_df = pd.DataFrame(data=np.zeros((numrows, num_epochs)),
                            index = index,
                            columns = ['epoch_'+str(n) for n in range(1,num_epochs+1)])
    #Make eval results dictionaries
    eval_results_valid = {'accuracy':copy.deepcopy(result_df),
        'auroc':copy.deepcopy(result_df),
        'avg_precision':copy.deepcopy(result_df)}
    eval_results_test = copy.deepcopy(eval_results_valid)
    return eval_results_valid, eval_results_test

def save(eval_dfs_dict, filename_prefix):
    """Variables
    <eval_dfs_dict> is a dict of pandas dataframes
    <filename_prefix> is a string"""
    for k in eval_dfs_dict.keys():
        eval_dfs_dict[k].to_csv(filename_prefix+'_'+k+'_Results.csv')

def print_epoch_summary(epoch, which_label, eval_dfs_dict, filename_prefix):
    for k in eval_dfs_dict.keys():
        print('\t\t',k,'=',str(eval_dfs_dict[k].loc[which_label, 'epoch_'+str(epoch)] ))
        
def print_final_summary(eval_dfs_dict, filename_prefix, best_valid_loss_epoch):
    print('***Final Summary for',filename_prefix,'***')
    for k in eval_dfs_dict.keys():
        df = eval_dfs_dict[k]
        print('\tEpoch',best_valid_loss_epoch,k)
        for label in df.index.values:
            print('\t\t',label,':',str(df.loc[label,'epoch_'+str(best_valid_loss_epoch)]))

def report_best_results(eval_dfs_dict_valid, eval_dfs_dict_test,
                             metric_to_judge_valid, grand_file,
                             filename_prefix):
    """Search through the epochs recorded in <eval_dfs_dict_valid> for the df
    corresponding to <metric_to_judge_valid> and determine the best epoch.
    Then report all test set metrics for that epoch in <grand_file>,
    also recording <filename_prefix> in <grand_file>."""
    print('Choosing epoch based on validation set',metric_to_judge_valid)
    chosen_df = eval_dfs_dict_valid[metric_to_judge_valid]
    chosen_epoch = chosen_df.idxmax(axis=1)[0]
    with open(grand_file,'a') as f:
        f.write(filename_prefix+'\tEpoch='+str(chosen_epoch))
        for key in eval_dfs_dict_test:
            metric = round(eval_dfs_dict_test[key][[chosen_epoch]].values.tolist()[0][0], 3)
            f.write('\t'+key+'='+str(metric))
        f.write('\n')


###################################
# First Way: Averaged Performance #---------------------------------------------
###################################
#These functions enable averaging the performance metrics for each fold of cross-validation
def sum_eval_dfs_dicts(new_eval_dfs_dict, old_eval_dfs_dict):
    """Return <new_eval_dfs_dict> + <old_eval_dfs_dict>"""
    #Check that columns and index are equivalent
    for key in ['accuracy','auroc','avg_precision']:
        assert old_eval_dfs_dict[key].columns.values.tolist()==new_eval_dfs_dict[key].columns.values.tolist()
        assert old_eval_dfs_dict[key].index.values.tolist()==new_eval_dfs_dict[key].index.values.tolist()
    
    #Sum together
    combined_dfs_dict={}
    combined_dfs_dict['accuracy'] = old_eval_dfs_dict['accuracy'].values + new_eval_dfs_dict['accuracy'].values
    combined_dfs_dict['auroc'] = old_eval_dfs_dict['auroc'].values + new_eval_dfs_dict['auroc'].values
    combined_dfs_dict['avg_precision'] = old_eval_dfs_dict['avg_precision'].values + new_eval_dfs_dict['avg_precision'].values
    return combined_dfs_dict

def update_and_save_cv_avg_perf_df(perf_all_models_avg, all_eval_dfs_dict, cv_fold_mlp,
                               mlp_args_specific, save_path):
    """Used in cross-validation. Save the performance across all folds."""
    num_epochs = mlp_args_specific['num_epochs']
    
    #Divide by the number of folds to get the average performance across all folds
    for key in ['accuracy','auroc','avg_precision']:
        all_eval_dfs_dict[key].loc[:,:] = all_eval_dfs_dict[key].values / cv_fold_mlp
    
    #Save performance
    best_epoch = None; best_acc = 0; best_auroc = 0; best_avg_prec = 0
    for epoch in range(1, num_epochs+1):
        avg_prec_epoch = all_eval_dfs_dict['avg_precision'].at['Label','epoch_'+str(epoch)]
        if avg_prec_epoch > best_avg_prec:
            best_epoch = epoch
            best_acc = all_eval_dfs_dict['accuracy'].at['Label','epoch_'+str(epoch)]
            best_auroc = all_eval_dfs_dict['auroc'].at['Label','epoch_'+str(epoch)]
            best_avg_prec = avg_prec_epoch
    
    #Save to the perf_all_models_avg:
    model_summary = {'MLP_Layer':mlp_args_specific['mlp_layers'],
                     'Learning_Rate':mlp_args_specific['learningrate'],
                     'Dropout_Rate':mlp_args_specific['dropout'],
                     'Ensemble_Size':num_ensemble,
                     'Best_Epoch':best_epoch,
                     'Accuracy':round(best_acc,4),
                     'AUROC':round(best_auroc,4),
                     'Avg_Precision':round(best_avg_prec,4)}
    perf_all_models_avg.append(model_summary)
    
    #Print
    print('The mean accuracy is', round(best_acc,4))
    print('The mean AUROC is',round(best_auroc,4))
    print('The mean Average Precision is',round(best_avg_prec))
    
    # save output to csv
    perf_all_models_avg.to_csv(save_path, index=False)
    return perf_all_models_avg

###################################
# Second Way: General Performance #---------------------------------------------
###################################
#These functions enable calculating the performance metrics based on the
#aggregated predictions for all examples across all the folds of cross-validation
def sum_test_outs(fold_test_out, all_test_out):
    pass

def update_and_save_cv_gen_perf_df(perf_all_models_avg, all_test_out, cv_fold_mlp,
                               mlp_args_specific, save_path):
    pass

#########################
# Calculation Functions #-------------------------------------------------------
#########################
def compute_accuracy(true_labels, labels_pred):
    """Print and save the accuracy of the model on the dataset"""    
    correct = (true_labels == labels_pred)
    correct_sum = correct.sum()
    return (float(correct_sum)/len(true_labels))

def compute_confusion_matrix(true_labels, labels_pred):
    """Return the confusion matrix"""
    cm = sklearn.metrics.confusion_matrix(y_true=true_labels,
                          y_pred=labels_pred)
    if cm.size < 4: #cm is too small to calculate anything
        return np.nan, np.nan, np.nan, np.nan, np.nan
    true_neg, false_pos, false_neg, true_pos = cm.ravel()
    sensitivity = float(true_pos)/(true_pos + false_neg)
    specificity = float(true_neg)/(true_neg + false_pos)
    ppv = float(true_pos)/(true_pos + false_pos)
    npv = float(true_neg)/(true_neg + false_neg)
    
    return((str(cm).replace("\n","_")), sensitivity, specificity, ppv, npv)

def compute_partial_auroc(fpr, tpr, thresh = 0.2, trapezoid = False, verbose=False):
    fpr_thresh, tpr_thresh = get_fpr_tpr_for_thresh(fpr, tpr, thresh)
    if len(fpr_thresh) < 2:#can't calculate an AUC with only 1 data point
        return np.nan 
    if verbose:
        print('fpr: '+str(fpr))
        print('fpr_thresh: '+str(fpr_thresh))
        print('tpr: '+str(tpr))
        print('tpr_thresh: '+str(tpr_thresh))
    return sklearn.metrics.auc(fpr_thresh, tpr_thresh)

def get_fpr_tpr_for_thresh(fpr, tpr, thresh):
    """The <fpr> and <tpr> are already sorted according to threshold (which is
    sorted from highest to lowest, and is NOT the same as <thresh>; threshold
    is the third output of sklearn.metrics.roc_curve and is a vector of the
    thresholds used to calculate FPR and TPR). This function figures out where
    to bisect the FPR so that the remaining elements are no greater than
    <thresh>. It bisects the TPR in the same place."""
    p = (bisect.bisect_left(fpr, thresh)-1) #subtract one so that the FPR
    #of the remaining elements is NO GREATER THAN <thresh>
    return fpr[: p + 1], tpr[: p + 1]

######################
# Plotting Functions #----------------------------------------------------------
######################
def plot_precision_recall_curve(true_labels, pred_probs, epoch, filename_prefix, which_label):
    """<filename_prefix> e.g. MLP_Test"""
    #http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
    average_precision = sklearn.metrics.average_precision_score(true_labels, pred_probs)
    precision, recall, _ = sklearn.metrics.precision_recall_curve(true_labels, pred_probs)
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
              average_precision))
    if len(which_label)>0:
        which_label = '_'+which_label
    plt.savefig(filename_prefix+'_PR_Curve'+which_label+'_Epoch'+str(epoch)+'.pdf')
    plt.close()

def plot_roc_curve(fpr, tpr, epoch, filename_prefix, which_label):
    #http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    if len(which_label)>0:
        which_label = '_'+which_label
    plt.savefig(filename_prefix+'_ROC_Curve'+which_label+'_Epoch'+str(epoch)+'.pdf')
    plt.close()

def plot_learning_curves(training_loss, valid_loss, filename_prefix):
    """Variables
    <training_loss> and <valid_loss> are numpy arrays with one numerical entry
    for each epoch quanitfying the loss for that epoch."""
    x = np.arange(0,len(training_loss))
    plt.plot(x, training_loss, color='blue', lw=2, label='train')
    plt.plot(x, valid_loss, color='green',lw = 2, label='valid')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc='lower right')
    plt.savefig(filename_prefix+'_Learning_Curves.pdf')
    plt.close()
    
    #save numpy array
    #np.save('training_loss.npy',training_loss)
    #np.save('valid_loss.npy',valid_loss)

def plot_heatmap(numeric_array, center, filename_prefix, yticklabels):
    """Save a heatmap based on numeric_array"""
    seaborn.set()
    seaplt = (seaborn.heatmap(numeric_array,
                           center=center,
                           yticklabels=yticklabels)).get_figure()
    seaplt.savefig(filename_prefix+'_Heatmap.pdf')
    seaplt.clf()
    