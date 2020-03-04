#visualization.py

import os
import numpy as np
import pandas as pd
from scipy import stats
import sklearn.metrics

import matplotlib
matplotlib.use('agg') #so that it does not attempt to display via SSH
import matplotlib.pyplot as plt
plt.ioff() #turn interactive plotting off
import matplotlib.lines as mlines

from . import calibr

class MakeAllFigures(object):
    def __init__(self, gene_name, results_dir):
        self.gene_name = gene_name
        self.results_dir = results_dir
        possible_files = os.listdir(results_dir)
        self.chosen_file_LR = [y for y in [x for x in possible_files if 'LR' in x] if 'all_test_out.csv' in y][0]
        self.chosen_file_MLP = [y for y in [x for x in possible_files if 'MLP' in x] if 'all_test_out.csv' in y][0]
        print('Making figures based on LR file',self.chosen_file_LR,'and MLP file',self.chosen_file_MLP)
                
        #self.test_out has columns Consensus,Change,Position,Conservation,
        #SigNoise,Pred_Prob,Pred_Label,True_Label and index of arbitrary ints
        test_out_LR = pd.read_csv(os.path.join(results_dir, self.chosen_file_LR),header=0,index_col=0)
        self.true_labels_LR = test_out_LR.loc[:,'True_Label']
        self.pred_probs_LR = test_out_LR.loc[:,'Pred_Prob']
        test_out_MLP = pd.read_csv(os.path.join(results_dir, self.chosen_file_MLP),header=0,index_col=0)
        self.true_labels_MLP = test_out_MLP.loc[:,'True_Label']
        self.pred_probs_MLP = test_out_MLP.loc[:,'Pred_Prob']
        
        #Plot characteristics
        #COlors: https://matplotlib.org/tutorials/colors/colors.html
        self.LR_color = 'crimson'
        self.LR_linestyle = 'solid'
        self.MLP_color = 'royalblue'
        self.MLP_linestyle = 'solid'
        self.neutral_color = 'k'
        self.neutral_linestyle = 'dashed'
        self.lw = 2
        
        #Plot
        self.plot_precision_recall_curve()
        self.plot_roc_curve()
        self.plot_calibration_curve()
    
    def plot_precision_recall_curve(self):
        #http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
        average_precision_LR = sklearn.metrics.average_precision_score(self.true_labels_LR, self.pred_probs_LR)
        precision_LR, recall_LR, _ = sklearn.metrics.precision_recall_curve(self.true_labels_LR, self.pred_probs_LR)
        LR_line, = plt.step(recall_LR, precision_LR, color=self.LR_color, alpha=0.2, where='post',linewidth=self.lw,linestyle=self.LR_linestyle)
        plt.fill_between(recall_LR, precision_LR, step='post', alpha=0.2, color=self.LR_color)
        
        average_precision_MLP = sklearn.metrics.average_precision_score(self.true_labels_MLP, self.pred_probs_MLP)
        precision_MLP, recall_MLP, _ = sklearn.metrics.precision_recall_curve(self.true_labels_MLP, self.pred_probs_MLP)
        MLP_line, = plt.step(recall_MLP, precision_MLP, color=self.MLP_color, alpha=0.2, where='post',linewidth=self.lw,linestyle=self.MLP_linestyle)
        plt.fill_between(recall_MLP, precision_MLP, step='post', alpha=0.2, color=self.MLP_color)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall Curves')
        
        plt.legend([LR_line, MLP_line], ['LR, AP=%0.2f' % average_precision_LR, 'MLP, AP=%0.2f' % average_precision_MLP], loc='lower right')
        plt.savefig(os.path.join(self.results_dir, self.gene_name+'_Best_Models_PR_Curves.pdf'))
        plt.close()
    
    def plot_roc_curve(self):
        #http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
        lw = 2
        fpr_LR, tpr_LR, _ = sklearn.metrics.roc_curve(self.true_labels_LR,self.pred_probs_LR,pos_label = 1)
        roc_auc_LR = sklearn.metrics.auc(fpr_LR, tpr_LR)
        LR_line, = plt.plot(fpr_LR, tpr_LR, color=self.LR_color, lw=self.lw, linestyle = self.LR_linestyle)
        
        fpr_MLP, tpr_MLP, _ = sklearn.metrics.roc_curve(self.true_labels_MLP,self.pred_probs_MLP,pos_label = 1)
        roc_auc_MLP = sklearn.metrics.auc(fpr_MLP, tpr_MLP)
        MLP_line, = plt.plot(fpr_MLP, tpr_MLP, color=self.MLP_color, lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_MLP, linestyle = self.MLP_linestyle)
        
        plt.plot([0, 1], [0, 1], color=self.neutral_color, lw=self.lw, linestyle=self.neutral_linestyle) #diagonal line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristics')
        plt.legend([LR_line, MLP_line], ['LR, AUC=%0.2f' % roc_auc_LR, 'MLP, AUC=%0.2f' % roc_auc_MLP], loc='lower right')
        plt.savefig(os.path.join(self.results_dir, self.gene_name+'_Best_Models_ROC_Curves.pdf'))
        plt.close()
    
    def plot_calibration_curve(self):
        #https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html
        #https://scikit-learn.org/stable/auto_examples/calibration/plot_compare_calibration.html#sphx-glr-auto-examples-calibration-plot-compare-calibration-py
        fig, ax = plt.subplots()
        plt.plot([0, 1], [0, 1], color=self.neutral_color, lw=self.lw, linestyle=self.neutral_linestyle) #diagonal line
        fraction_of_positives_LR, mean_predicted_prob_LR = calibr.calibration_curve_new(self.true_labels_LR,
                            self.pred_probs_LR, n_bins=20, strategy='quantile')
        LR_line, = plt.plot(mean_predicted_prob_LR, fraction_of_positives_LR,
                 color = self.LR_color, marker='o', markersize=3, linewidth=self.lw, linestyle = self.LR_linestyle)
        
        fraction_of_positives_MLP, mean_predicted_prob_MLP = calibr.calibration_curve_new(self.true_labels_MLP,
                            self.pred_probs_MLP, n_bins=20, strategy='quantile')
        MLP_line, = plt.plot(mean_predicted_prob_MLP, fraction_of_positives_MLP,
                 color = self.MLP_color, marker='o', markersize=3, linewidth=self.lw, linestyle = self.LR_linestyle)
        
        #Calculate the calibration slopes using a best fit line
        LR_slope, LR_intercept, _, _, _ = stats.linregress(mean_predicted_prob_LR, fraction_of_positives_LR)
        MLP_slope, MLP_intercept, _, _, _ = stats.linregress(mean_predicted_prob_MLP, fraction_of_positives_MLP)
        
        #Plot the calibration best-fit lines
        abline(LR_slope, LR_intercept, self.LR_color)
        abline(MLP_slope, MLP_intercept, self.MLP_color)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curves')
        plt.legend([LR_line, MLP_line], ['LR, Slope=%0.2f' % LR_slope, 'MLP, Slope=%0.2f' % MLP_slope], loc='lower right')
        plt.savefig(os.path.join(self.results_dir, self.gene_name+'_Best_Models_Calibration_Curves.pdf'))
        plt.close()

#Plot a line based on a slope and intercept in matplotlib
def abline(slope, intercept, color):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, color = color, linewidth = 1, linestyle = 'dotted')

