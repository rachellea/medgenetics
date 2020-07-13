#plot_for_sens_spec.py

import os
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
import matplotlib
matplotlib.rcParams.update({'font.size': 12})

class MakePanelFigure_SensSpec(object):
    """Make 2 plots x 4 genes x 2 models (LR, MLP):
    plot 1: plot 'threshold' on the X axis, and on the Y axis, plot 4 curves:
        sensitivity, specificity, positive predictive value, negative predictive
        value
    plot 2: plot 'threshold' on the X axis, and on the Y axis plot the number
        of predictions below the threshold (so eventually at a decision
        threshold of 1, 100% of all predictions are represented.)"""
    def __init__(self, base_results_dir):
        self.genes = ['ryr2','kcnq1','kcnh2','scn5a']
        self.define_decision_thresholds() #manually defined; see method
        base_results_dir = base_results_dir
        fig, self.ax = plt.subplots(nrows = 4, ncols = 4, figsize=(16,17.33))
        for idx in range(len(self.genes)):
            self.idx = idx #column number
            gene_name = self.genes[idx]
            results_dir = os.path.join(base_results_dir, gene_name)
            possible_files = os.listdir(results_dir)
            
            for model_name in ['MLP','LR']:
                chosen_file = [y for y in [x for x in possible_files if model_name in x] if 'all_test_out.csv' in y][0]
                print('Making',model_name,'figures based on file',chosen_file)
                
                #self.test_out has columns Consensus,Change,Position,Conservation,
                #SigNoise,Pred_Prob,Pred_Label,True_Label and index of arbitrary ints
                test_out = pd.read_csv(os.path.join(results_dir, chosen_file),header=0,index_col=0)
                true_labels = test_out.loc[:,'True_Label']
                pred_probs = test_out.loc[:,'Pred_Prob']
                
                self.model_name = model_name
                self.calculate_plotting_data(true_labels, pred_probs)
                self.plot_1_four_curves()
                self.plot_2_below_threshold()
                
                #Calculate
                chosen_mystery = [y for y in [x for x in possible_files if model_name in x] if 'all_mysteryAAs_out' in y][0]
                print('Calculating',model_name,'mysteryAA counts based on file',chosen_mystery)
                mystery_out = pd.read_csv(os.path.join(results_dir,chosen_mystery),header=0,index_col=0)
                self.calculate_mystery_counts_relative_to_thresholds(mystery_out, gene_name, model_name)
        
        fig.suptitle('  RYR2                      KCNQ1                    KCNH2                    SCN5A', fontsize=32)
        #Matplotlib tight layout doesn't take into account title so we pass
        #the rect argument
        #https://stackoverflow.com/questions/8248467/matplotlib-tight-layout-doesnt-take-into-account-figure-suptitle
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(base_results_dir, 'Visualization_All_Sens_Spec.pdf'))
        self.mystery_counts_df.to_csv(os.path.join(base_results_dir,'MysteryAA_Counts_By_Threshold.csv'),header=True,index=True)
    
    def define_decision_thresholds(self):
        """Manually chosen decision thresholds for each gene.
        Note that the gene order in the df must match that defined by
        <self.genes>"""
        mlp_thresholds = [[0.4,0.41], #ryr2
                      [0.4,0.6], #kcnq1
                      [0.35,0.4], #kcnh2
                      [0.3,0.31]] #scn5a
        self.mlp_decision_thresholds = pd.DataFrame(mlp_thresholds, columns=['lower','upper'], index=self.genes)
        lr_thresholds = [[0.5,0.51], #ryr2
                      [0.3,0.55], #kcnq1
                      [0.2,0.4], #kcnh2
                      [0.5,0.51]] #scn5a
        self.lr_decision_thresholds = pd.DataFrame(lr_thresholds, columns=['lower','upper'], index=self.genes)
        
        #Also init mystery_counts_df
        self.mystery_counts_df = pd.DataFrame(np.zeros((8,4)),
                index=['MLP_'+g for g in self.genes]+['LR_'+g for g in self.genes],
                columns=['benign_below_thresh','grey_between_thresh','path_above_thresh','total'])
    
    def calculate_plotting_data(self, true_labels, pred_probs):
        """Calculate the data needed to make the plots including sensitivity,
        specificity, ppv, npv, and predictions below a threshold"""
        # initialize thresholds and lists
        self.thresholds = np.linspace(0,1,1000)
        self.sensitivity = []
        self.specificity = []
        self.ppv = []
        self.npv = []
        self.preds_below = []
        
        # get metrics
        for thr in self.thresholds:
            tn, fp, fn, tp = confusion_matrix(true_labels, pred_probs, thr)
            self.sensitivity.append(tp/(tp+fn))
            self.specificity.append(tn/(tn+fp))
            if tp+fp == 0:
                self.ppv.append(0)
            else:
                self.ppv.append(tp/(tp+fp))
            if tn+fn == 0:
                self.npv.append(0)
            else:
                self.npv.append(tn/(tn+fn))
        
            # get number of predictions below threshold
            self.preds_below.append(len([i for i in pred_probs if i < thr]))

    def plot_1_four_curves(self):
        if self.model_name == 'MLP':
            row = 0
            decision_thresholds_df = self.mlp_decision_thresholds
        elif self.model_name == 'LR':
            row = 2
            decision_thresholds_df = self.lr_decision_thresholds
        
        #Plot sens, spec, ppv, npv:
        self.ax[row,self.idx].plot(self.thresholds, self.sensitivity, label='Sensitivity')
        self.ax[row,self.idx].plot(self.thresholds, self.specificity, label='Specificity')
        self.ax[row,self.idx].plot(self.thresholds, self.ppv, label='PPV')
        self.ax[row,self.idx].plot(self.thresholds, self.npv, label='NPV')
        
        #Plot decision thresholds:
        lower = decision_thresholds_df.at[self.genes[self.idx],'lower']
        upper = decision_thresholds_df.at[self.genes[self.idx],'upper']
        self.ax[row,self.idx].axvspan(lower, upper, alpha=0.3, color='black')
        
        self.ax[row,self.idx].legend(loc='lower right', prop={'size': 8})
        self.ax[row,self.idx].set_title(self.model_name+' Metrics Per Threshold')
        self.ax[row,self.idx].set_xlabel('Threshold')
    
    def plot_2_below_threshold(self):
        if self.model_name == 'MLP':
            row = 1
        elif self.model_name == 'LR':
            row = 3
        self.ax[row,self.idx].plot(self.thresholds, self.preds_below)
        self.ax[row,self.idx].set_title(self.model_name+' Predictions Below Threshold')
        self.ax[row,self.idx].set_ylabel('Number of Predictions')
        self.ax[row,self.idx].set_xlabel('Threshold')
    
    def calculate_mystery_counts_relative_to_thresholds(self, mystery_out, gene_name, model_name):
        """Count how many mysteryAAs fall into the benign, grey zone, or
        pathologic regions defined by the decision thresholds"""
        if self.model_name == 'MLP': decision_thresholds_df = self.mlp_decision_thresholds
        elif self.model_name == 'LR': decision_thresholds_df = self.lr_decision_thresholds
        lower = decision_thresholds_df.at[self.genes[self.idx],'lower']
        upper = decision_thresholds_df.at[self.genes[self.idx],'upper']
        pred_probs = mystery_out.loc[:,'Pred_Prob']
        below = np.sum(pred_probs<lower)
        between = np.sum((pred_probs>=lower)&(pred_probs<=upper))
        above = np.sum(pred_probs>upper)
        total = mystery_out.shape[0]
        assert (below+between+above)==total
        index = model_name+'_'+gene_name
        self.mystery_counts_df.at[index,'benign_below_thresh'] = below
        self.mystery_counts_df.at[index,'grey_between_thresh'] = between
        self.mystery_counts_df.at[index,'path_above_thresh'] = above
        self.mystery_counts_df.at[index,'total'] =  total

def confusion_matrix(true_label, pred_prob, threshold=0.5):
    pred_label = [0 if i < threshold else 1 for i in pred_prob]
    tn, fp, fn, tp = metrics.confusion_matrix(true_label, pred_label).ravel()
    return tn, fp, fn, tp

