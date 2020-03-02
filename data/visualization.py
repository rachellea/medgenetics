#visualization.py

import os
import pandas as pd
import sklearn.metrics
from sklearn.calibration import *

import matplotlib
matplotlib.use('agg') #so that it does not attempt to display via SSH
import matplotlib.pyplot as plt
plt.ioff() #turn interactive plotting off
import matplotlib.lines as mlines

class MakeAllFigures(object):
    def __init__(self, gene_name, results_dir):
        
        #TODO: modify this so that it does figures for both modeling approach, LR and MLP!
        
        self.results_dir = results_dir
        possible_files = os.listdir(results_dir)
        self.chosen_file = [y for y in [x for x in possible_files if modeling_approach in x] if 'all_test_out.csv' in y][0]
        #self.test_out has columns Consensus,Change,Position,Conservation,
        #SigNoise,Pred_Prob,Pred_Label,True_Label and index of arbitrary ints
        test_out = pd.read_csv(os.path.join(results_dir, self.chosen_file),header=0,index_col=0)
        self.true_labels = test_out.loc[:,'True_Label']
        self.pred_probs = test_out.loc[:,'Pred_Prob']
        
        #Plot
        self.plot_precision_recall_curve()
        self.plot_roc_curve()
        self.plot_calibration_curve()
    
    def plot_precision_recall_curve(self):
        #http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
        average_precision = sklearn.metrics.average_precision_score(self.true_labels, self.pred_probs)
        precision, recall, _ = sklearn.metrics.precision_recall_curve(self.true_labels, self.pred_probs)
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
        plt.savefig(os.path.join(self.results_dir, self.chosen_file.replace('all_test_out.csv','PR_Curve.pdf')))
        plt.close()
    
    def plot_roc_curve(self):
        #http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(self.true_labels,self.pred_probs,pos_label = 1)
        roc_auc = sklearn.metrics.auc(fpr, tpr)
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.savefig(os.path.join(self.results_dir, self.chosen_file.replace('all_test_out.csv','ROC_Curve.pdf')))
        plt.close()
    
    def plot_calibration_curve(self):
        #https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html
        #https://scikit-learn.org/stable/auto_examples/calibration/plot_compare_calibration.html#sphx-glr-auto-examples-calibration-plot-compare-calibration-py
        fraction_of_positives, mean_predicted_prob = calibration_curve_new(self.true_labels,self.pred_probs,n_bins=20,strategy='quantile')
        fig, ax = plt.subplots()
        plt.plot(mean_predicted_prob, fraction_of_positives, marker='o', markersize=3, linewidth=1) #label=modeling_approach #TODO plot logreg and mlp in same plots
        plt.savefig(os.path.join(self.results_dir, self.chosen_file.replace('all_test_out.csv','Calibration_Curve.pdf')))
        
        # ax_lim = max(max(logreg_x), max(mlp_x), max(logreg_y), max(mlp_y))
        # ax.set_xlim(left=0, right=ax_lim + 0.05)
        # ax.set_ylim(bottom=0, top=ax_lim + 0.05)
        # line = mlines.Line2D([0, 1], [0, 1], color='black')
        # transform = ax.transAxes
        # line.set_transform(transform)
        # ax.add_line(line)
        # fig.suptitle('Calibration plot for ' + self.gene_name)
        # ax.set_xlabel('Predicted probability')
        # ax.set_ylabel('True probability')
        # plt.legend()

# New Calibration Curve Function from Sklearn
# Pasted from https://github.com/scikit-learn/scikit-learn/blob/b194674c4/sklearn/calibration.py#L522
# on 3/2/2020
def calibration_curve_new(y_true, y_prob, normalize=False, n_bins=5,
                      strategy='uniform'):
    """Compute true and predicted probabilities for a calibration curve.
    The method assumes the inputs come from a binary classifier.
    Calibration curves may also be referred to as reliability diagrams.
    Read more in the :ref:`User Guide <calibration>`.
    Parameters
    ----------
    y_true : array, shape (n_samples,)
        True targets.
    y_prob : array, shape (n_samples,)
        Probabilities of the positive class.
    normalize : bool, optional, default=False
        Whether y_prob needs to be normalized into the bin [0, 1], i.e. is not
        a proper probability. If True, the smallest value in y_prob is mapped
        onto 0 and the largest one onto 1.
    n_bins : int
        Number of bins. A bigger number requires more data. Bins with no data
        points (i.e. without corresponding values in y_prob) will not be
        returned, thus there may be fewer than n_bins in the return value.
    strategy : {'uniform', 'quantile'}, (default='uniform')
        Strategy used to define the widths of the bins.
        uniform
            All bins have identical widths.
        quantile
            All bins have the same number of points.
    Returns
    -------
    prob_true : array, shape (n_bins,) or smaller
        The true probability in each bin (fraction of positives).
    prob_pred : array, shape (n_bins,) or smaller
        The mean predicted probability in each bin.
    References
    ----------
    Alexandru Niculescu-Mizil and Rich Caruana (2005) Predicting Good
    Probabilities With Supervised Learning, in Proceedings of the 22nd
    International Conference on Machine Learning (ICML).
    See section 4 (Qualitative Analysis of Predictions).
    """
    y_true = column_or_1d(y_true)
    y_prob = column_or_1d(y_prob)
    check_consistent_length(y_true, y_prob)

    if normalize:  # Normalize predicted values into interval [0, 1]
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
    elif y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError("y_prob has values outside [0, 1] and normalize is "
                         "set to False.")

    labels = np.unique(y_true)
    if len(labels) > 2:
        raise ValueError("Only binary classification is supported. "
                         "Provided labels %s." % labels)
    y_true = label_binarize(y_true, labels)[:, 0]

    if strategy == 'quantile':  # Determine bin edges by distribution of data
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_prob, quantiles * 100)
        bins[-1] = bins[-1] + 1e-8
    elif strategy == 'uniform':
        bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
    else:
        raise ValueError("Invalid entry to 'strategy' input. Strategy "
                         "must be either 'quantile' or 'uniform'.")

    binids = np.digitize(y_prob, bins) - 1

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    prob_true = (bin_true[nonzero] / bin_total[nonzero])
    prob_pred = (bin_sums[nonzero] / bin_total[nonzero])

    return prob_true, prob_pred


