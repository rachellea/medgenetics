#visualization.py

from scipy import stats

import matplotlib
matplotlib.use('agg') #so that it does not attempt to display via SSH
import matplotlib.pyplot as plt
plt.ioff() #turn interactive plotting off
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

class MakeAllFigures():
    def __init__(gene_name, modeling_approach, results_dir):
        possible_files = os.listdir(results_dir)
        chosen_file = [y for y in [x for x in possible_files if modeling_approach in x] if 'all_test_out.csv' in y]
        #self.test_out has columns Consensus,Change,Position,Conservation,
        #SigNoise,Pred_Prob,Pred_Label,True_Label and index of arbitrary ints
        self.test_out = pd.read_csv(os.path.join(results_dir, chosen_file),header=0,index_col=0)
        


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




     

def _calibration_plot(self):
    
    #self.path = "mlp_results/ryr2/cv_dropout_epoch1000_ensemble15/"
        #self.path = "mlp_results/ryr2/calibration/" # path for calibration plot
        # for calibration plot
    self.num_bins = 20
    self.calibration_strategy = 'quantile'
    
    # make calibration plot for the two logistic regression and mlp models
    lg = self._run_logreg()
    # debugging statements
    #print("CHECK IF TRUE LABELS ARE THE SAME\n\n")
    #for i in range(len(self.kfold_true)):
        #print((lg.true_test_labels_lst[i] == self.kfold_true[i]).all())
    #    for j in range(len(self.kfold_true[i])):
  #          print(lg.true_test_labels_lst[i][j] == self.kfold_true[i][j])
        #print(np.array(self.test_labels[i]) == np.array(self.kfold_true[i]))
        
    #print("\n\n")
    
    logreg_kfold_probability_stacked = np.hstack(lg.logreg_kfold_probability)
    kfold_prob_stacked = np.hstack(self.kfold_prob)
    kfold_true_stacked = np.hstack(self.kfold_true)
                                 
    if self.calibration_strategy == 'quantile':
        print("\n\n\n--------------------Starting calibration plots-----------\n\n\n")
        print("Strategy chosen: quantile")
        logreg_y, logreg_x = calibration.calibration_curve(kfold_true_stacked,
        logreg_kfold_probability_stacked, n_bins = self.num_bins, strategy='quantile')
        mlp_y, mlp_x = calibration.calibration_curve(kfold_true_stacked,
        kfold_prob_stacked, n_bins = self.num_bins, strategy='quantile')
    elif self.calibration_strategy == 'uniform': 
        logreg_y, logreg_x = calibration.calibration_curve(kfold_true_stacked,
        logreg_kfold_probability_stacked, strategy='uniform', n_bins=self.num_bins)
        mlp_y, mlp_x = calibration.calibration_curve(kfold_true_stacked,
        kfold_prob_stacked, strategy='uniform', n_bins=self.num_bins)
    else:
        sys.exit("Must choose either 'uniform' or 'quantile' as strategy for calibration plot under variable self.calibration_strategy")

    # plot calibration curves
    fig, ax = plt.subplots()
    plt.plot(logreg_x,logreg_y, marker='o', markersize=3, linewidth=1, label='logreg')
    plt.plot(mlp_x, mlp_y, marker='o',  markersize=3, linewidth=1, label='mlp')

    # reference line, legends, and axis labels
    #plt.xticks(np.arange(0.0, 1.1, 0.1))
    #plt.yticks(np.arange(0.0, 1.1, 0.1))
    
    # for a complete plot from 0 to 1
    #ax.set_xlim(left=0, right=1)
    #ax.set_ylim(bottom=0, top=1)

    # for a more compact plot
    ax_lim = max(max(logreg_x), max(mlp_x), max(logreg_y), max(mlp_y))
    ax.set_xlim(left=0, right=ax_lim + 0.05)
    ax.set_ylim(bottom=0, top=ax_lim + 0.05)
    line = mlines.Line2D([0, 1], [0, 1], color='black')
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    fig.suptitle('Calibration plot for ' + self.gene_name)
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('True probability')
    plt.legend()
    # save the plot
    if self.num_ensemble > 0:
        num_ensemble = "-" + str(self.num_ensemble)+ "ensemble-"
    else:
        num_ensemble = ""
    if self.cv_fold_mlp > 1:
        cv = "cv"
    else:
        cv = ""
    if self.calibration_strategy == 'uniform':
        num_bins = "-" + str(self.num_bins) + "bins"
    elif self.calibration_strategy == 'quantile':
        #num_bins = ""
        num_bins = "-" + str(self.num_bins) + "bins"
    figure_path = self.path + self.gene_name + '-python-calibration-' + cv 
    figure_path += num_bins + num_ensemble + self.calibration_strategy + ".png"
    plt.savefig(figure_path, dpi=100)  


    # caluclate the slopes of the best fit lines for both models
    logreg_linregress = stats.linregress(logreg_x, logreg_y)
    mlp_linregress = stats.linregress(mlp_x, mlp_y)
    print("-------------------Logreg Best Fit Line------------------")
    print(logreg_linregress)
    print('\n\n\n')
    print("-------------------MLP Best Fit Line---------------------")
    print(mlp_linregress)

