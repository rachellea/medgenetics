import matplotlib
matplotlib.use('agg') #so that it does not attempt to display via SSH
import matplotlib.pyplot as plt
plt.ioff() #turn interactive plotting off

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
