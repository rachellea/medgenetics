#regression.py
#Rachel Draelos

import numpy as np
import evaluate
from sklearn import linear_model, metrics

class LogisticRegression(object):
    #http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
    """Variables:
    <logreg_penalty>:can be 'l1' or 'l2'
    <C>: float, default: 1.0. Inverse of regularization strength; must be a
        positive float. Like in support vector machines, smaller values specify
        stronger regularization.
    
    Note on linear_model.LogisticRegression defaults:
    fit_intercept = True
    solver = 'liblinear' (can do l1 or l2 penalty; liblinear is good choice for
        small datasets; 'sag' and 'saga' are faster for large ones; 'sag' can
        handle l2, 'saga' can handle l1)"""
    def __init__(self, descriptor, split, logreg_penalty, C): 
        print('Running logistic regression with penalty=',str(logreg_penalty),'and C=',str(C))
        logreg = linear_model.LogisticRegression(penalty=logreg_penalty, C=C)
        logreg.fit(split.train.data, split.train.labels)
        test_predictions = logreg.predict(split.test.data) #shape e.g. (752,)
        test_true_labels = np.reshape(split.test.labels, (len(split.test.labels))) #shape e.g. (752,) rather than (752,1)
        
        #Save evaluation results
        filename_prefix = descriptor+'_LogReg_'+str(logreg_penalty)+'C'+str(C)+'_Test'
        evaluate.plot_precision_recall_curve(true_labels = test_true_labels,
            pred_probs = test_predictions, epoch = 'NA', filename_prefix = filename_prefix, which_label='')
        fpr, tpr, thresholds = metrics.roc_curve(y_true = test_true_labels,
                                     y_score = test_predictions,
                                     pos_label = 1) #admitted is 1; not admitted is 0
        evaluate.plot_roc_curve(fpr=fpr, tpr=tpr, epoch='NA', filename_prefix = filename_prefix, which_label='')
        with open(filename_prefix+'_Coefficients.txt', 'w') as outfile:
            for i in range(len(split.train.data_meanings)):
                outfile.write(str(split.train.data_meanings[i])+'\t'+str(logreg.coef_[0,i])+'\n')
        with open(filename_prefix+'_Results.txt','w') as f:
            f.write('AUC: '+str(metrics.auc(fpr, tpr)))
            f.write('\nAccuracy:'+str(evaluate.compute_accuracy(test_true_labels, test_predictions)))
            f.write('\nAverage Precision:'+str(metrics.average_precision_score(test_true_labels, test_predictions)))
