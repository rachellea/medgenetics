#regression.py
#Rachel Draelos
# edited by Farica Zhuang

import numpy as np
import scipy
from sklearn import linear_model, metrics, model_selection
import matplotlib.pyplot as plt
import evaluate

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
    def __init__(self, descriptor, split, logreg_penalty, C, fold = 10, figure_num=0):
        #--------------------------Set file path--------------------------------
        # this is the folder you want results to be stored
        filepath = 'regression_results/'

        #------------------------------Start logistic regression--------------------
        print('Running logistic regression with penalty=',str(logreg_penalty),'and C=',str(C))
        clf = linear_model.LogisticRegression(penalty=logreg_penalty, C=C)

        cv = model_selection.KFold(n_splits=fold,shuffle=True)

        plt.figure()

        # Compute ROC curve and ROC area with averaging
        tprs = []
        aucs_val = []
        accs = []
        prec = []
        auc_str = []
        base_fpr = np.linspace(0, 1, 101)
        x = np.array(split.clean_data)
        y = np.array(split.clean_labels)
        i = 1
        for train, test in cv.split(x,y):
            model = clf.fit(x[train], y[train])
            y_score = model.predict_proba(x[test])
            y_pred = model.predict(x[test])
            fpr, tpr, _ = metrics.roc_curve(y[test], y_score[:, 1])
            auc = metrics.roc_auc_score(y[test], y_score[:,1])
            plt.plot(fpr, tpr, 'b', alpha=0.15)
            auc_str.append("fold " + str(i) + " AUC: " + str(auc))
            
            tpr = scipy.interp(base_fpr, fpr, tpr)
            tpr[0] = 0.0
            tprs.append(tpr)
            aucs_val.append(auc)
            accs.append(metrics.accuracy_score(y[test], y_pred))
            prec.append(metrics.average_precision_score(y[test], y_pred))

            i += 1

        tprs = np.array(tprs)
        mean_tprs = tprs.mean(axis=0)
        std = tprs.std(axis=0)

        tprs_upper = np.minimum(mean_tprs + std, 1)
        tprs_lower = mean_tprs - std

        plt.plot(base_fpr, mean_tprs, 'b', label = "Average AUC:" + str(sum(aucs_val)/len(aucs_val)))
        plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)

        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim((0, 1))
        plt.ylim(0, 1)
        title = 'ROC for ' + descriptor + ' Log Reg '+str(logreg_penalty)+'C'+str(C)+ " "+str(fold) + ' Fold CV'
        plt.title(title)
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.axes().set_aspect('equal', 'datalim')
        plt.legend(loc="lower right")
        title = 'ROC_' + descriptor + '_LogReg_'+str(logreg_penalty)+'C'+str(C) + '_'+ str(fold) + '_Fold_CV' + '.png'
        plt.savefig(filepath + title)
        plt.show()

        # write the results to a file
        filename_prefix = filepath + descriptor +'_LogReg_'+str(logreg_penalty)+'C'+str(C)+ '_'+ str(fold) + '_Fold_CV_Test'
        with open(filename_prefix+'_CV_Results.txt','w') as f:
            for auc in auc_str:
                f.write(auc + "\n")
            f.write('Average AUC: '+ str(np.array(aucs_val).mean()))
            f.write('\nAverage Accuracy:'+ str(np.array(accs).mean()))
            f.write('\nAverage Precision:'+ str(np.array(prec).mean()))
      #-----------------------------Rachel's original stuff--------------------------------      
        # print('Running logistic regression with penalty=',str(logreg_penalty),'and C=',str(C))
        # logreg = linear_model.LogisticRegression(penalty=logreg_penalty, C=C)
        # logreg.fit(split.train.data, split.train.labels)
        # test_predictions = logreg.predict(split.test.data) #shape e.g. (752,)
        # test_true_labels = np.reshape(split.test.labels, (len(split.test.labels))) #shape e.g. (752,) rather than (752,1)
        
        # #Save evaluation results
        # filename_prefix = descriptor+'_LogReg_'+str(logreg_penalty)+'C'+str(C)+'_Test'
        # evaluate.plot_precision_recall_curve(true_labels = test_true_labels,
        #     pred_probs = test_predictions, epoch = 'NA', filename_prefix = filename_prefix, which_label='')
        # fpr, tpr, thresholds = metrics.roc_curve(y_true = test_true_labels,
        #                              y_score = test_predictions,
        #                              pos_label = 1) #admitted is 1; not admitted is 0
        # evaluate.plot_roc_curve(fpr=fpr, tpr=tpr, epoch='NA', filename_prefix = filename_prefix, which_label='')
        # with open(filename_prefix+'_Coefficients.txt', 'w') as outfile:
        #     for i in range(len(split.train.data_meanings)):
        #         outfile.write(str(split.train.data_meanings[i])+'\t'+str(logreg.coef_[0,i])+'\n')
        # with open(filename_prefix+'_Results.txt','w') as f:
        #     f.write('AUC: '+str(metrics.auc(fpr, tpr)))
        #     f.write('\nAccuracy:'+str(evaluate.compute_accuracy(test_true_labels, test_predictions)))
        #     f.write('\nAverage Precision:'+str(metrics.average_precision_score(test_true_labels, test_predictions)))
