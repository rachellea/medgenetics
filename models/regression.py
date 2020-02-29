#regression.py

import numpy as np
import pandas as pd
from sklearn import linear_model

class LogisticRegression(object):
    #http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
    """Parameters:
    <descriptor>: string describing the model
    <logreg_penalty>: can be 'l1' or 'l2'
    <C>: float, default: 1.0. Inverse of regularization strength; must be a
        positive float. Smaller values specify stronger regularization.
    <num_epochs>: used as the max_iter (max iterations) to reach a solution
    
    Note on linear_model.LogisticRegression defaults:
        fit_intercept = True
        solver = 'liblinear' (can do l1 or l2 penalty; liblinear is good choice
            for small datasets; 'sag' and 'saga' are faster for large ones;
            'sag' can handle l2, 'saga' can handle l1)"""
    def __init__(self, descriptor, split, logreg_penalty, C, decision_threshold,
                 num_epochs):
        print('\tLogistic Regresssion with penalty=',str(logreg_penalty),'and C=',str(C))
        self.split = split
        self.logreg_penalty = logreg_penalty
        self.C = C
        self.decision_threshold = decision_threshold
        self.num_epochs = num_epochs
    
    def run_all(self):
        """Train and evaluate a logistic regression model"""
        logreg = linear_model.LogisticRegression(penalty=self.logreg_penalty, C=self.C, max_iter=self.num_epochs)
        logreg.fit(self.split.train.data, self.split.train.labels)
        #predicted probabilities are first for class 0 and then for class 1
        test_pred_probs_array = logreg.predict_proba(self.split.test.data) #shape e.g. (752,)
        #We only want the predicted probabilities for class 1 (i.e. for mutation):
        test_pred_probs = test_pred_probs_array[:,1]
        test_pred_labels = (test_pred_probs >= self.decision_threshold).astype('int')
        
        #Save results
        #test_out is a dictionary with keys that are epochs and values
        #that are dataframes. We do use <num_epochs> as the max_iter, but since
        #we don't get performance at every iteration, just save logistic
        #regression results under 'epoch_0' (for consistent output format
        #between MLP and LR)
        test_out_df = pd.DataFrame(np.concatenate((self.split.test.data, np.expand_dims(test_pred_probs,1), np.expand_dims(test_pred_labels,1), self.split.test.labels),axis = 1),
                               columns=self.split.clean_data.columns.values.tolist()+['Pred_Prob','Pred_Label','True_Label'])
        test_out_df = test_out_df.sort_values(by='Position')
        self.test_out = {}
        self.test_out['epoch_0'] = test_out_df