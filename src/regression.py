#regression.py

import os
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
    <num_epochs>: passed in only for compatibility in input parameters with mlp
    <seed>: this arg isn't used; it's just for compatibility with the mlp class
    <results_dir>: directory to save the coefficients of the final model
    
    Note on linear_model.LogisticRegression defaults:
        fit_intercept = True
        solver = 'liblinear' (can do l1 or l2 penalty; liblinear is good choice
            for small datasets; 'sag' and 'saga' are faster for large ones;
            'sag' can handle l2, 'saga' can handle l1)"""
    def __init__(self, descriptor, split, logreg_penalty, C, decision_threshold,
                 num_epochs, mysteryAAs, seed, results_dir):
        print('\tLogistic Regresssion with penalty=',str(logreg_penalty),'and C=',str(C))
        self.split = split
        self.logreg_penalty = logreg_penalty
        self.C = C
        self.decision_threshold = decision_threshold
        self.mysteryAAs = mysteryAAs
        self.max_iter = 1000
        self.seed = seed
        self.results_dir = results_dir
    
    def run_all_train_test(self):
        """Train and evaluate a logistic regression model"""
        logreg = linear_model.LogisticRegression(penalty=self.logreg_penalty, C=self.C, max_iter=self.max_iter)
        logreg = logreg.fit(self.split.train.data, self.split.train.labels)
        #predicted probabilities are first for class 0 and then for class 1
        test_pred_probs_array = logreg.predict_proba(self.split.test.data) #shape e.g. (752,)
        #We only want the predicted probabilities for class 1 (i.e. for mutation):
        test_pred_probs = test_pred_probs_array[:,1]
        test_pred_labels = (test_pred_probs >= self.decision_threshold).astype('int')
        
        #Save results
        #test_out is a dictionary with keys that are epochs and values
        #that are dataframes. We do use 1000 as the max_iter, but since
        #we don't get performance at every iteration, just save logistic
        #regression results under 'epoch_0' (for consistent output format
        #between MLP and LR)
        test_out_df = pd.DataFrame(np.concatenate((self.split.test.data,
                                np.expand_dims(test_pred_probs,1),
                                np.expand_dims(test_pred_labels,1),
                                self.split.test.labels),axis = 1),
                                columns=self.split.clean_data.columns.values.tolist()+['Pred_Prob','Pred_Label','True_Label'])
        test_out_df = test_out_df.sort_values(by='Position')
        self.test_out = {}
        self.test_out['epoch_0'] = test_out_df
    
    def run_all_mysteryAA_preds(self):
        """Train logistic regression model on all available data and then
        make predictions on mysteryAAs"""
        logreg = linear_model.LogisticRegression(penalty=self.logreg_penalty, C=self.C, max_iter=self.max_iter)
        logreg = logreg.fit(self.split.train.data, self.split.train.labels)
        
        #Save coefficients since this is the final model
        coeff_df = pd.DataFrame(logreg.coef_,index=['Coefficients'],columns=self.split.train.data_meanings).transpose()
        coeff_filename = 'LR-Coefficients-Ep0-'+str(self.logreg_penalty)+'-C'+str(self.C)+'.csv'
        coeff_df.to_csv(os.path.join(self.results_dir, coeff_filename), header=True, index=True)
        
        #predicted probabilities are first for class 0 and then for class 1
        mysteryAAs_pred_probs_array = logreg.predict_proba(self.mysteryAAs.data) #shape e.g. (752,)
        #We only want the predicted probabilities for class 1 (i.e. for mutation):
        mysteryAAs_pred_probs = mysteryAAs_pred_probs_array[:,1]
        mysteryAAs_pred_labels = (mysteryAAs_pred_probs >= self.decision_threshold).astype('int')
        
        #Save results
        mysteryAAs_out_df = pd.DataFrame(np.concatenate((self.mysteryAAs.data,
                               np.expand_dims(mysteryAAs_pred_probs,1),
                               np.expand_dims(mysteryAAs_pred_labels,1),
                               self.mysteryAAs.labels),axis = 1),
                               columns=self.split.clean_data.columns.values.tolist()+['Pred_Prob','Pred_Label','True_Label'])
        mysteryAAs_out_df = mysteryAAs_out_df.sort_values(by='Position')
        self.mysteryAAs_out = {}
        self.mysteryAAs_out['epoch_0'] = mysteryAAs_out_df
        