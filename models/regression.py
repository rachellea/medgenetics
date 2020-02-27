#regression.py

import numpy as np
from sklearn import linear_model, metrics, model_selection

class LogisticRegression(object):
    #http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
    """Variables:
    <descriptor>: string describing the model
    <logreg_penalty>: can be 'l1' or 'l2'
    <C>: float, default: 1.0. Inverse of regularization strength; must be a
        positive float. Smaller values specify stronger regularization.
    
    Note on linear_model.LogisticRegression defaults:
        fit_intercept = True
        solver = 'liblinear' (can do l1 or l2 penalty; liblinear is good choice
            for small datasets; 'sag' and 'saga' are faster for large ones;
            'sag' can handle l2, 'saga' can handle l1)"""
    def __init__(self, descriptor, split, logreg_penalty, C, decision_threshold):
        print('\tLogistic Regresssion with penalty=',str(logreg_penalty),'and C=',str(C))
        self.split = split
        self.logreg_penalty = logreg_penalty
        self.C = C
        self.decision_threshold = decision_threshold
    
    def run_all(self):
        """Train and evaluate a logistic regression model"""
        logreg = linear_model.LogisticRegression(penalty=self.logreg_penalty, C=self.C)
        logreg.fit(split.train.data, split.train.labels)
        test_pred_probs = logreg.predict(split.test.data) #shape e.g. (752,)
        test_pred_labels = (test_pred_probs >= self.decision_threshold).astype('int')
        test_true_labels = np.reshape(split.test.labels, (len(split.test.labels))) #shape e.g. (752,) rather than (752,1)
        
        #Save results
        #test_out is a dictionary with keys that are epochs and values
        #that are dataframes. We're not using epochs for logistic regression
        #but for compatibility with the MLP output we'll call the result of
        #logistic regression 'epoch_0'
        #A dataframe contains the data set (entire_x)
        #as well as the predicted probabilities, predicted labels, and
        #true labels for all examples
        test_out_df = pd.DataFrame(np.concatenate((split.test.data, test_pred_probs, test_pred_labels, test_true_labels),axis = 1),
                               columns=self.train_set.data_meanings+['Pred_Prob','Pred_Label','True_Label'])
        test_out_df = test_out_df.sort_values(by='Position')
        self.test_out['epoch_0'] = test_out_df