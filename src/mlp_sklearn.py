#regression.py

import numpy as np
import pandas as pd
from sklearn import neural_network

class MLP(object):
    #https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
    """Parameters: see mlp_tf.py for documentation
    
    By default, sklearn MLP solver is Adam, nonlinearity is ReLU, learning rate
    is constant, and shuffle is True."""
    def __init__(self,
                 descriptor,
                 split,
                 decision_threshold,
                 num_epochs,
                 learningrate, #e.g. 1e-4
                 mlp_layers,
                 dropout,
                 mysteryAAs):
        print('\tsklearn MLP',descriptor)
        self.split = split
        self.decision_threshold = decision_threshold
        self.mysteryAAs = mysteryAAs
        self.num_epochs = num_epochs
        self.mlp_args = {'hidden_layer_sizes':tuple(mlp_layers),
                         'batch_size':split.batch_size,
                         'learning_rate_init':learningrate,
                         'max_iter':self.num_epochs,
                         'random_state':0, #TODO address random state for ensemble
                         'validation_fraction':0}
        self.test_out = {}
    
    def run_all_train_test(self):
        """Train and evaluate an MLP"""
        mlp = neural_network.MLPClassifier(**self.mlp_args)
        for epoch in range(0,self.num_epochs):
            mlp.partial_fit(self.split.train.data, self.split.train.labels, classes=np.unique(self.split.train.labels))
            #predicted probabilities are first for class 0 and then for class 1
            test_pred_probs_array = mlp.predict_proba(self.split.test.data) #shape e.g. (752,)
            #We only want the predicted probabilities for class 1 (i.e. for mutation):
            test_pred_probs = test_pred_probs_array[:,1]
            test_pred_labels = (test_pred_probs >= self.decision_threshold).astype('int')
            
            #Save results. See regression.py for documentation
            test_out_df = pd.DataFrame(np.concatenate((self.split.test.data,
                                    np.expand_dims(test_pred_probs,1),
                                    np.expand_dims(test_pred_labels,1),
                                    self.split.test.labels),axis = 1),
                                    columns=self.split.clean_data.columns.values.tolist()+['Pred_Prob','Pred_Label','True_Label'])
            test_out_df = test_out_df.sort_values(by='Position')
            self.test_out['epoch_'+str(epoch)] = test_out_df
    
    def run_all_mysteryAA_preds(self):
        """Train logistic regression model on all available data and then
        make predictions on mysteryAAs"""
        mlp = neural_network.MLPClassifier(**self.mlp_args)
        mlp.fit(self.split.train.data, self.split.train.labels)
        #predicted probabilities are first for class 0 and then for class 1
        mysteryAAs_pred_probs_array = mlp.predict_proba(self.mysteryAAs.data) #shape e.g. (752,)
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
        self.mysteryAAs_out['epoch_'+str(self.num_epochs)] = mysteryAAs_out_df
        