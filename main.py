#main.py

import copy
import os
import pickle
import pandas as pd
import numpy as np

#Custom imports
from data import utils as utils
from data import clean_data as clean_data
import mlp_model
import mlp_model_cv
import regression

class RunGeneModel(object):
    def __init__(self, gene_name, descriptor, shared_args, cols_to_delete=[], cv_fold_lg=10, cv_fold_mlp=10):
        """<gene_name> is a string, one of: 'kcnh2', 'kcnq1', 'ryr2', or 'scn5a'."""
        self.gene_name = gene_name
        self.descriptor = descriptor
        self.shared_args = shared_args
        self.cols_to_delete = cols_to_delete
        self.cv_fold_lg = cv_fold_lg
        self.cv_fold_mlp = cv_fold_mlp
    
    def do_all(self):
        self._prep_data()
        self._prep_split_data(self.inputx,self.split_args)
        self._prep_mysteryAAs()
        self._run_mlp()
        #self._run_logreg()
    
    def _prep_data(self):
        #Real data with healthy and diseased
        ag = clean_data.AnnotatedGene(self.gene_name)
        ag.annotate_everything()
        self.inputx = ag.inputx #make it self.inputx so you can access from testing script
        self.mysteryAAs = ag.mysteryAAs
        self.columns_to_ensure_here = [x for x in ag.columns_to_ensure if x not in self.cols_to_delete]
        self.split_args = {'train_percent':0.7,
                        'valid_percent':0.15,
                        'test_percent':0.15,
                        'max_position':ag.max_position,
                        'columns_to_ensure':self.columns_to_ensure_here}
        self.ag = ag
    
    def _prep_split_data(self, inputx, split_args):
        """There are arguments to this function to facilitate unit testing"""
        data = (copy.deepcopy(inputx)).drop(columns=['Label']+self.cols_to_delete)
        labels = copy.deepcopy(inputx[['Label']])
        print('Fraction of diseased:',str( np.sum(labels)/len(labels) ) )
        all_args = {**self.shared_args, **split_args}
        self.real_data_split = utils.Splits(data = data,
                             labels = labels,
                             **all_args)


    
    def _prep_mysteryAAs(self):
        #WES data, mysteryAAs (want predictions for these)
        mysteryAAs_data = (copy.deepcopy(self.mysteryAAs)).drop(columns=self.cols_to_delete)
        mysteryAAs_labels = pd.DataFrame(np.zeros((mysteryAAs_data.shape[0],1)), columns=['Label'])
        self.mysteryAAs_split = utils.Splits(data = mysteryAAs_data,
                                     labels = mysteryAAs_labels,
                                     train_percent = 1.0,
                                     valid_percent = 0,
                                     test_percent = 0,
                                     max_position = self.ag.max_position,
                                     columns_to_ensure = self.columns_to_ensure_here,
                                     **self.shared_args).train
        assert self.mysteryAAs_split.data.shape[0] == self.mysteryAAs.shape[0]
        
        #Save pickled split:
        print('Saving pickled split')
        pickle.dump(self.real_data_split, open(self.gene_name+'_'+self.descriptor+'.pickle', 'wb'),-1)
    
    def _run_mlp(self):
        #Run MLP
        print('Running MLP')
        # if no cross validation
        if self.cv_fold_mlp == 0:
            m = mlp_model.MLP(descriptor=self.gene_name+'_'+self.descriptor,
                        split=copy.deepcopy(self.real_data_split),
                        decision_threshold = 0.5,
                        num_epochs = 1000,
                        learningrate = 1e-4,
                        mlp_layers = copy.deepcopy([30,20]),
                        dropout=0.5,
                        exclusive_classes = True,
                        save_model = False,
                        mysteryAAs = self.mysteryAAs_split)
        # if cross validation
        else: 
            m = mlp_model_cv.MLP_cv(descriptor=self.gene_name+'_'+self.descriptor,
                                  split=copy.deepcopy(self.real_data_split),
                                  decision_threshold = 0.5,
                                  num_epochs = 1000,
                                  learningrate = 1e-4,
                                  mlp_layers = copy.deepcopy([30,20]),
                                  dropout=0.5,
                                  exclusive_classes = True,
                                  save_model = False,
                                  mysteryAAs = self.mysteryAAs_split,
                                  cv_fold = self.cv_fold_mlp)
        m.run_all()

    def _run_logreg(self):
        # Run Logistic Regression
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
        print("Running Log Reg")

        classifier_penalty= ['l1', 'l2']
        classifier_C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

        # set the k value for k fold cross validation (# of folds for cross validation. Set to 0 if 
        # we don't want to do cross validation)
        kfold = self.cv_fold_lg

        for pen in classifier_penalty:
          for C in classifier_C:
            lg = regression.LogisticRegression(descriptor=descriptor, split=copy.deepcopy(self.real_data_split),logreg_penalty=pen, C=C, fold=kfold)



if __name__=='__main__':
    variations = {'noSN':['Position','Conservation'],
        'withSN':['Position','Conservation','SigNoise']}
    for descriptor in variations:
        cont_vars = variations[descriptor]
        shared_args = {'impute':False,
                        'impute_these_categorical':[],
                        'impute_these_continuous':[],
                        'one_hotify':True,
                        'one_hotify_these_categorical':['Consensus','Change','Domain'],  #cat_vars
                        'normalize_data':True,
                        'normalize_these_continuous':cont_vars,
                        'seed':10393, #make it 12345 for original split
                        'batch_size':300}
        RunGeneModel(gene_name='ryr2', descriptor=descriptor,shared_args = shared_args, cols_to_delete=list(set(['Position','Conservation','SigNoise'])-set(cont_vars)), cv_fold_lg=10, cv_fold_mlp=10).do_all()

