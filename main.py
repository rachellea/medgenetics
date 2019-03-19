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

class RunGeneModel(object):
    def __init__(self, gene_name, cat_vars, cont_vars, descriptor):
        """<gene_name> is a string, one of: 'kcnh2', 'kcnq1', 'ryr2', or 'scn5a'.
        <cat_vars> is a list of which categorical variables to use e.g.
            ['Consensus','Change','Domain']
        <cont_vars> is a list of which continuous variables to use e.g.
            ['Position','Conservation','SigNoise'] """
        self.gene_name = gene_name
        self.descriptor = descriptor
        #Shared split args
        shared_args = {'impute':False,
                        'impute_these_categorical':[],
                        'impute_these_continuous':[],
                        'one_hotify':True,
                        'one_hotify_these_categorical':cat_vars,
                        'normalize_data':True,
                        'normalize_these_continuous':cont_vars,
                        'seed':10393, #make it 12345 for original split
                        'batch_size':300}
        
        #Real data with healthy and diseased
        ag = clean_data.AnnotatedGene(self.gene_name)
        self.inputx = ag.inputx #make it self.inputx so you can access from testing script
        mysteryAAs = ag.mysteryAAs
        data = copy.deepcopy(self.inputx[cat_vars+cont_vars])
        labels = copy.deepcopy(self.inputx[['Label']])
        print('Fraction of diseased:',str( np.sum(labels)/len(labels) ) )
        self.real_data_split = utils.Splits(data = data,
                             labels = labels,
                             train_percent = 0.7,
                             valid_percent = 0.15,
                             test_percent = 0.15,
                             max_position = ag.max_position,
                             columns_to_ensure = ag.columns_to_ensure,
                             **shared_args)
    
        #WES data, mysteryAAs (want predictions for these)
        mysteryAAs_data = copy.deepcopy(mysteryAAs[cat_vars+cont_vars])
        mysteryAAs_labels = pd.DataFrame(np.zeros((mysteryAAs_data.shape[0],1)), columns=['Label'])
        self.mysteryAAs_split = utils.Splits(data = mysteryAAs_data,
                                     labels = mysteryAAs_labels,
                                     train_percent = 1.0,
                                     valid_percent = 0,
                                     test_percent = 0,
                                     max_position = ag.max_position,
                                     columns_to_ensure = ag.columns_to_ensure,
                                     **shared_args).train
        assert self.mysteryAAs_split.data.shape[0] == mysteryAAs.shape[0]
        
        #Save pickled split:
        print('Saving pickled split')
        pickle.dump(self.real_data_split, open(self.gene_name+'_'+self.descriptor+'.pickle', 'wb'),-1)
    
    def run(self):
        #Run MLP
        print('Running MLP')
        m = mlp_model.MLP(descriptor=self.gene_name+'_'+self.descriptor,
                      split=copy.deepcopy(self.real_data_split),
                      decision_threshold = 0.5,
                      num_epochs = 1000,
                      learningrate = 1e-4,
                      mlp_layers = copy.deepcopy([30,20]),
                      exclusive_classes = True,
                      save_model = False,
                      mysteryAAs = self.mysteryAAs_split)
        m.run_all()
        

if __name__=='__main__':
    RunGeneModel(gene_name='ryr2',cat_vars = ['Consensus','Change','Domain'],
                   cont_vars=['Position','Conservation','SigNoise'],
                   descriptor='withSN').run()
    RunGeneModel(gene_name='ryr2',cat_vars = ['Consensus','Change','Domain'],
                   cont_vars=['Position','Conservation'],
                   descriptor='noSN').run()
  