#framingham_model.py

import copy
import os
import pickle
import pandas as pd
import numpy as np

#Custom imports
from data import utils as utils
from data import clean_data as clean_data
import mlp_model

def run_gene_model(gene_name):
    """<gene_name> is a string, one of: 'kcnh2', 'kcnq1', 'ryr2', or 'scn5a'."""
    #Shared split args
    shared_args = {'impute':False,
                    'impute_these_categorical':[],
                    'impute_these_continuous':[],
                    'one_hotify':True,
                    'one_hotify_these_categorical':['Consensus', 'Change', 'Domain'],
                    'normalize_data':True,
                    'normalize_these_continuous':['Position','Conservation'],
                    'seed':12345,
                    'batch_size':300}
    
    #Real data with healthy and diseased
    ag = clean_data.AnnotatedGene(gene_name)
    inputx = ag.inputx
    everyAA = ag.everyAA
    data = copy.deepcopy(inputx[['Position','Consensus','Change','Domain','Conservation']])
    labels = copy.deepcopy(inputx[['Label']])
    print('Fraction of diseased:',str( np.sum(labels)/len(labels) ) )
    real_data_split = utils.Splits(data = data,
                         labels = labels,
                         train_percent = 0.7,
                         valid_percent = 0.15,
                         test_percent = 0.15,
                         **shared_args)

    #Fake data with all possible combos of every AA at every position
    everyAA_data = copy.deepcopy(everyAA[['Position','Consensus','Change','Domain','Conservation']])
    everyAA_labels = copy.deepcopy(everyAA[['Label']])
    everyAA_split = utils.Splits(data = everyAA_data,
                                 labels = everyAA_labels,
                                 train_percent = 1.0,
                                 valid_percent = 0,
                                 test_percent = 0,
                                 **shared_args).train
    assert everyAA_split.data.shape[0] == everyAA.shape[0]
    
    #Run MLP
    m = mlp_model.MLP(descriptor=gene_name,
                  split=copy.deepcopy(real_data_split),
                  decision_threshold = 0.5,
                  num_epochs = 1000,
                  learningrate = 1e-4,
                  mlp_layers = copy.deepcopy([30,20]),
                  exclusive_classes = True,
                  save_model = False,
                  everyAA = everyAA_split)
    m.run_all()
    

if __name__=='__main__':
    run_gene_model('ryr2')
    run_gene_model('scn5a')
    run_gene_model('kcnq1')
    run_gene_model('kcnh2')