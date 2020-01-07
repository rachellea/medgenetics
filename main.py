# main.py

import copy
import os
import pickle
import pandas as pd
import numpy as np
from sklearn import model_selection, metrics, calibration
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
from scipy import stats
import itertools
from tqdm import tqdm

#Custom imports
from models import run_mlp
from models import run_regression
from data import clean_data

# for checking purposes/sanity check
from sklearn import neural_network


class Run(object):
    def __init__(self, gene_name, shared_args, what_to_run, what_models):
        """
        <what_to_run> is a list of strings that can contain:
            'perform_grid_search': this will do a grid search over different
                model setups
            'get_test_set_preds': this will save all of the test set
                predictions on all examples for a specified model setup
        <what_models> is a list of strings that can contain 'mlp' and/or 'lr'"""
        d = clean_data.PrepareData(gene_name, shared_args)
        real_data_split = d.real_data_split
        mysteryAAs_split = d.mysteryAAs_split
        
        #Multilayer Perceptron Model
        if 'mlp' in what_models:
            if 'perform_grid_search' in what_to_run:
                run_mlp.GridSearchMLP(gene_name, real_data_split, mysteryAAs_split)
            if 'get_test_set_preds' in what_to_run:
                pass #TODO IMPLEMENT THIS
            if 'make_mysteryAA_preds' in what_to_run:
                run_mlp.PredictMysteryAAs_MLP(gene_name, real_data_split, mysteryAAs_split)
        
        #Logistic Regression Model
        elif 'lr' in what_models:
            pass

if __name__=='__main__':
    shared_args = {'one_hotify':True,
                        'one_hotify_these_categorical':['Consensus','Change','Domain'],
                        'normalize_data':False, # change to false if performing best model
                        'normalize_these_continuous':['Position', 'Conservation', 'SigNoise'],
                        'seed':10393, #make it 12345 for original split
                        'batch_size':300}
    Run('scn5a',shared_args,what_to_run=[''],what_model=[])
