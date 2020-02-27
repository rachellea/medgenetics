# main.py

import copy
import os
import pickle
import datetime
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

def run(gene_name, what_to_run, modeling_approach):
    """Parameters:
    <gene_name> a string, either 'ryr2', 'kcnh2', 'kcnq1', or 'scn5a'
    <what_to_run> a string, either
        'perform_grid_search': this will do a grid search over different
            model setups
        or
        'get_mysteryAA_preds': this will get the predictions on the
            mysteryAAs for the best model
    <modeling_approach>: a string, either 'MLP' (for multilayer perceptron)
        or 'LR' for logistic regression"""
    results_dir = os.path.abspath(os.path.join('results',datetime.datetime.today().strftime('%Y-%m-%d')))
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    data_preproc_args = {'one_hotify_these_categorical':['Consensus','Change','Domain'],
                'normalize_these_continuous':['Position', 'Conservation', 'SigNoise'],
                'batch_size':256}
    d = clean_data.PrepareData(gene_name, data_preproc_args, results_dir)
    real_data_split = d.real_data_split
    mysteryAAs_Dataset = d.mysteryAAs_Dataset
    
    if what_to_run == 'perform_grid_search':
        run_mlp.GridSearch(gene_name, modeling_approach, results_dir, real_data_split, None, testing=True)
    if what_to_run == 'get_mysteryAA_preds':
        run_mlp.PredictMysteryAAs(gene_name, modeling_approach, results_dir, real_data_split, mysteryAAs_Dataset)
        

if __name__=='__main__':
    Run('ryr2',
        what_to_run='perform_grid_search',
        what_models='MLP')
