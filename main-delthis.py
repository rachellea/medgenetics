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
from src import run_models, circgenetics_replication, visualization
from data import clean_data

def run(gene_name, what_to_run, modeling_approach, results_dir):
    """Parameters:
    <gene_name> a string, either 'ryr2', 'kcnh2', 'kcnq1', or 'scn5a'
    <what_to_run> a string, one of:
        'grid_search': this will do a grid search over different model setups
        'test_pred': this will save the test set predictions for the best
            model setup identified in the grid_search (meaning that grid_search
            must be run before test_pred is run)
        'mysteryAA_pred': this will save the mysteryAA predictions for the
            best model setup identified in the grid_search (meaning that grid_search
            must be run before mysteryAA_pred is run)
            
    <modeling_approach>: a string, either 'MLP' (for multilayer perceptron)
        or 'LR' for logistic regression"""
    data_preproc_args = {'one_hotify_these_categorical':['Consensus','Change','Domain'],
                'normalize_these_continuous':['Position', 'Conservation', 'SigNoise'],
                'batch_size':256}
    d = clean_data.PrepareData(gene_name, data_preproc_args, results_dir)
    if what_to_run == 'grid_search':
        run_models.RunPredictiveModels(gene_name, modeling_approach, results_dir, d.real_data_split, what_to_run, testing=True)
    elif what_to_run == 'test_pred':
        run_models.RunPredictiveModels(gene_name, modeling_approach, results_dir, d.real_data_split, what_to_run, testing=True)
    elif what_to_run == 'mysteryAA_pred':
        run_models.PredictMysteryAAs(gene_name, modeling_approach, results_dir, d.real_data_split, d.mysteryAAs_dict)

def make_results_dirs():
    """Make directories for storing results"""
    date_dir = os.path.join('results',datetime.datetime.today().strftime('%Y-%m-%d'))
    if not os.path.exists(date_dir):
        os.mkdir(date_dir)
    
    results_dir_ryr2 = os.path.abspath(os.path.join(date_dir,'ryr2'))
    results_dir_kcnq1 = os.path.abspath(os.path.join(date_dir,'kcnq1'))
    results_dir_kcnh2 = os.path.abspath(os.path.join(date_dir,'kcnh2'))
    results_dir_scn5a = os.path.abspath(os.path.join(date_dir,'scn5a'))
    
    for directory in [results_dir_ryr2, results_dir_kcnq1, results_dir_kcnh2, results_dir_scn5a]:
        if not os.path.exists(directory):
            os.mkdir(directory)
    return results_dir_ryr2, results_dir_kcnq1, results_dir_kcnh2, results_dir_scn5a

if __name__=='__main__':
    results_dir_ryr2, results_dir_kcnq1, results_dir_kcnh2, results_dir_scn5a = make_results_dirs()
    results_dir_ryr2_first = os.path.join(results_dir_ryr2,'firstrun')
    if not os.path.exists(results_dir_ryr2_first):
        os.mkdir(results_dir_ryr2_first)
    run('ryr2',what_to_run='grid_search',modeling_approach='MLP',results_dir = results_dir_ryr2_first)
    results_dir_ryr2_second = os.path.join(results_dir_ryr2,'secondrun')
    if not os.path.exists(results_dir_ryr2_second):
        os.mkdir(results_dir_ryr2_second)
    run('ryr2',what_to_run='grid_search',modeling_approach='MLP',results_dir = results_dir_ryr2_second)
    