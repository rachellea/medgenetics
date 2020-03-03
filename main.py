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
from models import run_models, circgenetics_replication
from data import clean_data, visualization

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
        run_models.RunPredictiveModels(gene_name, modeling_approach, results_dir, d.real_data_split, what_to_run, testing=False)
    elif what_to_run == 'test_pred':
        run_models.RunPredictiveModels(gene_name, modeling_approach, results_dir, d.real_data_split, what_to_run, testing=False)
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

def replicate_entire_study():
    results_dir_ryr2, results_dir_kcnq1, results_dir_kcnh2, results_dir_scn5a = make_results_dirs()
    
    #Logistic Regression
    #run('ryr2',what_to_run='grid_search',modeling_approach='LR',results_dir = results_dir_ryr2)
    #run('ryr2',what_to_run='test_pred',modeling_approach='LR',results_dir = results_dir_ryr2)
    #run('ryr2',what_to_run='mysteryAA_pred',modeling_approach='LR',results_dir = results_dir_ryr2)
    
    #run('kcnq1',what_to_run='grid_search',modeling_approach='LR',results_dir = results_dir_kcnq1)
    #run('kcnq1',what_to_run='test_pred',modeling_approach='LR',results_dir = results_dir_kcnq1)
    #run('kcnq1',what_to_run='mysteryAA_pred',modeling_approach='LR',results_dir = results_dir_kcnq1)
    
    #run('kcnh2',what_to_run='grid_search',modeling_approach='LR',results_dir = results_dir_kcnh2)
    #run('kcnh2',what_to_run='test_pred',modeling_approach='LR',results_dir = results_dir_kcnh2)
    #run('kcnh2',what_to_run='mysteryAA_pred',modeling_approach='LR',results_dir = results_dir_kcnh2)
    
    run('scn5a',what_to_run='grid_search',modeling_approach='LR',results_dir = results_dir_scn5a)
    run('scn5a',what_to_run='test_pred',modeling_approach='LR',results_dir = results_dir_scn5a)
    run('scn5a',what_to_run='mysteryAA_pred',modeling_approach='LR',results_dir = results_dir_scn5a)
    
    
    
    #MLPs
    #run('ryr2',what_to_run='grid_search',modeling_approach='MLP',results_dir = results_dir_ryr2)
    #run('ryr2',what_to_run='test_pred',modeling_approach='MLP',results_dir = results_dir_ryr2)
    #run('ryr2',what_to_run='mysteryAA_pred',modeling_approach='MLP',results_dir = results_dir_ryr2)
    #visualization.MakeAllFigures('ryr2',results_dir_ryr2)
    
    #KCNQ1
    

if __name__=='__main__':
    replicate_entire_study()
    
    #delthis = 'results/delthis'
    #if not os.path.exists(delthis):
    #    os.mkdir(delthis)
    #circgenetics_replication.ReplicateCircGenetics(delthis)
