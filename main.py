# main.py

import os
import datetime

#Custom imports
from src import run_models, circgenetics_replication, visualization1, visualization1_all, visualization2_all
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
    all_features = ['Position', 'Conservation', 'SigNoise', 'Consensus', 'Change', 'PSSM', 'RateOfEvolution']
    d = clean_data.PrepareData(gene_name, results_dir, features_to_use=all_features)
    if what_to_run == 'grid_search':
        run_models.RunPredictiveModels(gene_name, modeling_approach, results_dir, d.real_data_split, what_to_run, testing=False)
    elif what_to_run == 'test_pred':
        run_models.RunPredictiveModels(gene_name, modeling_approach, results_dir, d.real_data_split, what_to_run, testing=False)
    elif what_to_run == 'mysteryAA_pred':
        run_models.PredictMysteryAAs(gene_name, modeling_approach, results_dir, d.real_data_split, d.mysteryAAs_dict)

def replicate_circgenetics():
    #Directories
    date_dir = os.path.join('results',datetime.datetime.today().strftime('%Y-%m-%d'))
    if not os.path.exists(date_dir):
        os.mkdir(date_dir)
    results_dir_circgenetics = os.path.abspath(os.path.join(date_dir,'circgenetics'))
    if not os.path.exists(results_dir_circgenetics):
        os.mkdir(results_dir_circgenetics)
    
    #Data and run
    d = clean_data.Prepare_KCNQ1_CircGenetics(results_dir_circgenetics)
    circgenetics_replication.ReplicateCircGenetics(results_dir_circgenetics, d.real_data_split)
    

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
    return date_dir, results_dir_ryr2, results_dir_kcnq1, results_dir_kcnh2, results_dir_scn5a

def replicate_entire_study():
    date_dir, results_dir_ryr2, results_dir_kcnq1, results_dir_kcnh2, results_dir_scn5a = make_results_dirs()
    
    #Logistic Regression
    run('ryr2',what_to_run='grid_search',modeling_approach='LR',results_dir = results_dir_ryr2)
    run('ryr2',what_to_run='test_pred',modeling_approach='LR',results_dir = results_dir_ryr2)
    run('ryr2',what_to_run='mysteryAA_pred',modeling_approach='LR',results_dir = results_dir_ryr2)
    
    run('kcnq1',what_to_run='grid_search',modeling_approach='LR',results_dir = results_dir_kcnq1)
    run('kcnq1',what_to_run='test_pred',modeling_approach='LR',results_dir = results_dir_kcnq1)
    run('kcnq1',what_to_run='mysteryAA_pred',modeling_approach='LR',results_dir = results_dir_kcnq1)
    
    run('kcnh2',what_to_run='grid_search',modeling_approach='LR',results_dir = results_dir_kcnh2)
    run('kcnh2',what_to_run='test_pred',modeling_approach='LR',results_dir = results_dir_kcnh2)
    run('kcnh2',what_to_run='mysteryAA_pred',modeling_approach='LR',results_dir = results_dir_kcnh2)
    
    run('scn5a',what_to_run='grid_search',modeling_approach='LR',results_dir = results_dir_scn5a)
    run('scn5a',what_to_run='test_pred',modeling_approach='LR',results_dir = results_dir_scn5a)
    run('scn5a',what_to_run='mysteryAA_pred',modeling_approach='LR',results_dir = results_dir_scn5a)
    
    #MLPs
    run('ryr2',what_to_run='grid_search',modeling_approach='MLP',results_dir = results_dir_ryr2)
    run('ryr2',what_to_run='test_pred',modeling_approach='MLP',results_dir = results_dir_ryr2)
    run('ryr2',what_to_run='mysteryAA_pred',modeling_approach='MLP',results_dir = results_dir_ryr2)
    
    run('kcnq1',what_to_run='grid_search',modeling_approach='MLP',results_dir = results_dir_kcnq1)
    run('kcnq1',what_to_run='test_pred',modeling_approach='MLP',results_dir = results_dir_kcnq1)
    run('kcnq1',what_to_run='mysteryAA_pred',modeling_approach='MLP',results_dir = results_dir_kcnq1)
    
    run('kcnh2',what_to_run='grid_search',modeling_approach='MLP',results_dir = results_dir_kcnh2)
    run('kcnh2',what_to_run='test_pred',modeling_approach='MLP',results_dir = results_dir_kcnh2)
    run('kcnh2',what_to_run='mysteryAA_pred',modeling_approach='MLP',results_dir = results_dir_kcnh2)
    
    run('scn5a',what_to_run='grid_search',modeling_approach='MLP',results_dir = results_dir_scn5a)
    run('scn5a',what_to_run='test_pred',modeling_approach='MLP',results_dir = results_dir_scn5a)
    run('scn5a',what_to_run='mysteryAA_pred',modeling_approach='MLP',results_dir = results_dir_scn5a)
    
    #Visualization - Separate Figure for Each Gene/Performance Metric
    visualization1.MakeAllFigures('ryr2',results_dir_ryr2)
    visualization1.MakeAllFigures('kcnq1',results_dir_kcnq1)
    visualization1.MakeAllFigures('kcnh2',results_dir_kcnh2)
    visualization1.MakeAllFigures('scn5a',results_dir_scn5a)
    
    #Visualization - One Figure Summarizing Everything
    visualization1_all.MakePanelFigure(date_dir)
    visualization2_all.MakePanelFigure_SensSpec(date_dir)
    
    
if __name__=='__main__':
    replicate_circgenetics()
    replicate_entire_study()
    