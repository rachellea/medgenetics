# main.py

import os
import datetime

#Custom imports
from src import run_models, circgenetics_replication, visualization1, visualization1_all, visualization2_all, visualization3_all
from data import clean_data

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
    
    #Visualization - One Figure Summarizing Everything
    visualization1_all.MakePanelFigure(date_dir)
    visualization2_all.MakePanelFigure_SensSpec(date_dir)
    visualization3_all.MakeFigure_MysteryViolin(date_dir)
    
    
if __name__=='__main__':
    replicate_entire_study()
    