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
from models import mlp
from models import regression

# for checking purposes/sanity check
from sklearn import neural_network




if __name__=='__main__':
    #variations = {'noSN':['Position','Conservation'],
    #    'withSN':['Position','Conservation','SigNoise']}
    variations = {'withSN':['Position', 'Conservation', 'SigNoise']}
    for descriptor in variations:
        cont_vars = variations[descriptor]
        shared_args = {'impute':False,
                        'impute_these_categorical':[],
                        'impute_these_continuous':[],
                        'one_hotify':True,
                        'one_hotify_these_categorical':['Consensus','Change','Domain'],
#cat_vars
                        'normalize_data':False, # change to false if performing best model
                        'normalize_these_continuous':cont_vars,
                        'seed':10393, #make it 12345 for original split
                        'batch_size':300}
        
        layers = [[20],[30,20],[60,20],[60,60, 20],[120,60,20],[40],[40,40],[60,40], [120,60,40]]
        layer = layers[8]
        RunGeneModel(gene_name='scn5a', descriptor=descriptor,shared_args = shared_args, \
                     cols_to_delete=list(set(['Position','Conservation','SigNoise'])-set(cont_vars)),\
                     num_ensemble=0, cv_fold_lg=10, cv_fold_mlp=10, layer = layer).do_all()



