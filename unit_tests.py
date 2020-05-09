#unit_tests.py

import os
import numpy as np
import pandas as pd

from data import clean_data

#############
# Functions #-------------------------------------------------------------------
#############
def testing_ryr2_prep_data():
    """Test that domain information and conservation information were
    added correctly to Healthy and Pathologic examples"""
    features_to_use= ['Position', 'Conservation', 'SigNoise','Consensus','Change','Domain','PSSM','RateOfEvolution']
    genemodel = clean_data.PrepareData('ryr2',results_dir='',features_to_use = features_to_use)
    x = genemodel.ag.inputx
    assert ((x['Position'] ==24) & (x['Consensus']=='C')
        & (x['Change'] == 'R') & (x['Label']==0)
        & (x['Domain']=='NTD')
        & ((x['Conservation']-0.548673) < 1e-5)
        & (x['SigNoise']==0)
        & (x['PSSM']==-4)
        & (x['RateOfEvolution']) -(-0.271) < 1e-5).any()
    assert ((x['Position'] ==2111) & (x['Consensus']=='V')
        & (x['Change'] == 'A') & (x['Label']==0)
        & (x['Domain']=='HD1')
        & ((x['Conservation']-0.725663717) < 1e-5)
        & (x['SigNoise']==2.433325988)
        & (x['PSSM']==0)
        & (x['RateOfEvolution'] - 1.169 < 1e-5)).any()
    assert ((x['Position'] ==4851) & (x['Consensus']=='F')
        & (x['Change'] == 'C') & (x['Label']==1)
        & (x['Domain']=='Channel-domain')
        & ((x['Conservation']-0.769911504) < 1e-5)
        & (x['SigNoise'] == 654.1233578)
        & (x['PSSM']==-3)
        & (x['RateOfEvolution']) - (-0.939) < 1e-5).any()
    os.remove('ryr2.pickle')
    os.remove('ryr2_data_cleaning_history.csv')
    print('Passed testing_ryr2_prep_data()')

if __name__=='__main__':
    testing_ryr2_prep_data()