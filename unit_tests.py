#ryr2_code_testing.py

import numpy as np
import pandas as pd

import main
from data import utils
from data import clean_data

#############
# Functions #-------------------------------------------------------------------
#############
def testing_ryr2_prep_data():
    """Test that domain information and conservation information were
    added correctly to Healthy and Pathologic examples"""
    genemodel = main.RunGeneModel('ryr2','',{})
    genemodel._prep_data()
    x = genemodel.inputx
    assert ((x['Position'] ==24) & (x['Consensus']=='C')
        & (x['Change'] == 'R') & (x['Label']==0)
        & (x['Domain']=='NTD')
        & ((x['Conservation']-0.548673) < 1e-5)
        & (x['SigNoise']==0)).any()
    assert ((x['Position'] ==2111) & (x['Consensus']=='V')
        & (x['Change'] == 'A') & (x['Label']==0)
        & (x['Domain']=='HD1')
        & ((x['Conservation']-0.725663717) < 1e-5)
        & (x['SigNoise']==1810.216864)).any()
    assert ((x['Position'] ==4851) & (x['Consensus']=='F')
        & (x['Change'] == 'C') & (x['Label']==1)
        & (x['Domain']=='Channel-domain')
        & ((x['Conservation']-0.769911504) < 1e-5)
        & (x['SigNoise'] == 369458.1281)).any()
    print('Passed testing_ryr2_prep_data()')

def testing_utils():
    fake = pd.DataFrame(np.array([['828','P','C','0'], ['2110','C','D','1'], ['4487','D','A','0']]),
                        columns=['Position','Consensus','Change','Label'])
    fake['Position'] = pd.to_numeric(fake['Position'], downcast = 'integer')
    fake['Label'] = pd.to_numeric(fake['Label'], downcast = 'integer')
    shared_args = {'impute':False,
                        'impute_these_categorical':[],
                        'impute_these_continuous':[],
                        'one_hotify':True,
                        'one_hotify_these_categorical':['Consensus','Change'],
                        'normalize_data':True,
                        'normalize_these_continuous':['Position','Conservation'],
                        'seed':5, #must be 5 for this test (order stays the same)
                        'batch_size':300}
    split_args = {'train_percent':1.0,
                'valid_percent':0,
                'test_percent':0,
                'max_position':4487,
                'columns_to_ensure':['Position','Consensus_P','Change_C',
            'Conservation','Domain_SPRY2-first','Consensus_C','Change_D',
            'Domain_Handle-domain','Consensus_D','Change_A',
            'Domain_Channel-domain']}
    
    ag = clean_data.AnnotatedGene('ryr2')
    ag.inputx = fake[['Position','Consensus','Change','Label']]
    ag.annotate_everything() #add Domain and Conservation
    inputx = ag.inputx
    
    #Get code output
    genemodel = main.RunGeneModel('ryr2','',shared_args)
    genemodel._prep_split_data(inputx, split_args)
    fakesplit = genemodel.real_data_split
    
    #match column order
    fakesplit_df = pd.DataFrame(fakesplit.train.data, columns = fakesplit.train.data_meanings)
    fakesplit_df = fakesplit_df[split_args['columns_to_ensure']]
    
    #Construct the expected answer
    expected_df = pd.DataFrame(np.zeros((3,11)), columns = split_args['columns_to_ensure'])
    expected_df.loc[0,'Position'] = 828
    expected_df.loc[0,'Consensus_P'] = 1
    expected_df.loc[0,'Change_C'] = 1
    expected_df.loc[0,'Conservation'] = 0.82300885
    expected_df.loc[0,'Domain_SPRY2-first'] = 1
    
    expected_df.loc[1,'Position'] = 2110
    expected_df.loc[1,'Consensus_C'] = 1
    expected_df.loc[1,'Change_D'] = 1
    expected_df.loc[1,'Conservation'] = 0.575221239
    expected_df.loc[1,'Domain_Handle-domain'] = 1
    
    expected_df.loc[2,'Position'] = 4487
    expected_df.loc[2,'Consensus_D'] = 1
    expected_df.loc[2,'Change_A'] = 1
    expected_df.loc[2,'Conservation'] = 0.699115044
    expected_df.loc[2,'Domain_Channel-domain'] = 1
    
    mean_pos = 2475; std_pos = 1515.91314615
    expected_df['Position'] = (expected_df['Position']-mean_pos)/std_pos
    
    cons = [0.82300885, 0.575221239, 0.699115044]
    expected_df['Conservation'] = (expected_df['Conservation']-np.mean(cons))/np.std(cons)
    
    assert utils.arrays_are_close_enough(fakesplit_df.values, expected_df.values, tol =  1e-6)
    print('Passed testing_utils()')


if __name__=='__main__':
    testing_ryr2_prep_data()
    testing_utils()