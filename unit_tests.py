#ryr2_code_testing.py

import numpy as np
import pandas as pd

import main
from data_handling import utils
###########
# Globals #---------------------------------------------------------------------
###########
COLUMNS = ['Position', 'Conservation', 'Consensus_A',
            'Consensus_C', 'Consensus_D', 'Consensus_E', 'Consensus_F',
            'Consensus_G', 'Consensus_H','Consensus_I', 'Consensus_K',
            'Consensus_L', 'Consensus_M', 'Consensus_N', 'Consensus_P',
            'Consensus_Q', 'Consensus_R', 'Consensus_S', 'Consensus_T',
            'Consensus_V', 'Consensus_W', 'Consensus_Y', 'Change_A', 'Change_C',
            'Change_D', 'Change_E', 'Change_F', 'Change_G', 'Change_H',
            'Change_I', 'Change_K', 'Change_L', 'Change_M', 'Change_N',
            'Change_P', 'Change_Q', 'Change_R', 'Change_S', 'Change_T',
            'Change_V', 'Change_W', 'Change_Y', 'Domain_Central-domain',
            'Domain_Channel-domain', 'Domain_HD1', 'Domain_HD2',
            'Domain_Handle-domain', 'Domain_NTD', 'Domain_Outside', 'Domain_P1',
             'Domain_P2', 'Domain_SPRY1-first', 'Domain_SPRY1-second',
             'Domain_SPRY1-third', 'Domain_SPRY2-first', 'Domain_SPRY2-second',
             'Domain_SPRY3-first', 'Domain_SPRY3-second']

#############
# Functions #-------------------------------------------------------------------
#############
def testing_add_domain_and_conservation():
    """Test that domain information and conservation information were
    added correctly to Healthy and Pathologic examples"""
    result = main.RYR2(inputx = main.prepare_real_data(),
                       train_percent = 0.7, valid_percent = 0.15,
                       test_percent = 0.15)
    x = result.inputx
    assert ((x['Position'] ==24) & (x['Consensus']=='C')
        & (x['Change'] == 'R') & (x['Label']==0)
        & (x['Domain']=='NTD')
        & ((x['Conservation']-0.548673) < 1e-5)).any()
    assert ((x['Position'] ==2111) & (x['Consensus']=='V')
        & (x['Change'] == 'A') & (x['Label']==0)
        & (x['Domain']=='HD1')
        & ((x['Conservation']-0.725663717) < 1e-5)).any()
    assert ((x['Position'] ==4851) & (x['Consensus']=='F')
        & (x['Change'] == 'C') & (x['Label']==1)
        & (x['Domain']=='Channel-domain')
        & ((x['Conservation']-0.769911504) < 1e-5)).any()
    assert result.split.train.data_meanings == COLUMNS
    print('Passed testing_add_domain_and_conservation()')

def testing_utils():
    fake = pd.DataFrame(np.array([['828','P','C','0'], ['2110','C','D','1'], ['4487','D','A','0']]),
                        columns=['Position','Consensus','Change','Label'])
    fake['Position'] = pd.to_numeric(fake['Position'], downcast = 'integer')
    fake['Label'] = pd.to_numeric(fake['Label'], downcast = 'integer')
    fakesplit = main.RYR2(inputx = fake,
                       train_percent = 1.0, valid_percent = 0,
                       test_percent = 0, seed = 5).split #seed of 5 means order stays same
    
    #Construct the expected answer
    expected_df = pd.DataFrame(np.zeros((3,58)), columns = COLUMNS)
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
    
    assert utils.arrays_are_close_enough(fakesplit.train.data, expected_df.values, tol =  1e-6)
    assert fakesplit.train.data_meanings == COLUMNS
    print('Passed testing_utils()')

if __name__=='__main__':
    testing_add_domain_and_conservation()
    testing_utils()
