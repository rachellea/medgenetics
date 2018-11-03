#framingham_model.py

import copy
import os
import pickle
import pandas as pd
import numpy as np

#Custom imports
import utils
import mlp_model

###########
# Classes #---------------------------------------------------------------------
###########

class RYR2(object):
    """Given data in <inputx>, add domain and conservation information, and then
    split into train, test, and valid sets."""
    def __init__(self,
                 inputx,
                 train_percent,
                 valid_percent,
                 test_percent):
        """<inputx> is a pandas dataframe potentially containing inputx healthy and
        diseased data, with columns Position, Consensus, and Change."""
        self.inputx = inputx
        
        #add a column denoting the component of the protein it is part of
        self.add_domain_info()
        #add a column with conservation score
        self.add_conservation_info()
        
        #testing only on real data (valid_percent>0), not everyAA (where train_percent=1)
        if valid_percent>0: self.test_domains_and_conservation() 
        
        #Create splits
        print('Shape of data and labels together',str(self.inputx.shape))
       # print('Columns of data and labels together',str(self.inputx.columns.values))
        clean_data = copy.deepcopy(self.inputx[['Position','Consensus','Change','Domain','Conservation']])
        clean_labels = copy.deepcopy(self.inputx[['Label']])
        print('Fraction of diseased:',str( np.sum(clean_labels)/len(clean_labels) ) )
        self.split = utils.Splits(data = clean_data,
                             labels = clean_labels,
                             train_percent = train_percent,
                             valid_percent = valid_percent,
                             test_percent = test_percent,
                             impute = False,
                             impute_these_categorical = [],
                             impute_these_continuous = [],
                             one_hotify = True,
                             one_hotify_these_categorical = ['Consensus', 'Change', 'Domain'],
                             normalize_data = True,
                             normalize_these_continuous = ['Position','Conservation'],
                             seed = 1029384,
                             batch_size = 300)
    
    def add_domain_info(self):
        self.domains = {'NTD':[1,642],
            'SPRY1-first':[643,837], 'SPRY1-second':[1459,1484],'SPRY1-third':[1606,1641],
            'P1':[861,1066],
            'SPRY2-first':[828,856],'SPRY2-second':[1084,1254],
            'SPRY3-first':[1255,1458],'SPRY3-second':[1485,1605],
            'Handle-domain':[1642,2110],
            'HD1':[2111,2679],'P2':[2701,2907],'HD2':[2982,3528],
            'Central-domain':[3613,4207],'Channel-domain':[4486,4968]}
        self.inputx['Domain'] = ''
        domain_added_count = 0
        domain_not_added_count = 0
        for rowname in self.inputx.index.values.tolist():
            domain_added = False
            position = self.inputx.loc[rowname,'Position']
            for key in self.domains:
                start = self.domains[key][0]
                stop = self.domains[key][1]
                if (start <= position <=stop):
                    self.inputx.loc[rowname,'Domain'] = key
                    domain_added = True
                    domain_added_count+=1
            if not domain_added:
                self.inputx.loc[rowname,'Domain'] = 'Outside'
                domain_not_added_count+=1
        print('Added domain annotation to',str(domain_added_count),'examples')
        print('No domain annotation for',str(domain_not_added_count),'examples')
    
    def add_conservation_info(self):
        conservation = pd.read_csv('/data/rlb61/Collaborations/Landstrom/datafiles/ryr2_conservation.csv',
                                   header = None,
                                   names=['Position','Conservation'])
        self.inputx = self.inputx.merge(conservation, how='inner', on='Position')

    def test_domains_and_conservation(self):
        assert ((self.inputx['Position'] ==24) & (self.inputx['Consensus']=='C')
            & (self.inputx['Change'] == 'R') & (self.inputx['Label']==0)
            & (self.inputx['Domain']=='NTD')
            & ((self.inputx['Conservation']-0.548673) < 1e-5)).any()
        assert ((self.inputx['Position'] ==2111) & (self.inputx['Consensus']=='V')
            & (self.inputx['Change'] == 'A') & (self.inputx['Label']==0)
            & (self.inputx['Domain']=='HD1')
            & ((self.inputx['Conservation']-0.725663717) < 1e-5)).any()
        assert ((self.inputx['Position'] ==4851) & (self.inputx['Consensus']=='F')
            & (self.inputx['Change'] == 'C') & (self.inputx['Label']==1)
            & (self.inputx['Domain']=='Channel-domain')
            & ((self.inputx['Conservation']-0.769911504) < 1e-5)).any()

#############
# Functions #-------------------------------------------------------------------
#############
def prepare_real_data():
    """Return 'inputx' dataframe for healthy and diseased"""
    #Data processing
    healthy = pd.read_csv('/data/rlb61/Collaborations/Landstrom/datafiles/Healthy_Variants_repaired.csv', header=0)
    healthy['Label'] = 0
    diseased = pd.read_csv('/data/rlb61/Collaborations/Landstrom/datafiles/Pathologic_Variants_repaired.csv', header=0)
    diseased['Label'] = 1
    #columns ['Position', 'Consensus', 'Change', 'Label']
    #healthy (1722, 4)
    #diseased (151, 4)
      
    #concat
    helen = healthy.shape[0]
    dislen = diseased.shape[0]
    diseased.index = range(helen,helen+dislen)
    return pd.concat([healthy,diseased])

def prepare_everyAA(real_data):
    """Create a pandas dataframe containing every possible amino acid change
    at every position, except for anything present in the real data."""
    print('Creating everyAA')
    #Read in the RYR2 sequence
    ryr2seq = ''
    with open('/data/rlb61/Collaborations/Landstrom/datafiles/RyR2_Protein_Sequence_Human.txt','r') as f:
        for line in f:
            ryr2seq = ryr2seq + line.rstrip()
    assert len(ryr2seq) == 4967
    ryr2seq = ryr2seq.upper()
    
    #List of all possible AAs:
    possible_aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
                    'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    max_length = len(ryr2seq)*(len(possible_aas)-1) #don't count C-->C (not a change)
    print('everyAA max length is',max_length)
    
    #Fill dataframe
    everyAA = pd.DataFrame(np.empty((max_length,3), dtype='str'), columns = ['Position','Consensus','Change'])
    dfindex = 0
    for position in range(len(ryr2seq)):
        consensus = ryr2seq[position]
        for aa in possible_aas:
            if aa != consensus: 
                everyAA.at[dfindex, 'Position'] = str(position+1) #one-based
                everyAA.at[dfindex, 'Consensus'] = consensus
                everyAA.at[dfindex, 'Change'] = aa
                dfindex+=1
        if position % 500 == 0: print('Done with',position)
    assert max_length == dfindex
    #Transform position to int
    everyAA['Position'] = pd.to_numeric(everyAA['Position'], downcast='integer')
    
    #Remove anything that's already present in the real data
    #Fast way: I already know the duplicates have been removed, so I will
    #create "false duplicates" by concatenating. Then I will remove all dups
    print('Shape of everyAA:',everyAA.shape)
    print('After removing real data, rows should be',everyAA.shape[0] - real_data.shape[0])
    #make it int16
    
    real_data['Position'] = pd.to_numeric(real_data['Position'], downcast='integer')
    everyAA = pd.concat([everyAA, real_data[['Position','Consensus','Change']]],axis=0)
    everyAA = everyAA.drop_duplicates(keep=False) #drop all dups
    print('After removing real data, rows are',everyAA.shape[0])
    everyAA['Label']=0 #dummy, never used
    return everyAA

def make_small_everyAA_for_testing():
    possible_aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
                    'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    everyAA = pd.DataFrame(np.transpose(np.array([possible_aas, possible_aas])),
                        columns=['Consensus','Change'])
    everyAA['Label'] = 0
    everyAA['Position'] = [x for x in range(1,21)]
    return everyAA

##### #
# Run #-------------------------------------------------------------------------
#######
def run_ryr2_model():
    """Prepare real data, prepare everyAA data, and run the MLP."""
    #Real data with healthy and diseased
    realdata = prepare_real_data()
    real_data_split = RYR2(inputx = realdata,
                       train_percent = 0.7, valid_percent = 0.15,
                       test_percent = 0.15).split
    
    #Fake data with all possible combos of every AA at every position
    everyAA = prepare_everyAA(realdata)
    
    everyAA_split = RYR2( inputx = everyAA,
                    train_percent = 1.0, valid_percent = 0,
                    test_percent = 0).split.train
    assert everyAA_split.data.shape[0] == everyAA.shape[0]
    
    #Run MLP
    m = mlp_model.MLP(descriptor='ryr2_model',
                  split=copy.deepcopy(real_data_split),
                  decision_threshold = 0.5,
                  num_epochs = 1000,
                  learningrate = 1e-4,
                  mlp_layers = copy.deepcopy([30,20]),
                  exclusive_classes = True,
                  save_model = False,
                  everyAA = everyAA_split)
    m.run_all()
    

if __name__=='__main__':
    run_ryr2_model()