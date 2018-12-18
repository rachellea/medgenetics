#Rachel Ballantyne Draelos

import os
import copy
import pandas as pd
import numpy as np

class BareGene(object):
    def __init__(self, gene_name):
        """Return one dataframe with the cleaned genetic data and the labels.
        Performs these tasks:
            > Remove non-allowed symbols
            > Remove entries that disagree with reference sequence
            > Remove duplicates
            > Merge self.healthy and self.diseased dataframes
        Also save a dataframe as <gene_name>_data_cleaning_history.csv that documents
        what was removed and why."""
        self.gene_name = gene_name
        self.obtain_reference_sequence() #creates self.geneseq
        self.history = pd.DataFrame(np.empty(100000,5),dtype=str), columns=['Healthy_or_Path','Consensus','Position','Change','Reason_Removed'])
        self.history_idx = 0
        
        self.healthy = pd.read_csv(os.path.join(gene_name,gene_name+'_variants_healthy_raw.csv'),header=0)
        self.diseased = pd.read_csv(os.path.join(gene_name,gene_name+'_variants_pathologic_raw.csv'),header=0)
        
        #Copies for comparison at the end
        self.healthy_original = copy.deepcopy(self.healthy)
        self.diseased_original = copy.deepcopy(self.diseased)
        
        #Clean the data
        for df in [self.healthy, self.diseased]:
            df = self.remove_disallowed_symbols(df)
            df = self.remove_consensus_equals_change(df)
            df = self.remove_consensus_disagrees_with_reference(df)
            df = self.remove_per_file_dups(df)
        
        #Merge healthy and diseased
        self.all = self.merge_healthy_and_diseased()
        
        #remove things listed as both healthy and pathologic
        self.all = self.all.drop_duplicates(subset=['Consensus','Position','Change'],
                                            keep=False)
        return self.all
    
    def obtain_reference_sequence(self):
        geneseq = ''
        with open(os.path.join(self.gene_name,self.gene_name+'_reference_sequence.txt', 'r') as f:
            for line in f:
                geneseq = geneseq + line.rstrip()
        if self.gene_name == 'ryr2':
            assert len(geneseq) == 4967
        elif self.gene_name == 'scn5a':
            pass #TODO (for scn5a and all other genes)
        self.geneseq = geneseq.upper()
    
    def remove_disallowed_symbols(self, df):
        return df
    
    def remove_consensus_equals_change(self, df):
        """Remove anything that is not a real mutation (i.e. the change is the
        same as the consensus sequence)"""
        print('remove_consensus_equals_change():\n\tRows before:', df.shape[0])
        df = df.loc[df['Consensus']!=df['Change']]
        print('\tRows after:',healthy.shape[0])
    
    def remove_consensus_disagrees_with_reference(self):
        """Remove anything where the consensus amino acid disagrees with the
        ground truth reference amino acid for that position.
        Note that 'healthy' and 'diseased' are one-based. So position 5 in self.healthy
        should be referenced as position 4 in self.geneseq (the reference
        sequence)"""
        reference_df = pd.DataFrame(np.transpose(np.array([list(self.geneseq), [x for x in range(1,len(self.geneseq)+1)]])),
                                    columns=['Reference','Position'])
        #pos columns must have same type in order to be merged
        reference_df['Position'] = pd.to_numeric(reference_df['Position'], downcast='integer')
        self.healthy['Position'] = pd.to_numeric(healthy['Position'], downcast='integer')
        self.diseased['Position'] = pd.to_numeric(diseased['Position'], downcast='integer')
        
        self.healthy = self.healthy.merge(copy.deepcopy(reference_df), how = 'left', on='Position')
        self.healthy = self.healthy.loc[healthy['Consensus']==healthy['Reference']]
        self.diseased = self.diseased.merge(copy.deepcopy(reference_df), how = 'left', on = 'Position')
        self.diseased = self.diseased.loc[diseased['Consensus']==diseased['Reference']]
        print('Healthy rows after removing self.healthy entries where consensus!=reference:',healthy.shape[0])
        print('Diseased rows after removing self.diseased entries where consensus!=reference:',diseased.shape[0])
    
    def remove_per_file_dups(self):
        """Remove any duplicates within a file"""
        print('Healthy rows before removing self.healthy dups:',healthy.shape[0])
        print('Diseased rows before removing self.diseased dups:',diseased.shape[0])
        self.healthy = self.healthy.drop_duplicates(keep = 'first')
        disease = self.diseased.drop_duplicates(keep = 'first')
        print('Healthy rows after removing self.healthy dups:',healthy.shape[0])
        print('Diseased rows after removing self.diseased dups:',diseased.shape[0])        
    
    def merge_healthy_and_diseased(self):
        """Return 'inputx' dataframe for self.healthy and self.diseased"""
        #add Label column, for overall columns ['Position', 'Consensus', 'Change', 'Label']
        self.healthy['Label'] = 0
        self.diseased['Label'] = 1
        
        #concat
        helen = self.healthy.shape[0]
        dislen = self.diseased.shape[0]
        self.diseased.index = range(helen,helen+dislen)
        return pd.concat([self.healthy,self.diseased])


class AnnotatedGene(object):
    def __init__(self, gene_name):
        """Add domain and conservation information to the bare gene and to everyAA;
        return gene and everyAA dataframes"""
        self.inputx = BareGene(gene_name)
        self.everyAA = prepare_everyAA(self.inputx)
        
        for df in [self.inputx, self.everyAA]:
            #add a column denoting the component of the protein it is part of
            df = self.add_domain_info(df)
            #add a column with conservation score
            df = self.add_conservation_info(df)
        
        return self.inputx, self.everyAA
       
    def add_domain_info(self, df):
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
        for rowname in df.index.values.tolist():
            domain_added = False
            position = df.loc[rowname,'Position']
            for key in self.domains:
                start = self.domains[key][0]
                stop = self.domains[key][1]
                if (start <= position <=stop):
                    df.loc[rowname,'Domain'] = key
                    domain_added = True
                    domain_added_count+=1
            if not domain_added:
                df.loc[rowname,'Domain'] = 'Outside'
                domain_not_added_count+=1
        print('Added domain annotation to',str(domain_added_count),'examples')
        print('No domain annotation for',str(domain_not_added_count),'examples')
        return df
    
    def add_conservation_info(self, df):
        conservation = pd.read_csv('/data/rlb61/Collaborations/Landstrom/datafiles/gene_conservation.csv',
                                   header = None,
                                   names=['Position','Conservation'])
        return df.merge(conservation, how='inner', on='Position')
        

#####################
# everyAA functions #-----------------------------------------------------------
#####################
def prepare_everyAA(real_data):
    """Create a pandas dataframe containing every possible amino acid change
    at every position, except for anything present in the real data."""
    print('Creating everyAA')
    #Read in the PreparedGeneticData sequence
    geneseq = ''
    with open('/data/rlb61/Collaborations/Landstrom/datafiles/Gene_Protein_Sequence_Human.txt','r') as f:
        for line in f:
            geneseq = geneseq + line.rstrip()
    assert len(geneseq) == 4967
    geneseq = geneseq.upper()
    
    #List of all possible AAs:
    possible_aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
                    'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    max_length = len(geneseq)*(len(possible_aas)-1) #don't count C-->C (not a change)
    print('everyAA max length is',max_length)
    
    #Fill dataframe
    everyAA = pd.DataFrame(np.empty((max_length,3), dtype='str'), columns = ['Position','Consensus','Change'])
    dfindex = 0
    for position in range(len(geneseq)):
        consensus = geneseq[position]
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
    """Dummy version of everyAA to be able to run the MLP code quickly"""
    possible_aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
                    'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    everyAA = pd.DataFrame(np.transpose(np.array([possible_aas, possible_aas])),
                        columns=['Consensus','Change'])
    everyAA['Label'] = 0
    everyAA['Position'] = [x for x in range(1,21)]
    return everyAA
