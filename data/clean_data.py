#clean_data.py

import os
import copy
import pickle
import numpy as np
import pandas as pd

import sys
sys.path.append('./data')

try:
    from . import utils
except:
    import utils

AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
               'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']
#X is not really an amino acid; it's a stop codon

class BareGene(object):
    def __init__(self, gene_name, results_dir):
        """Return one dataframe with the cleaned genetic data and the labels.
        Performs these tasks:
            > Remove non-allowed symbols
            > Remove entries that disagree with reference sequence
            > Remove duplicates
            > Merge self.healthy and self.diseased dataframes
        Also save a dataframe as <gene_name>_data_cleaning_history.csv that documents
        what was removed and why."""
        self.gene_name = gene_name
        self.obtain_reference_sequence() #creates self.reference_df
        self.history = pd.DataFrame(np.empty((100000,5),dtype=str), columns=['Description','Consensus','Position','Change','Reason_Removed'])
        self.history_idx = 0
        
        self.healthy = pd.read_csv(os.path.join('data',os.path.join(gene_name,gene_name+'_variants_healthy_raw.csv')),header=0)
        self.diseased = pd.read_csv(os.path.join('data',os.path.join(gene_name,gene_name+'_variants_pathologic_raw.csv')),header=0)
        wes = pd.read_csv(os.path.join('data',os.path.join(gene_name,gene_name+'_variants_wes_raw.csv')),header=0)
        clinvar = pd.read_csv(os.path.join('data',os.path.join(gene_name,gene_name+'_variants_clinvar_raw.csv')),header=0)
        self.mysteryAAs = pd.concat([wes,clinvar]).reset_index()
       
        #Copies for comparison at the end
        self.healthy_original = copy.deepcopy(self.healthy)
        self.diseased_original = copy.deepcopy(self.diseased)
        
        #Clean the data and add in signal to noise
        dfs = {'healthy':self.healthy, 'pathologic':self.diseased,
                'mysteryAAs':self.mysteryAAs}
        for key in dfs.keys():
            df = dfs[key]
            print('\n***Working on',key,'***')
            df = self.remove_missing_values(df, key)
            df = self.clean_up_symbols(df, key)
            df = self.remove_consensus_equals_change(df, key)
            df = self.remove_consensus_disagrees_with_reference(df, key)
            df = self.remove_dups(df, key, 'first', 'duplicate_within_file_removed_first')
            dfs[key] = df

        # update the variables with clean data
        self.healthy = dfs['healthy']
        self.diseased = dfs['pathologic']
        self.mysteryAAs = dfs['mysteryAAs']
        
        #Merge healthy and diseased
        merged = self.merge_healthy_and_diseased(self.healthy, self.diseased)
        print('merged shape:',merged.shape)
        merged = self.remove_dups(merged, 'merged', False, 'duplicate_across_healthy_sick_removed_both')
        self.merged = merged.reset_index(drop=True)#Reindex
        
        #make sure there are no overlaps between mystery and merged:
        #note that duplicates between healthy and sick have been removed
        #so those can stay in mysteryAAs (makes sense since if they're listed
        #as both healthy and sick we don't know the right answer.)
        print('mysteryAAs shape:',self.mysteryAAs.shape)
        temp = copy.deepcopy(self.merged).drop(columns='Label')
        mystmerged = self.merge_healthy_and_diseased(temp, self.mysteryAAs)
        mystmerged = self.remove_dups(mystmerged, 'mystmerged', False, 'duplicate_across_mystery_and_healthysick_removed_both')
        self.mysteryAAs = (mystmerged[mystmerged['Label']==1]).drop(columns='Label')
        print('mysteryAAs shape after:',self.mysteryAAs.shape)
        
        #Save history
        self.history.to_csv(os.path.join(results_dir, self.gene_name+'_data_cleaning_history.csv'))
        
        #TODO write function that checks that everything listed in self.history
        #is present in the original files and is absent from the current version
        #of the file.
    ####################
    # Cleaning Methods #--------------------------------------------------------
    ####################    
    def remove_missing_values(self, df, key):
        """Remove rows with any missing values"""
        print('remove_missing_values()\n\tRows before:',df.shape[0])
        temp = df[df.isnull().any(axis=1)]
        self.update_history_using_temp(temp, key, 'missing_values')
        df = df.dropna(axis='index', how='any')
        print('\tRows after:',df.shape[0])
        return df
    
    def clean_up_symbols(self, df, key):
        """Clean up symbols so that everything in the Consensus and Change columns
        corresponds to a letter in AMINO_ACIDS, or is:
            fs (frameshift)
            dup (duplication)
            empty (nothing there, corresponds to symbol -)
            misc (any other combination of symbols including multiple AAs)"""
        print('clean_up_symbols()')
        global AMINO_ACIDS
        #Just print off when you make a change what that change is
        for choice in ['Consensus','Change']:
            for rowidx in df.index.values:
                current_aa = df.at[rowidx,choice]
                #print('current_aa is:', current_aa)
                if current_aa not in AMINO_ACIDS:
                    if current_aa not in ['dup','fs']:
                        #Then it's a weird aa and must be cleaned
                        if 'dup' in current_aa:
                            print('\t[',rowidx,',',choice,']',current_aa,'-->','dup')
                            df.at[rowidx,choice] = 'dup'
                        elif 'fs' in current_aa:
                            print('\t[',rowidx,',',choice,']',current_aa,'-->','fs')
                            df.at[rowidx,choice] = 'fs'
                        elif '-' == current_aa:
                            print('\t[',rowidx,',',choice,']',current_aa,'-->','empty')
                            df.at[rowidx,choice] = 'empty'
                        else:
                            print('\t[',rowidx,',',choice,']',current_aa,'-->','misc')
                            df.at[rowidx,choice] = 'misc'
        return df
    
    def remove_consensus_equals_change(self, df, key):
        """Remove anything that is not a real mutation (i.e. the change is the
        same as the consensus sequence)"""
        #Document history of upcoming change
        temp = df.loc[df['Consensus']==df['Change']]
        self.update_history_using_temp(temp, key, 'consensus_equals_change')
        
        #Perform change
        df = df.loc[df['Consensus']!=df['Change']]
        print('remove_consensus_equals_change(): Rows after:',df.shape[0])
        
        return df
    
    def remove_consensus_disagrees_with_reference(self, df, key):
        """Remove anything where the consensus amino acid disagrees with the
        ground truth reference amino acid for that position."""
        #Prepare for the removal
        reference_df = copy.deepcopy(self.reference_df)
        df['Position'] = pd.to_numeric(df['Position'], downcast='integer')
        df = df.merge(copy.deepcopy(reference_df), how = 'left', on='Position')
        
        #Document history of upcoming change
        temp =  df.loc[df['Consensus']!=df['Reference']]
        self.update_history_using_temp(temp, key, 'consensus_disagrees_w_reference')
        
        #Perform change
        df = df.loc[df['Consensus']==df['Reference']]
        print('remove_consensus_disagrees_with_reference(): Rows after:',df.shape[0])
        return df
    
    def remove_dups(self, df, key, keep, reason_removed):
        """Remove duplicates.
        if keep == 'first' then keep first occurrence
        if keep == False then drop all occurrences """
        #Document history of upcoming change
        cols = ['Consensus','Position','Change']
        # get a dataframe of the duplicataed rows
        temp = df[df.duplicated(subset=cols,keep=keep)]
        self.update_history_using_temp(temp, key, reason_removed)
        
        #Perform change
        print('remove_dups():')
        print('\tRows before:', df.shape[0])
        df = df.drop_duplicates(subset=cols, keep=keep)
        print('\tRows after:',df.shape[0])
        
        #Sanity check - make sure duplicates are gone
        assert df[df.duplicated(subset=['Consensus', 'Position', 'Change'], keep=False)].shape[0] == 0
        return df
    
    def merge_healthy_and_diseased(self, healthy, diseased):
        """Return 'inputx' dataframe for self.healthy and self.diseased"""
        #add Label column, for overall columns ['Position', 'Consensus', 'Change', 'Label']
        healthy['Label'] = 0
        diseased['Label'] = 1
        
        #concat
        helen = healthy.shape[0]
        dislen = diseased.shape[0]
        diseased.index = range(helen,helen+dislen)
        return pd.concat([healthy,diseased])

    ##################
    # Helper Methods #----------------------------------------------------------
    ##################    
    def obtain_reference_sequence(self):
        geneseq = get_geneseq(self.gene_name)
        #In this df we create the index starting from 1 because 'healthy' and
        #'diseased' use one-based positions and we must be consistent.
        reference_df = pd.DataFrame(np.transpose(np.array([list(geneseq), [x for x in range(1,len(geneseq)+1)]])),
                                    columns=['Reference','Position'])
        #pos columns must have same type in order to be merged with other df later
        reference_df['Position'] = pd.to_numeric(reference_df['Position'], downcast='integer')
        self.reference_df = reference_df
        #Save max position
        assert len(reference_df.index.values.tolist())==max(reference_df['Position'].values.tolist())
        self.max_position = max(reference_df['Position'].values.tolist())

    def update_history_using_temp(self, temp, key, reason_removed):
        """Update history of the data cleaning process"""
        for rowidx in temp.index.values:
            self.history.at[self.history_idx,'Description'] = key
            self.history.at[self.history_idx,'Consensus'] = temp.at[rowidx,'Consensus']
            self.history.at[self.history_idx,'Position'] = temp.at[rowidx,'Position']
            self.history.at[self.history_idx,'Change'] = temp.at[rowidx,'Change']
            if reason_removed == 'consensus_disagrees_w_reference':
                reason_removed += ': '+temp.at[rowidx,'Reference']
            self.history.at[self.history_idx,'Reason_Removed'] = reason_removed
            self.history_idx += 1

class AnnotatedGene(object):
    def __init__(self, gene_name, results_dir):
        """Add domain, conservation, and signal to noise information to the
        bare gene and to mysteryAAs; produce gene and mysteryAAs dataframes"""
        global AMINO_ACIDS
        self.gene_name = gene_name
        
        b = BareGene(gene_name, results_dir)
        self.inputx = b.merged
       
        # check if we have a clean data
        print("Checking inputx in clean data annotated gene")
        for i in range(len(self.inputx.values)):
            consensus = self.inputx.values[i][0]
            change = self.inputx.values[i][2]
            position = self.inputx.values[i][1]
            #self.ori_dict[(consensus, change)] = position

            if consensus == change:
                print(consensus, change)

        self.max_position = b.max_position
        self.mysteryAAs = b.mysteryAAs
    
    def annotate_everything(self):
        #add a column denoting the domain of the protein it is part of
        self.create_domain_dictionary()
        self.inputx = self.add_domain_info(self.inputx)
        self.mysteryAAs = self.add_domain_info(self.mysteryAAs)
        
        #replace any domain annotation that is not used with 'Outside'
        self.domains_used = list(set(self.inputx.loc[:,'Domain'].values.tolist()))
        self.domains_not_used = list(set(self.domains.keys()) - set(self.domains_used))
        if 'Outside' in self.domains_used:
            assert len(self.domains_used)+len(self.domains_not_used) == len(self.domains.keys())+1 #+1 for Outside
        if len(self.domains_not_used) > 0:
            self.mysteryAAs = self.mysteryAAs.replace(to_replace = self.domains_not_used, value = 'Outside')
        print('Domains used:',self.domains_used)
        print('Domains not used:', self.domains_not_used)
        
        #add a column with conservation score
        self.inputx = self.add_conservation_info(self.inputx)
        self.mysteryAAs = self.add_conservation_info(self.mysteryAAs)
        
        #add a column with signal to noise info
        self.inputx = self.add_signoise_info(self.inputx)
        self.mysteryAAs = self.add_signoise_info(self.mysteryAAs)
        
        self.columns_to_ensure = (['Position', 'Conservation','SigNoise']
            +['Consensus_'+letter for letter in AMINO_ACIDS]
            +['Change_'+letter for letter in AMINO_ACIDS]
            +self.domains_used)
        print('Done with AnnotatedGene')
       
    def add_domain_info(self, df):
        print('Adding domain info')
        df['Domain'] = ''
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
    
    def create_domain_dictionary(self):
        self.domains = {}
        domain_df = pd.read_csv(os.path.join('data',os.path.join(self.gene_name,self.gene_name+'_domains.csv')),
                                header=0)
        for rowidx in domain_df.index.values:
            start = domain_df.at[rowidx,'Start']
            stop = domain_df.at[rowidx,'Stop']
            self.domains[domain_df.at[rowidx,'Domain']] = [start,stop]
    
    def add_conservation_info(self, df):
        print('Adding conservation info')
        con_file = os.path.join('data',os.path.join(self.gene_name,self.gene_name+'_conservation.csv'))
        conservation = pd.read_csv(con_file,
                                   header = None,
                                   names=['Position','Conservation'])
        return df.merge(conservation, how='inner', on='Position')
    
    def add_signoise_info(self, df):
        print('Adding signoise info')
        signoise_file = os.path.join('data',os.path.join(self.gene_name,self.gene_name+'_signal_to_noise.csv')) #Columns Position, SigNoise
        signoise = pd.read_csv(signoise_file,
                               header = None,
                               names = ['Position','SigNoise'])
        return df.merge(signoise, how = 'inner', on = 'Position')


class Prepare_KCNQ1_CircGenetics(object):
    def __init__(self, results_dir):
        """Add PSSM and Rate of Evolution to the bare gene and make split"""
        global AMINO_ACIDS
        self.gene_name = 'kcnq1'
        b = BareGene(self.gene_name, results_dir)
        self.inputx = b.merged
        self.max_position = b.max_position
        self._add_pssm()
        self._add_roe()
        self._prep_train_val_data()
        print('Done with Prepare_KCNQ1_CircGenetics')
        
    def _add_pssm(self):
        print('Adding PSSM info')
        PSSM_AAs = ['A','G','I','L','V','M','F','W','P','C',
                    'S','T','Y','N','Q','H','K','R','D','E']
        pssm_file = os.path.join(os.path.join('data','circgenetics'),'kcnq1_pssm_matrix.csv')
        pssm = pd.read_csv(pssm_file,header = 0, index_col=0) #index is Position
        self.inputx['PSSM'] = np.nan
        for idx in self.inputx.index.values.tolist():
            position = self.inputx.at[idx,'Position']
            consensus = self.inputx.at[idx,'Consensus']
            change = self.inputx.at[idx,'Change']
            assert consensus == pssm.at[position,'Consensus']
            if change in PSSM_AAs:
                self.inputx.at[idx,'PSSM'] = pssm.at[position,change]
            else: #change is some weird amino acid, e.g. 'X'
                max_for_idx = np.amax(pssm[PSSM_AAs].loc[idx,:].values)
                print('Using the max PSSM',max_for_idx,'for change',change)
                self.inputx.at[idx,'PSSM'] = max_for_idx
    
    def _add_roe(self):
        print('Adding rate of evolution info')
        roe_file = os.path.join(os.path.join('data','circgenetics'),'kcnq1_rate_of_evolution.csv')
        roe = pd.read_csv(roe_file,header = 0)
        self.inputx = self.inputx.merge(roe, how = 'inner', on = 'Position')
    
    def _prep_train_val_data(self):
        self.inputx = self.inputx[['PSSM','RateOfEvolution','Label']]
        self.real_data_split = utils.Splits(data = self.inputx[['PSSM','RateOfEvolution']],
             labels=self.inputx[['Label']],
             one_hotify_these_categorical=[],
             normalize_these_continuous=['PSSM','RateOfEvolution'],
             columns_to_ensure=['PSSM','RateOfEvolution'],
             batch_size=16) #circgenetics paper did not specify batch size


class PrepareData(object):
    def __init__(self, gene_name, results_dir):
        """This class produces self.real_data_split and self.mysteryAAs_split
        which are needed for all the modeling."""
        print('*** Preparing data for',gene_name,'***')
        self.gene_name = gene_name
        self.results_dir = results_dir
        self.shared_args = {'one_hotify_these_categorical':['Consensus','Change','Domain'],
                'normalize_these_continuous':['Position', 'Conservation', 'SigNoise'],
                'batch_size':256}
        
        #Load real data consisting of benign and pathologic mutations
        ag = AnnotatedGene(self.gene_name, results_dir)
        ag.annotate_everything()
        self.ag = ag
        
        self._prep_train_val_data() #creates self.real_data_split
        self._prep_mysteryAAs() #creates self.mysteryAAs_split and self.ori_position
            
    def _prep_train_val_data(self):
        inputx = self.ag.inputx
        self._run_sanity_check(inputx)
        
        #Prepare split data
        split_args = {'columns_to_ensure':self.ag.columns_to_ensure}
        all_args = {**self.shared_args, **split_args }
        data = (copy.deepcopy(inputx)).drop(columns=['Label'])
        labels = copy.deepcopy(inputx[['Label']])
        print('Fraction of diseased:',str( np.sum(labels)/len(labels) ) )
        self.real_data_split = utils.Splits(data = data,
                             labels = labels,
                             **all_args)    
        #Save pickled split:
        print('Saving pickled split')
        pickle.dump(self.real_data_split, open(os.path.join(self.results_dir, self.gene_name+'.pickle'), 'wb'),-1)
    
    def _prep_mysteryAAs(self):
        #mysteryAAs are from WES and ClinVar data. Will get predictions for these
        mysteryAAs_raw = self.ag.mysteryAAs
        self._run_sanity_check(mysteryAAs_raw)
        
        mysteryAAs_data = (copy.deepcopy(mysteryAAs_raw))
        mysteryAAs_labels = pd.DataFrame(np.zeros((mysteryAAs_data.shape[0],1)), columns=['Label'])
        mysteryAAs_split = utils.Splits(data = mysteryAAs_data,
                                     labels = mysteryAAs_labels,
                                     columns_to_ensure = self.ag.columns_to_ensure,
                                     **self.shared_args)

        #Normalize mysteryAAs based on the statistics of the entire train/val
        #set (because the final predictive model is trained based on all
        #available healthy/diseased AAs)
        #must deep copy self.real_data_split.clean_data or else you will
        #end up normalizing the real data twice due to modifying the
        #underlying dataframe!
        all_real_data = copy.deepcopy(self.real_data_split.clean_data)
        _, mysteryAAs_final_clean_data = mysteryAAs_split._normalize(all_real_data, mysteryAAs_split.clean_data)
        assert mysteryAAs_final_clean_data.shape[0] == mysteryAAs_raw.shape[0]
        self.mysteryAAs_dict = {}
        self.mysteryAAs_dict['scaler'] = mysteryAAs_split.scaler
        tempdata = copy.deepcopy(mysteryAAs_data)
        tempdata['True_Label'] = mysteryAAs_labels['Label']
        self.mysteryAAs_dict['raw_data'] = tempdata
        self.mysteryAAs_dict['Dataset'] = utils.Dataset(data = mysteryAAs_final_clean_data.values,
                                                labels = mysteryAAs_split.clean_labels.values,
                                                shuffle = False,
                                                data_meanings = mysteryAAs_final_clean_data.columns.values.tolist(),
                                                label_meanings = mysteryAAs_split.clean_labels.columns.values.tolist(),
                                                batch_size = 256)
    
    def _run_sanity_check(self, df):
        """A quick santiy check to ensure consensus!=change and to ensure no
        duplicates"""
        #Ensure consensus!=change
        assert (df.loc[:,'Consensus'].values == df.loc[:,'Change'].values).sum() == 0, 'Data is not clean: consensus==change at'+str(position)
        #Ensure no duplicates
        if len(df[df.duplicated(subset=['Consensus', 'Position', 'Change'], keep=False)]) > 0:
            assert False, 'Data is not clean: duplicates remain'

class PrepareEveryAA(object):
    """everyAA is an alternative to mysteryAAs. everyAA contains every possible
    amino acid mutation for a given gene"""
    def __init__(self, gene_name, size):
        """<size> is either 'full' for every possible AA mutation for the
        specified gene, or 'small' for a small version intended for testing"""
        self.gene_name = gene_name
        self.geneseq = get_geneseq(gene_name)
        
        if size == 'full':
            self.everyAA = self.prepare_everyAA()
        elif size == 'small':
            self.everyAA = self.make_small_everyAA_for_testing()
    
    def prepare_everyAA(self):
        """Create a pandas dataframe containing every possible amino acid change
        at every position, except for anything present in the real data."""
        global AMINO_ACIDS
        print('Creating everyAA')
        #List of all possible AAs:
        max_length = len(self.geneseq)*(len(AMINO_ACIDS)-1) #don't count C-->C (not a change)
        print('everyAA max length is',max_length)
        
        #Fill dataframe
        everyAA = pd.DataFrame(np.empty((max_length,3), dtype='str'), columns = ['Position','Consensus','Change'])
        dfindex = 0
        for position in range(len(self.geneseq)):
            consensus = self.geneseq[position]
            for aa in AMINO_ACIDS:
                if aa != consensus: 
                    everyAA.at[dfindex, 'Position'] = str(position+1) #one-based
                    everyAA.at[dfindex, 'Consensus'] = consensus
                    everyAA.at[dfindex, 'Change'] = aa
                    dfindex+=1
            if position % 500 == 0: print('Done with',position)
        assert max_length == dfindex
        #Transform position to int
        everyAA['Position'] = pd.to_numeric(everyAA['Position'], downcast='integer')
        everyAA['Label']=0 #dummy, never used
        
        #Reindex
        everyAA = everyAA.reset_index(drop=True)#Reindex
        return everyAA
    
    def make_small_everyAA_for_testing():
        """Dummy version of everyAA to be able to run the MLP code quickly"""
        global AMINO_ACIDS
        everyAA = pd.DataFrame(np.transpose(np.array([AMINO_ACIDS, AMINO_ACIDS])),
                            columns=['Consensus','Change'])
        everyAA['Label'] = 0
        everyAA['Position'] = [x for x in range(1,1+len(AMINO_ACIDS))]
        return everyAA

def get_geneseq(gene_name):
    geneseq = ''
    with open(os.path.join('data',os.path.join(gene_name, gene_name+'_reference_sequence.txt')), 'r') as f:
        for line in f:
            geneseq = geneseq + line.rstrip()
    if gene_name == 'ryr2':
        assert len(geneseq) == 4967
    return geneseq.upper()
