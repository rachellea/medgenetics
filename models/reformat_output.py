#reformat_output.py

import numpy as np
import pandas as pd

def make_output_human_readable(df, scaler):
    """This function takes a dataframe returned by a trained model,
       and reverses one hot encoding and normalization
       <df> is model.test_out with columns data meanings (i.e. various columns
           of the data like 'Conservation' and 'Position'), Pred_Prob,
           Pred_Label, and True_Label"""
    print('Making output human readable. Number of rows:', len(df.index))
    col = [col for col in df.columns.values if col.startswith('Consensus') or col.startswith('Change') or col.startswith('Position')]
    print('Number of duplicates:', len(df[df.duplicated(subset=col,keep=False)]))
    
    consensusAA = []
    changeAA = []
    # get the consensusAA and changeAA lists by reversing one hot encoding
    count = 0
    col_no_change = 0
    for index, row in df.iterrows():
        found = False
        # get the consensus and change of this row
        for column in df.columns:
            if column.startswith('Consensus_') and row[column]==1:
                consensus = column[-1]
                consensusAA.append(consensus)
            if column.startswith('Change_') and row[column]==1:
                found = True
                change = column[-1]
                changeAA.append(change)
        if not found:
            assert False, 'Found example with no change '+str(index)
        #increment count
        count += 1
    # convert consensusAA and changeAA lists to numpy
    consensusAA = np.array(consensusAA)
    changeAA = np.array(changeAA)

    # get the original continuous variables by performing inverse transform
    inverted_cont_vars = scaler.inverse_transform(df[['Position', 'Conservation', 'SigNoise']].values)
        
    # create a new dataframe with the necessary columns
    new_df = pd.DataFrame(np.concatenate((np.expand_dims(consensusAA,1),
                                          np.expand_dims(changeAA,1),
                                          inverted_cont_vars),axis=1),
                          columns=['Consensus', 'Change', 'Position','Conservation','SigNoise'])
    for colname in ['Pred_Prob','True_Label']:
        new_df[colname] = df[colname].values
    
    #change data types of columns
    new_df['Consensus'] = new_df['Consensus'].astype(str)
    new_df['Change'] = new_df['Change'].astype(str)
    new_df['Position'] = new_df['Position'].astype(int)
    new_df['Pred_Prob'] = new_df['Pred_Prob'].astype(float)
    new_df['True_Label'] = new_df['True_Label'].astype(int)
    
    #sort so that different members of an ensemble will all have the AAs in
    #the same order:
    new_df = new_df.sort_values(by='Position')
    new_df = check_human_readable_correctness(new_df)
    return new_df

def check_human_readable_correctness(new_df):
    """Check to make sure that the result of converting back to human readable
    format is correct and the entries match up with the original files"""
    clinvar_raw = pd.read_csv(os.path.join('data/'+gene_name,gene_name+'_variants_clinvar_raw.csv'),header= 0)
    wes_raw = pd.read_csv(os.path.join('data/'+gene_name,gene_name+'_variants_wes_raw.csv'),header = 0)
    test_raw = None #TODO
    return new_df

def make_fold_test_out_human_readable(fold_test_out, scaler):
    """Make the fold_test_out dictionary of dataframes contain only
    human-readable dataframes"""
    new_fold_test_out = {}
    for epoch_key in list(fold_test_out.keys()):
        new_fold_test_out[epoch_key] = make_output_human_readable(fold_test_out[epoch_key],scaler)
    return new_fold_test_out
    