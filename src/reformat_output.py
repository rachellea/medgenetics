#reformat_output.py

import os
import numpy as np
import pandas as pd

def make_output_human_readable(gene_name, df, scaler, raw_data):
    """This function takes a dataframe returned by a trained model,
       and reverses one hot encoding and normalization
       <gene_name> is a string e.g. 'ryr2' (used for loading files to do
           sanity checks on the human readable format of the df)
       <df> is a pandas dataframe with columns that are the data meanings
           (i.e. various columns of the data like 'Conservation' and 'Position')
           where the data is represented as one-hot variables (for categorical)
           or as normalized variables (for continuous). <df> also includes
           columns Pred_Prob, Pred_Label, and True_Label
       <scaler> contains the mean and scale needed to reverse normalization
       <raw_data> contains data and labels that can be used to verify that the
           data denormalization was successful and that the labels are in the
           right order"""
    col = [col for col in df.columns.values if col.startswith('Consensus') or col.startswith('Change') or col.startswith('Position')]
    #print('Number of duplicates:', len(df[df.duplicated(subset=col,keep=False)]))
    
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
    new_df = pd.DataFrame(np.concatenate((np.expand_dims(consensusAA,1),
                                          np.expand_dims(changeAA,1)),axis=1),
                          columns=['Consensus', 'Change'])
    
    # get the original continuous variables by performing inverse transform
    inverted_cont_vars = scaler.inverse_transform(df[['Position', 'Conservation', 'SigNoise']].values)
    #initialize df columns as numeric:
    new_df['Position'] = 0.0; new_df['Conservation'] = 0.0; new_df['SigNoise'] = 0.0
    #fill in df columns:
    new_df[['Position','Conservation','SigNoise']] = inverted_cont_vars
    #must round before casting to int otherwise some of the positions will be wrong
    new_df['Position'] = [int(round(x)) for x in new_df['Position'].values.tolist()]
    
    #add in predicted and true labels
    for colname in ['Pred_Prob','Pred_Label','True_Label']:
        new_df[colname] = df[colname].values
    
    if 'FoldNum' in df.columns.values.tolist(): #true for test set, not mysteryAAs
        new_df['FoldNum'] = df['FoldNum'].values
    
    #sort so that different members of an ensemble will all have the AAs in
    #the same order. Must sort by Position, Consensus, AND Change because this
    #is the minimum information needed to uniquely identify a mutation
    new_df = new_df.sort_values(by=['Position','Consensus','Change'])
    
    #Check that the <new_df> data is equal to the raw data and labels in
    #<raw_data> (where <raw_data> comes from the initial untransformed version
    #of the data set)
    raw_data = raw_data.sort_values(by=['Position','Consensus','Change'])
    assert (new_df[['Consensus','Change','Position','True_Label']].values == raw_data[['Consensus','Change','Position','True_Label']].values).all().all()
    assert (np.isclose(new_df[['Conservation','SigNoise']].values, raw_data[['Conservation','SigNoise']].values)).all().all()
    return new_df

def save_all_eval_dfs_dict(all_eval_dfs_dict, colname, outfilepath):
    """Save a CSV representing the <all_eval_dfs_dict> for the best epoch
    which is specified by <colname> (e.g. 'epoch_0') to <outfilepath>."""
    index_names = all_eval_dfs_dict['accuracy'].index.values.tolist()
    save_df = pd.DataFrame(np.zeros((len(index_names),3)),columns=['accuracy_'+colname,'auroc_'+colname,'avg_precision_'+colname],
                           index = index_names)
    for idx in save_df.index.values.tolist():
        save_df.at[idx,'accuracy_'+colname] = all_eval_dfs_dict['accuracy'].at[idx,colname]
        save_df.at[idx,'auroc_'+colname] = all_eval_dfs_dict['auroc'].at[idx,colname]
        save_df.at[idx,'avg_precision_'+colname] = all_eval_dfs_dict['avg_precision'].at[idx,colname]
    save_df.to_csv(outfilepath, header=True, index=True)

def tag_mysteryAAs_with_wes_and_clinvar(gene_name, new_df):
    """Check to make sure that the result of converting back to human readable
    format is correct and the entries match up with the original files"""
    clinvar_raw = pd.read_csv(os.path.join('data/', os.path.join(gene_name,gene_name+'_variants_clinvar_raw.csv')),header= 0)
    wes_raw = pd.read_csv(os.path.join('data/',os.path.join(gene_name,gene_name+'_variants_wes_raw.csv')),header = 0)
    sources = {'clinvar':clinvar_raw,'wes':wes_raw}
    new_df['Source'] = ''
    for idx in new_df.index.values.tolist():
        position = new_df.at[idx,'Position']
        consensus = new_df.at[idx,'Consensus']
        change = new_df.at[idx,'Change']
        for source in sources.keys():
            source_df = sources[source]
            same_position = source_df[source_df['Position']==position]
            if same_position.shape[0] > 0:
                for sidx in same_position.index.values.tolist():
                    if (same_position.at[sidx,'Consensus']==consensus) and (same_position.at[sidx,'Change']==change):
                        new_df.at[idx,'Source'] = source
    return new_df

def make_fold_test_out_human_readable(gene_name, fold_test_out, scaler, raw_data):
    """Make the fold_test_out dictionary of dataframes contain only
    human-readable dataframes"""
    new_fold_test_out = {}
    for epoch_key in list(fold_test_out.keys()):
        new_fold_test_out[epoch_key] = make_output_human_readable(gene_name,
                                fold_test_out[epoch_key], scaler, raw_data)
    return new_fold_test_out
    