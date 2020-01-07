#reformat_output.py

def make_output_human_readable(df, scaler):
    """This function takes a dataframe returned by a trained model,
       and reverses one hot encoding and normalization
       
       <df> is model.test_out with columns data meanings (i.e. various columns
           of the data like 'Conservation' and 'Position'), Pred_Prob,
           Pred_Label, and True_Label"""

    # debugging: checking for duplicates
    print("Number of rows:", len(df.index))
    col = [col for col in df.columns.values if col.startswith("Consensus") or col.startswith("Change") or col.startswith("Position")]
    print("Checking for duplicates in mysteryAA cleanup")
    print("Number of duplicates:", len(df[df.duplicated(subset=col,keep=False)]))

    consensusAA = []
    changeAA = []
    position = []
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
    new_df = pd.DataFrame(np.vstack((consensusAA, changeAA, inverted_cont_vars).T,
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
    return new_df

def mysteryAAs_make_output_human_readable(filename, gene_name, ori_position): #TODO replace "filename"
    '''This function converts the csv file outputted by test() in mlp 
       into a csv file with a format that we want (ConsensusAA, ChangeAA, 
       Position, Pred_Prob)'''
    print("Cleaning up the file", filename)
    # output file clean up
    df = pd.read_csv(filename)
    
    # for debugging purposes
    #pred_prob = sorted(df['Pred_Prob'].values, reverse=True)
    #print(pred_prob)
     
    # debugging: checking for duplicates
    print("Number of rows:", len(df.index))        
    col = [col for col in df.columns.values if col.startswith("Consensus") or col.startswith("Change") or col.startswith("Position")]
    print("Checking for duplicates in mysteryAA cleanup")
    print("Number of duplicates:", len(df[df.duplicated(subset=col,keep=False)]))

    consensusAA = []
    changeAA = []
    position = []
    pred_prob = df['Pred_Prob'].values

    # get the consensusAA and changeAA lists
    count = 0
    for index, row in df.iterrows():
        # get the consensus and change of this row
        for column in df.columns:
            if column.startswith('Consensus_') and row[column]==1:
                consensus = column[-1]
                consensusAA.append(consensus)
            if column.startswith('Change_') and row[column]==1:
                change = column[-1]
                changeAA.append(change)
        # get the position of this row
        position.append(int(ori_position[count]))
        #increment count
        count += 1
    # convert consensusAA and changeAA lists to numpy
    consensusAA = np.array(consensusAA)
    changeAA = np.array(changeAA)
   
    # stack the 4 columns
    new_np = np.vstack((consensusAA, changeAA, position, pred_prob)).T

    # sort the rows from highest predicted probability to the lowest
    index = np.argsort(new_np[:,-1])
    new_np = new_np[index[::-1]]

    
    # create a new dataframe with the 4 specified columns
    new_df = pd.DataFrame(new_np, columns=['ConsensusAA', 'ChangeAA', 'Position', 'Pred_Prob'])
    
    # add df to list of cleaned dataframes of each mlp in the ensemble to
    # be evaluated in _predict_mysteryAAs()
    
    new_df['Position'] = new_df['Position'].astype(int)
    new_df['ConsensusAA'] = new_df['ConsensusAA'].astype(str)
    new_df['ChangeAA'] = new_df['ChangeAA'].astype(str)

    #if self.ensemble == False or 
    # separate mysteryAAs into two files: wes or clinvar
    clinvar_raw = pd.read_csv(os.path.join('data/'+gene_name,gene_name+'_variants_clinvar_raw.csv'),header= 0)
    clinvar_raw.columns = ['ConsensusAA','Position','ChangeAA']
    clinvar_raw['Position'] = clinvar_raw['Position'].astype(int)
    clinvar_raw['ConsensusAA'] = clinvar_raw['ConsensusAA'].astype(str)
    clinvar_raw['ChangeAA'] = clinvar_raw['ChangeAA'].astype(str)

    wes_raw = pd.read_csv(os.path.join('data/'+gene_name,gene_name+'_variants_wes_raw.csv'),header = 0)
    wes_raw.columns = ['ConsensusAA','Position', 'ChangeAA']
    wes_raw['Position'] = wes_raw['Position'].astype(int)
    wes_raw['ConsensusAA'] = wes_raw['ConsensusAA'].astype(str)
    wes_raw['ChangeAA'] = wes_raw['ChangeAA'].astype(str)
    print(len(new_df))        
    # merge the raw and the resulting mysteryAAs to separate the output into two files
    clinvar_df = pd.merge(new_df, clinvar_raw, how='inner', on=['ConsensusAA','ChangeAA', 'Position'])
    wes_df = pd.merge(new_df, wes_raw, how='inner', on=['ConsensusAA', 'ChangeAA', 'Position'])

    print(len(new_df))
    # create a new csv file
    clinvar_df.to_csv('final_clinvar_' + filename, index=None, header=True)
    wes_df.to_csv('final_wes_' +filename, index=None, header=True)
        