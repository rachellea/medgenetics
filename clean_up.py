import pandas as pd
import numpy as np

filename = "ryr2_withSN_mysteryAAs_results_epoch_611.csv"
df = pd.read_csv(filename)
consensusAA = []
changeAA = []
position = df['Position'].values
pred_prob = df['Pred_Prob'].values

# get the consensusAA and changeAA 
for index, row in df.iterrows():
    for column in df.columns.values:
        if column.startswith('Consensus_') and row[column]==1:
             consensusAA.append(column[-1])
        if column.startswith('Change_') and row[column]==1:
             changeAA.append(column[-1])
        
# convert consensusAA and changeAA to numpy
consensusAA = np.array(consensusAA)
changeAA = np.array(changeAA)

# stack the 4 columns
new_np = np.vstack((consensusAA, changeAA, position, pred_prob)).T

# sort the rows from highest predicted probability
index = np.argsort(new_np[:,-1])
new_np = new_np[index[::-1]]

# create a new dataframe
new_df = pd.DataFrame(new_np, columns=['ConsenseusAA', 'ChangeAA', 'Position', 'Pred_Prob'])

# create a new csv file
new_df.to_csv('final_' + filename, index=None, header=True)
