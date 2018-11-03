import copy
import pandas as pd
import numpy as np

def remove_wrong():
    healthy = pd.read_csv('/data/rlb61/Collaborations/Landstrom/datafiles/Healthy_Variants_deduped.csv', header=0)
    diseased = pd.read_csv('/data/rlb61/Collaborations/Landstrom/datafiles/Pathologic_Variants_deduped.csv', header=0)
    
    #reference seq
    ryr2seq = ''
    with open('/data/rlb61/Collaborations/Landstrom/datafiles/RyR2_Protein_Sequence_Human.txt','r') as f:
        for line in f:
            ryr2seq = ryr2seq + line.rstrip()
    assert len(ryr2seq) == 4967
    ryr2seq = ryr2seq.upper()
    
    #Remove anything that is not a real mutation (i.e. the "change" is the consensus sequence)
    print('Healthy rows before removing healthy entries where consensus==change:',healthy.shape[0])
    print('Diseased rows before removing diseased entries where consensus==change:',diseased.shape[0])
    healthy = healthy.loc[healthy['Consensus']!=healthy['Change']]
    diseased = diseased.loc[diseased['Consensus']!=diseased['Change']]
    print('Healthy rows after removing healthy entries where consensus==change:',healthy.shape[0])
    print('Diseased rows after removing diseased entries where consensus==change:',diseased.shape[0])
    
    #Now remove anything that disagrees with the consensus sequence
    #note that 'healthy' and 'diseased' are one-based. So position 5 in healthy
    #should be referenced as position 4 in the reference sequence
    reference_df = pd.DataFrame(np.transpose(np.array([list(ryr2seq), [x for x in range(1,len(ryr2seq)+1)]])),
                                columns=['Reference','Position'])
    #pos columns must have same type in order to be merged
    reference_df['Position'] = pd.to_numeric(reference_df['Position'], downcast='integer')
    healthy['Position'] = pd.to_numeric(healthy['Position'], downcast='integer')
    diseased['Position'] = pd.to_numeric(diseased['Position'], downcast='integer')
    
    healthy = healthy.merge(copy.deepcopy(reference_df), how = 'left', on='Position')
    healthy = healthy.loc[healthy['Consensus']==healthy['Reference']]
    diseased = diseased.merge(copy.deepcopy(reference_df), how = 'left', on = 'Position')
    diseased = diseased.loc[diseased['Consensus']==diseased['Reference']]
    print('Healthy rows after removing healthy entries where consensus!=reference:',healthy.shape[0])
    print('Diseased rows after removing diseased entries where consensus!=reference:',diseased.shape[0])
    
    healthy[['Position','Consensus','Change']].to_csv('Healthy_Variants_repaired.csv', header=True, index=False)
    diseased[['Position','Consensus','Change']].to_csv('Pathologic_Variants_repaired.csv', header=True, index=False)
    print('Done')
    
if __name__=='__main__':
    remove_wrong()