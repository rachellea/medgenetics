import copy
import pandas as pd
import numpy as np

def remove_dups():
    healthy = pd.read_csv('/data/rlb61/Collaborations/Landstrom/datafiles/Healthy_Variants.csv', header=0)
    diseased = pd.read_csv('/data/rlb61/Collaborations/Landstrom/datafiles/Pathologic_Variants.csv', header=0)
    
    #Remove any file-specific dups
    print('Healthy rows before removing healthy dups:',healthy.shape[0])
    print('Diseased rows before removing diseased dups:',diseased.shape[0])
    healthy = healthy.drop_duplicates(keep = 'first')
    disease = diseased.drop_duplicates(keep = 'first')
    print('Healthy rows after removing healthy dups:',healthy.shape[0])
    print('Diseased rows after removing diseased dups:',diseased.shape[0])
    
    #Now remove cross-file dups (i.e. example that appear in each file)
    #I got these dups using unix
    #first force them to be a dup in the individual file, and then remove
    #all dups from the individual file
    print('Healthy rows after removing cross-file dups should be',healthy.shape[0]-25)
    print('Diseased rows after removing cross-file dups should be',diseased.shape[0]-25)
    dups = pd.DataFrame([[1013,'R','Q'], [1107,'T','M'], [186,'V','M'], [2113,'V','M'],
        [2145,'G','R'], [217,'I','V'], [2267,'R','H'], [2359,'R','Q'],
        [2392,'Y','C'], [240,'H','R'], [2487,'L','I'], [2504,'T','M'],
        [2510,'T','A'], [3800,'C','F'], [420,'R','W'], [4282,'A','V'],
        [4307,'R','C'], [4552,'H','R'], [4556,'A','T'], [4662,'G','S'],
        [466,'P','A'], [4790,'R','Q'], [62,'L','F'], [739,'R','H'],
        [77,'A','V']], columns = ['Position','Consensus','Change'])
    healthy = pd.concat([healthy,copy.deepcopy(dups)], axis = 0)
    healthy = healthy.drop_duplicates(keep = False)
    diseased = pd.concat([diseased, copy.deepcopy(dups)], axis = 0)
    diseased = diseased.drop_duplicates(keep = False)
    print('Healthy rows after removing healthy cross-file dups:',healthy.shape[0])
    print('Diseased rows after removing diseased cross-file dups:',diseased.shape[0])
    healthy.to_csv('Healthy_Variants_deduped.csv', header=True, index=False)
    diseased.to_csv('Pathologic_Variants_deduped.csv', header=True, index=False)
    print('Done')
    
if __name__=='__main__':
    remove_dups()