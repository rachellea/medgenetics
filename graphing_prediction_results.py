#graphing_prediction_results.py

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('agg') #so that it does not attempt to display via SSH
import matplotlib.pyplot as plt
plt.ioff() #turn interactive plotting off

def graph_everyAA_results():
    everyAA = pd.read_csv('/data/rlb61/Collaborations/Landstrom/2018-11-02-Neural-Net-All-Probabilities/everyAA_results_epoch_678.csv')
    everyAA.sort_values(by='Position',ascending=True,inplace=True)
    
    #I previously normalized the position to make it a float. Now I want to get
    #the integer position back.
    #Unscale using the reported mean and scale from nohup.out
    mean = [2.48823204e03, 6.56967399e-01] 
    scale = [1.43434546e03, 1.96735154e-01]
    everyAA['Position'] = pd.to_numeric((((everyAA['Position']*scale[0])+mean[0]).round(decimals=0)), downcast = 'integer')
    
    #Now get the mean predicted probability
    selected = everyAA[['Position','Pred_Prob']]
    result = selected.groupby('Position')['Pred_Prob'].mean()
    result.to_csv('/data/rlb61/Collaborations/Landstrom/2018-11-02-Neural-Net-All-Probabilities/everyAA_mean.csv',
                  header=True,index=False)
    
    #Load signal-to-noise data
    signoise = pd.read_csv('/data/rlb61/Collaborations/Landstrom/datafiles/RyR2_SCD-GnomAD_Signal_to_Noise.csv',
                           header=0, index_col = False)
    #the signal to noise ratio is not between 0 and 1 so I will force it there:
    signoise_vals = (signoise['SigNoise'].values)/np.max(signoise['SigNoise'].values)
    
    #Plot
    plt.plot([x for x in range(1,4968)], result.values, color='blue', lw=0.05, label='mlp-pred-prob' )
    plt.plot([x for x in range(1,4968)], signoise_vals, color='red', lw=0.05, label='sig-to-noise')
    plt.xlabel('Position')
    plt.ylabel('Pred Prob and Sig-to-Noise')
    plt.title('RYR2 Pred Mutation Pathogenicity and Sig-Noise')
    plt.legend(loc='lower right')
    plt.savefig('everyAA_pred_probs_figure.pdf')
    plt.close()

if __name__=='__main__':
    graph_everyAA_results()