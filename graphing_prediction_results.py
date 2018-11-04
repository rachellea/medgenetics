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
    
    #Plot
    plt.plot([x for x in range(1,4968)], result.values, color='blue', lw=0.05 )
    plt.xlabel('Position')
    plt.ylabel('Predicted Probability')
    plt.title('Predicted Mutation Pathogenicity for RYR2')
    plt.savefig('everyAA_pred_probs_figure.pdf')
    plt.close()

if __name__=='__main__':
    graph_everyAA_results()