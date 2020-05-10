#plot_for_sens_spec.py

import os
import pandas as pd
import numpy as np
import seaborn
import matplotlib.pyplot as plt

class MakeFigure_MysteryViolin(object):
    """Make 2 violin plots x 4 genes. Show the distribution of the
    predicted probabilities for the mysteryAAs for the WES and ClinVar data
    sets."""
    def __init__(self, base_results_dir):
        self.genes = ['ryr2','kcnq1','kcnh2','scn5a']
        base_results_dir = base_results_dir
        fig, self.ax = plt.subplots(nrows = 1, ncols = 4, figsize=(16,6))
        for idx in range(len(self.genes)):
            self.idx = idx #column number
            gene_name = self.genes[idx]
            results_dir = os.path.join(base_results_dir, gene_name)
            possible_files = os.listdir(results_dir)
            
            chosen_mystery_LR_fname = [y for y in [x for x in possible_files if 'LR' in x] if 'all_mysteryAAs_out' in y][0]
            chosen_mystery_LR = pd.read_csv(os.path.join(results_dir,chosen_mystery_LR_fname),header=0,index_col=0)
            chosen_mystery_LR['Model'] = 'LR'
            
            chosen_mystery_MLP_fname = [y for y in [x for x in possible_files if 'MLP' in x] if 'all_mysteryAAs_out' in y][0]
            chosen_mystery_MLP = pd.read_csv(os.path.join(results_dir,chosen_mystery_MLP_fname),header=0,index_col=0)
            chosen_mystery_MLP['Model'] = 'MLP'
            
            input_data = pd.concat([chosen_mystery_LR,chosen_mystery_MLP],ignore_index=True)
            self.plot_violin(input_data)
        
        fig.suptitle('  RYR2                  KCNQ1                 KCNH2                 SCN5A', fontsize=32)
        #Matplotlib tight layout doesn't take into account title so we pass
        #the rect argument
        #https://stackoverflow.com/questions/8248467/matplotlib-tight-layout-doesnt-take-into-account-figure-suptitle
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(base_results_dir, 'Visualization_All_ViolinPlots_MysteryAA.pdf'))
    
    def plot_violin(self,input_data):
        #https://seaborn.pydata.org/generated/seaborn.violinplot.html
        seaborn.violinplot(x='Model', y='Pred_Prob', hue='Source',
                    data=input_data, palette='muted', split=True, ax = self.ax[self.idx],
                    inner='stick', bw=.2)        
        self.ax[self.idx].legend(loc='lower right', prop={'size': 6})
        #self.ax[self.idx].set_title('Predicted Probabilities')
    
    