#plot_for_sens_spec.py

import os
import pandas as pd
import numpy as np
import seaborn
import matplotlib.pyplot as plt

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
import matplotlib
matplotlib.rcParams.update({'font.size': 14})

class MakeFigure_MysteryViolin(object):
    """Make 2 violin plots x 4 genes. Show the distribution of the
    predicted probabilities for the mysteryAAs for the WES and ClinVar data
    sets."""
    def __init__(self, base_results_dir):
        self.genes = ['ryr2','kcnq1','kcnh2','scn5a']
        base_results_dir = base_results_dir
        fig, self.ax = plt.subplots(nrows = 1, ncols = 4, figsize=(16,4.5))
        
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
            self.ax[self.idx].set_title(gene_name.upper(),fontsize=22)
            self.ax[self.idx].set_ylim(0,1.0)
                
        #Matplotlib tight layout doesn't take into account title so we pass
        #the rect argument
        #https://stackoverflow.com/questions/8248467/matplotlib-tight-layout-doesnt-take-into-account-figure-suptitle
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(base_results_dir, 'Visualization_All_ViolinPlots_MysteryAA.pdf'))
    
    def plot_violin(self,input_data):
        #Rename so plot labels look good:
        input_data = input_data.rename(columns={'Pred_Prob':'Predicted Probability'})
        input_data = input_data.replace(to_replace='wes',value='WES')
        input_data = input_data.replace(to_replace='clinvar',value='ClinVar')
        #Plot. https://seaborn.pydata.org/generated/seaborn.violinplot.html
        seaborn.violinplot(x='Model', y='Predicted Probability', hue='Source', hue_order=['ClinVar','WES'],
                    data=input_data, palette='muted', split=True, ax = self.ax[self.idx],
                    inner='stick')
        if self.idx == 3:
            self.ax[self.idx].legend(loc='upper right', prop={'size': 14})
        else:
            self.ax[self.idx].get_legend().remove()
        if self.idx != 0:
            self.ax[self.idx].set_ylabel('')
            self.ax[self.idx].set_yticks([])
        self.ax[self.idx].set_xlabel('')
    