#circgenetics_replication.py

import os
import copy
import numpy as np
import pandas as pd
from sklearn import model_selection

#Custom imports
from . import cv_agg
from . import ensemble_agg
from . import run_models

"""Replication of this paper (as closely as possible given the information
present in the paper):
        "Predicting the Functional Impact of KCNQ1 Variants of Unknown
        Significance" by Bian Li, Jeffrey L. Mendenhall, Brett M. Kroncke,
        Keenan C. Taylor, Hui Huang, Derek K. Smith, Carlos G. Vanoye,
        Jeffrey D. Blume, Alfred L. George, Charles R. Sanders, and Jens Meiler.
        Circ Cardiovasc Genet 2017
        
    Quote from the paper:
        The neural network in the present study was a fully connected 3-layer
        feed-forward network with a sigmoid transfer function. The input layer
        consists of 2 nodes, 1 for each predictive feature. The output layer
        consists of a single neuron that outputs a numeric prediction of the
        functional impact of a given variant on the scale of 0 to 1 with 1 being
        complete dysfunction. A hidden layer with 3 neurons was chosen
        considering the fact that the dropout technique was adopted [...]
        The learning rate was set to 0.05 and momentum was set to 0.8.
        Weights were updated after each presentation of a variant to the
        network, and a constant weight decay of 0.02 was applied
        
        Because the number of ways a data set can be split into k subsets is
        enormous, it is desirable to repeat the random splitting p times to
        reduce artifacts. In the current study we chose k = 3 and p = 200"""

class ReplicateCircGenetics(object):
    def __init__(self, results_dir, real_data_split):
        self.results_dir = results_dir
        self.real_data_split = real_data_split
        self.modeling_approach = 'MLP'
        self.gene_name = 'circgenetics'
        self.number_of_cv_folds = 3
        self.p_repeats = 200 #TODO DO THIS
        self.what_to_run = 'test_pred'
        self._run_one_model_setup()
    
    def _return_model_args_specific(self):
        return {'descriptor':self.gene_name,
                'decision_threshold':0.5,
                'num_epochs':1000, #paper did not specify
                'learningrate':0.05, #from paper
                'mlp_layers': copy.deepcopy([3]),
                'dropout':0.5, #paper did not specify dropout rate but did specify use of dropout
                'mysteryAAs':None}
    
    def _run_one_model_setup(self):
        print('Running one model setup')
        self.perf_all_models = pd.DataFrame(columns = ['Mean_Best_Epoch','Mean_Accuracy','Std_Accuracy',
                    'Mean_AUROC','Std_AUROC','Mean_Avg_Precision','Std_Avg_Precision',
                    'Gen_Best_Epoch','Gen_Accuracy','Gen_AUROC','Gen_Avg_Precision'])
        num_ensemble = 1
        fold_num = 1
        cv = model_selection.StratifiedKFold(n_splits=self.number_of_cv_folds, random_state=19357)
        data = self.real_data_split.clean_data #note that data is not yet normalized here
        label = self.real_data_split.clean_labels
        
        model_args_specific = self._return_model_args_specific()
        
        for train_indices, test_indices in cv.split(data, label):
            print('\n******For',self.gene_name+', in CV fold number',fold_num,'out of',self.number_of_cv_folds,'******')
            #Create a copy of the real_data_split
            split = copy.deepcopy(self.real_data_split)
            
            #Update the splits with the train and test indices for this cv loop and
            #normalize training data
            split._make_splits_cv(train_indices, test_indices)
            
            #Train and evaluate the model
            model_args = {**model_args_specific, **{'split':copy.deepcopy(split)}}
            fold_test_out, fold_eval_dfs_dict = ensemble_agg.train_and_eval_ensemble(self.modeling_approach, model_args, num_ensemble, fold_num)
            fold_test_out = cv_agg.add_fold_column(fold_test_out, fold_num)
            
            #Aggregate the fold_eval_dfs_dict (performance metrics) for FIRST WAY:
            if fold_num == 1:
                all_eval_dfs_dict = fold_eval_dfs_dict
            else:
                all_eval_dfs_dict = cv_agg.concat_eval_dfs_dicts(fold_eval_dfs_dict, all_eval_dfs_dict)
            
            #Aggregate the fold_test_out (data and predictions) for SECOND WAY:
            if fold_num == 1:
                all_test_out = fold_test_out
            else:
                all_test_out = cv_agg.concat_test_outs(fold_test_out, all_test_out)
            fold_num += 1
        
        #Save performance
        self.perf_all_models = cv_agg.update_and_save_cv_perf_df(self.modeling_approach,
                self.perf_all_models, all_eval_dfs_dict, all_test_out, self.number_of_cv_folds,
                num_ensemble, model_args_specific, 
                save_path = os.path.join(self.results_dir,self.gene_name+'_Performance_All_'+self.modeling_approach+'_Models.csv'))
        