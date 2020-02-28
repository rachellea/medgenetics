#run_models.py

import os
import copy
import pickle
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn import neural_network
from sklearn import model_selection, metrics, calibration

#Custom imports
from . import cv_agg
from . import ensemble_agg
from . import reformat_output

class SimpleMLP(object):
    def __init__(self, real_data_split, hidden_layer_sizes=(30,20),
                 batch_size=300, max_iter=1000, early_stopping=False):
        """Run the sklearn MLP implementation rather than the
        Tensorflow implementation"""
        # initialize an MLP
        nn = neural_network.MLPClassifier(hidden_layer_sizes, batch_size, max_iter, early_stopping)
        #train MLP using all the training data
        x_train = real_data_split.clean_data
        y_train = real_data_split.clean_labels
        x_test = mysteryAAs_split.data
        nn.fit(x_train, np.array(y_train).ravel())
        predict_proba = nn.predict_proba(x_test)
        print(sorted(predict_proba[:,1], reverse=True))

class GridSearch(object):
    """Perform a grid search across predefined architectures and hyperparameters
    for a given gene <gene_name> to determine the best MLP model setup.
    To run without ensembling, i.e. to run each architecture/hyperparameter
    grouping on only one instantiation of the model, set ensemble=[1]"""
    def __init__(self, gene_name, modeling_approach, results_dir,
                 real_data_split, testing):
        self.gene_name = gene_name
        self.modeling_approach = modeling_approach
        assert self.modeling_approach in ['MLP','LR']
        self.real_data_split = real_data_split
        self.number_of_cv_folds = 10 #ten fold cross validation
        self.max_epochs = 1000
        self.results_dir = os.path.join(results_dir, gene_name)
        if not os.path.exists(self.results_dir):
            os.mkdir(self.results_dir)
        self._initialize_perf_df()
        
        #Run
        if self.modeling_approach == 'MLP':
            if testing:
                self._initialize_search_params_mlp_testing()
            else:
                self._initialize_search_params_mlp()
            self._run_all_mlp_setups()
        
        elif self.modeling_approach == 'LR':
            self._initialize_search_params_lr()
            self._run_all_lr_setups()
    
    def _initialize_perf_df(self):
        """Initialize dataframe that will store the performance for all of the
        different model variants"""
        if self.modeling_approach == 'MLP':
            model_colnames = ['MLP_Layer','Learning_Rate','Dropout_Rate']
        elif self.modeling_approach == 'LR':
            model_colnames = ['Penalty','C']
        self.perf_all_models = pd.DataFrame(columns = model_colnames+['Ensemble_Size',
                    'Mean_Best_Epoch','Mean_Accuracy','Mean_AUROC','Mean_Avg_Precision',
                    'Gen_Best_Epoch','Gen_Accuracy','Gen_AUROC','Gen_Avg_Precision'])
        for colname in ['MLP_Layer','Penalty']:
            if colname in self.perf_all_models.columns.values.tolist():
                self.perf_all_models[colname] = self.perf_all_models[colname].astype('str')
    
    # MLP Methods #-------------------------------------------------------------
    def _initialize_search_params_mlp_testing(self):
        """Initialize a small list of hyperparameters and architectures to try
        for testing purposes"""
        learn_rate = [100]
        dropout = [0.3]
        ensemble = [1]
        layers = [[20],[120,60,20]]
        comb_lst = [learn_rate, dropout, ensemble, layers]
        self.combinations = list(itertools.product(*comb_lst))
    
    def _initialize_search_params_mlp(self):
        """Initialize lists of hyperparameters and architectures to assess"""
        learn_rate = [1e-4,1e-3,1e-2,1e-1,1,10,100,1000]
        dropout = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        ensemble = [1]
        layers = [[20],[30,20],[60,20],[60,60, 20],[120,60,20],[40],[40,40],[60,40], [120,60,40]]
        comb_lst = [learn_rate, dropout, ensemble, layers]
        self.combinations = list(itertools.product(*comb_lst))
    
    def _run_all_mlp_setups(self):
        """Run an MLP (or MLP ensemble) for each combination of hyperparameters
        and architectures"""
        for comb in tqdm(self.combinations):
            mlp_args_specific = {'descriptor':self.gene_name,
                'decision_threshold':0.5,
                'num_epochs':self.max_epochs, # fix number of epochs to 300
                'learningrate':comb[0],
                'mlp_layers': copy.deepcopy(comb[3]),
                'dropout':comb[1]}
            self._run_one_model_setup(mlp_args_specific, num_ensemble = comb[2])
    
    # LR Methods #--------------------------------------------------------------
    def _initialize_search_params_lr(self):
        """Initialize lists of hyperparmeters to assess"""
        logreg_penalty = ['l1', 'l2']
        C = [0.0001,0.001,0.01,0.1,1,10,100,1000]
        ensemble = [1]
        comb_lst = [logreg_penalty, C, ensemble]
        self.combinations = list(itertools.product(*comb_lst))
    
    def _run_all_lr_setups(self):
        """Run a logistic regression model (or LR ensemble) for each combination
        of hyperparameters"""
        for comb in tqdm(self.combinations):
            lr_args_specific = {'descriptor':self.gene_name,
                                'logreg_penalty':comb[0],
                                'C':comb[1],
                                'decision_threshold':0.5,
                                'num_epochs':1}
            self._run_one_model_setup(lr_args_specific, num_ensemble = comb[2])
    
    # Generic Method to Run a Model Setup #-------------------------------------
    def _run_one_model_setup(self, model_args_specific, num_ensemble):
        print('Running one model setup')
        fold_num = 1
        cv = model_selection.StratifiedKFold(n_splits=self.number_of_cv_folds, random_state=19357)
        data = self.real_data_split.clean_data #note that data is not yet normalized here
        label = self.real_data_split.clean_labels
        
        for train_indices, test_indices in cv.split(data, label):
            print('\n******In CV fold number',fold_num,'out of',self.number_of_cv_folds,'******')
            #Create a copy of the real_data_split
            split = copy.deepcopy(self.real_data_split)
            
            #Update the splits with the train and test indices for this cv loop and
            #normalize training data
            split._make_splits_cv(train_indices, test_indices)
            
            #Train and evaluate the model
            model_args = {**model_args_specific, **{'split':copy.deepcopy(split)}}
            fold_test_out, fold_eval_dfs_dict = ensemble_agg.train_and_eval_ensemble(self.modeling_approach, model_args, num_ensemble)
            
            #Aggregate the fold_eval_dfs_dict (performance metrics) for FIRST WAY:
            if fold_num == 1:
                all_eval_dfs_dict = fold_eval_dfs_dict
            else:
                all_eval_dfs_dict = cv_agg.sum_eval_dfs_dicts(fold_eval_dfs_dict, all_eval_dfs_dict)
            
            #Aggregate the fold_test_out (data and predictions) for SECOND WAY:
            if fold_num == 1:
                all_test_out = fold_test_out
            else:
                all_test_out = cv_agg.concat_test_outs(fold_test_out, all_test_out, num_epochs = model_args['num_epochs'])
            fold_num += 1
            
        # Calculating Performance in Two Ways (see cv_agg.py for documentation)
        self.perf_all_models = cv_agg.update_and_save_cv_perf_df(self.modeling_approach,
            self.perf_all_models, all_eval_dfs_dict, all_test_out, self.number_of_cv_folds, model_args_specific, 
            save_path = os.path.join(self.results_dir,self.gene_name+'_perf_'+self.modeling_approach+'.csv'))
    
class PredictMysteryAAs(object):
    def __init__(self, gene_name, results_dir, real_data_split, mysteryAAs_split):
        """This function uses mlp to predict the mysteryAAs"""
        #TODO UPDATE THIS FUNCTION SO THAT IT WORKS FOR ANY KIND OF MODELING APPROACH
        self.gene_name = gene_name
        self.real_data_split = real_data_split
        self.mysteryAAs_split = mysteryAAs_split
        
        # set hyperparameters
        self.learningrate = 1000
        self.dropout = 0.4
        self.max_epochs = 1000
        self.num_ensemble = 10 
        self.layer = [30,20]
        self.number_of_cv_folds = 10

        print("Predicting mysteryAAs using MLP...")
        print("Learning rate:", self.learningrate)
        print("Dropout:", self.dropout)
        print("Number of Epochs:", self.max_epochs)
        print("Number of MLPs in the ensemble:", self.num_ensemble)
 
        # define a list to store mlps for our ensemble
        self.ensemble_output_lst = []

        # initialize ensembles and store in the list
        for en_num in range(self.num_ensemble):
            
            print("Initializing ensemble number", en_num + 1, " out of", self.num_ensemble )

            # initialize mlp
            m = mlp.MLP(descriptor=self.gene_name,
                split=copy.deepcopy(self.real_data_split),
                decision_threshold = 0.5,
                num_epochs = self.max_epochs,
                learningrate = self.learningrate,
                mlp_layers = copy.deepcopy(self.layer),
                dropout=self.dropout,
                exclusive_classes = True,
                save_model = False,
                mysteryAAs = self.mysteryAAs_split,
                cv_fold = self.number_of_cv_folds,
                ensemble=self.num_ensemble)

            # set up graph and session for the model
            m.set_up_graph_and_session()

            # train as per normal
            m.train()

            # get the unnormalized positions
            self.ori_position = self.ori_position[m.perm_ind]

            if m.best_valid_loss_epoch == 0:
                print("best valid loss epoch was not initialized")
                m.best_valid_loss_epoch = self.max_epochs
            m.clean_up()
          
            print("Cleaning up output file of ensemble number:", en_num)

            # clean up the output file from training
            reformat_output.mysteryAAs_make_output_human_readable(m.mysteryAAs_filename +str(m.best_valid_loss_epoch)+'.csv',
                                                                  self.gene_name, self.ori_position)
      
        # loop through the list of cleaned dataframe to get the average of predicted probas
        ori_df = self.ensemble_output_lst[0]
        for j in range(len(ori_df.index)):
            tot_pred_proba = 0
            consensus_AA = ori_df.iloc[j]['ConsensusAA']
            change_AA = ori_df.iloc[j]['ChangeAA']
            position = ori_df.iloc[j]['Position']
            tot_pred_proba += float(ori_df.iloc[j]['Pred_Prob'])

            for i in range(1,len(self.ensemble_output_lst)):
                curr_df = self.ensemble_output_lst[i]
                row = curr_df.loc[(curr_df['ConsensusAA'] == consensus_AA) & (curr_df['ChangeAA']== change_AA) & (curr_df['Position'] == position)]
                tot_pred_proba += float(row["Pred_Prob"])
             
            # get the average predict proba
            avg_pred_prob = tot_pred_proba / self.num_ensemble
            
            # replace the pred_proba value of ori_df with this average
            ori_df.iloc[j]['Pred_Prob'] = avg_pred_prob
  
        # sort the dataframe by pred proba column of ori_df
        ori_df.sort_values('Pred_Prob')

        # save dataframe to a file
        ori_df.to_csv('mysteryAAs_predictions_bestMLP_with_ensemble.csv', index=False)          
