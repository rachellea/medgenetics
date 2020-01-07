#run_models.py

import os
import copy
import pickle
import datetime
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn import neural_network
from sklearn import model_selection, metrics, calibration

#Custom imports
import mlp
import mlp_loops
import reformat_output

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

class GridSearchMLP(object):
    """Perform a grid search across predefined architectures and hyperparameters
    for a given gene <gene_name> to determine the best MLP model setup."""
    def __init__(self, gene_name, real_data_split, mysteryAAs_split):
        """<gene_name> is a string, one of: 'kcnh2', 'kcnq1', 'ryr2', or 'scn5a'."""
        self.gene_name = gene_name
        self.real_data_split = real_data_split
        self.mysteryAAs_split = mysteryAAs_split
        self.cv_fold_mlp = 10 #ten fold cross validation
        self.max_epochs = 1000
        self.results_dir = os.path.join(os.path.join('results',gene_name),datetime.datetime.today().strftime('%Y-%m-%d'))
        self.big_test_data_and_predsput_df = pd.DataFrame(
            
            
            
            columns = ['MLP_Layer','Learning_Rate','Dropout_Rate','Ensemble_Size','Best_Epoch','Accuracy','AUROC','Avg_Precision']
            
        )
        
        #Run
        self._initialize_search_params()
        self._get_best_mlp()
        self._save_test_preds_of_best_mlp()
    
    def _initialize_search_params(self):
        #Initialize lists of hyperparameters and architectures to try
        learn_rate = [1e-4,1e-3,1e-2,1e-1,1,10,100,1000]
        dropout = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        ensemble = [0]
        layers = [[20],[30,20],[60,20],[60,60, 20],[120,60,20],[40],[40,40],[60,40], [120,60,40]]
        comb_lst = [learn_rate, dropout, ensemble, layers]
        self.combinations = list(itertools.product(*comb_lst))
    
    def _get_best_mlp(self):
        self.save_test_out = False

        # for each combination of hyperparameters, get set the epoch and run mlp
        for comb in tqdm(self.combinations):
            best_epoch = 0
            epoch_max_avg_prec = 0
            # set the hyperparameters
            self.num_ensemble = comb[2]
            self.get_best_epoch = True
            
            mlp_args_specific = {'descriptor':self.gene_name,
                'decision_threshold':0.5,
                'num_epochs':self.max_epochs, # fix number of epochs to 300
                'learningrate':comb[0],
                'mlp_layers': copy.deepcopy(comb[3]),
                'dropout':comb[1],
                'exclusive_classes':True,
                'save_model':False,
                'mysteryAAs':self.mysteryAAs_split,
                'cv_fold':self.cv_fold_mlp,
                'ensemble':self.num_ensemble,
                'save_test_out':self.save_test_out}
            
            self._run_mlp(mlp_args)
    
    def _run_mlp(self, mlp_args_specific):
        print('Running MLP')
        #Initializations
        test_labels = [] #TODO delete this if it turns out you don't need it
        fold_num = 1
        
        cv = model_selection.StratifiedKFold(n_splits=self.cv_fold_mlp, random_state=19357)
        data = self.real_data_split.clean_data #note that data is not yet normalized here
        label = self.real_data_split.clean_labels
        
        for train, test in cv.split(data, label):
            print("In CV fold number", fold_num, " out of", self.cv_fold_mlp)
            
            # create a copy of the real_data_split
            split = copy.deepcopy(self.real_data_split)
            
            # update the splits with the train and test indices for this cv loop and
            # normalize training data
            split._make_splits_cv(train, test)
            test_labels.append(split.test.labels)
            
            #Gather MLP args
            mlp_args = {**mlp_args_specific, **{'split':copy.deepcopy(split)}}
            
            #Train and evaluate the model
            if self.num_ensemble > 0: #do ensembling
                fold_eval_dfs_dict, fold_test_data_and_preds = mlp_loops.train_and_eval_ensemble(mlp_args, self.num_ensemble)
            else: #no ensembling (only one MLP)
                fold_eval_dfs_dict = mlp_loops.train_and_eval_one_mlp(mlp_args)
                
            #Sum the fold_eval_dfs_dict across all the folds
            #For ensemble, the eval_dfs_dict will contain only the final epoch
            #For a single MLP, the eval_dfs_dict will contain all of the epochs
            if fold_num == 1:
                eval_dfs_dict = evaluate.sum_eval_dfs_dicts(fold_eval_dfs_dict, None)
            else:
                eval_dfs_dict = evaluate.sum_eval_dfs_dicts(fold_eval_dfs_dict, eval_dfs_dict)
            
            # save the dataframe of all the validation sets          
            if self.save_test_out:
                # if this is the first fold, initialize the dataframe
                if fold_num == 1:
                    self.test_data_and_preds = self.fold_test_data_and_preds
                    # otherwise, concatenate with existing dataframe
                else:
                    self.test_data_and_preds = pd.concat([self.test_data_and_preds, self.fold_test_data_and_preds], ignore_index=True)
                    
            fold_num += 1
        
        # save the results for the best average precision
        self.big_test_data_and_predsput_df = evaluate.update_and_save_cv_perf_df(eval_dfs_dict,
                    self.cv_fold_mlp, self.num_ensemble, mlp_args_specific,self.big_test_data_and_predsput_df,
                    save_path = os.path.join(self.results_dir,self.gene_name+'_all_mlp_results.csv'))
    
    def _save_test_preds_of_best_MLP(self):
        """Save the test predictions of the best MLP model"""
          # get the best model
        best_mod = output_df.iloc[output_df['Avg Precision'].idxmax()]
        self.layer = best_mod["MLP Layer"]
        self.learningrate =  best_mod["Learning Rate"]
        self.dropout = best_mod["Dropout Rate"]
        self.num_ensemble =  best_mod["Ensemble Size"]
        self.max_epochs = best_mod["Best Epoch"]
        # save best param to txt file
        with open(os.path.join(self.results_dir,self.gene_name+'_best_mlp_summary.txt'), 'w') as f:
            f.write("---------Hyperparameters for the best model--------")
            f.write("\nMLP Layer:" + str(self.layer))
            f.write("\nLearning Rate:" + str(self.learningrate))
            f.write("\nDropout Rate:" + str(self.dropout))
            f.write("\nEnsemble Size:" + str(self.num_ensemble))
            f.write("\nBest Epoch:" + str(self.max_epochs))
            f.write("\nAccuracy:" + str(round(best_mod["Accuracy"],4)))
            f.write("\nAUROC:" + str(round(best_mod['AUROC'], 4)))
            f.write("\nAvg Precision:" + str(round(best_mod["Avg Precision"],4)))
        print("Done saving all results")                    
        # run the best model again to obtain true label and predicted probabilities
        self.save_test_out = True
        self._run_mlp()
        self.test_data_and_preds.to_csv(os.path.join(self.results_dir,self.gene_name+'_best_mlp_test_preds.csv'),index=False)
    
class PredictMysteryAAs_MLP(object):
    def __init__(self, gene_name, real_data_split, mysteryAAs_split):
        '''This function uses mlp to predict the mysteryAAs'''
        self.gene_name = gene_name
        self.real_data_split = real_data_split
        self.mysteryAAs_split = mysteryAAs_split
        
        # set hyperparameters
        self.learningrate = 1000
        self.dropout = 0.4
        self.max_epochs = 1000
        self.num_ensemble = 10 
        self.layer = [30,20]
        self.cv_fold_mlp = 10

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
                cv_fold = self.cv_fold_mlp,
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
