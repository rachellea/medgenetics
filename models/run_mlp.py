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
from . import mlp
from . import mlp_loops
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

class GridSearchMLP(object):
    """Perform a grid search across predefined architectures and hyperparameters
    for a given gene <gene_name> to determine the best MLP model setup."""
    def __init__(self, gene_name, results_dir, real_data_split, mysteryAAs_split, testing=False):
        """<gene_name> is a string, one of: 'kcnh2', 'kcnq1', 'ryr2', or 'scn5a'."""
        self.gene_name = gene_name
        self.real_data_split = real_data_split
        self.mysteryAAs_split = mysteryAAs_split
        self.cv_fold_mlp = 10 #ten fold cross validation
        self.max_epochs = 1000
        self.results_dir = os.path.join(results_dir, gene_name)
        
        #Initialize dataframe that will store the performance for all of the
        #different models:
        self.perf_all_models = pd.DataFrame(np.zeros(1,12),
                columns = ['MLP_Layer','Learning_Rate','Dropout_Rate',
                           'Ensemble_Size','Mean_Best_Epoch','Mean_Accuracy',
                           'Mean_AUROC','Mean_Avg_Precision','Gen_Best_Epoch',
                           'Gen_Accuracy','Gen_AUROC','Gen_Avg_Precision'])
        self.perf_all_models['MLP_Layer'] = self.perf_all_models['MLP_Layer'].astype('str')
        
        #Run
        if testing:
            self._initialize_testing_search_params()
        else:
            self._initialize_search_params()
        self._get_best_mlp()
        self._save_test_preds_of_best_mlp()
    
    def _initialize_testing_search_params(self):
        #Initialize lists of hyperparameters and architectures to try
        learn_rate = [1e-2,100]
        dropout = [0.3]
        ensemble = [0]
        layers = [[20],[120,60,20]]
        comb_lst = [learn_rate, dropout, ensemble, layers]
        self.combinations = list(itertools.product(*comb_lst))
    
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
                fold_test_out, fold_eval_dfs_dict = mlp_loops.train_and_eval_ensemble(mlp_args, self.num_ensemble)
            else: #no ensembling (only one MLP)
                fold_test_out, fold_eval_dfs_dict = mlp_loops.train_and_eval_one_mlp(mlp_args)
            
            #Aggregate the fold_eval_dfs_dict (performance metrics) for FIRST WAY:
            if fold_num == 1:
                all_eval_dfs_dict = fold_eval_dfs_dict
            else:
                all_eval_dfs_dict = evaluate.sum_eval_dfs_dicts(fold_eval_dfs_dict, all_eval_dfs_dict)
            
            #Aggregate the fold_test_out (data and predictions) for SECOND WAY:
            if fold_num == 1:
                all_test_out = fold_test_out
            else:
                all_test_out = evaluate.concat_test_outs(fold_test_out, all_test_out, num_epochs = mlp_args['num_epochs'])
            
            fold_num += 1
            
        # Calculating Performance in Two Ways #---------------------------------
        #Now we are done with the folds of cross-validation.
        #We need to calculate performance. We will do this in two ways:
        #FIRST WAY: 'Averaged Perf':averaging the performance of each fold.
        #This will help us see if there is a lot of variability in performance
        #between folds. Technically if we're using average precision we also
        #would need to account for the number of positives in each fold to do
        #a proper weighting. But due to computational cost we are not going to
        #do that.
        #SECOND WAY: 'Generalized Perf': we concatenate the actual predictions
        #for each example in the test set of each fold so that we get a
        #predicted probability for every example in the data set. Then we
        #calculate the performance metrics on all this data at the same time.
        #This implicitly does all of the weighting correctly based on number of
        #true positives, number of false positives, and so on. But it hides
        #any 'variability between folds' that may exist with this model setup.
        self.perf_all_models = evaluate.update_and_save_cv_perf_df(
            self.perf_all_models, all_eval_dfs_dict, all_test_out, self.cv_fold_mlp, mlp_args_specific, 
            save_path = os.path.join(self.results_dir,self.gene_name+'_perf_all_models.csv'))
    
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
        self.all_folds_test_out.to_csv(os.path.join(self.results_dir,self.gene_name+'_best_mlp_test_preds.csv'),index=False)
    
class PredictMysteryAAs_MLP(object):
    def __init__(self, gene_name, results_dir, real_data_split, mysteryAAs_split):
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