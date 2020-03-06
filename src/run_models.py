#run_models.py

import os
import copy
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn import model_selection

#Custom imports
from . import cv_agg
from . import ensemble_agg
from . import reformat_output

#####################################################################
# Grid Search - Calculate Performance Across Different Model Setups #-----------
#####################################################################
class RunPredictiveModels(object):
    """Run models using cross validation and ensembling.
    Parameters:
    <gene_name>: a string e.g. 'ryr2'
    <modeling_approach>: a string, either 'MLP' or 'LR' (for logistic regression)
    <results_dir>: path to directory to save results
    <real_data_split>: data split defined in clean_data.py PrepareData class
    <what_to_run>: a string, either 'grid_search' (to perform grid search
        over many predefined model setups) or 'test_pred' (after a grid search
        is complete this will select the best model and then save the test set
        predictions of that model for use in e.g. visualization functions)
    <testing>: if True, and if <what_to_run>=='grid_search' then only run
        on a small number of possible settings (for code testing purposes)
    
    To run without ensembling, i.e. to run each architecture/hyperparameter
    grouping on only one instantiation of the model, set ensemble=[1]"""
    def __init__(self, gene_name, modeling_approach, results_dir,
                 real_data_split, what_to_run, testing):
        self.gene_name = gene_name
        self.modeling_approach = modeling_approach
        assert self.modeling_approach in ['MLP','LR']
        self.real_data_split = real_data_split
        if testing:
            self.number_of_cv_folds = 2
        else:
            self.number_of_cv_folds = 10 #ten fold cross validation
        self.max_epochs = 1000
        self.results_dir = results_dir
        self._initialize_perf_df()
        self.what_to_run = what_to_run
        self.testing = testing
        
        if what_to_run == 'grid_search':
            self._run_grid_search()
        elif what_to_run == 'test_pred':
            self._run_test_pred()
            
    # Grid Search Method #------------------------------------------------------
    def _run_grid_search(self):
        """Perform a grid search across predefined architectures and
        hyperparameters for a given gene <gene_name> to determine the best
        MLP model setup."""
        if self.modeling_approach == 'MLP':
            if self.testing:
                self._initialize_search_params_mlp_testing()
            else:
                self._initialize_search_params_mlp()
            self._run_all_mlp_setups()
            
        elif self.modeling_approach == 'LR':
            if self.testing:
                self._initialize_search_params_lr_testing()
            else:
                self._initialize_search_params_lr()
            self._run_all_lr_setups()
    
    # Test Predictions Methods #------------------------------------------------
    def _run_test_pred(self):
        """Load the setup of the best model and save the test set predictions
        for that model"""
        model_args_specific, num_ensemble = \
                return_best_model_args(self.gene_name,
                        self.results_dir, self.modeling_approach)
        #because self.what_to_run == 'test_pred', calling _run_one_model_setup
        #will result in the test predictions being saved for this model setup.
        self._run_one_model_setup(model_args_specific, num_ensemble)
    
    # Init Perf Df #------------------------------------------------------------
    def _initialize_perf_df(self):
        """Initialize dataframe that will store the performance for all of the
        different model variants"""
        if self.modeling_approach == 'MLP':
            model_colnames = ['MLP_Layer','Learning_Rate','Dropout_Rate']
        elif self.modeling_approach == 'LR':
            model_colnames = ['Penalty','C']
        self.perf_all_models = pd.DataFrame(columns = model_colnames+['Ensemble_Size',
                    'Mean_Best_Epoch','Mean_Accuracy','Std_Accuracy',
                    'Mean_AUROC','Std_AUROC','Mean_Avg_Precision','Std_Avg_Precision',
                    'Gen_Best_Epoch','Gen_Accuracy','Gen_AUROC','Gen_Avg_Precision',
                    'Calibration_Slope'])
        for colname in ['MLP_Layer','Penalty']:
            if colname in self.perf_all_models.columns.values.tolist():
                self.perf_all_models[colname] = self.perf_all_models[colname].astype('str')
    
    # MLP Methods #-------------------------------------------------------------
    def _initialize_search_params_mlp_testing(self):
        """Initialize a small list of hyperparameters and architectures to try
        for testing purposes"""
        learn_rate = [100]
        dropout = [0.3]
        ensemble = [3]
        layers = [[20],[120,60,20]]
        comb_lst = [learn_rate, dropout, ensemble, layers]
        self.combinations = list(itertools.product(*comb_lst))
    
    def _initialize_search_params_mlp(self):
        """Initialize lists of hyperparameters and architectures to assess"""
        #running this grid search should take about 10 hours on one GPU
        learn_rate = [1e-3,1e-2,1e-1,1,10,100,1000]
        dropout = [0,0.2,0.4,0.6]
        ensemble = [1]
        layers = [[30,20],[60,60],[60,60,40],[120,60,20],[120,60,40]]
        comb_lst = [learn_rate, dropout, ensemble, layers]
        self.combinations = list(itertools.product(*comb_lst))
    
    def _run_all_mlp_setups(self):
        """Run an MLP (or MLP ensemble) for each combination of hyperparameters
        and architectures"""
        for comb in tqdm(self.combinations):
            mlp_args_specific = {'descriptor':self.gene_name,
                'decision_threshold':0.5,
                'num_epochs':self.max_epochs,
                'learningrate':comb[0],
                'mlp_layers': copy.deepcopy(comb[3]),
                'dropout':comb[1],
                'mysteryAAs':None}
            self._run_one_model_setup(mlp_args_specific, num_ensemble = comb[2])
    
    # LR Methods #--------------------------------------------------------------
    def _initialize_search_params_lr_testing(self):
        """Initialize a small list of hyperparameters to try for testing purposes"""
        logreg_penalty = ['l1']; C = [0.1]; ensemble = [1]
        comb_lst = [logreg_penalty, C, ensemble]
        self.combinations = list(itertools.product(*comb_lst))
    
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
                                'num_epochs':self.max_epochs,
                                'mysteryAAs':None}
            self._run_one_model_setup(lr_args_specific, num_ensemble = comb[2])
    
    # Generic Method to Run a Model Setup #-------------------------------------
    def _run_one_model_setup(self, model_args_specific, num_ensemble):
        print('Running one model setup')
        fold_num = 1
        #Set the numpy random seed to 0 because scikit-learn does not have its own
        #global random state but rather it uses the numpy random state instead.
        #So if you want to have reproducible results, you need to set the np
        #seed before you use scikit-learn. 
        np.random.seed(0)
        cv = model_selection.StratifiedKFold(n_splits=self.number_of_cv_folds, random_state=19357)
        data = self.real_data_split.clean_data #note that data is not yet normalized here
        label = self.real_data_split.clean_labels
        
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
            
            #If you want to save the test predictions in a human readable format
            #then you need to convert test_out to a human readable format here,
            #because the scaler mean and scale will change slightly with each fold.
            if self.what_to_run == 'test_pred':
                raw_test_data = copy.deepcopy(self.real_data_split.raw_data.iloc[test_indices])
                raw_test_labels = copy.deepcopy(self.real_data_split.raw_labels.iloc[test_indices])
                raw_test_data['True_Label'] = raw_test_labels['Label']
                fold_test_out = \
                    reformat_output.make_fold_test_out_human_readable(self.gene_name,
                    fold_test_out, split.scaler, raw_test_data)
            
            #Aggregate the fold_test_out (data and predictions) for SECOND WAY:
            if fold_num == 1:
                all_test_out = fold_test_out
            else:
                all_test_out = cv_agg.concat_test_outs(fold_test_out, all_test_out)
            fold_num += 1
            
        # Calculating Performance in Two Ways (see cv_agg.py for documentation)
        if self.what_to_run == 'grid_search':
            #don't do this for 'test_pred' because then you will overwrite the file
            #that was saved during the 'grid_search'
            self.perf_all_models = cv_agg.update_and_save_cv_perf_df(self.modeling_approach,
                self.perf_all_models, all_eval_dfs_dict, all_test_out, self.number_of_cv_folds,
                num_ensemble, model_args_specific, 
                save_path = os.path.join(self.results_dir,self.gene_name+'_Performance_All_'+self.modeling_approach+'_Models.csv'))
        
        #Save test set predictions if indicated
        if self.what_to_run == 'test_pred':
            best_model = select_best_model_setup(self.gene_name, self.results_dir, self.modeling_approach)
            bestmodelstring = return_best_model_string(self.gene_name,self.results_dir,self.modeling_approach)
            all_test_out['epoch_'+str(best_model['Gen_Best_Epoch'])].to_csv(os.path.join(self.results_dir, self.gene_name+'_'+bestmodelstring+'_all_test_out.csv'))
            reformat_output.save_all_eval_dfs_dict(all_eval_dfs_dict, colname = 'epoch_'+str(best_model['Gen_Best_Epoch']),
                outfilepath = os.path.join(self.results_dir, self.gene_name+'_'+bestmodelstring+'_all_eval_dfs_dict.csv'))

################################
# Select Best-Performing Model #------------------------------------------------
################################
def select_best_model_setup(gene_name, results_dir, modeling_approach):
    """Return a pandas series describing the best model. The best model setup
    is selected based on highest general average precision."""
    path_to_perf_results = os.path.join(results_dir,gene_name+'_Performance_All_'+modeling_approach+'_Models.csv')
    perf_all_models = pd.read_csv(path_to_perf_results,index_col=False)
    #First remove anything with a calibration slope outside the range 0 - 2
    #to eliminate poorly-calibrated models
    perf_all_models = perf_all_models[perf_all_models['Calibration_Slope']>0]
    perf_all_models = perf_all_models[perf_all_models['Calibration_Slope']<2]
    #Now out of the remaining models, choose the one with the best avg precision
    perf_all_models = perf_all_models.sort_values(by='Gen_Avg_Precision',ascending=False)
    best_model = perf_all_models.iloc[0,:]
    print('Model with highest Gen_Avg_Precision selected:\n',best_model)
    return best_model
    
def return_best_model_args(gene_name, results_dir, modeling_approach):
    """Return a dictionary of model args and the ensemble size for the best model"""
    best_model = select_best_model_setup(gene_name,results_dir,modeling_approach)
    if modeling_approach == 'MLP':
        model_args = {'descriptor':gene_name,
            'decision_threshold':0.5,
            #add one to the best epoch because the MLP code will run for
            #range(0,num_epochs) which means that if you want to include num_epochs
            #you need to run for range(0,num_epochs+1)
            'num_epochs':best_model['Gen_Best_Epoch']+1,
            'learningrate':best_model['Learning_Rate'],
            'mlp_layers':[int(x) for x in best_model['MLP_Layer'].replace(']','').replace('[','').split(',')],
            'dropout':best_model['Dropout_Rate'],
            'mysteryAAs':None}
    
    elif modeling_approach == 'LR':        
        model_args = {'descriptor':gene_name,
            'logreg_penalty':best_model['Penalty'],
            'C':best_model['C'],
            'decision_threshold':0.5,
            'num_epochs':best_model['Gen_Best_Epoch'],
            'mysteryAAs':None}
    return model_args, int(best_model['Ensemble_Size'])

def return_best_model_string(gene_name, results_dir, modeling_approach):
    """Return a string that describes the setup of the best model"""
    best_model = select_best_model_setup(gene_name, results_dir,modeling_approach)
    if modeling_approach == 'MLP':
        return ('MLP-Ep'+str(best_model['Gen_Best_Epoch'])
                +'-'+str(best_model['MLP_Layer'].replace(' ','-'))
                +'-L'+str(best_model['Learning_Rate'])
                +'-D'+str(best_model['Dropout_Rate']))
    elif modeling_approach == 'LR':
        return ('LR-Ep'+str(best_model['Gen_Best_Epoch'])
                +'-'+str(best_model['Penalty'])
                +'-C'+str(best_model['C']))

###########################################
# Run Best-Performing Model on MysteryAAs #-------------------------------------
###########################################
class PredictMysteryAAs(object):
    def __init__(self, gene_name, modeling_approach, results_dir,
                 real_data_split, mysteryAAs_dict):
        """Use MLP model to predict mutation pathogenicity of mysteryAAs"""
        self.gene_name = gene_name
        self.modeling_approach = modeling_approach
        self.results_dir = results_dir
        self.real_data_split_tocopy = real_data_split
        
        #mysteryAAs_dict is produced in clean_data.py and has keys 'scaler',
        #'Dataset' and 'raw_data'
        self.mysteryAAs_dict = mysteryAAs_dict
        
        #note that the mysteryAAs are already normalized according to the
        #statistics of all the real data
        self.model_args, self.num_ensemble = return_best_model_args(gene_name, results_dir, modeling_approach)
        self.model_args['mysteryAAs'] = self.mysteryAAs_dict['Dataset']
        
        #Run
        self._prepare_data()
        self._run_model_on_mysteryAAs()
    
    def _prepare_data(self):
        #Put all the real data into the train split and leave the test split empty
        self.real_data_split = copy.deepcopy(self.real_data_split_tocopy)
        train_indices = np.array([x for x in range(self.real_data_split.clean_data.shape[0])])
        test_indices = np.array([])
        self.real_data_split._make_splits_cv(train_indices, test_indices)
        
        #Sanity checks, including that the scaler used to normalize the real
        #data here should be identical to that previously used to normalize
        #the mysteryAAs, because both scalers are calculated on the full
        #available labeled dataset
        assert (self.real_data_split.scaler.mean_ == self.mysteryAAs_dict['scaler'].mean_).all()
        assert (self.real_data_split.scaler.scale_ == self.mysteryAAs_dict['scaler'].scale_).all()
        assert self.real_data_split.train.data.shape[0] == self.real_data_split.clean_data.shape[0]
        assert self.real_data_split.test.data.shape[0] == 0
        
        self.model_args['split'] = self.real_data_split
    
    def _run_model_on_mysteryAAs(self):
        #Train on all available data and make predictions on mysteryAAs
        best_model = select_best_model_setup(self.gene_name, self.results_dir, self.modeling_approach)
        ensemble_lst = ensemble_agg.train_ensemble(self.modeling_approach, self.model_args, self.num_ensemble, 'mysteryAA_pred')
        mysteryAA_raw_preds = ensemble_agg.create_fold_test_out(ensemble_lst, self.model_args['decision_threshold'], 'mysteryAA_pred')
        mysteryAA_raw_preds_df = mysteryAA_raw_preds['epoch_'+str(best_model['Gen_Best_Epoch'])]
        
        #Convert predictions to human readable format and save
        mysteryAA_readable_preds_df = reformat_output.make_output_human_readable(self.gene_name,
                        mysteryAA_raw_preds_df, self.mysteryAAs_dict['scaler'],
                        self.mysteryAAs_dict['raw_data'])
        mysteryAA_readable_preds_df = reformat_output.tag_mysteryAAs_with_wes_and_clinvar(self.gene_name, mysteryAA_readable_preds_df)
        bestmodelstring = return_best_model_string(self.gene_name,self.results_dir,self.modeling_approach)
        mysteryAA_readable_preds_df.to_csv(os.path.join(self.results_dir, self.gene_name+'_'+bestmodelstring+'_all_mysteryAAs_out_df.csv'))
