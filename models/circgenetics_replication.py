#circgenetics_replication.py

import copy
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers, regularizers
from sklearn import model_selection
import numpy as np
import pandas as pd

from data import utils
from . import cv_agg

"""Attempt to replicate this paper:
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
    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.gene_name = 'circgenetics'
        self.number_of_cv_folds = 3
        self.p_repeats = 200 #TODO DO THIS
        self.load_data()
        self.run_one_model_setup()
    
    def load_data(self):
        print('Loading data')
        #df columns: Residue_ID, Wild_Type, Variant, PSSM, Rate_of_Evolution, Label
        #The only features used in training are PSSM and Rate_of_Evolution
        df = pd.read_csv('data/circgenetics/kcnq1_training_dataset.csv',header=0)
        print('Data shape:',df.shape)
        self.real_data_split = utils.Splits(data = df[['PSSM','Rate_of_Evolution']],
             labels=df[['Label']],
             one_hotify_these_categorical=[],
             normalize_these_continuous=['PSSM','Rate_of_Evolution'],
             columns_to_ensure=['PSSM','Rate_of_Evolution'],
             batch_size=16)
    
    def run_one_model_setup(self):
        print('Running one model setup')
        self.perf_all_models = pd.DataFrame(columns = ['Mean_Best_Epoch','Mean_Accuracy','Std_Accuracy',
                    'Mean_AUROC','Std_AUROC','Mean_Avg_Precision','Std_Avg_Precision',
                    'Gen_Best_Epoch','Gen_Accuracy','Gen_AUROC','Gen_Avg_Precision'])
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
            fold_test_out, fold_eval_dfs_dict = train_and_eval_circgenetics(split, fold_num)
            
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
        self.perf_all_models = cv_agg.update_and_save_cv_perf_df('',
            self.perf_all_models, all_eval_dfs_dict, all_test_out, self.number_of_cv_folds,
            num_ensemble, {}, 
            save_path = os.path.join(self.results_dir,self.gene_name+'_Performance_CircGenetics.csv'))
        
def train_and_eval_circgenetics(split, fold_num):
    """Train and evaluate the Circ Genetics model"""
    #Model setup
    num_epochs = 1000 #paper did not specify
    learningrate = 0.05 #from paper
    dropout_rate = 0.5 #paper did not specify dropout rate but did specify use of dropout
    momentum = 0.8 #from paper
    weight_decay = 0.02 #from paper (L2 regularization)
    batch_size = 16 #paper did not specify
    
    #Keras model training and predictions
    model = Sequential()
    model.add(Dense(3, input_dim=2, activation='sigmoid',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    #paper did not specify the optimizer so we use SGD
    #paper did not specify the loss function so we use cross entropy
    sgd = keras.optimizers.SGD(learning_rate=learningrate, momentum=momentum)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(split.train.data, split.train.labels, epochs=num_epochs, batch_size=batch_size)
    entire_pred_probs = model.predict(split.test.data)
    entire_pred_labels = (entire_pred_probs > decision_threshold).astype('int')
    
    #Create fold_test_out
    test_out_df = pd.DataFrame(np.concatenate((split.test.data, entire_pred_probs, entire_pred_labels, split.test.labels),axis = 1),
                               columns=split.test.data_meanings+['Pred_Prob','Pred_Label','True_Label'])
    test_out_df = test_out_df.sort_values(by='Position')
    fold_test_out['epoch_0'] = test_out_df #actually 1000 epochs but we only record results for the last epoch
    
    #Create fold_eval_dfs_dict
    idx ='fold_num'+str(fold_num)
    result_df = pd.DataFrame(data=np.zeros((1, num_epochs)),
                        index = [idx],columns = ['epoch_0'])
    fold_eval_dfs_dict = {'accuracy':copy.deepcopy(result_df),
        'auroc':copy.deepcopy(result_df),'avg_precision':copy.deepcopy(result_df)}
    fold_eval_dfs_dict['accuracy'].at[idx,col] = sklearn.metrics.accuracy_score(split.test.labels, entire_pred_labels)
    fold_eval_dfs_dict['auroc'].at[idx,col] = sklearn.metrics.roc_auc_score(split.test.labels, entire_pred_probs)
    fold_eval_dfs_dict['avg_precision'].at[idx,col] = sklearn.metrics.average_precision_score(split.test.labels, entire_pred_probs)
    
    return fold_test_out, fold_eval_dfs_dict

