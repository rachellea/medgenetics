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
from data import utils
from data import clean_data
from models import mlp
from models import regression


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
    def __init__(self, gene_name, descriptor, real_data_split, mysteryAAs_split):
        """<gene_name> is a string, one of: 'kcnh2', 'kcnq1', 'ryr2', or 'scn5a'."""
        self.gene_name = gene_name
        self.descriptor = descriptor
        self.real_data_split = real_data_split
        self.mysteryAAs_split = mysteryAAs_split
        self.cv_fold_mlp = 10 #ten fold cross validation
        self.results_dir = os.path.join(os.path.join('results',gene_name),datetime.datetime.today().strftime('%Y-%m-%d'))
        
        #Run
        self._initialize_search_params()
        self._get_best_mlp()
    
    def _initialize_search_params(self):
        #Initialize lists of hyperparameters and architectures to try
        learn_rate = [1e-4,1e-3,1e-2,1e-1,1,10,100,1000]
        dropout = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        ensemble = [0]
        max_epochs = 1000
        layers = [[20],[30,20],[60,20],[60,60, 20],[120,60,20],[40],[40,40],[60,40], [120,60,40]]
        comb_lst = [learn_rate, dropout, ensemble, layers]
        self.combinations = list(itertools.product(*comb_lst))
    
    def _initialize_results_dirs(self):
        
    
    def _get_best_mlp(self):
        self.save_test_out = False
        
       

        # initalize output list to populate filename_csv
        output = []

        # for each combination of hyperparameters, get set the epoch and run mlp
        for comb in tqdm(self.combinations):
            best_epoch = 0
            epoch_max_avg_prec = 0
            # set the hyperparameters
            self.learningrate = comb[0]
            self.dropout = comb[1]
            self.num_ensemble = comb[2]
            self.num_epochs = max_epochs
            self.layer = comb[3]
            self.get_best_epoch = True
            self._run_mlp()
            # save the results for the best average precision
            for epoch in range(1, self.num_epochs+1):
                self.epoch_res[epoch][0] /= self.cv_fold_mlp
                self.epoch_res[epoch][1] /= self.cv_fold_mlp
                self.epoch_res[epoch][2] /= self.cv_fold_mlp
                avg_prec_res = self.epoch_res[epoch][2]
                if avg_prec_res > epoch_max_avg_prec:
                    best_epoch = epoch
                    curr_acc, curr_auroc, epoch_max_avg_prec = self.epoch_res[epoch][0], self.epoch_res[epoch][1], self.epoch_res[epoch][2]
            # save current results to dataframe
            lst = [self.layer, self.learningrate, self.dropout, self.num_ensemble, best_epoch,round(curr_acc,4), round(curr_auroc,4), round(epoch_max_avg_prec,4)]
            # append to output list
            output.append(lst)

            # save output to csv
            cols = ["MLP Layer", "Learning Rate", "Dropout Rate", "Ensemble Size", "Best Epoch", "Accuracy", "AUROC", "Avg Precision"]
            output_df = pd.DataFrame.from_records(output,columns=cols)
            output_df.to_csv(filename_csv, index=False)

        # get the best model
        best_mod = output_df.iloc[output_df['Avg Precision'].idxmax()]
        self.layer = best_mod["MLP Layer"]
        self.learningrate =  best_mod["Learning Rate"]
        self.dropout = best_mod["Dropout Rate"]
        self.num_ensemble =  best_mod["Ensemble Size"]
        self.num_epochs = best_mod["Best Epoch"]
        # save best param to txt file
        with open(filename_txt, 'w') as f:
            f.write("---------Hyperparameters for the best model--------")
            f.write("\nMLP Layer:" + str(self.layer))
            f.write("\nLearning Rate:" + str(self.learningrate))
            f.write("\nDropout Rate:" + str(self.dropout))
            f.write("\nEnsemble Size:" + str(self.num_ensemble))
            f.write("\nBest Epoch:" + str(self.num_epochs))
            f.write("\nAccuracy:" + str(round(best_mod["Accuracy"],4)))
            f.write("\nAUROC:" + str(round(best_mod['AUROC'], 4)))
            f.write("\nAvg Precision:" + str(round(best_mod["Avg Precision"],4)))
        print("Done saving all results")                    
        # run the best model again to obtain true label and predicted probabilities
        self.save_test_out = True
        self._run_mlp()
        self.final_out.to_csv(specificity_csv,index=False)
    
    def _run_mlp(self):
        #Run MLP
        print('Running MLP')
        # for debugging
        self.true_labels_lst = []
        # initialize an empty list for mlp predicted probabilities
        self.mlp_kfold_probability = []
        # initialize an empty list to store true labels for each fold
        self.kfold_true_label = []
        # initialize an empty list to store test accuracy for each fold
        fold_acc = []
        fold_auroc = []
        fold_avg_prec = []
        cv = model_selection.StratifiedKFold(n_splits=self.cv_fold_mlp, random_state=19357)
        fold_num = 1
        # note that data is not yet normalized here
        data = self.real_data_split.clean_data
        label = self.real_data_split.clean_labels
        self.test_labels = []
        self.cv_num = 1
        self.mlp_kfold_val_data = []
        self.epoch_res = {}
        for train, test in cv.split(data, label):
            # create a copy of the real_data_split
            split = copy.deepcopy(self.real_data_split)
            
            # update the splits with the train and test indices for this cv loop and
            # normalize training data
            split._make_splits_cv(train, test)
            self.test_labels.append(split.test.labels)
            
            # Doing ensembling:
            if self.num_ensemble > 0: #do ensembling
                self._train_and_eval_ensemble()
            else: #no ensembling (only one MLP)
                self._train_and_eval_one_mlp()
            
            # save the dataframe of all the validation sets          
            if self.save_test_out:
                # if this is the first fold, initialize the dataframe
                if fold_num == 1:
                    self.final_out = self.fold_df
                    # otherwise, concatenate with existing dataframe
                else:
                    self.final_out = pd.concat([self.final_out, self.fold_df], ignore_index=True)

            self.cv_num += 1
            fold_num += 1
        self._print_performance_across_folds()
    
    
    def _train_and_eval_ensemble(self):
        # initialize ensembles
        print("Calling init_ensemble()")
        self._train_ensemble(self.num_ensemble, split)

        # evaluate the accuracy of this fold using the ensemble
        # initialized
        accuracy, auroc, avg_prec = self._evaluate_ensemble()
        fold_acc.append(accuracy)
        fold_auroc.append(auroc)
        fold_avg_prec.append(avg_prec)
    
    def _print_performance_across_folds(self):
        """Print the accuracy, AUROC, and average precision for the cross
        validation folds."""
        # print the individual accuracies and the average accuracy
        print("\n\n The accuracies of the cross validation folds are:\n")
        tot_acc = 0
        for k in range(len(fold_acc)):
            print("Fold ",str(k+1), ": ", str(fold_acc[k]))
            tot_acc += fold_acc[k]
        # print the individual accuracies and the average accuracy
        print("\n\n The auroc of the cross validation folds are:\n")
        tot_auroc = 0
        for k in range(len(fold_auroc)):
            print("Fold ",str(k+1), ": ", str(fold_auroc[k]))
            tot_auroc += fold_auroc[k]
        # print the individual accuracies and the average accuracy
        print("\n\n The average precision of the cross validation folds are:\n")
        tot_avg_prec = 0
        for k in range(len(fold_avg_prec)):
            print("Fold ",str(k+1), ": ", str(fold_avg_prec[k]))
            tot_avg_prec += fold_avg_prec[k]
        
        # get the overall metrics
        print('The overall mean accuracy is', tot_acc/self.cv_fold_mlp)
        print('The overall mean AUROC is',tot_auroc/self.cv_fold_mlp)
        print('The overall mean Average Precision is',tot_avg_prec/self.cv_fold_mlp)
        print('Done')
        
    def _train_and_eval_one_mlp(self):
        #redefine mlp object with the new split
        m = mlp.MLP(descriptor=self.gene_name+'_'+self.descriptor,
            split=split,
            decision_threshold = 0.5,
            num_epochs = self.num_epochs,
            learningrate = self.learningrate,
            mlp_layers = copy.deepcopy(self.layer),
            dropout=self.dropout,
            exclusive_classes = True,
            save_model = False,
            mysteryAAs = self.mysteryAAs_split,
            cv_fold = self.cv_fold_mlp,
            ensemble=self.num_ensemble,
            save_test_out = self.save_test_out)

        # set up graph and session for the model
        m.set_up_graph_and_session()
        # train as per normal if we are not doing ensembling
        m.train()
        
        # if we are finding best mlp, then we have a dictionary of all the epochs to evaluate
        if self.find_best_mlp:
            for epoch in range(1, self.num_epochs+1):
                pred_prob = m.pred_prob_dict[epoch]
                true_label = m.true_label_dict[epoch]
                pred_label = m.pred_label_dict[epoch]
                acc = metrics.accuracy_score(true_label, pred_label)
                auc = metrics.roc_auc_score(true_label, pred_label)
                avg_prec = metrics.average_precision_score(true_label, pred_label)
                if epoch not in self.epoch_res:
                    self.epoch_res[epoch] = [acc, auc, avg_prec]
                else: 
                    self.epoch_res[epoch][0] = self.epoch_res[epoch][0]+acc
                    self.epoch_res[epoch][1] = self.epoch_res[epoch][1]+auc
                    self.epoch_res[epoch][2] = self.epoch_res[epoch][2]+avg_prec
                
        # append test accuracy to list
        df = m.eval_results_test['accuracy']
        for label in df.index.values:
            acc = df.loc[label,'epoch_'+str(m.num_epochs)]
            print("The accuracy for fold number ", str(fold_num), " is ",str(acc))
        fold_acc.append(acc)
        # append auroc to list
        df = m.eval_results_test['auroc']
        for label in df.index.values:
            auroc = df.loc[label,'epoch_'+str(m.num_epochs)]
            print("The auroc for fold number ", str(fold_num), " is ",str(auroc))
        fold_auroc.append(auroc)
        #append average precision to list
        df = m.eval_results_test['avg_precision']
        for label in df.index.values:
            avg_prec = df.loc[label,'epoch_'+str(m.num_epochs)]
            print("The average precision for fold number ",str(fold_num), " is ", str(avg_prec))
        fold_avg_prec.append(avg_prec)
        
        # update lists for calibration
        self.mlp_kfold_probability.append(m.selected_pred_probs)
        self.kfold_true_label.append(m.selected_labels_true)
    
        # get the resulting dataframe for this fold
        if self.save_test_out:
            self.fold_df = self.output_cleanup(m.test_out, split.scaler)
        
    
    

    def _train_ensemble(self, num_ensemble, split):
        """This function initializes mlps for the ensemble.
           Inputs: num_ensemble, the number of mlps in the ensemble
                   split, the split object to specify training and testing data
        """
        # define a list to store mlps for our ensemble
        self.ensemble_lst = []

        # initialize ensembles and store in the list
        for i in range(num_ensemble):
            print("In CV fold number", self.cv_num, " out of", self.cv_fold_mlp)
            print("Initializing mlp number", i+1, " out of", num_ensemble, "for ensemble")
            # initialize mlp
            m = mlp.MLP(descriptor=self.gene_name+'_'+self.descriptor,
                split=copy.deepcopy(split),
                decision_threshold = 0.5,
                num_epochs = self.num_epochs, # fix number of epochs to 300
                learningrate = self.learningrate,
                mlp_layers = copy.deepcopy(self.layer),
                dropout=self.dropout,
                exclusive_classes = True,
                save_model = False,
                mysteryAAs = self.mysteryAAs_split,
                cv_fold = self.cv_fold_mlp,
                ensemble=self.num_ensemble,
                save_test_out = self.save_test_out)

            # set up graph and session for the model
            m.set_up_graph_and_session()

            # train as per normal
            m.train()

            # store to list
            self.ensemble_lst.append(m)

    def _evaluate_ensemble(self):
        """This function evaluates the test set for the ensemble of mlps
            output: accuracy, auroc, and average precision of the ensemble"""
        print("Evaluating ensemble")
        # true label for calibration
        self.kfold_true_label.append(self.ensemble_lst[0].selected_labels_true)

        # get the true label
        true_label = self.ensemble_lst[0].selected_labels_true
        pred_label_lst = []
        pred_prob_lst = []
        for i in range(len(true_label)):
            pred_label = []
            pred_prob = 0
            # for each mlp, get the predicted label and predicted proba
            for j in range(len(self.ensemble_lst)):
                m = self.ensemble_lst[j]
                pred_label.append(m.selected_pred_labels[i])
                #print("Adding the predicted probability: ", m.selected_pred_probs.shape)
                pred_prob += m.selected_pred_probs[i]
            # for predicted labels, get the most frequent predicted label
            if pred_label.count(0) > pred_label.count(1):
                pred_label_lst.append(0)
            else:
                pred_label_lst.append(1)
            # for predicted probability, get the average predicted probability
            pred_prob_lst.append(pred_prob/len(self.ensemble_lst))

        # calculate accuracy, auroc, and average precision
        accuracy = metrics.accuracy_score(true_label, pred_label_lst)
        auroc = metrics.roc_auc_score(true_label, pred_prob_lst)
        avg_prec = metrics.average_precision_score(true_label, pred_prob_lst)

        # update list for calibration
        self.mlp_kfold_probability.append(pred_prob_lst)

        # if we are saving test output
        if self.save_test_out:
            # get the list of cleanedup df
            clean_df = []
            for model in self.ensemble_lst:
                clean_df.append(self.output_cleanup(model.test_out, model.split.scaler))
            # merge the dataframes
            curr_df = ""
            for df in clean_df:
                if len(curr_df) == 0:
                    curr_df = df
                else:
                    curr_df = pd.merge(curr_df, df, on=['Consensus', 'Position', 'Change'])
                    curr_df['Pred_Prob'] = curr_df['Pred_Prob_x'] + curr_df['Pred_Prob_y']
                    curr_df = curr_df.drop(['Pred_Prob_x', 'Pred_Prob_y'], axis=1)
            # divide pred prob by number of mlps in this ensemble
            curr_df['Pred_Prob'] = curr_df['Pred_Prob'].div(len(self.ensemble_lst)).round(3)
            # for debugging purposes only
            print("The resulting df has", curr_df.isnull().sum(), " null values")
  
            self.fold_df = curr_df          
        return accuracy, auroc, avg_prec




    def output_cleanup(self, df, scaler):
        '''This function takes a dataframe returned by a trained model,
           and reverses one hot encoding and normalization'''

        # debugging: checking for duplicates
        print("Number of rows:", len(df.index))
        col = [col for col in df.columns.values if col.startswith("Consensus") or col.startswith("Change") or col.startswith("Position")]
        print("Checking for duplicates in mysteryAA cleanup")
        print("Number of duplicates:", len(df[df.duplicated(subset=col,keep=False)]))

        consensusAA = []
        changeAA = []
        position = []
        pred_prob = df['Pred_Prob'].values
        # get the consensusAA and changeAA lists by reversing one hot encoding
        count = 0
        col_no_change = 0
        for index, row in df.iterrows():
            found = False
            # get the consensus and change of this row
            for column in df.columns:
                if column.startswith('Consensus_') and row[column]==1:
                    consensus = column[-1]
                    consensusAA.append(consensus)
                if column.startswith('Change_') and row[column]==1:
                    found = True
                    change = column[-1]
                    changeAA.append(change)
            if not found:
                print("Found one with no change", index)
            #increment count
            count += 1
        # convert consensusAA and changeAA lists to numpy
        consensusAA = np.array(consensusAA)
        changeAA = np.array(changeAA)

        # get the original positions by performing inverse transform
        res = scaler.inverse_transform(df[['Position', 'Conservation', 'SigNoise']].values)
        position = [int(row[0]) for row in res]
        
        # stack the 4 columns
        new_np = np.vstack((consensusAA, changeAA, position, pred_prob)).T

        # sort the rows from highest predicted probability to the lowest
        index = np.argsort(new_np[:,-1])
        new_np = new_np[index[::-1]]


        # create a new dataframe with the 4 specified columns
        new_df = pd.DataFrame(new_np, columns=['Consensus', 'Change', 'Position', 'Pred_Prob'])

        new_df['Position'] = new_df['Position'].astype(int)
        new_df['Consensus'] = new_df['Consensus'].astype(str)
        new_df['Change'] = new_df['Change'].astype(str)
        new_df['Pred_Prob'] = new_df['Pred_Prob'].astype(float)

        return new_df

    def _mysteryAAs_output_cleanup(self, filename):
        '''This function converts the csv file outputted by test() in mlp 
           into a csv file with a format that we want (ConsensusAA, ChangeAA, 
           Position, Pred_Prob)'''
        
        print("Cleaning up the file", filename)
        
        # output file clean up
        df = pd.read_csv(filename)
      
        # for debugging purposes
        #pred_prob = sorted(df['Pred_Prob'].values, reverse=True)
        #print(pred_prob)
         
        # debugging: checking for duplicates
        print("Number of rows:", len(df.index))        
        col = [col for col in df.columns.values if col.startswith("Consensus") or col.startswith("Change") or col.startswith("Position")]
        print("Checking for duplicates in mysteryAA cleanup")
        print("Number of duplicates:", len(df[df.duplicated(subset=col,keep=False)]))

        consensusAA = []
        changeAA = []
        position = []
        pred_prob = df['Pred_Prob'].values

        # get the consensusAA and changeAA lists
        count = 0
        for index, row in df.iterrows():
            # get the consensus and change of this row
            for column in df.columns:
                if column.startswith('Consensus_') and row[column]==1:
                    consensus = column[-1]
                    consensusAA.append(consensus)
                if column.startswith('Change_') and row[column]==1:
                    change = column[-1]
                    changeAA.append(change)
            # get the position of this row
            position.append(int(self.ori_position[count]))
            #increment count
            count += 1
        # convert consensusAA and changeAA lists to numpy
        consensusAA = np.array(consensusAA)
        changeAA = np.array(changeAA)
       
        # ------------testing----------------------
        #print("--------check if the same------------")
        #print(consensusAA)
        #print(position)
        #print(changeAA)
        #print("-------Done checking-------------\n\n\n")
        #print("The positions found are")
        #print(len(sorted(position))) 
        
        # stack the 4 columns
        new_np = np.vstack((consensusAA, changeAA, position, pred_prob)).T

        # sort the rows from highest predicted probability to the lowest
        index = np.argsort(new_np[:,-1])
        new_np = new_np[index[::-1]]

        
        # create a new dataframe with the 4 specified columns
        new_df = pd.DataFrame(new_np, columns=['ConsensusAA', 'ChangeAA', 'Position', 'Pred_Prob'])
        
        # add df to list of cleaned dataframes of each mlp in the ensemble to
        # be evaluated in _predict_mysteryAAs()
        # uncomment this if we're predicting using mlp
        #self.ensemble_output_lst.append(new_df)
        
        new_df['Position'] = new_df['Position'].astype(int)
        new_df['ConsensusAA'] = new_df['ConsensusAA'].astype(str)
        new_df['ChangeAA'] = new_df['ChangeAA'].astype(str)
  
        #if self.ensemble == False or 
        # separate mysteryAAs into two files: wes or clinvar
        clinvar_raw = pd.read_csv(os.path.join('data/'+self.gene_name,self.gene_name+'_variants_clinvar_raw.csv'),header= 0)
        clinvar_raw.columns = ['ConsensusAA','Position','ChangeAA']
        clinvar_raw['Position'] = clinvar_raw['Position'].astype(int)
        clinvar_raw['ConsensusAA'] = clinvar_raw['ConsensusAA'].astype(str)
        clinvar_raw['ChangeAA'] = clinvar_raw['ChangeAA'].astype(str)
 
        wes_raw = pd.read_csv(os.path.join('data/'+self.gene_name,self.gene_name+'_variants_wes_raw.csv'),header = 0)
        wes_raw.columns = ['ConsensusAA','Position', 'ChangeAA']
        wes_raw['Position'] = wes_raw['Position'].astype(int)
        wes_raw['ConsensusAA'] = wes_raw['ConsensusAA'].astype(str)
        wes_raw['ChangeAA'] = wes_raw['ChangeAA'].astype(str)
        print(len(new_df))        
        # merge the raw and the resulting mysteryAAs to separate the output into two files
        clinvar_df = pd.merge(new_df, clinvar_raw, how='inner', on=['ConsensusAA','ChangeAA', 'Position'])
        wes_df = pd.merge(new_df, wes_raw, how='inner', on=['ConsensusAA', 'ChangeAA', 'Position'])

        print(len(new_df))
        # create a new csv file
        clinvar_df.to_csv('final_clinvar_' + filename, index=None, header=True)
        wes_df.to_csv('final_wes_' +filename, index=None, header=True)
        

class PredictMysteryAAs_MLP(object):
    def __init__(self, gene_name, descriptor, real_data_split,
                 mysteryAAs_split):
        '''This function uses mlp to predict the mysteryAAs'''
        self.gene_name = gene_name
        self.descriptor = descriptor
        self.real_data_split = real_data_split
        self.mysteryAAs_split = mysteryAAs_split
        
        # set hyperparameters
        self.learningrate = 1000
        self.dropout = 0.4
        self.num_epochs = 1000
        self.num_ensemble = 10 
        self.layer = [30,20]
        self.cv_fold_mlp = 10

        print("Predicting mysteryAAs using MLP...")
        print("Learning rate:", self.learningrate)
        print("Dropout:", self.dropout)
        print("Number of Epochs:", self.num_epochs)
        print("Number of MLPs in the ensemble:", self.num_ensemble)
 
        # define a list to store mlps for our ensemble
        self.ensemble_output_lst = []

        # initialize ensembles and store in the list
        for en_num in range(self.num_ensemble):
            
            print("Initializing ensemble number", en_num + 1, " out of", self.num_ensemble )

            # initialize mlp
            m = mlp.MLP(descriptor=self.gene_name+'_'+self.descriptor,
                split=copy.deepcopy(self.real_data_split),
                decision_threshold = 0.5,
                num_epochs = self.num_epochs,
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
                m.best_valid_loss_epoch = self.num_epochs
            m.clean_up()
          
            print("Cleaning up output file of ensemble number:", en_num)

            # clean up the output file from training
            self._mysteryAAs_output_cleanup(m.mysteryAAs_filename +str(m.best_valid_loss_epoch) + ".csv")
      
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
