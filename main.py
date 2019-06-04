#main.py

import copy
import os
import pickle
import pandas as pd
import numpy as np
from sklearn import model_selection, metrics

#Custom imports
from data import utils as utils
from data import clean_data as clean_data
import mlp_model
import regression

class RunGeneModel(object):
    def __init__(self, gene_name, descriptor, shared_args, cols_to_delete=[], ensemble=False,cv_fold_lg=10, cv_fold_mlp=10):
        """<gene_name> is a string, one of: 'kcnh2', 'kcnq1', 'ryr2', or 'scn5a'."""
        self.gene_name = gene_name
        self.descriptor = descriptor
        self.shared_args = shared_args
        self.cols_to_delete = cols_to_delete
        self.ensemble = ensemble
        self.cv_fold_lg = cv_fold_lg
        self.cv_fold_mlp = cv_fold_mlp
    
    def do_all(self):
        self._prep_data()
        self._prep_split_data(self.inputx,self.split_args)
        self._prep_mysteryAAs()
        self._run_mlp()
        #self._run_logreg()
    
    def _prep_data(self):
        #Real data with healthy and diseased
        ag = clean_data.AnnotatedGene(self.gene_name)
        ag.annotate_everything()
        self.inputx = ag.inputx #make it self.inputx so you can access from testing script
        self.mysteryAAs = ag.mysteryAAs
        self.columns_to_ensure_here = [x for x in ag.columns_to_ensure if x not in self.cols_to_delete]
        self.split_args = {'train_percent':0.7,
                        'valid_percent':0.15,
                        'test_percent':0.15,
                        'max_position':ag.max_position,
                        'columns_to_ensure':self.columns_to_ensure_here}
        self.ag = ag
    
    def _prep_split_data(self, inputx, split_args):
        """There are arguments to this function to facilitate unit testing"""
        data = (copy.deepcopy(inputx)).drop(columns=['Label']+self.cols_to_delete)
        labels = copy.deepcopy(inputx[['Label']])
        print('Fraction of diseased:',str( np.sum(labels)/len(labels) ) )
        all_args = {**self.shared_args, **split_args}
        self.real_data_split = utils.Splits(data = data,
                             labels = labels,
                             **all_args)


    
    def _prep_mysteryAAs(self):
        #WES data, mysteryAAs (want predictions for these)
        mysteryAAs_data = (copy.deepcopy(self.mysteryAAs)).drop(columns=self.cols_to_delete)
        mysteryAAs_labels = pd.DataFrame(np.zeros((mysteryAAs_data.shape[0],1)), columns=['Label'])
        self.mysteryAAs_split = utils.Splits(data = mysteryAAs_data,
                                     labels = mysteryAAs_labels,
                                     train_percent = 1.0,
                                     valid_percent = 0,
                                     test_percent = 0,
                                     max_position = self.ag.max_position,
                                     columns_to_ensure = self.columns_to_ensure_here,
                                     **self.shared_args).train
        assert self.mysteryAAs_split.data.shape[0] == self.mysteryAAs.shape[0]
        
        #Save pickled split:
        print('Saving pickled split')
        pickle.dump(self.real_data_split, open(self.gene_name+'_'+self.descriptor+'.pickle', 'wb'),-1)
    
    def _run_mlp(self):
        #Run MLP
        print('Running MLP')

        # set hyperparameters here
        self.learningrate = 1e-2
        self.dropout = 0

        # if we are performing cross validation
        if self.cv_fold_mlp > 1:
            # initialize an empty list to store test accuracy for each fold
            fold_acc = []
            fold_auroc = []
            fold_avg_prec = []
            cv = model_selection.KFold(n_splits=self.cv_fold_mlp)
            fold_num = 1
            data = self.real_data_split.clean_data
            label = self.real_data_split.clean_labels
            for train, test in cv.split(data, label):
                # create a copy of the real_data_split
                split = copy.deepcopy(self.real_data_split)
                
                # update the splits with the train and test indices for this cv loop
                split._make_splits_cv(train, test)

                # check if we're doing ensembling
                if self.ensemble:
                    # set number of mlps in the ensemble
                    num_ensemble = 5

                    # initialize ensembles
                    self._init_ensemble(num_ensemble, split)

                    # evaluate the accuracy of this fold using the ensemble initialized
                    accuracy, auroc, avg_prec = self._evaluate_ensemble()

                    fold_acc.append(accuracy)
                    fold_auroc.append(auroc)
                    fold_avg_prec.append(avg_prec)

                else:
                    #redefine mlp object with the new split
                    m = mlp_model.MLP(descriptor=self.gene_name+'_'+self.descriptor,
                        split=split,
                        decision_threshold = 0.5,
                        num_epochs = 300, # fix number of epochs to 300
                        learningrate = self.learningrate,
                        mlp_layers = copy.deepcopy([30,20]),
                        dropout=self.dropout,
                        exclusive_classes = True,
                        save_model = False,
                        mysteryAAs = self.mysteryAAs_split,
                        cv_fold = self.cv_fold_mlp,
                        ensemble=self.ensemble)

                    # set up graph and session for the model
                    m.set_up_graph_and_session()
                    # train as per normal if we are not doing ensembling
                    m.train()

                    # append test accuracy to list
                    df = m.eval_results_test['accuracy']
                    for label in df.index.values:
                        acc = df.loc[label,'epoch_'+str(m.num_epochs)]
                        print("The accuracy for fold number ", str(fold_num), " is ", str(acc))
                    fold_acc.append(acc)
                    # append auroc to list
                    df = m.eval_results_test['auroc']
                    for label in df.index.values:
                        auroc = df.loc[label,'epoch_'+str(m.num_epochs)]
                        print("The auroc for fold number ", str(fold_num), " is ", str(auroc))
                    fold_auroc.append(auroc)
                    #append average precision to list
                    df = m.eval_results_test['avg_precision']
                    for label in df.index.values:
                        avg_prec = df.loc[label,'epoch_'+str(m.num_epochs)]
                        print("The average precision for fold number ", str(fold_num), " is ", str(avg_prec))
                    fold_avg_prec.append(avg_prec)

                fold_num += 1

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

            # write results to a txtfile
            path = "mlp_results/"
            filename = self.descriptor+'_' +str(self.cv_fold_mlp)+'cv_' + str(learningrate) + 'learnrate_' + str(dropout) + 'drop_'+str(self.ensemble)+'_ensemble_results.txt'
            with open(path+filename, 'w') as f:
                f.write("\n\n The average cross validation accuracy is :"+ str(tot_acc/self.cv_fold_mlp)+ "\n\n\n")
                f.write("\n\n The average cross validation auroc is :"+ str(tot_auroc/self.cv_fold_mlp)+ "\n\n\n")
                f.write("\n\n The average cross validation average precision is :"+ str(tot_avg_prec/self.cv_fold_mlp)+ "\n\n\n")

            print("\n\n\nDone\n\n\n")
        
        # otherwise, run all
        else:
            # create an mlp object
            m = mlp_model.MLP(descriptor=self.gene_name+'_'+self.descriptor,
                split=copy.deepcopy(self.real_data_split),
                decision_threshold = 0.5,
                num_epochs = 300, # set number of epochs to 300
                learningrate = self.learningrate,
                mlp_layers = copy.deepcopy([30,20]),
                dropout=self.dropout,
                exclusive_classes = True,
                save_model = False,
                mysteryAAs = self.mysteryAAs_split,
                cv_fold = self.cv_fold_mlp,
                ensemble=self.ensemble)
            m.run_all()

    def _init_ensemble(self, num_ensemble, split):
        """This function initializes mlps for the ensemble.
           Inputs: num_ensemble, the number of mlps in the ensemble
                   split, the split object to specify training and testing data
        """
        # define a list to store mlps for our ensemble
        self.ensemble_lst = []

        # initialize ensembles and store in the list
        for _ in range(num_ensemble):
            # initialize mlp
            m = mlp_model.MLP(descriptor=self.gene_name+'_'+self.descriptor,
                split=split,
                decision_threshold = 0.5,
                num_epochs = 300, # fix number of epochs to 300
                learningrate = self.learningrate,
                mlp_layers = copy.deepcopy([30,20]),
                dropout=self.dropout,
                exclusive_classes = True,
                save_model = False,
                mysteryAAs = self.mysteryAAs_split,
                cv_fold = self.cv_fold_mlp,
                ensemble=self.ensemble)

            # set up graph and session for the model
            m.set_up_graph_and_session()

            # train as per normal
            m.train()

            # store to list
            self.ensemble_lst.append(m)

    def _evaluate_ensemble(self):
        """This function evaluates the test set for the ensemble of mlps
            output: accuracy, auroc, and average precision of the ensemble"""

        # get the true label
        true_label = self.ensemble_lst[0].selected_labels_true
        pred_label_lst = []
        pred_prob_lst = []
        for i in range(len(true_label)):
            label = true_label[i]
            pred_label = []
            pred_prob = 0
            # for each mlp, get the predicted label and predicted proba
            for i in range(0,5):
                m = self.ensemble_lst[i]
                pred_label.append(m.selected_pred_labels[i])
                pred_prob += m.selected_pred_probs[i]
            # for predicted labels, get the most frequent predicted label
            if pred_label.count(0) > pred_label.count(1):
                pred_label_lst.append(0)
            else:
                pred_label_lst.append(1)
            # for predicted probability, get the average predicted probability
            pred_prob_lst.append(pred_prob/len(true_label))

        # calculate accuracy, auroc, and average precision
        accuracy = metrics.accuracy_score(true_label, pred_label_lst)
        auroc = metrics.roc_auc_score(true_label, pred_prob_lst)
        avg_prec = metrics.average_precision_score(true_label, pred_label_lst)

        return accuracy, auroc, avg_prec

    def _run_logreg(self):
        # Run Logistic Regression
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
        print("Running Log Reg")

        classifier_penalty= ['l1', 'l2']
        classifier_C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

        # set the k value for k fold cross validation (# of folds for cross validation. Set to 0 if 
        # we don't want to do cross validation)
        kfold = self.cv_fold_lg
        k = 0
        for pen in classifier_penalty:
          for C in classifier_C:
            lg = regression.LogisticRegression(descriptor=descriptor, split=copy.deepcopy(self.real_data_split),logreg_penalty=pen, C=C, figure_num=k, fold=kfold)
            k += 2


if __name__=='__main__':
    variations = {'noSN':['Position','Conservation'],
        'withSN':['Position','Conservation','SigNoise']}
    for descriptor in variations:
        cont_vars = variations[descriptor]
        shared_args = {'impute':False,
                        'impute_these_categorical':[],
                        'impute_these_continuous':[],
                        'one_hotify':True,
                        'one_hotify_these_categorical':['Consensus','Change','Domain'],  #cat_vars
                        'normalize_data':True,
                        'normalize_these_continuous':cont_vars,
                        'seed':10393, #make it 12345 for original split
                        'batch_size':300}
        RunGeneModel(gene_name='ryr2', descriptor=descriptor,shared_args = shared_args, cols_to_delete=list(set(['Position','Conservation','SigNoise'])-set(cont_vars)), ensemble=True, cv_fold_lg=0, cv_fold_mlp=5).do_all()


