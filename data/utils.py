#utils.py

import math
import copy
import numpy as np
import pandas as pd
import sklearn.preprocessing

class Splits(object):
    """Split the provided data into self.train and self.test,
    where each is a Dataset object.
    self.train.data is a numpy array; self.train.labels is a numpy array"""
    def __init__(self,
             data,
             labels,
             one_hotify_these_categorical, #columns to one hotify
             normalize_these_continuous, #columns to normalize
             max_position, #int for the last position of the gene
             columns_to_ensure, #list of column names you must have
             batch_size):
        """Variables:
        <data> is a pandas dataframe where the index is example IDs
            and the values (rows) are the data for that example. 
        <labels> is a pandas dataframe where the index is example IDs
            and the values (rows) are the labels for that example.
        <categorical_variables> is a list of strings with the names
            of the columns that are categorical.
        The indices of <data> and <labels> must match.
        
        Produces self.clean_data and self.clean_labels which are pandas
        dataframes that contain the data and labels respectively."""
        assert data.index.values.tolist()==labels.index.values.tolist()
        self.raw_data = copy.deepcopy(data) #for checks on human readable format later
        self.raw_labels = copy.deepcopy(labels) #for checks on human readable format later
        self.clean_data = data
        self.clean_labels = labels
        self.one_hotify_these_categorical = one_hotify_these_categorical
        self.normalize_these_continuous = normalize_these_continuous
        self.max_position = max_position
        self.columns_to_ensure = columns_to_ensure
        self.batch_size = batch_size
        
        #One-hotify:
        self._one_hotify()
        self._ensure_all_columns()
    
    def _one_hotify(self):
        """Modify self.clean_data so that each categorical column is turned
        into many columns that together form a one-hot vector for that variable.
        E.g. if you have a column 'Gender' with values 'M' and 'F', split it into
        two binary columns 'Gender_M' and 'Gender_F', and add a corresponding
        entry to the one hot indicies in self.date_dict['one_hot_indices']"""
        print('One-hotifying',str(len(self.one_hotify_these_categorical)),'categorical variables')
        print('\tData shape before one-hotifying:',str(self.clean_data.shape))
        #one hotify the categorical variables
        self.clean_data = pd.get_dummies(data = self.clean_data, columns = self.one_hotify_these_categorical, dummy_na = False)
        print('\tData shape after one-hotifying:',str(self.clean_data.shape))
    
    def _ensure_all_columns(self):
        """For small input data, 'one-hotifying' might not produce all of the
        columns or it might not produce them in the right order. Ensure that
        the columns needed are all present and are all in the right order."""
        print('Ensuring columns',self.columns_to_ensure)
        for required_column in self.columns_to_ensure:
            if required_column not in self.clean_data.columns.values:
                self.clean_data[required_column] = 0
        self.clean_data = self.clean_data[self.columns_to_ensure]
    
    def _normalize(self, train_data, test_data):#TODO test this
        """Return train_data and test_data such that the features specified
        in self.normalize_these_continuous have been normalized to
        approximately zero mean and unit variance, based on the
        training dataset only."""
        #http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler
        # normalize the training data
        train_selected = train_data[self.normalize_these_continuous].values
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(train_selected)
        print('scaler mean:',self.scaler.mean_,'\nscaler scale:',self.scaler.scale_)
        assert len(self.normalize_these_continuous)==self.scaler.mean_.shape[0]==self.scaler.scale_.shape[0]
        train_data.loc[:,self.normalize_these_continuous] = self.scaler.transform(train_selected)
        
        # normalize the test data if there is test data
        #(there will be no test data when doing the prediction on mysteryAAs
        #because in that case all data is used for training)
        if test_data.shape[0] > 0:
            test_selected = test_data[self.normalize_these_continuous].values
            test_data.loc[:,self.normalize_these_continuous] = self.scaler.transform(test_selected)
        #print('Normalized data:\n\tscaler.mean_',str(scaler.mean_), '\n\tscaler.scale_',str(scaler.scale_))
        return train_data, test_data
        
    def _make_splits_cv(self, train_indices, test_indices):
        """Split up self.clean_data and self.clean_labels into train and test
        data. <train_indices> and <test_indices> are the output of a function
        like model_selection.StratifiedKFold.
        Perform normalization based on the training data."""
        assert self.clean_data.index.values.tolist()==self.clean_labels.index.values.tolist()
        extra_args = {'data_meanings':self.clean_data.columns.values.tolist(),
            'label_meanings': self.clean_labels.columns.values.tolist(),
            'batch_size': self.batch_size}

        # get train data
        train_data = self.clean_data.iloc[train_indices]
        train_labels = self.clean_labels.iloc[train_indices]

        # get test data
        test_data = self.clean_data.iloc[test_indices]
        test_labels = self.clean_labels.iloc[test_indices]
        
        # normalize
        train_data, test_data = self._normalize(train_data, test_data)
         
        # convert everything to array
        train_data = train_data.values
        train_labels = train_labels.values
        test_data = test_data.values
        test_labels = test_labels.values
        
        #Note: you want to shuffle the training set between each epoch
        #so that the model can't cheat and learn the order of the training
        #data. You don't want to bother shuffling the test set because it is
        #just used for evaluation.
        self.train = Dataset(train_data, train_labels, shuffle = True, **extra_args)
        self.test = Dataset(test_data, test_labels, shuffle = False, **extra_args)
        print('\tTrain data shape:',str(train_data.shape))
        print('\tTest data shape:',str(test_data.shape))
        print('\tLength of one label:',str(train_labels.shape[1]))


class Dataset(object):
    def __init__(self,
                data,
                labels,
                shuffle,
                data_meanings,
                label_meanings,
                batch_size):
        """Variables
        <data> is a numpy array
        <data_meanings> is a list of strings describing the meanings of
            the columns of <data>
        <labels> is a numpy array
        <label_meanings> is a list of strings describing the meanings of
            the columns of <labels>
        Note: it is assumed that the order of examples in <data> matches the
        order of examples in <labels>"""
        assert data.shape[0] == labels.shape[0]
        self.data = data
        self.data_meanings = data_meanings
        self.labels = labels
        self.label_meanings = label_meanings
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_examples = np.shape(data)[0]
        self.epochs_completed = 0
        self.index_in_epoch = 0
        
    #Methods
    def next_batch(self):
        """Return the next set of examples from this data set.
        The last batch in an epoch may be smaller than <self.batch_size> if
        <self.batch_size> does not divide evenly into the total number of examples.
        In this case, the last batch in an epoch will contain only the
        remaining examples in the epoch."""
        start = self.index_in_epoch
        # Shuffle for the beginning of an epoch if indicated
        if start == 0 and self.shuffle: 
            perm0 = np.arange(self.num_examples)
            # shuffle the indices
            np.random.shuffle(perm0)
            # store the index permutation for when we have to retrieve unnormalized positions
            self.perm_ind = perm0
            self.data = self.data[perm0]
            self.labels = self.labels[perm0]
 
        #If you're in the middle of an epoch:
        if start + self.batch_size < self.num_examples:
            self.index_in_epoch += self.batch_size
            end = self.index_in_epoch
            all_data = self.data[start:end]
            all_labels = self.labels[start:end]
            return all_data, all_labels
            
        # If you're ready to continue to the next epoch, i.e. if
        #start + self.batch_size >= self.num_examples
        else:
            # Get the rest examples in this epoch
            rest_num_examples = self.num_examples - start
            data_rest_part = self.data[start:self.num_examples]
            labels_rest_part = self.labels[start:self.num_examples]

            #Prepare for the next epoch:
            start = 0
            self.index_in_epoch = 0
            self.epochs_completed += 1
            return data_rest_part, labels_rest_part
