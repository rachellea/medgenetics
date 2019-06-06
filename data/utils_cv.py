#utils.py
#Rachel Ballantyne Draelos

import math
import numpy as np
import pandas as pd
import sklearn.preprocessing
from sklearn import model_selection

class SplitsCV(object):
    """Split the provided data into self.train, self.valid, and self.test,
    where each is a Dataset object.
    self.train.data is a numpy array; self.train.labels is a numpy array"""
    def __init__(self,
             data,
             labels,
             seed,
             batch_size,
             cross_val_fold):
        """Variables:
        <data> is a pandas dataframe where the index is example IDs
            and the values (rows) are the data for that example. 
        <labels> is a pandas dataframe where the index is example IDs
            and the values (rows) are the labels for that example.
        <categorical_variables> is a list of strings with the names
            of the columns that are categorical.
        The indices of <data> and <labels> must match."""
        assert data.index.values.tolist()==labels.index.values.tolist()
        self.clean_data = data
        self.clean_labels = labels
        
        self.train_indices = train_indices
        self.valid_percent = valid_percent
        self.test_indices = test_indices
        if seed is not None:
            self.seed = seed
        else:
            self.seed = np.random.randint(0,10e6)
        self.batch_size = batch_size

        self.cross_val_fold = cross_val_fold
 
        self._make_splits_cv()
        
    
 

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
            np.random.shuffle(perm0)
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


#Used for testing
def arrays_are_close_enough(array1, array2, tol =  1e-6):
    """Because the following stuff doesn't work at all:
    numpy.testing.assert_almost_equal 
    np.all
    np.array_equal
    np.isclose"""
    assert array1.shape == array2.shape
    if len(array1.shape)==1: #one-dimensional
        for index in range(len(array1)):
            if np.isnan(array1[index]) and np.isnan(array2[index]):
                pass
            else:
                assert array1[index] - array2[index] < tol, 'Error at index '+str(index)
    else: #two-dimensional
        for row in range(array1.shape[0]):
            for col in range(array1.shape[1]):
                if np.isnan(array1[row,col]) and np.isnan(array2[row,col]):
                    pass
                else:
                    assert array1[row,col] - array2[row,col] < tol, 'Error at '+str(row)+', '+str(col)
    return True