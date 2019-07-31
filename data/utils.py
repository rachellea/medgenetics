#utils.py
#Rachel Ballantyne Draelos

import math
import numpy as np
import pandas as pd
import sklearn.preprocessing

class Splits(object):
    """Split the provided data into self.train, self.valid, and self.test,
    where each is a Dataset object.
    self.train.data is a numpy array; self.train.labels is a numpy array"""
    def __init__(self,
             data,
             labels,
             train_percent,
             valid_percent,
             test_percent,
             impute, #if True, impute
             impute_these_categorical, #columns to impute with mode
             impute_these_continuous, #columns to impute with median
             one_hotify, #if True, one-hotify specified categorical variables
             one_hotify_these_categorical, #columns to one hotify
             normalize_data, #if True, normalize data based on training set
             normalize_these_continuous, #columns to normalize
             seed, #seed to determine shuffling order before making splits; used for testing
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
        The indices of <data> and <labels> must match."""

        # initialize args for cv
        self.cv_args = {'data': data,
                        'labels': labels,
                        'valid_percent': valid_percent,
                        'impute': impute,
                        'impute_these_categorical': impute_these_categorical,
                        'impute_these_continuous': impute_these_continuous,
                        'one_hotify': one_hotify,
                        'one_hotify_these_categorical': one_hotify_these_categorical,
                        'normalize_data': normalize_data,
                        'normalize_these_continuous': normalize_these_continuous,
                        'seed': seed,
                        'max_position': max_position,
                        'columns_to_ensure': columns_to_ensure,
                        'batch_size': batch_size}

        assert (train_percent+valid_percent+test_percent)==1
        assert data.index.values.tolist()==labels.index.values.tolist()
        self.clean_data = data
        self.clean_labels = labels
        self.impute_these_categorical = impute_these_categorical
        self.impute_these_continuous = impute_these_continuous
        self.one_hotify_these_categorical = one_hotify_these_categorical
        self.normalize_these_continuous = normalize_these_continuous
        self.max_position = max_position
        self.columns_to_ensure = columns_to_ensure
        
        self.train_percent = train_percent
        self.valid_percent = valid_percent
        self.test_percent = test_percent
        if seed is not None:
            self.seed = seed
        else:
            self.seed = np.random.randint(0,10e6)
        self.batch_size = batch_size
        
        self._get_split_indices() #defines self.trainidx and self.testidx
        self._shuffle_before_splitting()
        
        #Further data prep:
        if impute:
            self._impute()
        if one_hotify:
            self._one_hotify()
            self._ensure_all_columns()
        if normalize_data:
            self._normalize()
            pass
        else:
            print('WARNING: you elected not to normalize your data. This could lead to poor performance.')
        self._make_splits() #creates self.train, self.valid, and self.test

    def _get_split_indices(self):
        """Get indices that will be used to split the data into train, test,
        and validation."""
        self.trainidx = int(self.clean_data.shape[0] * self.train_percent)
        self.testidx = int(self.clean_data.shape[0] * (self.train_percent+self.test_percent))

    def _shuffle_before_splitting(self):
        idx = np.arange(0, self.clean_data.shape[0])
        np.random.seed(self.seed)
        print('Creating splits based on seed',str(self.seed))
        np.random.shuffle(idx)
        self.clean_data = self.clean_data.iloc[idx]
        self.clean_labels = self.clean_labels.iloc[idx]
        
        # get position column before splitting
        self.position = self.clean_data['Position'].values

    def _impute(self):
        """Impute categorical variables using the mode of the training data
        and continuous variables using the median of the training data."""
        #impute missing categorical values with the training data mode https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.mode.html
        print('Imputing categorical variables with mode:\n',str(self.impute_these_categorical))
        training_data = self.clean_data.iloc[0:self.trainidx,:]
        imputed_with_modes = (self.clean_data[self.impute_these_categorical]).fillna((training_data[self.impute_these_categorical]).mode().iloc[0])
        self.clean_data[self.impute_these_categorical] = imputed_with_modes  
        
        #impute missing continuous values with the training data median
        print('Imputing continuous variables with median:\n',str(self.impute_these_continuous))
        imputed_with_medians = (self.clean_data[self.impute_these_continuous]).fillna((training_data[self.impute_these_continuous]).median())
        self.clean_data[self.impute_these_continuous] = imputed_with_medians
        
        print('Done imputing')

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
    
        #now that you've replaced all the original categorical columns with
        #multiple binary columns, find the indices of the new columns
        one_hot_indices = []
        for catvar in self.one_hotify_these_categorical:
            indices = []
            for colname in self.clean_data.columns.values:
                if (str(catvar)+'_') in colname:
                    indices.append(self.clean_data.columns.get_loc(colname))
            if len(indices) > 0: #TODO: figure out why it's 0 sometimes
                one_hot_indices.append(indices)
        self.one_hot_indices = one_hot_indices
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
    
    def _normalize(self):
        #TODO test this
        """Provide the features specified in self.normalize_these_continuous
        with approximately zero mean and unit variance, based on the
        training dataset only."""
        train_data = (self.clean_data[self.normalize_these_continuous].values)[0:self.trainidx,:]
        #http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler
        scaler = sklearn.preprocessing.StandardScaler().fit(train_data)
        print('Normalizing data:\n\tscaler.mean_',str(scaler.mean_),
              '\n\tscaler.scale_',str(scaler.scale_))
        assert len(self.normalize_these_continuous)==scaler.mean_.shape[0]==scaler.scale_.shape[0]
        self.clean_data[self.normalize_these_continuous] = scaler.transform((self.clean_data[self.normalize_these_continuous]).values)
    
    def _make_splits(self):
        """Split up self.clean_data and self.clean_labels
        into train, test, and valid data."""
        assert self.clean_data.index.values.tolist()==self.clean_labels.index.values.tolist()
        extra_args = {'data_meanings':self.clean_data.columns.values.tolist(),
            'label_meanings': self.clean_labels.columns.values.tolist(),
            'batch_size': self.batch_size}
    
        data_matrix = self.clean_data.values
        labels_matrix = self.clean_labels.values
    
        train_data = data_matrix[0:self.trainidx,:]
        train_labels = labels_matrix[0:self.trainidx]
        test_data = data_matrix[self.trainidx:self.testidx,:]
        test_labels = labels_matrix[self.trainidx:self.testidx]
        valid_data = data_matrix[self.testidx:,:]
        valid_labels = labels_matrix[self.testidx:]
    
        #Note: you want to shuffle the training set between each epoch
        #so that the model can't cheat and learn the order of the training
        #data. You don't want to bother shuffling the validation or test
        #sets because those are just evaluated on a fixed model.
        self.train = Dataset(train_data, train_labels, shuffle = True, **extra_args)
        self.test = Dataset(test_data, test_labels, shuffle = False, **extra_args)
        self.valid = Dataset(valid_data, valid_labels, shuffle= False, **extra_args)
        print('Finished making splits')
        print('\tTrain data shape:',str(train_data.shape))
        print('\tValid data shape:',str(valid_data.shape))
        print('\tTest data shape:',str(test_data.shape))
        print('\tLength of one label:',str(train_labels.shape[1]))

    def _make_splits_cv(self, train_indices, test_indices):
        """Split up self.clean_data and self.clean_labels
        into train, test, and valid data."""
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

        # convert everything to array
        train_data = train_data.values
        train_labels = train_labels.values
        test_data = test_data.values
        test_labels = test_labels.values
        

        #Note: you want to shuffle the training set between each epoch
        #so that the model can't cheat and learn the order of the training
        #data. You don't want to bother shuffling the validation or test
        #sets because those are just evaluated on a fixed model.
        self.train = Dataset(train_data, train_labels, shuffle = True, **extra_args)
        self.test = Dataset(test_data, test_labels, shuffle = False, **extra_args)

        print('Finished making cross validation splits')
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
