#mlp.py

import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf

class MLP(object):
    """Multilayer perceptron."""
    def __init__(self,
                 descriptor,
                 split,
                 decision_threshold,
                 num_epochs,
                 learningrate, #e.g. 1e-4
                 mlp_layers,
                 dropout,
                 mysteryAAs):
        """Variables
        <descriptor>: string that will be attached to the beginning of all
            model output files.
        <split>: created by Splits class in utils.py
        <mlp_layers>: list of ints e.g. [50, 30, 25] means the first layer of the
            MLP has size 50 and is followed by two hidden layers, of sizes
            30 and 25 respectively.
        <mysteryAAs>: contains all unknown mutations ('predict set' on which we
            must make predictions)"""
        print('\n\n\n\n**********',descriptor,'**********')
        self.descriptor = descriptor
        
        #Data sets
        self.train_set = split.train
        self.test_set = split.test
        self.mysteryAAs = mysteryAAs 
               
        #Tracking
        assert self.train_set.batch_size == self.test_set.batch_size
        self.num_train_batches = math.ceil((self.train_set.num_examples)/self.train_set.batch_size)
        self.num_test_batches = math.ceil((self.test_set.num_examples)/self.test_set.batch_size)
        if self.mysteryAAs is not None: self.num_mysteryAAs_batches = math.ceil((self.mysteryAAs.num_examples)/self.mysteryAAs.batch_size)
        self.num_epochs = num_epochs
        self.training_loss = np.zeros((self.num_epochs))
        self.num_batches_done = 0
        self.num_epochs_done = 0
        
        #Architecture and model setup
        self.decision_threshold = decision_threshold #used to calculate accuracy
        self.x_length = self.train_set.data.shape[1] #length of a single example
        self.y_length = self.train_set.labels.shape[1] #length of one example's label vector
        self.learningrate = learningrate
        self.mlp_layers = mlp_layers
        self.mlp_layers.append(self.y_length) #ensure predictions will have correct dimensions
        print('Mlp_layers is',str(self.mlp_layers))
        self.dropout = dropout
        self.test_out = {} #this will be a dictionary of data and predictions for every epoch
    
    def run_all(self):
        """Run all key methods"""
        self.set_up_graph_and_session()
        self.train_and_test()
        self.session.close()

    #~~~Key Methods~~~#
    def set_up_graph_and_session(self):
        #Build the graph
        tf.logging.set_verbosity(tf.logging.INFO) #Set output detail level (options: DEBUG, INFO, WARN, ERROR, or FATAL)
        self.graph = tf.Graph()
        self._build_graph()
        self.session = tf.Session(graph=self.graph)
        self.session.run(self.initialize)            
    
    def train_and_test(self):
        #Train
        for j in range(self.num_epochs):
            epoch_loss = 0
            for i in range(self.num_train_batches):
                x_data_batch, y_labels_batch = self.train_set.next_batch()
                feed_dict_train = {self.x_input: x_data_batch,
                                   self.y_labels: y_labels_batch,
                                   self.keep_prob:1-self.dropout}
                curr_loss, curr_opti = self.session.run([self.loss, self.optimizer], feed_dict=feed_dict_train)
                self.num_batches_done+=1
                epoch_loss+=curr_loss
            self.num_epochs_done+=1
            self.training_loss[self.num_epochs_done-1] = epoch_loss
            #Test
            self.test('Test')
    
    def test(self, chosen_dataset):
        if chosen_dataset == 'Test':
            chosen_set = self.test_set
            num_batches = self.num_test_batches
            all_eval_results = self.eval_results_test
        elif chosen_dataset == 'mysteryAAs':
            chosen_set = self.mysteryAAs
            num_batches = self.num_mysteryAAs_batches
        else:
            assert False, "chosen_dataset must be 'Test' or 'mysteryAAs' but you passed "+str(chosen_dataset)
        
        epoch_loss = 0
        for i in range(num_batches):
            x_data_batch, y_labels_batch = chosen_set.next_batch()
            feed_dict = {self.x_input: x_data_batch,
                         self.y_labels: y_labels_batch,
                         self.keep_prob: 1}
            curr_loss, batch_pred_probs, batch_pred_labels = self.session.run([self.loss, self.pred_probs,self.pred_labels], feed_dict=feed_dict)
            epoch_loss+=curr_loss
            #Gather the outputs of subsequent batches together:
            if i == 0:
                self.entire_pred_probs = batch_pred_probs
                self.entire_pred_labels = batch_pred_labels
                self.labels_true = y_labels_batch
                entire_x = x_data_batch
            else:
                #concatenate results on assuming that the zeroth dimension is the training example dimension
                self.entire_pred_probs = np.concatenate((self.entire_pred_probs,batch_pred_probs),axis = 0)
                self.entire_pred_labels = np.concatenate((self.entire_pred_labels,batch_pred_labels),axis=0)
                self.labels_true = np.concatenate((self.labels_true, y_labels_batch),axis=0)
                #$self.labels_true = labels_true
                entire_x = np.concatenate((entire_x, x_data_batch),axis = 0)
       
        #test_out is a dictionary with keys that are epochs and values
        #that are dataframes. A dataframe contains the data set (entire_x)
        #as well as the predicted probabilities, predicted labels, and
        #true labels for all examples
        test_out_df = pd.DataFrame(np.concatenate((entire_x, self.entire_pred_probs, self.entire_pred_labels, self.labels_true),axis = 1),
                               columns=self.train_set.data_meanings+['Pred_Prob','Pred_Label','True_Label'])
        test_out_df = test_out_df.sort_values(by='Position')
        self.test_out['epoch_'+str(self.num_epochs_done)] = test_out_df
        
        #~~~Save outputs for mysteryAAs~~~#
        if chosen_dataset == 'mysteryAAs':
            out = pd.DataFrame(np.concatenate((entire_x, entire_pred_probs, entire_pred_labels),axis = 1),
                               columns=self.train_set.data_meanings+['Pred_Prob','Pred_Label'])
            self.mysteryAAs_filename = self.descriptor + '_mysteryAAs_results_epoch_' 
            out.to_csv(self.mysteryAAs_filename + str(self.num_epochs_done) + ".csv", header=True,index=False)
    
    #~~~Helper Methods for Graph Building~~~#
    def _build_graph(self):
        self._define_placeholders_and_layers()
        self._define_loss()
        self._define_optimizer_and_performance()
    
    def _define_placeholders_and_layers(self):
        with self.graph.as_default():
            #~~~ Placeholders ~~~#
            self.x_input = tf.placeholder(tf.float32,
                           shape = [None, self.x_length],
                           name='self.x_input')
            print("Shape of self.x_input: "+str(self.x_input.get_shape().as_list()))
            
            self.y_labels = tf.placeholder(tf.float32,
                           shape = [None, self.y_length],
                           name='self.y_labels')
            print("Shape of self.y_labels: "+str(self.y_labels.get_shape().as_list()))
        
            self.keep_prob = tf.placeholder(tf.float32)

            #~~~ Model ~~~#
            self.pred_raw = self._create_mlp(variable_scope_name = 'mlp',
                               inputx = self.x_input,
                               num_inputs_zero = self.x_length,
                               hidden_layers = self.mlp_layers)
            
    def _define_loss(self):
        with self.graph.as_default():
            with tf.variable_scope('self.loss'):
                if self.y_length == 1: #binary classification
                    print('Binary classifier: using sigmoid cross entropy')
                    self.loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=self.pred_raw, labels=self.y_labels) )
    
    def _define_optimizer_and_performance(self):
        with self.graph.as_default():
            with tf.variable_scope('optimizer'):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss)
            
            with tf.variable_scope('performance_measures'):
                if self.y_length == 1: #binary classification
                    self.pred_probs = tf.nn.sigmoid(self.pred_raw)
                    self.pred_labels = tf.cast((self.pred_probs >= self.decision_threshold), tf.int32)
                print('Shape of self.pred_probs',str(self.pred_probs.get_shape().as_list()))
                print('Shape of self.pred_labels:',str(self.pred_labels.get_shape().as_list()))
            self.initialize = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
            
    def _create_mlp(self, variable_scope_name, inputx, num_inputs_zero, hidden_layers):
        with tf.variable_scope(variable_scope_name):
            use_relu_flag = True
            zero = self._new_fc_layer(inputx=inputx,
                                   num_inputs = num_inputs_zero,
                                   num_outputs = hidden_layers[0],
                                   name='layer_0',
                                   use_relu = use_relu_flag)
            print("Shape of",variable_scope_name,"layer_0:",str(zero.get_shape().as_list()),', relu=',str(use_relu_flag))
            
            for i in range(1, len(hidden_layers)): #hidden layers
                if i==1:
                    tmp = zero
                if i == (len(hidden_layers)-1):
                    use_relu_flag = False #no relu on last layer
                hi = self._new_fc_layer(inputx=tmp,
                                 num_inputs=hidden_layers[i-1],
                                 num_outputs=hidden_layers[i],
                                 name='layer_'+str(i),
                                 use_relu=use_relu_flag)
                tmp = hi
                print("Shape of", variable_scope_name, "layer_"+str(i),":",str(hi.get_shape().as_list()),', relu=',str(use_relu_flag))
        return hi
    
    def _new_fc_layer(self, inputx, num_inputs, num_outputs, name, use_relu):
        """Create a new fully-connected layer."""
        weights = tf.Variable(tf.truncated_normal([num_inputs, num_outputs], stddev=0.05), name=(name+'_weights'))
        biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]), name=(name+'_biases'))
        layer = tf.matmul(inputx, weights) + biases
        if use_relu:
            layer = tf.nn.relu(layer)
        return tf.nn.dropout(layer, self.keep_prob)