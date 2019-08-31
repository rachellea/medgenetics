#autoencoder.py
#Rachel Draelos

##########
#Imports #----------------------------------------------------------------------
##########
import numpy as np
import pandas as pd
import tensorflow as tf

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import math
import random

#Custom
import evaluate

##############
# Multisense #------------------------------------------------------------------
##############
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
                 exclusive_classes,
                 save_model,
                 mysteryAAs,
                 cv_fold,
                 ensemble):
        """
        Variables
        <descriptor>: string that will be attached to the beginning of all
            model output files.
        <split>: created by Splits class in utils.py
        <mlp_layers>: list of ints e.g. [50, 30, 25] means the first layer of the
            MLP has size 50 and is followed by two hidden layers, of sizes
            30 and 25 respectively.
        <exclusive_classes>: if True use softmax; if False use sigmoid
        <mysteryAAs>: contains all possible mutations (excluding anything
            in the train, test, or validation sets)."""
        print('\n\n\n\n**********',descriptor,'**********')
        self.descriptor = descriptor

        self.split = split
        
        #Data sets
        self.mysteryAAs = mysteryAAs 
        self.train_set = split.train
        self.test_set = split.test
        self.valid_set = split.valid
        self.preserved_data_meanings = split.train.data_meanings
        
        #Number of batches per epoch
        assert self.train_set.batch_size == self.test_set.batch_size == self.valid_set.batch_size
        self.num_train_batches = math.ceil((self.train_set.num_examples)/self.train_set.batch_size)
        self.num_test_batches = math.ceil((self.test_set.num_examples)/self.test_set.batch_size)
        self.num_valid_batches = math.ceil((self.valid_set.num_examples)/self.valid_set.batch_size)
        # self.num_mysteryAAs_batches = math.ceil((self.mysteryAAs.num_examples)/self.mysteryAAs.batch_size)
        self.num_epochs = num_epochs
        
        #Tracking losses and evaluation results
        self.training_loss = np.zeros((self.num_epochs))
        self.valid_loss = np.zeros((self.num_epochs))
        self.eval_results_valid, self.eval_results_test = evaluate.initialize_evaluation_dfs(self.train_set.label_meanings, self.num_epochs)
        self.num_batches_done = 0
        self.num_epochs_done = 0
        
        #For early stopping:
        self.initial_patience = 30
        self.best_valid_loss = np.inf
        self.best_valid_loss_epoch = 0
        self.patience_remaining = 30
        
        #Architecture and model setup
        self.exclusive_classes = exclusive_classes #Determines loss function
        self.save_model = save_model
        self.decision_threshold = decision_threshold #used to calculate accuracy
        self.x_length = self.train_set.data.shape[1] #length of a single example
        self.y_length = self.train_set.labels.shape[1] #length of one example's label vector
        self.learningrate = learningrate
        self.mlp_layers = mlp_layers
        self.mlp_layers.append(self.y_length) #ensure predictions will have correct dimensions
        print('Mlp_layers is',str(self.mlp_layers))

        self.cv_fold = cv_fold
        self.dropout = dropout
        self.ensemble = ensemble
    
    
    def run_all(self):
        """Run all key methods"""
        self.set_up_graph_and_session()
        self.train_test_evaluate()
        self.view_trainable_variables(view_values=False)
        self.plot_layer_0_weights()
        self.report_large_and_small_inputs()
        self.close_session()
        self.clean_up()
        evaluate.print_final_summary(self.eval_results_valid, self.descriptor+'_MLP_Valid', self.best_valid_loss_epoch)
        evaluate.print_final_summary(self.eval_results_test, self.descriptor+'_MLP_Test', self.best_valid_loss_epoch)

    #~~~Key Methods~~~#
    def set_up_graph_and_session(self):
        #Build the graph
        tf.logging.set_verbosity(tf.logging.INFO) #Set output detail level (options: DEBUG, INFO, WARN, ERROR, or FATAL)
        self.graph = tf.Graph()
        self._build_graph()
        self.session = tf.Session(graph=self.graph)
        self.session.run(self.initialize)            
    
    def train_test_evaluate(self):
        self.train()
        if self.save_model: self.save()
        self.save_evals()
    
    def close_session(self):
        self.session.close()
    
    def clean_up(self):
        """Delete eval results for mysteryAAs for every epoch
        except the best validation epoch as determined by early stopping"""
        files = [f for f in os.listdir('.') if os.path.isfile(f)]
        for f in files:
            if 'mysteryAAs' in f:
                if str(self.best_valid_loss_epoch) not in f:
                    os.remove(f)
    
    #~~~Methods~~~#
    def train(self):
        #TODO redo all of this using queues so you don't need a feed dict at all
        """For cross validation, we are doing all 300 epochs so no early stopping"""
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
#            if self.num_epochs_done % (int(self.num_epochs/3)) == 0: print('Finished Epoch',str(self.num_epochs_done)+'.\n\tTraining loss=',str(epoch_loss))
            # only validating when not doing cross validation because this is used for early stopping
            if self.cv_fold < 2:
                self.test('Valid')
            self.test('Test')
            #self.test('mysteryAAs')
            
            # Early stopping is only for non cross validation
            #Early stopping. TODO: consider other early stopping methods
            #patience gets reset every time a new global minimum is reached
            #patience is calculated in the validation 
            if self.cv_fold < 2:
                if self.patience_remaining == 0:
                    print('No more patience left at epoch',self.num_epochs_done)
                    print('--> Implementing early stopping. Best epoch was:',self.best_valid_loss_epoch)
                    break

    def test(self, chosen_dataset):
        #TODO redo this to use tensorflow streaming auc and other built in evaluations (so you don't need scikit learn)
        """One-vs-all evaluation for all labels, using accuracy, AUC, pAUC,
        and average precision.
        Variables
        <chosen_dataset> is either 'Valid' or 'Test'
        <save_outputs> if True, save predictions."""
        if chosen_dataset == 'Valid':
            chosen_set = self.valid_set
            num_batches = self.num_valid_batches
            all_eval_results = self.eval_results_valid
        elif chosen_dataset == 'Test':
            chosen_set = self.test_set
            num_batches = self.num_test_batches
            all_eval_results = self.eval_results_test
        elif chosen_dataset == 'mysteryAAs':
            chosen_set = self.mysteryAAs
            num_batches = self.num_mysteryAAs_batches
        else:
            assert False, "chosen_dataset must be 'Test' or 'Valid' or 'mysteryAAs' but you passed "+str(chosen_dataset)
        
        epoch_loss = 0
        for i in range(num_batches):
            x_data_batch, y_labels_batch = chosen_set.next_batch()
            feed_dict = {self.x_input: x_data_batch,
                         self.y_labels: y_labels_batch,
                         self.keep_prob: 1.0}
            curr_loss, batch_pred_probs, batch_pred_labels = self.session.run([self.loss, self.pred_probs,self.pred_labels], feed_dict=feed_dict)
            epoch_loss+=curr_loss
            #Gather the outputs of subsequent batches together:
            #TODO: make this more efficient e.g. use streaming_auc
            if i == 0:
                entire_pred_probs = batch_pred_probs
                entire_pred_labels = batch_pred_labels
                labels_true = y_labels_batch
                if chosen_dataset == 'mysteryAAs':
                    entire_x = x_data_batch
            else:
                #concatenate results on assuming that the zeroth dimension is the training example dimension
                entire_pred_probs = np.concatenate((entire_pred_probs,batch_pred_probs),axis = 0)
                entire_pred_labels = np.concatenate((entire_pred_labels,batch_pred_labels),axis=0)
                labels_true = np.concatenate((labels_true, y_labels_batch),axis=0)
                self.entire_pred_probs = entire_pred_probs
                self.entire_pred_labels = entire_pred_labels
                self.labels_true = labels_true
                if chosen_dataset == 'mysteryAAs':
                    entire_x = np.concatenate((entire_x, x_data_batch),axis = 0)
        
        #~~~Track validation loss and control early stopping~~~#
        if chosen_dataset == 'Valid':
            self.valid_loss[self.num_epochs_done-1] = epoch_loss
            if self.num_epochs_done % (int(self.num_epochs/3)) == 0: print('\tValid loss=',str(epoch_loss))
            #Set early stopping signal
            if epoch_loss < self.best_valid_loss:
                self.best_valid_loss = epoch_loss
                self.best_valid_loss_epoch = self.num_epochs_done
                self.patience_remaining = self.initial_patience
            else: #it's worse
                #so, it has to get worse <self.initial_patience> number of times
                #without ever getting better in order for all the patience
                #to run out.
                self.patience_remaining -= 1
        
        #~~~Save outputs for mysteryAAs~~~#
        if chosen_dataset == 'mysteryAAs':
            out = pd.DataFrame(np.concatenate((entire_x, entire_pred_probs, entire_pred_labels),axis = 1),
                               columns=self.train_set.data_meanings+['Pred_Prob','Pred_Label'])
            self.mysteryAAs_filename = self.descriptor + '_mysteryAAs_results_epoch_' 
            out.to_csv(self.mysteryAAs_filename + str(self.num_epochs_done) + ".csv", header=True,index=False)
 
            return #Don't perform "evaluations" on mysteryAAs
    
        #~~~ Run Evaluations on Valid or Test Results ~~~#
        for label_number in range(self.y_length):
            current_label = self.train_set.label_meanings[label_number]
            self.selected_labels_true = labels_true[:,label_number]
            self.selected_pred_labels = entire_pred_labels[:,label_number]
            self.selected_pred_probs = entire_pred_probs[:,label_number]
            
            #Update results dictionary of dataframes
            all_eval_results = evaluate.evaluate_all(all_eval_results,
                                                     self.num_epochs_done,
                                                     current_label,
                                                     self.selected_labels_true,
                                                     self.selected_pred_labels,
                                                     self.selected_pred_probs,
                                                     self.descriptor+'MLP_'+chosen_dataset,
                                                     self.num_epochs)
        

    def update(self):
        """ Update train, test, and related variables after updating split"""
        self.train_set = split.train
        self.test_set = split.test
        self.preserved_data_meanings = split.train.data_meanings

        #Number of batches per epoch
        self.num_train_batches = math.ceil((self.train_set.num_examples)/self.train_set.batch_size)
        self.num_test_batches = math.ceil((self.test_set.num_examples)/self.test_set.batch_size)
            
    def save(self):
        self.saver.save(self.session, os.path.join(os.getcwd(),'model_MLP'))
    
    def save_evals(self):
        evaluate.save(self.eval_results_valid, self.descriptor+'_MLP_Valid')
        evaluate.plot_learning_curves(self.training_loss, self.valid_loss, self.descriptor+'_MLP')
     
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
                if self.y_length == 1:
                    print('Binary classifier: using sigmoid cross entropy')
                    self.loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=self.pred_raw, labels=self.y_labels) )
                else: #Multilabel
                    if self.exclusive_classes:
                        print('Multilabel classifier (exclusive classes): using softmax cross entropy')
                        self.loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits = self.pred_raw, labels = self.y_labels) )
                    else:
                        print('Multilabel classifier (non-exclusive classes): using sigmoid cross entropy')
                        self.loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=self.pred_raw, labels=self.y_labels) )
    
    def _define_optimizer_and_performance(self):
        with self.graph.as_default():
            with tf.variable_scope('optimizer'):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss)
            
            with tf.variable_scope('performance_measures'):
                if self.y_length == 1:
                    self.pred_probs = tf.nn.sigmoid(self.pred_raw)
                    self.pred_labels = tf.cast((self.pred_probs >= self.decision_threshold), tf.int32)
                else:
                    if self.exclusive_classes:
                        self.pred_probs = tf.nn.softmax(self.pred_raw)
                        self.pred_labels = tf.one_hot ( indices = tf.argmax(self.pred_probs, axis=1),
                                                       depth = self.y_length,
                                                       on_value = 1,
                                                       off_value = 0,
                                                       axis = -1)
                    else:
                        self.pred_probs = tf.nn.sigmoid(self.pred_raw)
                        self.pred_labels = tf.cast((self.pred_probs >= self.decision_threshold), tf.int32)
                
                print('Shape of self.pred_probs',str(self.pred_probs.get_shape().as_list()))
                print('Shape of self.pred_labels:',str(self.pred_labels.get_shape().as_list()))
                #accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred_labels, tf.cast(self.y_labels, tf.int32)), tf.float32))
                #~~~Done with graph construction~~~
                #TODO: add in calculation of other metrics using tensorflow (e.g. AUC)
                
            self.initialize = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
            
    def _create_mlp(self, variable_scope_name, inputx, num_inputs_zero, hidden_layers):
        with tf.variable_scope(variable_scope_name):
            use_sigmoid_flag = True
            zero = self._new_fc_layer(inputx=inputx,
                                   num_inputs = num_inputs_zero,
                                   num_outputs = hidden_layers[0],
                                   name='layer_0',
                                   use_sigmoid = use_sigmoid_flag)
            print("Shape of",variable_scope_name,"layer_0:",str(zero.get_shape().as_list()),', sigmoid=',str(use_sigmoid_flag))
            
            for i in range(1, len(hidden_layers)): #hidden layers
                if i==1:
                    tmp = zero
                if i == (len(hidden_layers)-1):
                    use_sigmoid_flag = False #no sigmoid on last layer
                hi = self._new_fc_layer(inputx=tmp,
                                 num_inputs=hidden_layers[i-1],
                                 num_outputs=hidden_layers[i],
                                 name='layer_'+str(i),
                                 use_sigmoid=use_sigmoid_flag)
                tmp = hi
                print("Shape of", variable_scope_name,
"layer_"+str(i),":",str(hi.get_shape().as_list()),', sigmoid=',str(use_sigmoid_flag))
        return hi
    
    def _new_fc_layer(self, inputx, num_inputs, num_outputs, name, use_sigmoid):
        """Create a new fully-connected layer."""
        weights = tf.Variable(tf.truncated_normal([num_inputs, num_outputs], stddev=0.05), name=(name+'_weights'))
        biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]), name=(name+'_biases'))
        layer = tf.matmul(inputx, weights) + biases
        if use_sigmoid:
            layer = tf.nn.sigmoid(layer)
        return tf.nn.dropout(layer, rate=self.dropout)

    
    #~~~Methods for Viewing Variables~~~#
    def view_trainable_variables(self, view_values):
        """Print names of all trainable variables
        if <view_values>==True then also display their final learned values"""
        with self.graph.as_default():
            #https://stackoverflow.com/questions/41951657/using-tf-trainable-variables-to-show-names-of-trainable-variables?noredirect=1&lq=1
            variables_names = [v.name for v in tf.trainable_variables()]
            values = self.session.run(variables_names)
            for k, v in zip(variables_names, values):
                print('Variable:', str(k))
                print('Shape:', str(v.shape))
                if view_values:
                    print(str(v))

    def plot_layer_0_weights(self):
        """Save a heat map based on layer_0 weights"""
        with self.graph.as_default():
            layer_0_weights = self.session.run('mlp/layer_0_weights:0')
        evaluate.plot_heatmap(layer_0_weights, center=0,
                              filename_prefix = self.descriptor+'_layer_0_weights',
                              yticklabels=self.preserved_data_meanings)
    
    def report_large_and_small_inputs(self):
        """Save a file reporting the input variables corresponding to rows
        in layer_0_weights with the largest and smallest average values."""
        with self.graph.as_default():
            layer_0_weights = self.session.run('mlp/layer_0_weights:0')
        avgs = np.average(layer_0_weights, axis = 1)
        report = pd.DataFrame(avgs, index = self.preserved_data_meanings)
        report = report.astype(dtype='float')
        report.sort_values(by=report.columns.values.tolist(),
                           ascending=False,
                           inplace=True,
                           na_position='first')
        report.to_csv(self.descriptor+'_layer_0_weight_avgs_sorted.csv')
    
