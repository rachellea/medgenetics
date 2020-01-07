 
    def _predict_mysteryAAs_lg(self):
        '''This function predicts mysteryAAs using logistic regression'''
        
        # set hyperparameters
        c = 0.1
        pen = "l1"
    
        # train logistic regression
        lg = regression.LogisticRegression(descriptor=self.descriptor,
split=copy.deepcopy(self.real_data_split) ,logreg_penalty=pen, C=c, figure_num=1,
fold=0).get_lg()
        
        # use the trained model to get the predicted probabilities of mysteryAAs
        pred_prob = lg.predict_proba(self.mysteryAAs_split.data)[:,1]

        # put in a format for mysteryAAs_output_cleanup
        #print(self.mysteryAAs_split.data_meanings) 
        # create a datafame for the data
        df = pd.DataFrame.from_records(self.mysteryAAs_split.data,
columns=self.mysteryAAs_split.data_meanings)
        #print(df.head())

        # add predicted probability column
        df['Pred_Prob'] = pd.Series(pred_prob)

        # convert to a csv file
        filename = "kcnq1_withSN_mysteryAAs_results.csv"
        df.to_csv(filename, index=False)

        self._mysteryAAs_output_cleanup(filename)

        #print(df['Pred_Prob'])


  def _run_logreg_full(self):
        # Run Logistic Regression for all hyperparameters
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
        print("Running Log Reg")

        classifier_penalty= ['l1', 'l2']
        classifier_C = [0.0001 ,0.001, 0.01, 0.1, 1, 10, 100, 1000]

        # set the k value for k fold cross validation (# of folds for cross
        # validation. Set to 0 if 
        # we don't want to do cross validation)
        kfold = self.cv_fold_lg
        k = 0
        for pen in classifier_penalty:
          for C in classifier_C:
            lg = regression.LogisticRegression(descriptor=descriptor,
split=copy.deepcopy(self.real_data_split),logreg_penalty=pen, C=C, figure_num=k,
fold=kfold)
            k += 2

    def _run_logreg(self):
        # Run Logistic Regression for a specified C and penalty

        # set hyperparameters
        c = 0.1
        pen = 'l1'
        # run logistic regression
        lg = regression.LogisticRegression(descriptor=self.descriptor,
split=copy.deepcopy(self.real_data_split) ,logreg_penalty=pen, C=c, figure_num=1,
fold=self.cv_fold_lg)
        return lg