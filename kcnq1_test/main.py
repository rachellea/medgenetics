import pandas as pd
import numpy as np
from sklearn import metrics, model_selection
import copy
import mlp_model
import utils

# path to the training dataset
data_path = 'kcnq1_training_dataset.csv'
#data_path = "small_training_dataset.csv"

# prepare the data
df = pd.read_csv(data_path)
# the only two features are rate of evolution and PSSM
data = df.loc[:,['Rate of Evolution', 'PSSM']]
# get labels
labels = df.loc[:,['Label']]

# set hyperparameters from the paper
learningrate = 0.05
dropout = 0.5
momentum = 0.8 #TODO
weight_decay = 0.02 #TODO

# set our own hyperparameters
cv_fold_mlp = 3
repeat_num = 200
ensemble = False
num_epochs = 15

# create a split object
shared_args = {'impute':False,
                'impute_these_categorical':[],
                'impute_these_continuous':[],
                'one_hotify':False,
                'one_hotify_these_categorical':[],
#cat_vars
                'normalize_data':True,
                'normalize_these_continuous':['Rate of Evolution', 'PSSM'],
                'seed':10393, #make it 12345 for original split
                'batch_size':300}
split_args = {'train_percent':0.7,
                'valid_percent':0.15,
                'test_percent':0.15,
                'max_position':0, # this value will not be used
                'columns_to_ensure':['Rate of Evolution', 'PSSM']} # list of col names you must have

all_args = {**shared_args, **split_args }
split = utils.Splits(data = data,
                      labels = labels,
                       **all_args)

# initialize variables to store each average cross validation evaluation metrics
auc_tot = 0
mcc_tot = 0
ppv_tot = 0
npv_tot = 0
acc_tot = 0
tpr_tnr_tot = 0
tpr_tot = 0
tnr_tot = 0

# repeat cross validation repeat_num number of times
for rep_num in range(repeat_num):
    print(str(rep_num/2) + "% done")
    # initialize variables to store metrics for each fold
    fold_auc = 0
    fold_mcc = 0
    fold_ppv = 0
    fold_npv = 0
    fold_acc = 0
    fold_tpr_tnr = 0
    fold_tpr = 0
    fold_tnr = 0
    # shuffle the folds so we get a different split for every cross validation
    cv = model_selection.StratifiedKFold(n_splits=cv_fold_mlp, shuffle=True)
    fold_num = 1
    for train, test in cv.split(data, labels):
        print("\n\nDoing fold number", fold_num, " out of", cv_fold_mlp)
        print("Repeat num number", rep_num + 1, " out of", repeat_num)
        # create a copy of the real_data_split
        split = copy.deepcopy(utils.Splits(data = data, 
                                           labels = labels, 
                                           **all_args))
                
        # update the splits with the train and test indices for this cv loop
        split._make_splits_cv(train, test)

         #redefine mlp object with the new split
        m = mlp_model.MLP(descriptor='kcnq1_test',
                          split=copy.deepcopy(split),
                          decision_threshold = 0.5,
                          num_epochs = num_epochs, 
                          learningrate = learningrate,
                          mlp_layers = copy.deepcopy([2,3]),
                          dropout=dropout,
                          exclusive_classes = True,
                          save_model = False,
                          mysteryAAs = "", # this value will not be used
                          cv_fold = cv_fold_mlp,
                          ensemble=ensemble)
        # set up graph and session for the model
        m.set_up_graph_and_session()
        # train as per normal if we are not doing ensembling
        m.train()
         
        # evaluate
        # get the threshold that maximized MCC
        thresholds = np.linspace(0.01, 0.99, 100000)
        true_labels = m.selected_labels_true
        pred_probs = m.selected_pred_probs
        print(pred_probs)
        mcc = []
        for thr in thresholds:
            if thr == 0.5:
                print("We're halfway through the thresholds") 
            curr_pred_labels = [1 if prob >= thr else 0 for prob in pred_probs]
            curr_mcc = metrics.matthews_corrcoef(true_labels, curr_pred_labels)
            mcc.append(curr_mcc)
        bestscore = max(mcc)
        best_thr = thresholds[mcc.index(bestscore)]
        print("The best threshold is", best_thr)
        pred_labels = [ 1 if prob >= best_thr else 0 for prob in pred_probs]  
        print(true_labels)
        print(pred_labels)
        # calculate tp, tn, fp, fn
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for i in range(len(true_labels)): 
            if true_labels[i]==pred_labels[i]==1:
                TP += 1
            if pred_labels[i]==1 and true_labels[i]!=pred_labels[i]:
                FP += 1
            if true_labels[i]==pred_labels[i]==0:
                TN += 1
            if pred_labels[i]==0 and true_labels[i]!=pred_labels[i]:
                FN += 1
        # auroc - area under roc curve
        auroc = metrics.roc_auc_score(true_labels, pred_probs)
        # mcc - matthews correlation coefficient
        mcc = metrics.matthews_corrcoef(true_labels, pred_labels)
        # ppv - positive predictive value (tp/(tp+fp))
        if TP == 0:
            ppv = 0
        else:
            ppv = (TP / (TP + FP))
        # npv - negative predictive value (tn/(tn+fn))
        if TN == 0:
            npv = 0
        else:
            npv = (TN / (TN + FN))
        # accuracy 
        accuracy = metrics.accuracy_score(true_labels, pred_labels)
        # tpr
        if TP == 0:
            tpr = 0
        else:
            tpr = (TP / (TP + FN))
        # tnr
        if TN == 0:
            tnr = 0
        else:
            tnr = (TN / (TN + FP))
        # tpr_tnr
        tpr_tnr = tpr + tnr
        
        # store the values in the list
        fold_auc += auroc
        fold_mcc += mcc
        fold_ppv += ppv
        fold_npv += npv
        fold_acc += accuracy
        fold_tpr_tnr += tpr_tnr
        fold_tpr += tpr
        fold_tnr += tnr
          
        # append values to list 
        fold_num += 1
    
    # append the averages of the k fold cross validation into the lists
    auc_tot += fold_auc/cv_fold_mlp
    mcc_tot += fold_mcc/cv_fold_mlp
    ppv_tot += fold_ppv/cv_fold_mlp
    npv_tot += fold_npv/cv_fold_mlp
    acc_tot += fold_acc/cv_fold_mlp
    tpr_tnr_tot += fold_tpr_tnr/cv_fold_mlp
    tpr_tot += fold_tpr/cv_fold_mlp
    tnr_tot += fold_tnr/cv_fold_mlp    

print("\n\n-------------------Results---------------------")
print("AUC: ", auc_tot/repeat_num)
print('MCC: ', mcc_tot/repeat_num)
print('PPV:', ppv_tot/repeat_num)
print('NPV: ', npv_tot/repeat_num)
print("Accuracy: ", acc_tot/repeat_num)
print('TPR+TNR: ', tpr_tnr_tot/repeat_num)
print('TPR: ', tpr_tot/repeat_num)
print('TNR: ', tnr_tot/repeat_num)

print('\n\nDone\n\n')
