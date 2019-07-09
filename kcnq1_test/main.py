import pandas as pd
from sklearn import metrics
import copy
import mlp_model
import utils

# path to the training dataset
data_path = 'kcnq1_training_dataset.csv'

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
cv_fold_mlp = 0
ensemble = False

# create a split object
shared_args = {'impute':False,
                'impute_these_categorical':[],
                'impute_these_continuous':[],
                'one_hotify':False,
                'one_hotify_these_categorical':[],
#cat_vars
                'normalize_data':False,
                'normalize_these_continuous':[],
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

# create mlp object
m = mlp_model.MLP(descriptor='kcnq1_test',
    split=copy.deepcopy(split),
    decision_threshold = 0.5,
    num_epochs = 1000, 
    learningrate = learningrate,
    mlp_layers = copy.deepcopy([2,3]),
    dropout=dropout,
    exclusive_classes = True,
    save_model = False,
    mysteryAAs = "", # this value will not be used
    cv_fold = cv_fold_mlp,
    ensemble=ensemble)
m.run_all()

if m.best_valid_loss_epoch == 0:
    print("best valid loss epoch was not initialized")
    m.best_valid_loss_epoch = self.num_epochs
    m.clean_up()

# evaluate
true_labels = m.selected_labels_true
pred_labels = m.selected_pred_labels
pred_probs = m.selected_pred_probs
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
# report the resutls
print("\n\n-------------------Results---------------------")
print("AUC: ", auroc)
print('MCC: ', mcc)
print('PPV:', ppv)
print('NPV: ', npv)
print("Accuracy: ", accuracy)
print('TPR+TNR: ', tpr_tnr)
print('TPR: ', tpr)
print('TNR: ', tnr)

print('\n\nDone\n\n')
