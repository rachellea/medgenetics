import pandas as pd
import mlp_model

# path to the training dataset
data_path = 'data/kcnq1_test/kcnq1_training_dataset.csv'

# prepare the data
df = pd.read_csv(data_path)
data = df[['Rate of Evolution', 'PSSM']].values
label = df['Label'].values

print(data)
print(label)

# evaluate
# auroc
# mcc
# ppv
# npv
# accuracy
# tpr_tnr
# tpr
# tnr
