import pandas as pd

# read the csv file
df = pd.read_csv("kcnq1_training_dataset.csv")
df = df.head(3)

df.to_csv("small_training_dataset.csv")
