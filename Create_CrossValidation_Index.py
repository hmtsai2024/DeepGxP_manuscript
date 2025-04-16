import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import random

dir_output='/home/CBBI/tsaih/data/'

# Create a length matches the number of samples in training/validation/testing sets
l=[x for x in range(0, 6813)]

num_folds = 10

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1

index_train=[]
index_val=[]
index_test=[]
for train, test in kfold.split(l):
    random.shuffle(train)
    index_train.append(train[0:5054])
    index_val.append(train[5054:len(train)])
    index_test.append(test)
    fold_no = fold_no + 1

index_train_df=pd.DataFrame(index_train).T
index_val_df=pd.DataFrame(index_val).T
index_test_df=pd.DataFrame(index_test).T

index_val_df.iloc[1077, 0]=6803
index_val_df.iloc[1077, 1]=6777
index_val_df.iloc[1077, 2]=6812

index_test_df=index_test_df[:-1]

index_train_df.to_csv(dir_output + 'RandomIndex_train_10xCV_example.txt', sep='\t', index=False)
index_val_df.to_csv(dir_output + 'RandomIndex_val_10xCV_example.txt', sep='\t', index=False)
index_test_df.to_csv(dir_output + 'RandomIndex_test_10xCV_example.txt', sep='\t', index=False)