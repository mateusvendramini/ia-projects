from numpy import asarray
from sklearn.tree import DecisionTreeRegressor  
from sklearn.model_selection import KFold
import numpy as np  
from pandas import read_csv


if __name__ == '__main__':
    # Load dataset
    input_file_path = "reg02.csv"
    dataset = read_csv(input_file_path, header=0)
    # Separate input features from target values
    train_inputs = asarray(dataset[dataset.columns[:20]])
    train_targets = asarray(dataset[dataset.columns[-1]])
    kf = KFold(n_splits=5, shuffle=False)
    kf.get_n_splits(train_inputs, train_targets)
    for train,test in kf.split(train_inputs, train_targets):
        #print(train)
        #print(test)
        train_batch = None
        target_batch = None
        for id in train:
            if train_batch is not None:
                train_batch.append(train_inputs[id])
                target_batch.append(train_targets[id])
            else:
                train_batch = [train_inputs[id]]
                target_batch = [train_targets[id]]
        print(train_batch)
        print(target_batch)

    