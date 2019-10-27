from numpy import asarray
from sklearn.tree import DecisionTreeRegressor  
from sklearn.model_selection import KFold
import numpy as np  
from pandas import read_csv
from sklearn.metrics import mean_absolute_error

if __name__ == '__main__':
    # Load dataset
    input_file_path = "reg02.csv"
    dataset = read_csv(input_file_path, header=0)
    # Separate input features from target values
    train_inputs = asarray(dataset[dataset.columns[:20]])
    train_targets = asarray(dataset[dataset.columns[-1]])
    kf = KFold(n_splits=5, shuffle=False)
    kf.get_n_splits(train_inputs, train_targets)
    for k, (train,test) in enumerate(kf.split(train_inputs, train_targets)):
        dtr = DecisionTreeRegressor(criterion ="mse")
        dtr.fit(train_inputs[train], train_targets[train])
        print("Fold %d" %k)
        print("MAE for Train %f" %mean_absolute_error(train_targets[train], dtr.predict(train_inputs[train])))
        print("MAE for Validation %f" %mean_absolute_error(train_targets[test], dtr.predict(train_inputs[test])))

    