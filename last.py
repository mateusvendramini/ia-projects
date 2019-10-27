from numpy import asarray
from sklearn.neighbors  import KNeighborsClassifier
from pandas import read_csv
import numpy as np

if __name__ == '__main__':
    # Load dataset and split it in train and test
    input_file_path = "class02.csv"
    dataset = read_csv(input_file_path, header=0)
    folds = []
    folds.append(dataset[:200])
    folds.append(dataset[200:400])
    folds.append(dataset[400:600])
    folds.append(dataset[600:800])
    folds.append(dataset[800:1000])