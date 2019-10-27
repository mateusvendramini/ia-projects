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
    
    for fold in folds:
        print("lenFold %d" %len(fold))

    train_inputs = []
    train_targets = []
    # Separate input features from target values
    for i in range(len(folds)):
        train_inputs.append((folds[i][folds[i].columns[:100]]))
        train_targets.append((folds[i][folds[i].columns[-1]]))
    # train is all folds except k
    
    for k in range(len(folds)):
        train_input = []
        train_target = []
        for j in range(len(folds)):
            if (k != j):
                train_input.append(train_inputs[j])
                train_target.append(train_targets[j])
        
        # Train classifier
        print(train_target)
        print(asarray(train_input))
        print(asarray(train_target))
        knn = KNeighborsClassifier(n_neighbors=10, metric='euclidean')
        knn.fit(X=asarray(train_input), y=asarray(train_target))



        # Infer and print outputs
        def calculate_error_rate(dataset_inputs, dataset_targets, model):
            predictions = model.predict(dataset_inputs)
            errors = (predictions != dataset_targets)
            return errors.sum()/len(errors)


        train_accuracy = 1 - calculate_error_rate(train_input, train_target, knn)
        print("Train accuracy: {:5.3f}".format(train_accuracy*100))
        validation_accuracy = 1 - calculate_error_rate(train_inputs[i], train_targets[i], knn)
        print("Validation accuracy: {:5.3f}".format(validation_accuracy * 100))