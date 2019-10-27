from numpy import asarray
from sklearn.neighbors  import KNeighborsClassifier
from pandas import read_csv
import numpy as np

# Infer and print outputs
def calculate_error_rate(dataset_inputs, dataset_targets, model):
    predictions = model.predict(dataset_inputs)
    errors = (predictions != dataset_targets)
    return errors.sum()/len(errors)
    
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
    validation_array = []
    train_array = []
    for i in range (5):
        print("K fold = %d" %i)
        train_fold = None
        train_target = None
        validation_fold = folds[i]
        # Assemble train array
        for j in range(5):
            if i != j:
                if train_fold is not None:
                    train_fold = np.concatenate((train_fold, folds[j][folds[j].columns[:100]]))
                    train_target = np.concatenate((train_target, asarray(folds[j][folds[j].columns[-1]])))
                else:
                    train_fold = folds[j][folds[j].columns[:100]]
                    train_target = asarray(folds[j][folds[j].columns[-1]])
        
        validation_inputs = asarray(validation_fold[validation_fold.columns[:100]])
        validation_targets = asarray(validation_fold[validation_fold.columns[-1]])
        
        # "Train" model
        knn = KNeighborsClassifier(n_neighbors=10, metric='euclidean')
        knn.fit(X=asarray(train_fold), y=asarray(train_target))
        
        train_accuracy = 1 - calculate_error_rate(asarray(train_fold), asarray(train_target), knn)
        print("Train accuracy: {:5.3f}".format(train_accuracy*100))
        train_array.append(train_accuracy)
        validation_accuracy = 1 - calculate_error_rate(validation_inputs, validation_targets, knn)
        validation_array.append(validation_accuracy)
        print("Validation accuracy: {:5.3f}".format(validation_accuracy * 100))
        
    print("Final result")
    print("Train accuracy: {:5.3f} & Validation accuracy {:5.3f}".format(100*min(train_array), 100*min(validation_array)))
    