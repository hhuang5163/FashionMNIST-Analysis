from random import random
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import learning_curve
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, log_loss
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def preprocess_data(train, test, divider):
    '''
    Preprocesses the MNIST dataset for binary classification

    Inputs:
        train: pandas dataframe representing the training dataset
        test: pandas dataframe representing the testing dataset
        divider: scaling factor to rescale the RGB colors

    Outputs:
        mnist_train: preprocessed training dataset
        xmnist_test: preprocessed testing data without the classifications
        ymnist_test: preprocessed testing data classifications
    '''
    # Filters out data labeled as 3 or 8 for binary classification
    mnist_train = train[(train['label'] == 3) | (train['label'] == 8)]
    scaler_train = preprocessing.StandardScaler().fit(mnist_train.iloc[:, 1:])
    mnist_train.iloc[:, 1:] = scaler_train.transform(mnist_train.iloc[:, 1:])
    mnist_test = test[(test['label'] == 3) | (test['label'] == 8)]
    scaler_test = preprocessing.StandardScaler().fit(mnist_test.iloc[:, 1:])
    mnist_test.iloc[:, 1:] = scaler_test.transform(mnist_test.iloc[:, 1:])

    xmnist_test = mnist_test.iloc[:, 1:] / divider
    ymnist_test = mnist_test.iloc[:, 0]

    return mnist_train, xmnist_test, ymnist_test

def shuffle_split_data(data, numfolds):
    '''
    Shuffles and splits data into K folds at random

    Inputs:
        data: the data to split
        numfolds: represents K, the number of folds to split into

    Outputs:
        folds: an array in which the object at each index represents 
                one fold
    '''
    i = 0
    numrows = data.shape[0]
    # Number of data points in each fold
    chunksize = numrows // numfolds
    folds = []
    # Randomly shuffles the data before splitting into folds
    mnist_train = shuffle(data)
    while i < numrows:
        # Last fold will contain the remaining data, may not be size chunksize
        if len(folds) == numfolds - 1:
            folds.append(mnist_train.tail(numrows - i))
            break
        folds.append(mnist_train[i: i + chunksize])
        i += chunksize
    return folds

def k_fold_cv(mlmethod, numfolds, lambdas, train_folds, xtest, ytest, divider):
    '''
    K-fold cross validation implemented from scratch

    Inputs:
        mlmethod: string representing the type of ML method to apply CV on. Must 
            be "logreg" or "linsvm"
        numfolds: represents K, the number of folds to split into
        lambdas: the lambda values used in tuning the regularization hyperparameter
        train_folds: the training data split into K folds
        xtest: the testing data without the classification
        ytest: classification of the testing data
        divider: scaling factor to rescale the RGB colors
    
    Outputs:
        
    '''
    cv_err = np.zeros((numfolds, len(lambdas)))
    train_err = np.zeros((numfolds, len(lambdas)))
    test_err = np.zeros((numfolds, len(lambdas)))
    for k in range(numfolds):
        # Creates the training and 
        xtrain = pd.DataFrame()
        for i in range(numfolds):
            if i != k:
                xtrain = pd.concat([xtrain, train_folds[i]])
        ytrain = xtrain.iloc[:, 0]
        xtrain = xtrain.iloc[:, 1:] / divider
        ycv = train_folds[k].iloc[:, 0]
        xcv = train_folds[k].iloc[:, 1:] / divider

        for j in range(len(lambdas)):
            if mlmethod == "logreg":
                # Logistic Regression
                clf = LogisticRegression(C=lambdas[j], random_state=0, solver='sag', tol=5e-3, max_iter=1000, n_jobs=-1).fit(xtrain, ytrain)
                clf = clf.sparsify()
                cv_err[k, j] = log_loss(ycv, clf.predict_proba(xcv))
                train_err[k, j] = log_loss(ytrain, clf.predict_proba(xtrain))
                test_err[k, j] = log_loss(ytest, clf.predict_proba(xtest))
            elif mlmethod == "linsvm":
                # Linear SVM
                clf = LinearSVC(dual=False, random_state=0, C=lambdas[i], tol=1e-2, max_iter=1000).fit(xtrain, ytrain)
                cv_err[k, j] = 1 - accuracy_score(ycv, clf.predict(xcv))
                train_err[k, j] = 1 - accuracy_score(ytrain, clf.predict(xtrain))
                test_err[k, j] = 1 - accuracy_score(ytest, clf.predict(xtest))

    best_lambda = lambdas[np.argmin(mean_cv_err)]
    mean_cv_err = np.mean(cv_err, axis=0)
    mean_train_err = np.mean(train_err, axis=0)
    mean_test_err = np.mean(test_err, axis=0)
    
    return best_lambda, mean_cv_err, mean_train_err, mean_test_err



K = 5
DIVIDER = np.array(255)

mnist_train = pd.read_csv("MNIST/mnist_train.csv")
mnist_test = pd.read_csv("MNIST/mnist_test.csv")
mnist_train, xmnist_test, ymnist_test = preprocess_data(mnist_train, mnist_test, DIVIDER)
mnist_train_folds = shuffle_split_data(mnist_train, K)

# Logistic Regression
lambdas = np.linspace(1e0, 1e2, num=100)
best_lambda, mean_cv_err, mean_train_err, mean_test_err = k_fold_cv("logreg", K, lambdas, mnist_train_folds, xmnist_test, ymnist_test, DIVIDER)

# Linear SVM
lambdas = np.array([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2])
best_lambda, mean_cv_err, mean_train_err, mean_test_err = k_fold_cv("linsvm", K, lambdas, mnist_train_folds, xmnist_test, ymnist_test, DIVIDER)


# Plots Error Curves
print("Lambdas: ", lambdas)
print("Mean CV Error: ", mean_cv_err)
print("Mean Test Error: ", mean_test_err)
print("Best Lambda: ", best_lambda)

plt.plot(lambdas, mean_cv_err, color='r', label='CV Error')
plt.plot(lambdas, mean_train_err, color='g', label='Training Error')
plt.plot(lambdas, mean_test_err, color='b', label='Test Error')
plt.xlabel("Lambda (Hyperparameter) Value")
plt.ylabel("Accuracy Error")
plt.title("Error Curves vs. Hyperparameter Choice for Linear SVM")
plt.legend()
logregfig = plt.gcf()
plt.show()
logregfig.savefig('SVMErrorCurves.png')

