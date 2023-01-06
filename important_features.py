from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import pandas as pd
import numpy as np
from mlxtend.plotting import heatmap
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

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

    xtrain = mnist_train.iloc[:, 1:] / divider
    ytrain = mnist_train.iloc[:, 0]
    xtest = mnist_test.iloc[:, 1:] / divider
    ytest = mnist_test.iloc[:, 0]

    return xtrain, ytrain, xtest, ytest

def sparse_logreg(xtrain, ytrain, xtest):
    '''
    Fits a sparse logistic regression model to determine important features 
    in classifying handwritten numbers

    Inputs:
        xtrain: the training data without the classifications
        ytrain: the classifications for the training data
        xtest: the test data without the classifications
    
    Outputs:
        Plots that indicate which pixels were the most important in 
        classifying handwriting samples of the numbers 3 and 8.
        Examples of handwriting samples that were easy and hard to classify.
    '''
    clf = LogisticRegression(C=BEST_LOG_LAMBDA, random_state=0, solver='sag', tol=5e-3, max_iter=1000, n_jobs=-1).fit(xtrain, ytrain)
    heatmap(clf.coef_.reshape((28, 28)))
    logregfig = plt.gcf()
    plt.show()
    logregfig.savefig('LogImportantFeatures.png')

    # Determines examples of numbers that were easy and hard to classify
    predict_prob = clf.predict_proba(xtest)
    predict_prob = np.absolute(predict_prob[:, 0] - predict_prob[:, 1])
    unsure_predict_idx = np.flatnonzero(predict_prob < 0.1)
    confident_predict_idx = np.flatnonzero(predict_prob > 0.95)

    for idx in unsure_predict_idx:
        heatmap(xtest.iloc[idx, :].to_numpy().reshape((28, 28)))
        fig2 = plt.gcf()
        fig2.savefig('Hard2ClassifyDataLogReg/Uncertain' + str(idx))

    for idx in confident_predict_idx:
        heatmap(xtest.iloc[idx, :].to_numpy().reshape((28, 28)))
        fig2 = plt.gcf()
        fig2.savefig('Easy2ClassifyDataLogReg/Certain' + str(idx))

def linsvm(xtrain, ytrain, xtest):
    '''
    Fits a linear SVM to determine important features in classifying handwritten numbers

    Inputs:
        xtrain: the training data without the classifications
        ytrain: the classifications for the training data
        xtest: the test data without the classifications
    
    Outputs:
        Plots that indicate which pixels were the most important in 
        classifying handwriting samples of the numbers 3 and 8.
        Examples of handwriting samples that were easy and hard to classify.
    '''
    clf = LinearSVC(dual=False, random_state=0, C=BEST_SVM_LAMBDA, tol=1e-3, max_iter=500).fit(xtrain, ytrain)
    heatmap(clf.coef_.reshape((28, 28)))
    logregfig = plt.gcf()
    plt.show()
    logregfig.savefig('SVMImportantFeatures.png')

    # Determines examples of numbers that were easy and hard to classify
    conf_scores = np.absolute(clf.decision_function(xtest))
    unsure_predict_idx = np.flatnonzero(conf_scores < 0.1)
    for idx in unsure_predict_idx:
        heatmap(xtest.iloc[idx, :].to_numpy().reshape((28, 28)))
        fig2 = plt.gcf()
        fig2.savefig('Hard2ClassifyDataSVM/Uncertain' + str(idx))

    sure_predict_idx = np.flatnonzero(conf_scores > 4)
    for idx in sure_predict_idx:
        heatmap(xtest.iloc[idx, :].to_numpy().reshape((28, 28)))
        fig2 = plt.gcf()
        fig2.savefig('Easy2ClassifyDataSVM/Certain' + str(idx))


# Values as determined by cross validation from scratch
BEST_LOG_LAMBDA = 100
BEST_SVM_LAMBDA = 50
DIVIDER = np.array(255)

mnist_train = pd.read_csv("MNIST/mnist_train.csv")
mnist_test = pd.read_csv("MNIST/mnist_test.csv")
xtrain, ytrain, xtest, ytest = preprocess_data(mnist_train, mnist_test, DIVIDER)
sparse_logreg(xtrain, ytrain, xtest)
linsvm(xtrain, ytrain, xtest)
