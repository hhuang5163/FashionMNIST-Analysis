from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from sklearn.metrics import log_loss
import pandas as pd
import numpy as np

# Preprocess data
mnist_train = pd.read_csv("MNIST/mnist_train.csv")
mnist_train = mnist_train[(mnist_train['label'] == 3) | (mnist_train['label'] == 8)]
scaler_train = preprocessing.StandardScaler().fit(mnist_train.iloc[:, 1:])
mnist_train.iloc[:, 1:] = scaler_train.transform(mnist_train.iloc[:, 1:])

DIVIDER = np.array(255)

Xtrain = mnist_train.iloc[:, 1:] /  DIVIDER
ytrain = mnist_train.iloc[:, 0]

# Determines best lambda for sparse logistic regression via scikit-learn's built-in cross validation techniques
lambdas = np.linspace(1e0, 1e2, num=100)
logclf = LogisticRegressionCV(Cs=lambdas, cv = 5, random_state=0, solver='sag', tol=1e-2, max_iter=1000, n_jobs=-1).fit(Xtrain, ytrain)
print("LogisticRegressionCV Parameters: ", logclf.get_params())
print("LogisticRegressionCV Tuned Hyperparameters: ", logclf.C_)

# Determines best lambda for linear SVM using built-in methods
lambdas = np.linspace(50, 100, num=50)
# Stores mean accuracy of a given lambda
cv_score = np.zeros(len(lambdas))
for i in range(len(lambdas)):
    svmclf = LinearSVC(dual=False, random_state=0, C=lambdas[i], tol=1e-2, max_iter=1000)
    scores = cross_val_score(svmclf, Xtrain, ytrain, cv=5)
    cv_score[i] = np.mean(scores)

best_lambda = lambdas[np.argmax(cv_score)]
print("Best lambda chosen for SVC: ", best_lambda)