# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 19:48:18 2018

EE559 Homework Week 3, Prof. Jenkins, Spring 2018
@author: Hrishikesh Hippalgaonkar
Tested in Python 3.6.3 :: Anaconda custom (64-bit), Windows 10

Question 2(c) - (f).
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import perceptron
from sklearn.linear_model import LinearRegression
from sklearn.multiclass import OneVsRestClassifier

class MSE_binary(LinearRegression):
    def __init__(self):
        #print("Calling newly created MSE_binary function...")
        super(MSE_binary, self).__init__()
    def predict(self, X):
        thr = 0.5
        y = self._decision_function(X)
        y_new = np.zeros(y.shape)
        y_new[y>thr] = 1
        return y_new

def OneVsAll(train_data, test_data, train_label, test_label, numFeatures):
    binary_model = MSE_binary()
    model = OneVsRestClassifier(binary_model)
    model.fit(train_data[:, :numFeatures], train_label)
    print('Using', numFeatures, 'features:')
    print("Training Accuracy:  " + str(model.score(train_data[:, :numFeatures], train_label)*100) + " %")
    print("Testing Accuracy:   " + str(model.score(test_data[:, :numFeatures], test_label)*100) + " %")

def TrainPerceptron(train_data, test_data, train_label, test_label, numFeatures, itr):

    net = perceptron.Perceptron(max_iter = itr, shuffle = True)
    net.fit(train_data[:, :numFeatures], train_label)
    
    # Print the results
    print('Using', numFeatures, 'features:')
    print("Training Accuracy:  " + str(net.score(train_data[:, :numFeatures], train_label)*100) + " %")
    print("Testing Accuracy:   " + str(net.score(test_data[:, :numFeatures], test_label)*100) + " %")
    print('\nWeight vectors: ')
    print(net.coef_)
    print('\nIntercept vector: ')
    print(net.intercept_,'\n')

    
def TrainPerceptronRandomWeight(train_data, test_data, train_label, test_label, numFeatures, itr):
    
    weight = []
    intercept = []
    train_accuracy = np.empty([100])
    
    for i in range(0,100):
        net = perceptron.Perceptron(max_iter = itr, shuffle = True)
        net.fit(train_data[:, :numFeatures], train_label, coef_init = np.random.rand(3, numFeatures), intercept_init = np.random.rand(3))
        train_accuracy[i] = net.score(train_data[:, :numFeatures], train_label)
        weight.append(net.coef_)
        intercept.append(net.intercept_)
        train_accuracy[i] = net.score(train_data[:, :numFeatures], train_label)
    
    # Print the results
    max_accuracy = np.argmax(train_accuracy)
    net1 = perceptron.Perceptron(max_iter = itr, shuffle = True)
    net1.fit(train_data[:, :numFeatures], train_label, coef_init = weight[max_accuracy], intercept_init = intercept[max_accuracy])
    print('Using', numFeatures, 'features:')
    print("Maximum Training Accuracy:  " + str(train_accuracy[max_accuracy]*100) + " %")
    print("Corresponding Testing Accuracy:   " + str(net1.score(test_data[:, :numFeatures], test_label)*100) + " %")
    print('\nWeight vectors: ')
    print(net.coef_)
    print('\nIntercept vector: ')
    print(net.intercept_,'\n')

def main():
    
    # Load the training and testing data
    train_data = np.genfromtxt("wine_train.csv", delimiter = ",")
    test_data  = np.genfromtxt("wine_test.csv",  delimiter = ",")
    
    X_train = train_data[:, :13]
    X_test = test_data[:, :13]

    X_trainLabel = train_data[:, 13]
    X_testLabel = test_data[:, 13]
    
    print('\n********************** Part (a) **************************************')
    print('Before standardization:')
    print('Mean: ', np.mean(X_train, axis = 0))
    print('Std: ', np.std(X_train, axis = 0), '\n')

    # Standardize data
    train_13d = StandardScaler()  # initializes a StandardScalar object
    train_13d.fit(X_train)
    X_train_norm = train_13d.fit_transform(X_train)
    X_test_norm = train_13d.fit_transform(X_test)

    print('After standardization:')
    print('Mean: ', np.mean(X_train_norm, axis = 0))
    print('Std: ', np.std(X_train_norm, axis = 0), '\n')
    

    print('\n********************** Part (d) **************************************')
    print('\nUsing Perceptron')
    TrainPerceptron(X_train_norm, X_test_norm, X_trainLabel, X_testLabel, 2, 1000000)
    TrainPerceptron(X_train_norm, X_test_norm, X_trainLabel, X_testLabel, 13, 100)
    
    print('\n********************** Part (e) **************************************')
    print('\nUsing Perceptron with random initial weights')
    TrainPerceptronRandomWeight(X_train_norm, X_test_norm, X_trainLabel, X_testLabel, 2, 100)
    TrainPerceptronRandomWeight(X_train_norm, X_test_norm, X_trainLabel, X_testLabel, 13, 1000)

    print('\n********************** Part (g) **************************************')
    print('\nUsing MSE and One Vs Rest Model: (Un-normalized Data)')
    OneVsAll(X_train, X_test, X_trainLabel, X_testLabel, 2)
    OneVsAll(X_train, X_test, X_trainLabel, X_testLabel, 13)
    
    print('\n********************** Part (h) **************************************')
    print('\nUsing MSE and One Vs Rest Model: (Normalized Data)')
    OneVsAll(X_train_norm, X_test_norm, X_trainLabel, X_testLabel, 2)
    OneVsAll(X_train_norm, X_test_norm, X_trainLabel, X_testLabel, 13)
 
if __name__ == "__main__":
    main()
