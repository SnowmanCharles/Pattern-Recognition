# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 16:55:29 2018

EE559 Homework Week 3, Prof. Jenkins, Spring 2018
@author: Hrishikesh Hippalgaonkar
Tested in Python 3.6.3 :: Anaconda custom (64-bit), Windows 10

Question 1.

"""

import numpy as np
import trainPerceptron as tp
import plotDecBoundaries as pd

def shuffleData(dataArray):
    np.random.shuffle(dataArray)
    features = np.array(dataArray[:, [0,1]])
    labels = np.array(dataArray[:,2])
    return (features, labels)

def trainTestPlot(training_data, training_label, testing_data, testing_label, dataNum):
    print(" For Synthetic dataset ", dataNum)
    weight = tp.trainPerceptron(training_data, training_label)
    print("\nWeights: ",weight)
    weight = weight/np.max(np.abs(weight))
    print("\nNormalilzed Weight: ",weight)
    
    calculated_labels_train = tp.testingPerceptron(training_data, weight)
    training_error = tp.calculateError(training_label, calculated_labels_train)
    print("\nTraining Error: ",training_error)
    
    calculated_labels_test = tp.testingPerceptron(testing_data, weight)
    testing_error = tp.calculateError(testing_label, calculated_labels_test)
    print("\nTesting Error: ",testing_error)
    #pd.plotDecBoundaries(training_data, training_label, weight)
    #pd.plotDecBoundaries(testing_data, testing_label, weight)


def main():
    
    # Load the training and testing data
    train1 = np.genfromtxt("synthetic1_train.csv", delimiter = ",")
    test1  = np.genfromtxt("synthetic1_test.csv",  delimiter = ",")
    train2 = np.genfromtxt("synthetic2_train.csv", delimiter = ",")
    test2  = np.genfromtxt("synthetic2_test.csv",  delimiter = ",")
    
    temp1  = np.genfromtxt("feature_train.csv", delimiter = ",")
    temp2  = np.genfromtxt("feature_test.csv",  delimiter = ",")
    label1 = np.genfromtxt("label_train.csv", delimiter = ",")
    label2 = np.genfromtxt("label_test.csv",  delimiter = ",")
    
    train3 = np.column_stack((temp1, label1))
    test3 = np.column_stack((temp2, label2))
    
    synthetic1_train, synthetic1_train_label = shuffleData(train1)
    synthetic1_test,  synthetic1_test_label  = shuffleData(test1)
    synthetic2_train, synthetic2_train_label = shuffleData(train2)
    synthetic2_test,  synthetic2_test_label  = shuffleData(test2)
    synthetic3_train, synthetic3_train_label = shuffleData(train3)
    synthetic3_test,  synthetic3_test_label  = shuffleData(test3)
    
    trainTestPlot(synthetic1_train, synthetic1_train_label, synthetic1_test,  synthetic1_test_label, 1)
    trainTestPlot(synthetic2_train, synthetic2_train_label, synthetic2_test,  synthetic2_test_label, 2)
    trainTestPlot(synthetic3_train, synthetic3_train_label, synthetic3_test,  synthetic3_test_label, 3)
    
    
if __name__ == "__main__":
    main()



    