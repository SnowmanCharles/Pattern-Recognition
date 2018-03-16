# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 20:14:06 2018

EE559 Homework Week 3, Prof. Jenkins, Spring 2018
@author: Hrishikesh Hippalgaonkar
Tested in Python 3.6.3 :: Anaconda custom (64-bit), Windows 10

Question 1. 
"""

import numpy as np

def trainPerceptron(input_data, label):
    
    w_initial = np.array([0.1, 0.1, 0.1])
    numSamples = np.size(input_data, 0)
    t1 = np.ones((numSamples, 1))
    training_data = np.column_stack((t1, input_data))
    
    z = np.empty([numSamples,1])
    for i in range(0, numSamples):
        if(label[i] == 1):
            z[i] = 1
        else:
            z[i] = -1
    
    epoch = 0
    cost = np.empty([numSamples + 1])
    flag = 1
    w = np.empty([numSamples + 1, 3])
    final_w = np.empty([3])
    
    while (epoch < 1000 and flag == 1):
        #print(epoch)
        w[0,:] = w_initial
        for i in range(0, numSamples):
            k = np.multiply(w[i,:], training_data[i,:]) 
            cost[i] = np.sum(k)*z[i]
            if(cost[i] < 0):
                w[i + 1, :] = w[i, :] + z[i] * training_data[i, :]
            else:
                w[i + 1, :] = w[i, :]
                
        if(np.array_equal(w[numSamples, :], w[0, :])):
            flag = 0
        epoch = epoch + 1
        
        if(epoch != 1000):
            final_w = w[numSamples, :]
        else:
            t = np.argmax(cost)
            final_w = w[t,:]
            #print(final_w)
        w_initial = w[numSamples, :]
        

    return final_w 

def testingPerceptron(input_data, w):
    
    numSamples = np.size(input_data, 0)
    t1 = np.ones((numSamples, 1))
    training_data = np.column_stack((t1, input_data))
    
    cost = np.empty([numSamples])

    calculated_label = np.empty([numSamples])
 #   error = 0
            
    for i in range(0, numSamples):
        k = np.multiply(w, training_data[i,:]) 
        cost[i] = np.sum(k)
        if(cost[i] < 0):
            calculated_label[i] = 2
        else:
            calculated_label[i] = 1
    
#    for i in range(0, numSamples):
#        if(calculated_label[i] != input_label[i]):
#            error = error + 1
    
    return calculated_label


def calculateError(input_label, calculated_label):
    
    numSamples = np.size(input_label, 0)
    error = 0
    for i in range(0, numSamples):
        if(calculated_label[i] != input_label[i]):
            error = error + 1
    
    return (error/numSamples)*100
    
        
    
    