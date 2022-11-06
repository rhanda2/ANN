import math
import os, sys
import numpy as np
from sklearn import datasets

#import libraries as needed

def readDataLabels(): 
	#read in the data and the labels to feed into the ANN
	data = datasets.load_digits()
	X = data.data
	y = data.target

	return X,y

def to_categorical(y):
	
	#Convert the nominal y values tocategorical
	y_gt = np.zeros((y.shape[0], np.amax(y)+1))
	for i in range(y.shape[0]):
		y_gt[i, y[i]] = 1
	return y_gt
	
def train_test_split(data,labels,n=0.8): #TODO

	#split data in training and testing sets
	train = math.floor(n * data.shape[0])
	data_train = data[:train]
	labels_train = labels[:train]
	data_test = data[train:]
	labels_test = labels[train:]
	split = [data_train,labels_train,data_test,labels_test]
	return split

def normalize_data(data): #TODO

	# normalize/standardize the data
	l2 = np.atleast_1d(np.linalg.norm(data, ord=2, axis=1))
	l2[l2 == 0] = 1
	return data / np.expand_dims(l2, axis=1)
