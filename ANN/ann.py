import os, sys
import numpy as np
import math
import matplotlib.pyplot as plt 

from data import readDataLabels, normalize_data, train_test_split, to_categorical
from utils import accuracy_score, SigmoidActivation, SoftmaxActivation, CrossEntropyLoss

# Create an MLP with 8 neurons
# Input -> Hidden Layer -> Output Layer -> Output
# Neuron = f(w.x + b)
# Do forward and backward propagation

mode = 'train'      # train/test... Optional mode to avoid training incase you want to load saved model and test only.

class ANN:
    def __init__(self, num_input_features, num_hidden_units, num_outputs, hidden_unit_activation, output_activation, loss_function):
        self.num_input_features = num_input_features
        self.num_hidden_units = num_hidden_units
        self.num_outputs = num_outputs

        self.hidden_unit_activation = hidden_unit_activation()
        self.output_activation = output_activation()
        self.loss_func = loss_function()
        self.learning_rate = 0.001
        self.loss = []
        self.accuracy = []

    def initialize_weights(self):   # TODO
        # Create and Initialize the weight matrices
        # Never initialize to all zeros. Not Cool!!!
        # Try something like uniform distribution. Do minimal research and use a cool initialization scheme.

        self.w1 = np.random.uniform(0, 1, size=(self.num_input_features,self.num_hidden_units))
        self.b1 = np.zeros((1,self.num_hidden_units))
    
        self.w2 = np.random.uniform(0, 1, size=(self.num_hidden_units, self.num_outputs))
        self.b2 = np.zeros((1,self.num_outputs))
        return 

    def forward(self):      # TODO
        # x = input matrix
        # hidden activation y = f(z), where z = w.x + b
        # output = g(z'), where z' =  w'.y + b'
        # Trick here is not to think in terms of one neuron at a time
        # Rather think in terms of matrices where each 'element' represents a neuron
        # and a layer operation is carried out as a matrix operation corresponding to all neurons of the layer
        self.z1 = self.X_train.dot(self.w1) + self.b1
        self.a1 = self.hidden_unit_activation(self.z1)
        self.z2 = self.a1.dot(self.w2) + self.b2
        self.y_pred = self.output_activation(self.z2)
        
        accuracy = accuracy_score(self.y_pred, self.y_train)
        self.accuracy.append(accuracy)
        losses = self.loss_func(self.y_pred, self.y_train)
        self.loss.append(np.sum(losses))
        
        return

    def backward(self):     # TODO
        self.da2 = self.loss_func.grad(self.y_pred,self.y_train)
        self.dz2 = self.da2 * self.output_activation.grad()
        self.dw2 = self.a1.T.dot(self.dz2)
        self.db2 = np.sum(self.dz2, axis=0)

        self.da1 = self.dz2.dot(self.w2.T)
        self.dz1 = self.hidden_unit_activation.grad() * self.da1
        self.dw1 = self.X_train.T.dot(self.dz1)
        self.db1 = np.sum(self.dz1, axis=0)
        return

    def update_params(self):    # TODO
        self.w2 -= self.learning_rate * self.dw2
        self.b2 -= self.learning_rate * self.db2
        self.w1 -= self.learning_rate * self.dw1
        self.b1 -= self.learning_rate * self.db1
        return

    def train(self, dataset, learning_rate=0.001, num_epochs=1000):
        self.X_train = dataset[0]
        self.y_train = dataset[1]
        self.initialize_weights()
        self.learning_rate = learning_rate
        for epoch in range(num_epochs):
            self.forward()
            self.backward()
            self.update_params()
            print("==============")
            print(f"Epoch number: {epoch}\nAccuracy: {self.accuracy[-1]}\nLoss: {self.loss[-1]}")
        
        # plt.plot(self.accuracy)
        # print(self.accuracy)
        return self.y_pred

    def test(self, test_dataset):
        accuracy = 0    # Test accuracy
        # Get predictions from test dataset
        X_test, y_test = test_dataset
        z1 = X_test.dot(self.w1) + self.b1
        a1 = self.hidden_unit_activation(z1)
        z2 = a1.dot(self.w2) + self.b2
        y_pred = self.output_activation(z2)
        # Calculate the prediction accuracy, see utils.py
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy


def main(argv):
    ann = ANN(64, 16, 10, SigmoidActivation, SoftmaxActivation, CrossEntropyLoss)

    # Load dataset
    X, y = readDataLabels()      # dataset[0] = X, dataset[1] = y
    X_norm = normalize_data(X)
    y_gt = to_categorical(y)
    X_train, y_train, X_test, y_test = train_test_split(X_norm, y_gt)

    # Split data into train and test split. call function in data.py

    # call ann->train()... Once trained, try to store the model to avoid re-training everytime
    if mode == 'train':
        y_pred = ann.train(dataset=[X_train,y_train])
        pass        # Call ann training code here
    else:
        # Call loading of trained model here, if using this mode (Not required, provided for convenience)
        raise NotImplementedError
    print(y_pred[0], y_train[0])
    # Call ann->test().. to get accuracy in test set and print it.
    accuracy = ann.test([X_test, y_test])
    print(accuracy)

if __name__ == "__main__":
    main(sys.argv)
