# -*- coding: utf-8 -*-
"""
Created on Sat May 30 11:03:28 2020

@author: Corbi
"""

import numpy as np
import math
import random

class neuron:
    
    weights = []
    learning_rate = 0.1
    last_input = []
    last_weighted_input = []
    last_output = []
    delta_weights = []
    convergence_value = 0.001
    dropout_rate = 0
    
    function = 0 # 0 for linear, 1 for relu, 2 for sigma, 3 for tanh
    
    def __init__(self, inputs, function):
        self.weights = np.random.rand(inputs+1)-0.5
        self.delta_weights = np.zeros(np.shape(self.weights))
        self.function = function
        
    def relu(self, x):
        return (x if x > 0 else 0)
    
    def relu_prime(self, x):
        return (1 if x > 0 else 0)
        
    def sigmoid(self, x):
        return 1. / (1. + math.exp(-x))
    
    def sigmoid_prime(self, x):
        return self.last_output * (1. - self.last_output)
    
    def tanh(self, x):
        return math.tanh(x)
    
    def tanh_prime(self, x):
        return 1 - math.pow(self.last_output, 2)
        
    def feed_forward(self, data):
        #assert len(data) == (len(self.weights)-1)
        if (random.random() < self.dropout_rate):
            data = np.zeros(np.shape(data))
            self.last_input = np.append(data, np.array([0.]))
            self.last_weighted_input = np.dot(self.weights, self.last_input) # Ensure the right size
            self.last_output = self.last_weighted_input
        else:
            self.last_input = np.append(data, np.array([1.]))
            self.last_weighted_input = np.dot(self.weights, self.last_input)
            if (self.function == 1):
                self.last_output = self.relu(self.last_weighted_input)
            elif (self.function == 2):    
                self.last_output = self.sigmoid(self.last_weighted_input)
            elif (self.function == 3):
                self.last_output = self.tanh(self.last_weighted_input)
            else:
                self.last_output = self.last_weighted_input
        
        return self.last_output
    
    def back_prop(self, weighted_errors):
        summation = np.sum(weighted_errors)
        if (self.function == 1):
            summed_input = np.sum(self.last_input)
            error = self.relu_prime(self.last_weighted_input)
        elif (self.function == 2):
            error = self.sigmoid_prime(None)
        elif (self.function == 3):
            error = self.tanh_prime(None)
        else:
            error = 1
        error *= summation
        self.delta_weights = np.add(self.delta_weights, error * self.last_input * self.learning_rate)
        return error * self.weights[0:-1]
    
    def apply_back_prop(self, batch_size):
        self.delta_weights /= batch_size
        self.weights = self.weights - self.delta_weights
        diff = math.fabs(np.sum(self.delta_weights) / len(self.delta_weights))
        self.delta_weights = np.zeros(np.shape(self.weights))
        return (diff < self.convergence_value)
    
class network:
    
    layers = []
    batch_size = 1
    batch_count = 0
    dropout_rate = 0
    
    def __init__(self, layer_shapes, neuron_types):
        assert len(layer_shapes) > 0
        layer_count = len(layer_shapes)
        input_len = layer_shapes[0]
        layer_num = 0
        for i in layer_shapes:
            layer = []
            for j in range(i):
                layer = layer + [neuron(input_len, neuron_types[layer_num])]
            self.layers = self.layers + [layer]
            layer_num += 1
            input_len = i
            
    def feed_forward(self, data, dropout = False):
        #assert len(data) == len(self.layers[0])
        data = np.array(data)
        
        dropout_rate = self.dropout_rate
        if (not dropout and self.dropout_rate != 0):
            self.set_dropout_rate(0.0)
    
        for layer in self.layers:
            result = []
            for neuron in layer:
                result = result + [neuron.feed_forward(data)]
            data = result.copy()
            
        if (not dropout and dropout_rate != 0):
            self.set_dropout_rate(dropout_rate)
            
        return np.array(data)
    
    def backprop(self, y, yHat):
        #assert type(y) is np.ndarray 
        #assert type(yHat) is np.ndarray
        errors = np.repeat(y - yHat, len(self.layers[-1]))
        for i in reversed(range(len(self.layers))):
            j = 0
            if i > 0:
                new_errors = np.empty([len(self.layers[i]), len(self.layers[i-1])])
                for neuron in self.layers[i]:
                    new_errors[j] = np.array(neuron.back_prop(errors[j]))
                    j += 1
                errors = np.sum(new_errors, 0)
            else:
                for neuron in self.layers[i]:
                    np.array(neuron.back_prop(errors[j]))
                    j += 1
        self.batch_count += 1
        converged = False
        if (self.batch_count == self.batch_size):
            converged = True
            for i in reversed(range(len(self.layers))):
                for neuron in self.layers[i]:
                    converged = neuron.apply_back_prop(self.batch_count) and converged
            self.batch_count = 0
        return converged
    
    def huber(self, yHat, y, delta=1.):
        return np.where(np.abs(y - yHat) < delta,
                        .5*(y-yHat)**2,
                        delta*(np.abs(y-yHat)-0.5*delta))
    
    def cross_entropy(self, y, yHat):
        yHat = np.array(yHat)
        y = np.array(y)
        np.seterr(divide = 'ignore') 
        logY = np.log(y)
        log1mY = np.log(np.array([1.]) - y)
        np.seterr(divide = 'warn') 
        return -1./len(y)*np.sum(yHat*logY + (np.array([1.]) - yHat)*log1mY)
    
    def MSE(self, y, yHat):
        return 0.5 * np.sum(np.power(np.subtract(y, yHat), 2))
    
    def set_learning_rate(self, learning_rate):
        for layer in self.layers:
            for neuron in layer:
                neuron.learning_rate = learning_rate
                
    def set_convergence_value(self, convergence_value):
        for layer in self.layers:
            for neuron in layer:
                neuron.convergence_value = convergence_value
                
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        
    def set_dropout_rate(self, dropout_rate):
        self.dropout_rate = dropout_rate
        for layer in self.layers:
            for neuron in layer:
                neuron.dropout_rate = dropout_rate
        
    
def main():
    #x = np.array([1, 2, 3])
    xs = np.array([np.array([0]), np.array([1]), np.array([2]), np.array([3])])
    yHats = np.array([np.array([5.]), np.array([6.]), np.array([7.]), np.array([8.])])
    
    #net = network(np.array([3, 3, 1]), np.array([1, 1, 0]))
    net = network(np.array([1]), np.array([0]))
    
    i = 1
    while True:
        print("Step: {:d}".format(i))
        MSE = 0
        for j in range(len(xs)):
            y = net.feed_forward(xs[j])
            print("x = {:02f} YHat = {:02f} Y = {:02f}".format(xs[j][0], yHats[j][0], y[0]))
            net.backprop(y, yHats[j])
            MSE += net.MSE(y, yHats[j])
        MSE /= len(xs)
        print("MSE: {:02f}".format(MSE))
        if (MSE < 0.001):
            break
        print()
        i += 1
        if (i > 1000):
            break

if __name__ == "__main__":
    main()