# -*- coding: utf-8 -*-
"""
Created on Sat May 30 11:03:28 2020

@author: Corbi
"""

import numpy as np
import math

class neuron:
    
    weights = []
    learning_rate = 0.1
    last_input = []
    last_output = []
    delta_weights = []
    convergence_value = 0.01
    
    function = 0 # 0 for linear, 1 for sigma
    
    def __init__(self, inputs, function):
        self.weights = np.random.rand(inputs+1)
        self.function = function
        
    def sigmoid(self, x):
        return 1. / (1. + math.exp(-x))
    
    def sigmoid_prime(self):
        return self.last_output * (1. - self.last_output)
        
    def feed_forward(self, data):
        #assert len(data) == (len(self.weights)-1)
        
        self.last_input = np.append(data, np.array([1.]))
        self.last_output = np.dot(self.weights, self.last_input)
        if (self.function == 1):    
            self.last_output = self.sigmoid(self.last_output)
        
        return self.last_output
    
    def back_prop(self, weighted_errors):
        summation = np.sum(weighted_errors)
        if (self.function == 1):
            error = self.sigmoid_prime() * summation
        else:
            error = summation
        self.delta_weights = error * self.last_input * self.learning_rate
        return error * self.weights[0:-1]
    
    def apply_back_prop(self):
        self.weights = self.weights - self.delta_weights
        return (math.fabs(np.sum(self.delta_weights)/len(self.delta_weights)) < self.convergence_value)
    
class network:
    
    layers = []
    
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
            
    def feed_forward(self, data):
        #assert len(data) == len(self.layers[0])
        data = np.array(data)
    
        for layer in self.layers:
            result = []
            for neuron in layer:
                result = result + [neuron.feed_forward(data)]
            data = result.copy()
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
        converged = True
        for i in reversed(range(len(self.layers))):
            for neuron in self.layers[i]:
                converged = neuron.apply_back_prop() and converged
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