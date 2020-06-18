# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:18:07 2020

@author: Corbi
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import neural_network as nn
from random import shuffle

p_cnt = 50
noise_size = 0.1

xs = np.linspace(-10, 10, p_cnt)
yHats = np.sin(xs * .4 * math.pi) #xs * 3 + 1
yHats = np.add(yHats, np.sin(xs * .2 * math.pi + 1.)) / 2.
yHats = yHats + (np.random.rand(p_cnt)*noise_size) - 0.5

net = nn.network(np.array([1, 20, 1]), np.array([0, 1, 0]))
net.set_learning_rate(0.1)
net.set_batch_size(10)
net.set_dropout_rate(0.0)

MSE = 1
error_threshold = 0.01
i = 0
stop = 10000
print_mod = 100
converged = False
while MSE > error_threshold and i < stop and not converged:
    MSE = 0
    converged = True
    i_shuffled = list(range(len(xs)))
    shuffle(i_shuffled)
    for x in i_shuffled:
        y = net.feed_forward(np.array(xs[x]), True)
        converged = net.backprop(y, yHats[x]) and converged
        MSE += net.MSE(y, yHats[x])
    #ys = net.feed_forward(np.array(xs))
    #converged = converged and net.backprop(ys, yHats)
    #MSE = net.MSE(ys, yHats)
    MSE /= p_cnt
    if (i % print_mod == 0):
        ys = []
        for x in range(p_cnt):
            y = net.feed_forward(np.array(xs[x]))
            ys.append(y)
        #ys = net.feed_forward(np.array(xs))
        t_primes = []
        ts = np.linspace(0, 1, 11)
        for t in ts:
            t_primes.append(net.feed_forward(np.array(t)))
        #print("Iteration: {:d} MSE: {:02f}".format(i, MSE))
        fig = plt.figure()
        plt.scatter(xs, yHats, c='blue', figure=fig, label="y^")
        plt.scatter(xs, ys, c='red', figure=fig, label="y")
        plt.legend(loc='upper right')
        plt.title("Iteration #{:d}".format(i))
        #line = lines.Line2D([0, 0], net.layers[0][0].weights, figure=fig)
        #m = net.layers[0][0].weights[0]
        #b = net.layers[0][0].weights[1]
        #plt.plot([0, 1], [b, m+b])
        #plt.plot(ts, t_primes)
        plt.legend()
        plt.show()
    i += 1
    if (converged):
        print("\nConverged")
    if (MSE <= error_threshold):
        print("\nError minimized")
    if (i >= stop):
        print("\nMaximum iterations reached")
    
ys = []
for x in range(p_cnt):
    y = net.feed_forward(np.array(xs[x]))
    ys.append(y)
#ys = net.feed_forward(np.array(xs))
t_primes = []
ts = np.linspace(0, 1, 11)
for t in ts:
    t_primes.append(net.feed_forward(np.array(t)))
#print("Iteration: {:d} MSE: {:02f}".format(i, MSE))
fig = plt.figure()
plt.scatter(xs, yHats, c='blue', figure=fig, label="y^")
plt.scatter(xs, ys, c='red', figure=fig, label="y")
plt.legend(loc='upper right')
plt.title("Iteration #{:d}".format(i))
#line = lines.Line2D([0, 0], net.layers[0][0].weights, figure=fig)
#m = net.layers[0][0].weights[0]
#b = net.layers[0][0].weights[1]
#plt.plot([0, 1], [b, m+b])
#plt.plot(ts, t_primes)
plt.legend()
plt.show()

print()
print("Iterations: {:d}".format(i))
print("Error: {:f}".format(MSE))
print()