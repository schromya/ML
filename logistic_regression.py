"""
Demo of logistic regression.
Author: Mya Schroder
Date: 2/5/2024
"""

import matplotlib.pyplot as plt
import numpy as np
import math

def cross_entropy_gradients(data:np.array, weights) -> (float, np.array):
    """
    @brief      Finds Cross Entropy Error and gradients of the data given the weights
    @param data      Array of tuples of any length Array of tuples [(x1, x2, ...,  y), (x1, x2, ..., y), ...]
    """

    gradients = np.zeros(len(weights))
    total_error = 0
    len_data = len(data)

    for i in range(len_data):
        x = data[i][:-1]
        y = data[i][-1]

        y_hat = 1 / (1 + math.e ** -(weights[0] + np.dot(weights[1:], x)))

        # Cross Entropy Error
        error = -y * math.log(y_hat, 10) - (1 - y) * math.log(1-y_hat, 10)  

        # Derivatives of Cross Entropy Loss over len_data
        gradients[0] += 1/len_data * (y_hat - y) 
        gradients[1:] += 1/len_data * (y_hat - y)  * x

        total_error += error / len_data
        
        if (math.isinf(total_error)):
            print("\n!!WARNING: Total error value was too large and overflowed!!")
            exit()

    return total_error, gradients



def regression(data: [(float, ...)], learning_rate=0.001, iterations=1000) -> [float, ... ]:
    """
    @brief      Finds best fit line for data 
    @param data      Array of tuples of any length [(x,y, ...), (x,y, ...), ...]
    @param learning_rate    Learning rate (de)
    @param iterations       Number of iterations
    """
    len_weights = len(data[0])
    weights = np.zeros(len_weights)

    
    for i in range(iterations):
        print("----------- Iteration", i)
        
        total_error, gradients = cross_entropy_gradients(data, weights)

        weights = weights - learning_rate * gradients

        print("   Total Error:", total_error)
        print("   Weights:", weights)

    return weights


def graph_2D_data(data: [(float, float)], line_x:np.array = None, line_y:np.array = None):
    """
    @brief      Graphs data in scatter plot
    @param data      Array of tuples [(x,y), (x,y), ...]
    """

    x, y = zip(*data)  # Breaks into two arrays
    plt.scatter(x, y)

    if line_x.any() and line_y.any():
        plt.plot(line_x, line_y, '-r', label='y=mx+b')

    plt.show()


def main():
    np_data = np.array([[120,125, 161, 119, 118, 117, 159, 150],
                     [0, 0, 1, 0, 0, 0, 1, 1]]).T
    print("Data:", np_data)

    weights = regression(np_data, 0.001, 5000)
    print("WEIGHTS:", weights)

    line_x = np.linspace(-100, 200, 100)
    line_y = 1 / (1 + math.e ** -(weights[0] + weights[1:]*line_x))
    graph_2D_data(np_data, line_x, line_y)
    

if __name__ == "__main__":
    main()