"""
Demo of multi variable linear regression and gradient descent.
Author: Mya Schroder
Date: 1/25/2024
"""

import matplotlib.pyplot as plt
import numpy as np
import math

def MSE_gradients(data:np.array, weights) -> (float, np.array):
    """
    @brief      Finds MSE and gradients of the data given the weights
    @param data      Array of tuples of any length Array of tuples [(x1, x2, ...,  y), (x1, x2, ..., y), ...]
    """

    gradients = np.zeros(len(weights))
    mse = 0
    len_data = len(data)

    for i in range(len_data):
        x = data[i][:-1]
        y = data[i][-1]

        error = weights[0] + np.dot(weights[1:], x) - y

        gradients[0] += 1/len_data * error 
        gradients[1:] += 1/len_data * error * x

        mse += error ** 2 / (2 * len_data)
        
        if (math.isinf(mse)):
            print("\n!!WARNING: MSE value was too large and overflowed!!")
            exit()

    return mse, gradients



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
        
        mse, gradients = MSE_gradients(data, weights)

        weights = weights - learning_rate * gradients

        print("   MSE:", mse)
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

    np_data = np.array([
        [1, 3, 40],
        [2, 7, 80],
        [3, 10, 130],
        [4, 52, 140],
        [4, 75, 150],
        [5, 3, 170],
        [6, 2, 173]
    ])


    np_data = np.array([[8.6,9.6,3.6,9.6,7.6,2.6,4.6,6.6,9.6,9.6],[20,58,29,42,20,47,26,50,52,55], 
                     [33.694,40.314,22.484,38.554,31.254,22.024,24.594,32.114,39.654,39.984]]).T
    print("Data:", np_data)

    weights = regression(np_data, 0.001, 50000)
    print("WEIGHTS:", weights)

    # line_x = np.linspace(0, 10, 100)
    # line_y = weights[1] * line_x + weights[0]
    # graph_2D_data(np_data, line_x, line_y)
    

if __name__ == "__main__":
    main()