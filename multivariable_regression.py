"""
Demo of multi variable linear regression and gradient descent.
Author: Mya Schroder
Date: 1/25/2024
"""

import matplotlib.pyplot as plt
import numpy as np

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

    return mse, gradients



def regression(data: [(float, ...)]) -> [float, ... ]:
    """
    @brief      Finds best fit line for data 
    @param data      Array of tuples of any length [(x,y, ...), (x,y, ...), ...]
    """
    len_weights = len(data[0])
    weights = np.zeros(len_weights)

    LEARNING_RATE = 0.01
    ITERATIONS = 500
    
    for _ in range(ITERATIONS):
        
        mse, gradients = MSE_gradients(data, weights)
        print("MSE:", mse)

        weights = weights - LEARNING_RATE * gradients

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
        [1, 40],
        [2,80],
        [3,130],
        [4,140],
        [4, 150],
        [5, 170],
        [6,173]
    ])

    weights = regression(np_data)
    print("WEIGHTS:", weights)

    line_x = np.linspace(0, 10, 100)
    line_y = weights[1] * line_x + weights[0]
    graph_2D_data(np_data, line_x, line_y)
    

if __name__ == "__main__":
    main()