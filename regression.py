"""
Demo of linear regression and gradient descent.
Author: Mya Schroder
Date: 1/22/2024
"""

import matplotlib.pyplot as plt
import numpy as np

def graph_data(data: [(float, float)], line_x:np.array = None, line_y:np.array = None):
    """
    @brief      Graphs data in scatter plot
    @param data      Array of tuples [(x,y), (x,y), ...]
    """

    x, y = zip(*data)  # Breaks into two arrays
    plt.scatter(x, y)

    if line_x.any() and line_y.any():
        plt.plot(line_x, line_y, '-r', label='y=mx+b')

    plt.show()


def regression(data: [(float, float)]):
    """
    @brief      Finds best fit line for data 
    @param data      Array of tuples [(x,y), (x,y), ...]
    """
    m, b = 0, 0

    LEARNING_RATE = 0.01
    ITERATIONS = 500
    
    for _ in range(ITERATIONS):

        total_error = 0
        m_derivative, b_derivative = 0, 0
        for data_point in data:
            x = data_point[0]
            y = data_point[1]

            y_hat = m * x + b 

            m_derivative += (m * x + b - y) * x
            b_derivative += (m * x + b - y)
            
            total_error += (y_hat - y) ** 2

        
        n = len(data)    
        mse = total_error / (2*n)  # Mean Squared Error over 2 (to make derivative easier)
        print("MSE:", mse, ", m: ", m, ", b: ", b)
        m_derivative /= n
        b_derivative /= n

        m = m - LEARNING_RATE * m_derivative

        b = b - LEARNING_RATE * b_derivative

    return m, b






def main():
    data = [
        (1, 40),
        (2,80),
        (3,130),
        (4,140),
        (4, 150),
        (5, 170),
        (6,173)
    ]

    #graph_data(data)
    m,b =regression(data)
    
    line_x = np.linspace(0, 10, 100)
    line_y = m * line_x + b
    graph_data(data, line_x, line_y)

if __name__ == "__main__":
    main()