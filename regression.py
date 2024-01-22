"""
Demo of linear regression and gradient descent.
Author: Mya Schroder
Date: 1/22/2024
"""

import matplotlib.pyplot as plt

def graph_data(data: [(float, float)]):
    """
    @brief      Graphs data in scatter plot
    @param data      Array of tuples [(x,y), (x,y), ...]
    """

    x, y = zip(*data)  # Breaks into two arrays
    plt.scatter(x, y)
    plt.show()


def regression(data: [(float, float)]):
    """
    @brief      Finds best fit line for data 
    @param data      Array of tuples [(x,y), (x,y), ...]
    """
    m, b, total_error = 0

    for data_point in data:
        x = data_point[0]
        y = data_point[1]

        y_hat = m * x + b 

        total_error += (y_hat - y) ** 2

    n = len(data)    
    mse = total_error / n  # Mean Squared Error


    # TODO Add gradient descent




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

    graph_data(data)
    regression(data)

if __name__ == "__main__":
    main()