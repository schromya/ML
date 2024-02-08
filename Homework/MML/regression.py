
import numpy as np
import math
from enum import Enum


class Regression:
    
    def __init__(self):
        return

    @classmethod
    def train(cls, data: [(float, ...)], learning_rate=0.001, iterations=1000) -> ([float, ... ], [float, ...]):
        """
        @brief      Finds best fit line for data 
        @param data      Array of tuples of any length [(x,y, ...), (x,y, ...), ...]
        @param learning_rate    Learning rate (de)
        @param iterations       Number of iterations
        """
        len_weights = len(data[0])
        weights = np.zeros(len_weights)
        mse_list = np.array([])

        
        for _ in range(iterations):
            mse, gradients = cls.calculate_error_and_gradients(data, weights)
            mse_list = np.append(mse_list, mse)

            weights = weights - learning_rate * gradients

        return mse_list, weights
    
    @classmethod
    def test(cls, data:np.array, weights:np.array) -> (float, np.array):
        """
        @brief Wrapper for calculating error and gradients 
        @param data      Array of tuples of any length Array of tuples [(x1, x2, ...,  y), (x1, x2, ..., y), ...]
        @param weights   Array of weights (thetas)
        """
        return cls.calculate_error_and_gradients(data, weights)

    @classmethod
    def calculate_error_and_gradients(cls, data:np.array, weights:np.array) -> (float, np.array):
        """
        @brief This is specific to the child classes, so should be implemented there
        """
        return

class LinearMSE(Regression):
    def __init__(self):
        super().__init__()

    @classmethod
    def calculate_error_and_gradients(cls, data:np.array, weights:np.array) -> (float, np.array):
        """
        @brief      Finds MSE and gradients of the data given the weights
        @param data      Array of tuples of any length Array of tuples [(x1, x2, ...,  y), (x1, x2, ..., y), ...]
        @param weights   Array of weights (thetas)
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

class LogisticBinaryCrossEntropy(Regression):
    def __init__(self):
        super().__init__()

    @classmethod
    def calculate_error_and_gradients(cls, data:np.array, weights) -> (float, np.array):
        """
        @brief      Finds Cross Entropy Error and gradients of the data given the weights
        @param data      Array of tuples of any length Array of tuples [(x1, x2, ...,  y), (x1, x2, ..., y), ...]
        @param weights   Array of weights (thetas)
        """

        gradients = np.zeros(len(weights))
        cross_entropy = 0
        len_data = len(data)
        epsilon = 1e-10 # prevents taking log of 0 (Used ChatGPT to diagnose issue)

        for i in range(len_data):
            x = data[i][:-1]
            y = data[i][-1]

            y_hat = 1 / (1 + math.e ** -(weights[0] + np.dot(weights[1:], x)))

            # Cross Entropy Error
            error = -y * math.log(y_hat + epsilon, 10) - (1 - y) * math.log(1-y_hat + epsilon, 10)  

            print("y_hat:", y_hat)
            print("error:", error)
            # Derivatives of Cross Entropy Loss over len_data
            gradients[0] += 1/len_data * (y_hat - y) 
            gradients[1:] += 1/len_data * (y_hat - y)  * x

            cross_entropy += error / len_data
            
            if (math.isinf(cross_entropy)):
                print("\n!!WARNING: Total error value was too large and overflowed!!")
                exit()

        return cross_entropy, gradients


