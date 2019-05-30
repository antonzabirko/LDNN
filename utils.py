import numpy as np


def relu(Z):
    A = np.maximum(0, Z)
    activation_cache = Z

    return A, activation_cache


def relu_backwards(dA, activation_cache):
    Z = activation_cache
    r = relu(Z)
    dZ = np.array(dA, copy=True)

    dZ[Z <= 0] = 0
    return dZ


def sigmoid(Z):
    """
    Sigmoid function

    :param Z: Numpy array of any shape
    :return A: the activity
    :return cache: saved Z for backpropagation
    """

    A = np.divide(1, (1 + np.power(np.e, -Z)))
    activation_cache = Z

    return A, activation_cache


def sigmoid_backwards(dA, activation_cache):
    Z = activation_cache
    s = sigmoid(Z)
    dZ = dA * s * (1 - s)

    return dZ