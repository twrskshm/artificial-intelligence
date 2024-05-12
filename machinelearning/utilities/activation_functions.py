from typing import Union

from numpy import ndarray, where, exp


def step_activation(weighted_sum: Union[float, ndarray]) -> Union[float, ndarray]:
    """
    Computes the binary class output using a step activation function, commonly referred to as a Heaviside step
    function. This function is often used in binary classifiers such as the Perceptron. It returns a binary output where
    all positive input values yield 1, and zero or negative input values yield 0. This sharp threshold is what defines
    the decision boundary in a Perceptron model.

    Parameters:
    weighted_sum (float or ndarray): The weighted sum of inputs plus the bias. This can be either a single
                                     floating-point value or a numpy array of floats, depending on whether the input is
                                     for a single data point or multiple data points in a batch.

    Returns:
    float or ndarray: The binary output (1 or 0) for each input value, either as a single value or a numpy array of
                      binary values corresponding to the input array.
    """
    return where(weighted_sum > 0, 1, 0)


def sigmoid_activation(weighted_sum: Union[float, ndarray]) -> Union[float, ndarray]:
    """
    Computes the output of the sigmoid activation function, which is crucial in logistic regression for mapping
    predictions to probabilities. The sigmoid function formula is 1 / (1 + exp(-x)), where x represents the weighted sum
    of inputs plus the bias. This function smoothly varies from 0 to 1, and it is especially useful when a probability
    is required to describe the outcome, such as in logistic regression classifiers.

    Parameters:
    weighted_sum (float or ndarray): The weighted sum of inputs plus the bias, applicable to either a single data point
                                    or an entire batch of data points represented as a numpy array.

    Returns:
    float or ndarray: The continuous output values of the sigmoid function for each input weighted sum, where each value
                      lies between 0 and 1, representing the probability of the data point belonging to the positive
                      class.
    """
    return 1 / (1 + exp(-weighted_sum))
