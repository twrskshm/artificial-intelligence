from machinelearning.utilities.activation_functions import step_activation

from numpy import ndarray, zeros, dot


class Perceptron:
    def __init__(self,  num_iterations: int = 1000, learning_rate: float = 0.01):
        """
        Initializes a new instance of the Perceptron class. This constructor sets up the model with key parameters that
        influence the behavior during the training process. The model initializes weights as None and will be set during
        the training phase, starting with zero bias.

        Parameters:
        num_iterations (int): Specifies how many times the training algorithm will loop through the entire dataset. A
                              higher number of iterations allows more opportunities for the model to learn the data, but
                              increases computational complexity.
        learning_rate (float): Determines the magnitude of the model's adjustments to the weights with each update.
                               Higher learning rates may lead to faster convergence but can overshoot the minimum loss.

        Returns:
        None: This constructor method is used for initialization purposes only and does not return a value.
        """
        self.weights: ndarray = None
        self.num_iterations = num_iterations
        self.bias: float = 0
        self.learning_rate = learning_rate

    def train(self, feature_list: ndarray, target_list: ndarray):
        """
        Trains the Perceptron model on a set of features and associated target class labels using a simple learning
        algorithm that iteratively updates weights based on classification errors. The training process employs the step
        activation function to make predictions and updates the model weights and bias if the prediction is incorrect.

        Parameters:
        feature_list (ndarray): A 2D numpy array where each row represents a sample and each column represents a
                                feature.
        target_list (ndarray): A 1D numpy array of binary class labels (0 or 1) corresponding to each sample in the
                               feature list.

        Returns:
        None: This method updates the model's weights and bias in-place and does not return any value.
        """
        num_samples, num_features = feature_list.shape
        self.weights = zeros(num_features)

        for _ in range(self.num_iterations):
            for features_index, features in enumerate(feature_list):
                weighted_sum = dot(features, self.weights) + self.bias
                predicted_class = step_activation(weighted_sum)
                error = self.learning_rate * (target_list[features_index] - predicted_class)

                self.weights += error * features
                self.bias += error

    def predict(self, feature_list: ndarray) -> ndarray:
        """
        Predicts class labels for a given set of features using the learned weights and bias from the trained Perceptron
        model. The model uses the step activation function to convert the weighted sum of inputs and bias into a binary
        class label.

        Parameters:
        feature_list (ndarray): A 2D numpy array where each row represents a sample and each column a feature, similar
                                to the training data structure.

        Returns:
        ndarray: A 1D numpy array containing the predicted class labels (0 or 1) for each sample based on the learned decision boundary.
        """
        weighted_sum = dot(feature_list, self.weights) + self.bias

        return step_activation(weighted_sum)
