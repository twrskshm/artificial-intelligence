from machinelearning.utilities.activation_functions import sigmoid_activation

from numpy import ndarray, zeros, dot, where


class LogisticRegression:
    def __init__(self,  num_iterations: int = 1000, learning_rate: float = 0.01):
        """
        Initializes a new instance of the LogisticRegression class. This constructor sets up the model with the
        necessary parameters that control the learning process. The model's weights and bias are initialized to None and
        0, respectively, and will be set during the training phase.

        Parameters:
        num_iterations (int): The number of iterations to run the gradient descent algorithm during training. More
                              iterations can lead to more accurate models but may increase computation time.
        learning_rate (float): The step size at each iteration of gradient descent. A higher learning rate can cause the
                               training to converge faster, but too high a value might lead to divergence.

        Returns:
        None: This method does not return any value and is only used to initialize the model parameters.
        """
        self.weights: ndarray = None
        self.num_iterations = num_iterations
        self.bias: float = 0
        self.learning_rate = learning_rate

    def train(self, feature_list: ndarray, target_list: ndarray):
        """
        Train the LogisticRegression model by adjusting the weights and bias based on the provided training data. This
        method iteratively updates the model using gradient descent to minimize the error between predicted and actual
        class labels. The model uses sigmoid activation to produce probabilities, which are then used to compute the
        gradient of the loss function.

        Parameters:
        feature_list (ndarray): A 2D numpy array of input features where each row represents a sample and each column
                                represents a feature.
        target_list (ndarray): A 1D numpy array of binary target class labels corresponding to each row in feature_list.

        Returns:
        None: The function modifies the model's weights and bias in-place and does not return any value.
        """
        num_samples, num_features = feature_list.shape
        self.weights = zeros(num_features)

        for _ in range(self.num_iterations):
            predictions = sigmoid_activation(dot(feature_list, self.weights) + self.bias)
            errors = target_list - predictions
            self.weights += self.learning_rate * dot(feature_list.T, errors)
            self.bias += self.learning_rate * errors.sum()

    def predict(self, feature_list: ndarray) -> ndarray:
        """
        Predict class labels for the given features using the logistic regression model. This method calculates the
        probability that each input sample belongs to the positive class (label 1) based on the learned weights and
        bias. A threshold of 0.5 is applied to these probabilities to determine the class labels.

        Parameters:
        feature_list (ndarray): A 2D numpy array of input features where each row represents a sample and each column
                                represents a feature.

        Returns:
        ndarray: A 1D numpy array containing the predicted binary class labels (0 or 1) for each input sample.
        """
        probabilities = sigmoid_activation(dot(feature_list, self.weights) + self.bias)

        return where(probabilities >= 0.5, 1, 0)
