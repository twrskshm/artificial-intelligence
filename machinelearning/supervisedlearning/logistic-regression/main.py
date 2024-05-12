from logistic_regression import LogisticRegression

from sklearn.utils import Bunch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from numpy import ndarray, float64, mean

iris_data: Bunch = load_iris()
# Select features and targets for binary classification ('setosa' vs 'versicolor'). Exclude 'virginica' class which
# corresponds to target value 2.
binary_feature_list: ndarray = iris_data.data[iris_data.target != 2]
binary_target_list: ndarray = iris_data.target[iris_data.target != 2]
logistic_regression = LogisticRegression()
# Split the data into training and testing sets.
feature_list_train, feature_list_test, target_list_train, target_list_test = train_test_split(
    binary_feature_list,
    binary_target_list,
    # 20% of the data is used for testing.
    test_size=0.2,
    # Sets a seed for the random number generator used in splitting the dataset. This ensures that the split is the same
    # every time the code is run, providing reproducible results.
    random_state=42
)

logistic_regression.train(feature_list_train, target_list_train)

predictions: ndarray = logistic_regression.predict(feature_list_test)
accuracy: float64 = mean(predictions == target_list_test) * 100
print(f'Logistic Regression Accuracy: {accuracy:.2f}%')
