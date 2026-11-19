from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset and select two classes for binary classification
iris = load_iris()
X = iris.data[:100]
y = iris.target[:100]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train Perceptron model
model = Perceptron(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Prediction and evaluation
y_pred = model.predict(X_test)
print("Predictions:", y_pred)
print("Actual values:", y_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

new_sample = [[5.1, 3.5, 1.4, 0.2]]
print("New sample prediction:", model.predict(new_sample)[0])
