# Import necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Sample data (you should replace this with real data)
# Consider features such as boat size, weather conditions, load capacity, etc.
data = np.array([
    [10, 1, 0],
    [5, 0, 1],
    [8, 1, 1],
    [3, 0, 0],
    # Add more data samples here
])

# Labels indicating whether the boat is safe (1) or not (0)
labels = np.array([1, 0, 1, 0])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Create a decision tree classifier (you can choose a different model based on your data)
classifier = DecisionTreeClassifier()

# Train the classifier on the training data
classifier.fit(X_train, y_train)

# Make predictions on the test data
predictions = classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

# Print the accuracy
print(f"Accuracy: {accuracy * 100:.2f}%")

# Predict whether a new scenario is safe or not
new_scenario = np.array([[7, 1, 0]])  # Replace with your own input
prediction_new = classifier.predict(new_scenario)

if prediction_new[0] == 1:
    print("The person is safe from sinking.")
else:
    print("The person is not safe from sinking.")
