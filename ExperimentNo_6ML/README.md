# Experiment: Decision Tree

## 🎯 Aim

To implement the Decision Tree algorithm for classification and understand how it makes predictions using decision rules.

## 📖 Theory

A **Decision Tree** is a supervised Machine Learning algorithm used for classification and regression.

It represents decisions in a tree-like structure. It splits data into smaller groups based on different features until it reaches a final prediction.

### Important Concepts

1. **Root Node:** The starting point of the tree.
2. **Decision Node:** A node where data is divided based on a condition.
3. **Leaf Node:** The final node that provides the prediction.
4. **Splitting:** Dividing data into smaller groups based on feature values.
5. **Entropy:** Measures the impurity or uncertainty in a dataset.
6. **Gini Impurity:** Measures how mixed the classes are in a group.

### Types of Decision Trees

* **Classification Tree:** Predicts categorical values, such as Yes/No.
* **Regression Tree:** Predicts numerical values, such as house prices.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Jupyter Notebook

## 📋 Steps Performed

1. Import the required libraries.
2. Load the dataset.
3. Separate input features and the target variable.
4. Split the dataset into training and testing sets.
5. Create a Decision Tree model.
6. Train the model using training data.
7. Predict outputs for the test data.
8. Evaluate the model using suitable performance metrics.
9. Visualize the decision tree if required.

## 💻 Important Python Functions

* `train_test_split()` – Splits the dataset into training and testing sets.
* `DecisionTreeClassifier()` – Creates a classification model.
* `DecisionTreeRegressor()` – Creates a regression model.
* `fit()` – Trains the model.
* `predict()` – Predicts the output.
* `accuracy_score()` – Calculates classification accuracy.
* `plot_tree()` – Visualizes the decision tree.

## 📊 Expected Output

* The Decision Tree model is trained.
* Predictions are generated for test data.
* Model performance is evaluated.
* The tree structure can be visualized.

## ✅ Result

The Decision Tree algorithm is implemented to make predictions based on decision rules and evaluate its performance on the dataset.
