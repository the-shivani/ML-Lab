# Experiment: Support Vector Machine (SVM)

## 🎯 Aim

To implement the Support Vector Machine (SVM) algorithm for classification and understand how it separates data into different classes.

## 📖 Theory

**Support Vector Machine (SVM)** is a supervised Machine Learning algorithm used mainly for classification. It can also be used for regression.

SVM finds the best decision boundary, called a **hyperplane**, to separate data points belonging to different classes.

The data points closest to the decision boundary are called **support vectors**. These points help determine the position of the hyperplane.

### Important Concepts

1. **Hyperplane:** A decision boundary that separates different classes.
2. **Support Vectors:** Data points closest to the decision boundary.
3. **Margin:** The distance between the decision boundary and the nearest data points from each class.
4. **Kernel:** A function that helps SVM classify data that cannot be separated using a straight line.

### Types of SVM

* **Linear SVM:** Used when classes can be separated using a straight line or linear boundary.
* **Non-linear SVM:** Uses kernels to separate classes with more complex boundaries.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Jupyter Notebook

## 📋 Steps Performed

1. Import the required Python libraries.
2. Load the dataset.
3. Separate input features and the target variable.
4. Split the dataset into training and testing sets.
5. Standardize the features if required.
6. Create an SVM classifier.
7. Train the model using the training dataset.
8. Predict the classes of test data.
9. Evaluate the model using suitable performance metrics.
10. Display the results.

## 💻 Important Python Functions

* `train_test_split()` – Splits the dataset into training and testing sets.
* `StandardScaler()` – Standardizes numerical features.
* `SVC()` – Creates an SVM classifier.
* `fit()` – Trains the model.
* `predict()` – Predicts the output classes.
* `accuracy_score()` – Calculates prediction accuracy.
* `confusion_matrix()` – Shows correct and incorrect classifications.
* `classification_report()` – Displays precision, recall, and F1-score.

## 📊 Expected Output

* The SVM model is trained on the dataset.
* Predictions are generated for test data.
* Model performance is evaluated using suitable metrics.
* Classification results are displayed.

## ✅ Result

The Support Vector Machine algorithm is implemented to classify data into different classes and evaluate the model's performance.
