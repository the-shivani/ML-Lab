# Experiment: Linear Regression

## 🎯 Aim

To implement Linear Regression to predict a continuous numerical value based on one or more input features.

## 📖 Theory

Linear Regression is a supervised Machine Learning algorithm used to predict continuous values such as house prices, temperature, and salary.

It establishes a linear relationship between input variables and the target variable.

### Formula

For simple linear regression:

**y = mx + c**

Where:

* `y` = Predicted output
* `x` = Input variable
* `m` = Slope of the line
* `c` = Intercept

### Example

Suppose we want to predict a student's marks based on study hours.

| Study Hours | Marks |
| ----------- | ----: |
| 2           |    40 |
| 4           |    60 |
| 6           |    80 |

The model learns the relationship between study hours and marks to predict marks for new study hours.

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
3. Select the input features and target variable.
4. Split the dataset into training and testing sets.
5. Create a Linear Regression model.
6. Train the model using the training data.
7. Predict values using the testing data.
8. Evaluate the model's performance.
9. Visualize actual and predicted values.

## 💻 Important Functions

| Function                | Purpose                                    |
| ----------------------- | ------------------------------------------ |
| `train_test_split()`    | Splits data into training and testing sets |
| `LinearRegression()`    | Creates a Linear Regression model          |
| `fit()`                 | Trains the model                           |
| `predict()`             | Predicts numerical values                  |
| `mean_absolute_error()` | Calculates mean absolute error             |
| `mean_squared_error()`  | Calculates mean squared error              |
| `r2_score()`            | Measures the model's goodness of fit       |

## 📊 Expected Output

* Trained Linear Regression model.
* Predicted numerical values.
* Comparison of actual and predicted values.
* Model evaluation metrics.
* Visualization of the regression line, where applicable.

## ✅ Result

Linear Regression is implemented to learn the relationship between input features and a continuous target variable and predict numerical values.
