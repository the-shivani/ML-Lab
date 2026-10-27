# Experiment: Single Layer Neural Network

## 🎯 Aim

To implement a Single Layer Neural Network and understand how it learns patterns from data to make predictions.

## 📖 Theory

A **Single Layer Neural Network** is a simple neural network consisting of an input layer and an output layer, without any hidden layer.

It processes input data using weights and biases to produce an output. During training, the weights and biases are adjusted to reduce prediction errors.

### Main Components

1. **Input Layer:** Receives input features from the dataset.
2. **Weights:** Determine the importance of each input feature.
3. **Bias:** Helps adjust the output of the neuron.
4. **Activation Function:** Determines the neuron's output.
5. **Output Layer:** Produces the final prediction.

### Working of a Single Layer Neural Network

1. Input data is provided to the network.
2. Each input is multiplied by its corresponding weight.
3. The weighted inputs are added together with the bias.
4. An activation function processes the result.
5. The network produces an output.
6. During training, the weights and biases are updated to improve predictions.

### Mathematical Representation

The weighted sum is:

`z = w1*x1 + w2*x2 + ... + wn*xn + b`

The output is:

`y = f(z)`

Where:

* `x` = Input features
* `w` = Weights
* `b` = Bias
* `f(z)` = Activation function
* `y` = Predicted output

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Jupyter Notebook

## 📋 Steps Performed

1. Import the required libraries.
2. Load or create the dataset.
3. Separate input features and target values.
4. Preprocess the data if required.
5. Create a single-layer neural network model.
6. Train the model using the training data.
7. Generate predictions.
8. Evaluate the model's performance.
9. Display the results.

## 💻 Important Python Functions

* `train_test_split()` – Splits data into training and testing sets.
* `Perceptron()` – Creates a simple linear neural classifier.
* `fit()` – Trains the model.
* `predict()` – Generates predictions.
* `accuracy_score()` – Calculates classification accuracy.
* `confusion_matrix()` – Evaluates classification results.

## 📊 Expected Output

* The neural network is trained on the dataset.
* Predictions are generated for the input data.
* Model performance is evaluated using suitable metrics.
* Results are displayed.

## ✅ Result

A Single Layer Neural Network is implemented to understand its basic structure, learning process, and prediction capabilities.

