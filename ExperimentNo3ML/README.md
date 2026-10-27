# Experiment: Principal Component Analysis (PCA)

## 🎯 Aim

To implement Principal Component Analysis (PCA) on a dataset to reduce its dimensionality while preserving as much information as possible.

## 📖 Theory

Principal Component Analysis (PCA) is an unsupervised dimensionality reduction technique used in Machine Learning. It transforms a dataset with many features into a smaller set of new features called principal components.

PCA helps reduce the number of features, simplify data analysis, remove redundancy, and visualize high-dimensional data.

The first principal component captures the maximum possible variance in the data. The second principal component captures the maximum remaining variance while being perpendicular to the first, and so on.

### How PCA Works

1. Load the dataset.
2. Handle missing values if required.
3. Standardize the numerical features.
4. Calculate the covariance matrix or use an equivalent decomposition method.
5. Find the principal components.
6. Select the required number of components.
7. Transform the original data into the reduced-dimensional representation.
8. Analyze the explained variance.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Jupyter Notebook

## 💻 Important Python Libraries

| Library / Function          | Purpose                                                      |
| --------------------------- | ------------------------------------------------------------ |
| `pandas`                    | Loads and processes datasets                                 |
| `StandardScaler`            | Standardizes numerical features                              |
| `PCA`                       | Performs Principal Component Analysis                        |
| `fit_transform()`           | Learns the components and transforms the data                |
| `explained_variance_ratio_` | Shows the proportion of variance explained by each component |
| `Matplotlib`                | Visualizes PCA results                                       |

## 📋 Steps Performed

1. Import the required libraries.
2. Load the dataset.
3. Select the numerical features.
4. Handle missing values if necessary.
5. Standardize the features.
6. Apply PCA to the standardized data.
7. Select the required number of principal components.
8. Calculate the explained variance ratio.
9. Visualize the transformed data.
10. Compare the original and reduced dimensions.

## 📊 Expected Output

* Reduced-dimensional dataset.
* Principal component values.
* Explained variance ratio.
* Visualization of the transformed data.
* Comparison of the original number of features with the reduced number.

## ✅ Result

Principal Component Analysis reduces the dimensionality of numerical data by transforming the original features into principal components. It helps simplify datasets while retaining as much variance as possible in the selected components.
