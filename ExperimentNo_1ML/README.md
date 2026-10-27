# Experiment: Data Preprocessing

## 🎯 Aim

To perform data preprocessing techniques to clean and prepare a dataset for Machine Learning algorithms.

## 📖 Theory

Data preprocessing is the process of converting raw data into a clean and suitable format for Machine Learning.

Real-world datasets may contain missing values, duplicate records, inconsistent data, and features with different scales. Data preprocessing helps improve data quality and prepares the dataset for analysis and model training.

### Types of Data Preprocessing

1. **Handling Missing Values:** Identify and fill or remove missing values.
2. **Removing Duplicates:** Remove repeated records from the dataset.
3. **Encoding Categorical Data:** Convert categorical values into numerical form.
4. **Feature Scaling:** Bring numerical features to a comparable scale.
5. **Handling Outliers:** Identify and handle unusually high or low values when appropriate.
6. **Data Splitting:** Divide the dataset into training and testing sets.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Jupyter Notebook

## 📋 Steps Performed

1. Import the required Python libraries.
2. Load the dataset using Pandas.
3. Examine the dataset using `head()`, `info()`, and `describe()`.
4. Identify missing values using `isnull().sum()`.
5. Handle missing values using suitable techniques.
6. Identify and remove duplicate records if necessary.
7. Encode categorical features if required.
8. Scale numerical features if required.
9. Split the dataset into training and testing sets.
10. Verify the processed dataset.

## 💻 Important Python Functions

| Function            | Purpose                                     |
| ------------------- | ------------------------------------------- |
| `read_csv()`        | Loads a CSV dataset                         |
| `isnull().sum()`    | Counts missing values                       |
| `fillna()`          | Fills missing values                        |
| `dropna()`          | Removes rows or columns with missing values |
| `drop_duplicates()` | Removes dup_                                |
