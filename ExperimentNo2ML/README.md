# Experiment: Exploratory Data Analysis (EDA)

## 🎯 Aim

To perform Exploratory Data Analysis (EDA) on a dataset to understand its structure, identify patterns, detect missing values, and discover relationships between variables.

## 📖 Theory

Exploratory Data Analysis (EDA) is the process of analyzing and visualizing a dataset to understand its main characteristics before applying Machine Learning algorithms.

EDA helps us identify missing values, duplicate records, outliers, data distributions, and relationships between features.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Jupyter Notebook

## 📋 Steps Performed

1. Import the required Python libraries.
2. Load the dataset.
3. Display the first few rows using `head()`.
4. Check dataset dimensions using `shape`.
5. Examine column names and data types using `info()`.
6. Calculate statistical summaries using `describe()`.
7. Check for missing values using `isnull().sum()`.
8. Identify duplicate records.
9. Visualize numerical data using histograms and box plots.
10. Analyze relationships between variables using a correlation heatmap.
11. Draw conclusions from the analysis.

## 💻 Important Python Functions

| Function             | Purpose                                                |
| -------------------- | ------------------------------------------------------ |
| `head()`             | Displays the first five rows                           |
| `shape`              | Returns the number of rows and columns                 |
| `info()`             | Displays column names, data types, and non-null counts |
| `describe()`         | Generates descriptive statistics                       |
| `isnull().sum()`     | Counts missing values in each column                   |
| `duplicated().sum()` | Counts duplicate rows                                  |
| `hist()`             | Displays data distributions                            |
| `corr()`             | Calculates correlations between numerical columns      |

## 📊 Expected Output

* Dataset overview and summary statistics.
* Identification of missing values and duplicate records.
* Graphical representation of data distributions.
* Detection of potential outliers.
* Correlation analysis between numerical features.

## ✅ Result

Exploratory Data Analysis helps us understand the dataset, identify data quality issues, discover patterns, and prepare the data for further preprocessing and Machine Learning tasks.
