# Experiment: K-Means Clustering

## 🎯 Aim

To implement the K-Means Clustering algorithm to group similar data points into different clusters.

## 📖 Theory

**K-Means Clustering** is an unsupervised Machine Learning algorithm used to divide a dataset into groups called clusters.

It groups similar data points together without using predefined target labels.

### Working of K-Means Clustering

1. **Choose K:** Select the number of clusters to create.
2. **Initialize Centroids:** Select initial cluster centers.
3. **Assign Data Points:** Assign each data point to its nearest centroid.
4. **Update Centroids:** Calculate new centroids using the mean of the points in each cluster.
5. **Repeat:** Continue assigning points and updating centroids until the centroids stop changing significantly or a stopping condition is reached.

### Important Terms

* **Cluster:** A group of similar data points.
* **Centroid:** The center of a cluster.
* **K:** The number of clusters.
* **Elbow Method:** A technique used to help select a suitable number of clusters by examining the within-cluster sum of squares.

### Example

A shopping company can use K-Means to group customers according to their spending habits, such as low, medium, and high spending groups.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Jupyter Notebook

## 📋 Steps Performed

1. Import the required Python libraries.
2. Load or create the dataset.
3. Select the features for clustering.
4. Preprocess or scale the data if required.
5. Choose the number of clusters (K).
6. Create the K-Means model.
7. Fit the model to the dataset.
8. Assign data points to clusters.
9. Identify cluster centroids.
10. Visualize the clusters if required.

## 💻 Important Python Functions

* `KMeans()` – Creates a K-Means clustering model.
* `fit()` – Trains the model on the dataset.
* `fit_predict()` – Fits the model and returns cluster labels.
* `cluster_centers_` – Returns the cluster centroids.
* `labels_` – Returns the cluster labels.
* `inertia_` – Returns the within-cluster sum of squared distances.
* `StandardScaler()` – Standardizes numerical features if required.
* `plt.scatter()` – Displays data points in a scatter plot.

## 📊 Expected Output

* Data points are grouped into clusters.
* Each data point is assigned a cluster label.
* Cluster centroids are identified.
* A visualization of the clusters may be generated.

## ✅ Result

The K-Means Clustering algorithm is implemented to group similar data points into clusters and analyze the resulting groups.
