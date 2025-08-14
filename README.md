 ## Data Mining Project README
## Project Overview

This project demonstrates a complete data mining workflow on the Iris dataset (from scikit-learn) and synthetic transactional data for association rule mining. The project covers data preprocessing, exploratory data analysis, clustering, classification, and association rules, using Python libraries such as pandas, scikit-learn, seaborn, matplotlib, and mlxtend.

The primary objectives are:

* To preprocess and explore the Iris dataset.

* To perform K-Means clustering and evaluate cluster quality.

* To train and compare Decision Tree and KNN classifiers.

* To generate synthetic transactional data and extract association rules using the Apriori algorithm.

* To visualize results and interpret findings for potential real-world applications.

Project Structure
data_mining_project/
│
├── preprocessing_iris.py         # Data preprocessing and exploration
├── clustering_iris.py            # K-Means clustering implementation
├── classification_iris.py        # Classification (Decision Tree & KNN)
├── mining_iris_basket.py         # Synthetic transaction generation + Apriori rules
├── images/                       # Folder containing visualizations
│   ├── pairplot.png
│   ├── correlation_heatmap.png
│   ├── kmeans_clusters_scatter.png
│   ├── kmeans_elbow_curve.png
│   ├── decision_tree.png
├── README.md                     # Project documentation (this file)
├── synthetic_transactions.csv    # Optional: saved generated baskets

## Task 1: Data Preprocessing and Exploration

Objective: Prepare the Iris dataset for analysis and modeling.

Steps Taken:

Loading Data:

Used scikit-learn’s built-in load_iris dataset.

Handling Missing Values:

Checked for null values; none were found.

Feature Normalization:

Applied Min-Max scaling to all numeric features to ensure uniform range [0,1] for clustering and classification.

Label Encoding:

Species were encoded numerically for modeling.

Also retained original categorical names for visualization.

Exploratory Data Analysis (EDA):

Computed descriptive statistics: mean, standard deviation, min, max using pandas.describe().

## Visualizations:

Pairplot to examine relationships between features and species.
Correlation heatmap to check feature correlations.

Boxplots to identify potential outliers.

Output: Clean, normalized dataset with encoded species, ready for clustering and classification.

Task 2: K-Means Clustering

Objective: Group Iris data into clusters and evaluate alignment with true species.

Implementation:

K-Means Clustering:

Primary clustering: k=3 (matching the 3 species in Iris).

Predicted clusters stored in a new column.

Evaluation:

Adjusted Rand Index (ARI) to measure similarity between clusters and true species.

Tested k=2 and k=4 to validate optimal cluster count.

Elbow curve used to visually justify k=3.

Visualization:

Scatter plot of petal length vs petal width, colored by cluster and shaped by species.

Misclassifications identified where cluster color differs from species label.

Insights:

Clusters generally align with species.

Most misclassifications occur between versicolor and virginica, which have overlapping feature ranges.

Demonstrates how clustering can segment natural groupings in real-world data (e.g., customer segmentation).

Task 3 Part A: Classification

Objective: Train supervised classifiers to predict species based on features.

Steps:

Train/Test Split:

80/20 split, stratified to preserve species distribution.

Decision Tree Classifier:

Trained using all features.

Evaluation metrics: Accuracy, Precision, Recall, F1-score.

Tree visualized using plot_tree for interpretability.

K-Nearest Neighbors (KNN):

k=5 neighbors, trained on normalized data.

Metrics computed for comparison.

Classifier Comparison:

Decision Tree vs KNN accuracy evaluated.

Decision Tree offers interpretability via tree visualization; KNN is sensitive to scaling and works well on small datasets.

Findings:

Both classifiers achieve high accuracy (~93% for Decision Tree).

Misclassifications primarily between versicolor and virginica.

Demonstrates predictive modeling on multi-class datasets.

Task 3 Part B: Association Rule Mining

Objective: Identify patterns in synthetic transactional data.

Steps:

Synthetic Transaction Generation:

40 transactions, each with 3–6 items from a pool of 20.

Patterns introduced (e.g., milk & bread frequently co-occur) to ensure rules exist.

Apriori Algorithm (mlxtend):

Min support: 0.15

Min confidence: 0.4

Frequent itemsets extracted and rules generated.

Rule Analysis:

Sorted by lift to highlight strongest associations.

Top rule interpreted: “If a customer buys X, they are likely to also buy Y.”

Practical application: retail product placement, cross-selling, and promotions.

Output: Top 5 association rules with support, confidence, and lift metrics.

Libraries Used

pandas: Data manipulation

numpy: Numerical operations

scikit-learn: Preprocessing, K-Means, Decision Tree, KNN

seaborn & matplotlib: Visualization

mlxtend: Apriori and association rule mining

Visualizations

Pairplot, correlation heatmap, boxplots (EDA)

K-Means cluster scatter plots and elbow curve

Decision Tree diagram

Optional: top association rules printed

Conclusion & Insights

Data preprocessing ensures clean, scaled, and encoded datasets for modeling.

Clustering identifies natural groupings; supports exploratory segmentation.

Classification achieves high predictive accuracy, with Decision Tree providing interpretability.

Association rule mining identifies actionable co-purchase patterns in transactional data.

Synthetic data generation helps simulate scenarios for testing, though may artificially inflate pattern significance.

Overall, the project demonstrates an end-to-end data mining pipeline: from preprocessing to exploration, modeling, and actionable insights.
