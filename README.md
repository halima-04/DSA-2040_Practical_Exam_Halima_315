# Data Mining Project

## Project Overview
This project demonstrates a complete data mining workflow on the **Iris dataset** (from scikit-learn) and **synthetic transactional data** for association rule mining. The project covers **data preprocessing, exploratory data analysis, clustering, classification, and association rules**, using Python libraries such as **pandas, scikit-learn, seaborn, matplotlib, and mlxtend**.

The primary objectives are:  
1. To preprocess and explore the Iris dataset.  
2. To perform **K-Means clustering** and evaluate cluster quality.  
3. To train and compare **Decision Tree and KNN classifiers**.  
4. To generate **synthetic transactional data** and extract **association rules** using the Apriori algorithm.  
5. To visualize results and interpret findings for potential real-world applications.  

---

## Project Structure

data_mining_project/

│

├── preprocessing_iris.py # Data preprocessing and exploration

├── clustering_iris.py # K-Means clustering implementation

├── classification_iris.py # Classification (Decision Tree & KNN)

├── mining_iris_basket.py # Synthetic transaction generation + Apriori rules

├── images/ # Folder containing visualizations

│ ├── pairplot.png

│ ├── correlation_heatmap.png

│ ├── kmeans_clusters_scatter.png

│ ├── kmeans_elbow_curve.png

│ ├── decision_tree.png

├── README.md # Project documentation 

├── synthetic_transactions.csv # Optional: saved generated baskets


---

## Task 1: Data Preprocessing and Exploration

**Objective:** Prepare the Iris dataset for analysis and modeling.

**Steps Taken:**
1. **Loading Data** using `scikit-learn`’s built-in `load_iris` dataset.
2. **Handling Missing Values**: Checked for null values; none were found.
3. **Feature Normalization**: Applied **Min-Max scaling** to numeric features for clustering and classification.
4. **Label Encoding**: Encoded species numerically and retained original categorical names for visualization.
5. **Exploratory Data Analysis (EDA)**:
   - Descriptive statistics: mean, std, min, max using `pandas.describe()`.
   - Visualizations:
     - **Pairplot** to examine feature relationships.
     - **Correlation heatmap** to check feature correlations.
     - **Boxplots** to identify potential outliers.

**Output:** Clean, normalized dataset ready for clustering and classification.

---

## Task 2: K-Means Clustering

**Objective:** Group Iris data into clusters and evaluate alignment with true species.

**Implementation:**
1. **K-Means Clustering**:
   - Primary clustering: **k=3** (matching the 3 species).
   - Predicted clusters stored in a new column.
2. **Evaluation**:
   - **Adjusted Rand Index (ARI)** to compare clusters with true species.
   - Tested **k=2** and **k=4** for optimal cluster count.
   - **Elbow curve** visualized to justify k=3.
3. **Visualization**:
   - Scatter plot of **petal length vs petal width**, colored by cluster and shaped by species.
   - Misclassifications identified where cluster color differs from species label.

**Insights:**
- Clusters generally align with species.
- Misclassifications mainly between **versicolor** and **virginica**.
- Shows clustering’s utility in **natural segmentation**.

---

## Task 3 Part A: Classification

**Objective:** Train supervised classifiers to predict species.

**Steps:**
1. **Train/Test Split**: 80/20 split, stratified by species.
2. **Decision Tree Classifier**:
   - Trained on all features.
   - Evaluated using Accuracy, Precision, Recall, F1-score.
   - Tree visualized using `plot_tree`.
3. **K-Nearest Neighbors (KNN)**:
   - k=5 neighbors.
   - Evaluated metrics for comparison.
4. **Classifier Comparison**:
   - Decision Tree vs KNN accuracy.
   - Decision Tree offers interpretability; KNN works well on small datasets.

**Findings:**
- Both classifiers achieve high accuracy (~93% for Decision Tree).
- Misclassifications mainly between **versicolor** and **virginica**.
- Demonstrates predictive modeling on multi-class datasets.

---

## Task 3 Part B: Association Rule Mining

**Objective:** Identify patterns in synthetic transactional data.

**Steps:**
1. **Synthetic Transaction Generation**:
   - 40 transactions with 3–6 items each from 20-item pool.
   - Patterns introduced (e.g., milk & bread frequently co-occur).
2. **Apriori Algorithm** (`mlxtend`):
   - Min support: 0.15
   - Min confidence: 0.4
   - Frequent itemsets extracted; rules generated.
3. **Rule Analysis**:
   - Sorted by **lift** to highlight strongest associations.
   - Top rule interpreted: “If a customer buys X, they are likely to also buy Y.”
   - Practical application: **product placement, cross-selling, promotions**.

**Output:** Top 5 association rules with support, confidence, and lift metrics.

---

## Libraries Used
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **scikit-learn**: Preprocessing, K-Means, Decision Tree, KNN
- **seaborn** & **matplotlib**: Visualization
- **mlxtend**: Apriori and association rule mining

---

## Visualizations
- Pairplot
 <img width="1117" height="1000" alt="iris_pairplot" src="https://github.com/user-attachments/assets/67dee35b-f7ca-455d-9477-9b87c14a3a24" />
 
- correlation heatmap
<img width="800" height="600" alt="iris_correlation_heatmap" src="https://github.com/user-attachments/assets/4d63b3c8-7de7-49a1-a789-a382636d6f3d" />

- boxplots
<img width="640" height="480" alt="iris_boxplots" src="https://github.com/user-attachments/assets/a3cfcc67-cad2-40ec-ab37-4831921b72eb" />



- K-Means cluster scatter plots 
 <img width="800" height="600" alt="kmeans_clusters_scatter" src="https://github.com/user-attachments/assets/bd1607cf-1719-4e88-8c17-0dcb5f2fe1ef" />

- elbow curve
<img width="800" height="500" alt="kmeans_elbow_curve" src="https://github.com/user-attachments/assets/523af341-ab24-4c93-b077-853f925aa696" />


- Decision Tree diagram
- <img width="1200" height="800" alt="decision_tree" src="https://github.com/user-attachments/assets/7cae1e1f-4555-49fb-ad89-83af9bfb81d6" />

- Optional: top association rules printed

---

## Conclusion & Insights
- **Data preprocessing** ensures clean, scaled, and encoded datasets.
- **Clustering** identifies natural groupings; supports exploratory segmentation.
- **Classification** achieves high predictive accuracy; Decision Tree is interpretable.
- **Association rule mining** identifies actionable co-purchase patterns.
- **Synthetic data** helps simulate scenarios for testing, though may inflate pattern significance.
- Demonstrates an **end-to-end data mining pipeline** from preprocessing to actionable insights.


