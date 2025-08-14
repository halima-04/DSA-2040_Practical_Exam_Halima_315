### Practical Exam: Data Warehousing and Data Mining (HALIMA-315)
## Section 1: Data Warehousing (50 Marks)
# Task 1: Data Warehouse Design (15 Marks)
# Design a star schema for this data warehouse. Include at least one fact table and 3-4
dimension tables. 
import sqlite3


   
# Retail Data Warehouse Project

## Section 1: Data Warehousing (50 Marks)

### Task 1: Data Warehouse Design (15 Marks)

#### 1. Star Schema Design

**Fact Table: SalesFact**
| Column        | Description                         |
|---------------|-------------------------------------|
| SalesID       | Primary key, auto-increment          |
| CustomerID    | Foreign key to CustomerDim           |
| ProductID     | Foreign key to ProductDim            |
| TimeID        | Foreign key to TimeDim               |
| Quantity      | Number of units sold                 |
| TotalSales    | Quantity * UnitPrice                 |

**Dimension Tables:**

**CustomerDim**
| Column      | Description                    |
|-------------|--------------------------------|
| CustomerID  | Primary key                    |
| Name        | Customer name                  |
| Country     | Customer country               |
| Email       | Optional                       |

**ProductDim**
| Column      | Description                    |
|-------------|--------------------------------|
| ProductID   | Primary key                    |
| Name        | Product name                   |
| Category    | Product category               |
| Price       | Unit price                     |

**TimeDim**
| Column      | Description                    |
|-------------|--------------------------------|
| TimeID      | Primary key                    |
| Date        | Full date                      |
| Month       | Month number                   |
| Quarter     | Quarter number                 |
| Year        | Year                           |

**Diagram:**  

<img width="434" height="312" alt="image" src="https://github.com/user-attachments/assets/31503909-41e1-484c-be6d-bc0a7b0c4607" />

#### 2. Why Star Schema
The star schema was chosen over a snowflake schema because it provides simpler queries, better performance for read-heavy analytical workloads, and easier aggregation across dimensions, which suits OLAP queries and reporting requirements.

#### 3. SQL CREATE TABLE Statements (SQLite Syntax)
```sql
CREATE TABLE CustomerDim (
    CustomerID INTEGER PRIMARY KEY,
    Name TEXT,
    Country TEXT,
    Email TEXT
);

CREATE TABLE ProductDim (
    ProductID INTEGER PRIMARY KEY,
    Name TEXT,
    Category TEXT,
    Price REAL
);

CREATE TABLE TimeDim (
    TimeID INTEGER PRIMARY KEY,
    Date TEXT,
    Month INTEGER,
    Quarter INTEGER,
    Year INTEGER
);

CREATE TABLE SalesFact (
    SalesID INTEGER PRIMARY KEY AUTOINCREMENT,
    CustomerID INTEGER,
    ProductID INTEGER,
    TimeID INTEGER,
    Quantity INTEGER,
    TotalSales REAL,
    FOREIGN KEY (CustomerID) REFERENCES CustomerDim(CustomerID),
    FOREIGN KEY (ProductID) REFERENCES ProductDim(ProductID),
    FOREIGN KEY (TimeID) REFERENCES TimeDim(TimeID)
);
```
##Task 2: ETL Process Implementation (20 Marks)
## Step 1 — Extract
## What we do:
* Read the dataset (Online_Retail.csv) from disk into a pandas DataFrame.
* Remove rows missing essential values:
* InvoiceNo → needed to identify transactions.
* StockCode → product identification.
* Quantity and UnitPrice → required for sales calculations.
* InvoiceDate → needed for time-based analysis.
* Convert InvoiceDate to a proper datetime type so we can filter and group by time later.
* Remove any rows where the date could not be parsed.

## Why we do it:
* Ensures we are working only with valid, complete data before transformations.
* Makes sure the InvoiceDate column is in a format that allows filtering and aggregations.
* Avoids issues in later steps from missing or invalid values.
import zipfile
import pandas as pd
```
# Correct path to your ZIP file
zip_path = r"C:\Users\Salma\Downloads\online+retail.zip"

# Inspect ZIP contents
with zipfile.ZipFile(zip_path, 'r') as z:
    print("Files in zip:", z.namelist())

# Read the Excel inside the ZIP
excel_name = z.namelist()[0]
with zipfile.ZipFile(zip_path) as z:
    with z.open(excel_name) as f:
        df = pd.read_excel(f)

# Take a random sample of 1000 rows
df_sample = df.sample(n=1000, random_state=42)

print(df_sample.head())
print(df_sample.info())
```
<img width="960" height="540" alt="image" src="https://github.com/user-attachments/assets/2cf11847-04e5-4e6f-9e8b-e87636386a93" />

## Step 2 — Transform
## What we do:
* Remove invalid transactions:
* Negative or zero Quantity values.
* Zero or negative UnitPrice.
* Create a new column:
* TotalSales = Quantity * UnitPrice → This is the key sales measure.
* Filter transactions to the last year relative to 2025-08-12 (exam requirement).
* Create dimension-like tables:
* CustomerDim: unique CustomerID and Country.
* TimeDim: unique dates with TimeID, Month, Quarter, Year for time-based OLAP.
* Prepare fact table:
* SalesFact: contains CustomerID, TimeID, Quantity, and TotalSales.

## Why we do it:
* Removes bad data so our metrics are accurate.
* Adds new calculated metrics for reporting.
* Structures the data into star schema format to make OLAP queries easier in Task 3.
* Filters for recent transactions to keep analysis relevant and within the scope.
  
```
# Add TotalSales
df_sample['TotalSales'] = df_sample['Quantity'] * df_sample['UnitPrice']

# Customer Dimension
customer_dim = df_sample.groupby('CustomerID').agg({
    'Country': 'first',
    'TotalSales': 'sum'
}).reset_index()

# Time Dimension
time_dim = df_sample[['InvoiceDate']].drop_duplicates().reset_index(drop=True)
time_dim['TimeID'] = time_dim.index + 1
time_dim['Date'] = time_dim['InvoiceDate']
time_dim['Month'] = time_dim['InvoiceDate'].dt.month
time_dim['Quarter'] = time_dim['InvoiceDate'].dt.quarter
time_dim['Year'] = time_dim['InvoiceDate'].dt.year
time_dim = time_dim.drop(columns=['InvoiceDate'])

# Map TimeID to SalesFact
df_sample = df_sample.merge(time_dim[['Date','TimeID']], left_on='InvoiceDate', right_on='Date', how='left')
sales_fact = df_sample[['CustomerID','TimeID','Quantity','TotalSales']].copy()

```
## Step 3 — Load
## What we do:
* Connect to a SQLite database (retail_dw.db).
* Create tables:
* CustomerDim
* TimeDim
* SalesFact
* Load the cleaned/transformed data into these tables.
* Enforce foreign key constraints to maintain referential integrity.

## Why we do it:
* Moves data into a data warehouse structure for analysis.
* Allows running SQL queries efficiently in later steps (Task 3).
* Ensures we follow proper relational database design.
  
```
# 4. Load into SQLite
# -----------------------------
db_name = "retail_dw_sample.db"
conn = sqlite3.connect(db_name)
cursor = conn.cursor()

# Drop tables if they exist
cursor.executescript("""
DROP TABLE IF EXISTS SalesFact;
DROP TABLE IF EXISTS TimeDim;
DROP TABLE IF EXISTS CustomerDim;
""")

# Create tables
cursor.executescript("""
CREATE TABLE CustomerDim (
    CustomerID INTEGER PRIMARY KEY,
    Country TEXT,
    TotalSales REAL
);

CREATE TABLE TimeDim (
    TimeID INTEGER PRIMARY KEY,
    Date TEXT,
    Month INTEGER,
    Quarter INTEGER,
    Year INTEGER
);

CREATE TABLE SalesFact (
    SalesID INTEGER PRIMARY KEY AUTOINCREMENT,
    CustomerID INTEGER,
    TimeID INTEGER,
    Quantity INTEGER,
    TotalSales REAL,
    FOREIGN KEY (CustomerID) REFERENCES CustomerDim(CustomerID),
    FOREIGN KEY (TimeID) REFERENCES TimeDim(TimeID)
);
""")

# Insert data
customer_dim.to_sql('CustomerDim', conn, if_exists='append', index=False)
time_dim.to_sql('TimeDim', conn, if_exists='append', index=False)
sales_fact.to_sql('SalesFact', conn, if_exists='append', index=False)

conn.commit()
conn.close()

print(f"[ETL] Completed: {db_name} created with:")
print("CustomerDim rows:", len(customer_dim))
print("TimeDim rows:", len(time_dim))
print("SalesFact rows:", len(sales_fact))
```
<img width="960" height="282" alt="image" src="https://github.com/user-attachments/assets/e3e6ac08-6b33-4ab2-b66f-d0cd293e5b27" />

```import sqlite3
conn = sqlite3.connect("retail_dw_sample.db")
cursor = conn.cursor()

cursor.execute("SELECT * FROM CustomerDim LIMIT 5")
print(cursor.fetchall())

cursor.execute("SELECT * FROM SalesFact LIMIT 5")
print(cursor.fetchall())
conn.close()
```


## 1. OLAP Queries
Roll-up – total sales by Country and Quarter
```
import pandas as pd
import matplotlib.pyplot as plt

# Connect to your sample DB
conn = sqlite3.connect("retail_dw_sample.db")

# --- 1. Roll-up: Total sales by country and quarter ---
rollup_query = """
SELECT c.Country, t.Quarter, SUM(s.TotalSales) AS TotalSales
FROM SalesFact s
JOIN CustomerDim c ON s.CustomerID = c.CustomerID
JOIN TimeDim t ON s.TimeID = t.TimeID
GROUP BY c.Country, t.Quarter
ORDER BY c.Country, t.Quarter;
"""
rollup = pd.read_sql_query(rollup_query, conn)
print("Roll-up (Country x Quarter):")
print(rollup)
```
<img width="1920" height="1080" alt="Screenshot 2025-08-14 221017" src="https://github.com/user-attachments/assets/f97081b5-da76-425c-b10c-9b6715d40b2e" />

## Drill-down – monthly sales for UK
```
# --- 2. Drill-down: Sales details for a specific country (e.g., United Kingdom) by month ---
drilldown_query = """
SELECT t.Month, SUM(s.TotalSales) AS TotalSales
FROM SalesFact s
JOIN CustomerDim c ON s.CustomerID = c.CustomerID
JOIN TimeDim t ON s.TimeID = t.TimeID
WHERE c.Country = 'United Kingdom'
GROUP BY t.Month
ORDER BY t.Month;
"""
drilldown = pd.read_sql_query(drilldown_query, conn)
print("\nDrill-down (UK Sales by Month):")
print(drilldown)
```

<img width="1920" height="1080" alt="Screenshot 2025-08-14 221144" src="https://github.com/user-attachments/assets/27d0a54a-af52-42c0-8db4-e964fba2b607" />


## Slice – total sales for Electronics category
```
# For this sample, let's assume Description contains the word 'ELECTRONICS' for filtering
slice_query = """
SELECT SUM(TotalSales) AS TotalSales
FROM SalesFact
WHERE Description LIKE '%ELECTRONICS%';
"""
slice_result = pd.read_sql_query(slice_query, conn)
print("\nSlice (Electronics Sales):")
print(slice_result)

conn.close()
```


<img width="1920" height="539" alt="image" src="https://github.com/user-attachments/assets/692530eb-0e40-4f7a-bdbd-a43613a82c03" />


## 2. Visualization Example – bar chart for roll-up result
```
plt.figure(figsize=(10,6))
for country in rollup['Country'].unique():
    data = rollup[rollup['Country'] == country]
    plt.bar(data['Quarter'] + 0.1*list(rollup['Country'].unique()).index(country), data['TotalSales'], width=0.1, label=country)
plt.xlabel('Quarter')
plt.ylabel('Total Sales')
plt.title('Total Sales by Country and Quarter')
plt.legend()
plt.tight_layout()
plt.savefig("sales_by_country_quarter.png")
plt.show()
```

<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/414a53a8-e6f2-4a84-85e5-b020508f9f5e" />



   ###  Data Mining Project

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


