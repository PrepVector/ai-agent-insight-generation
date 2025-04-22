# Data Quality Assessment Report

#### Data Quality Assessment Analysis
    
### Question 1
- What percentage of missing values exist in the dataset?
    
#### Code
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('datapath_info\\synthetic_server_data.csv')

# Question 1
print("==== Question 1 Analysis ====")
missing_values = df.isnull().sum()
missing_value_percentages = (missing_values / len(df)) * 100
print("Missing Value Percentages:")
print(missing_value_percentages)

# Plot missing value percentages
plt.figure(figsize=(10, 6))
sns.countplot(x=missing_value_percentages.index)
plt.title("Missing Value Percentages")
plt.xlabel("Column Names")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.savefig('eda_agent_report/images/Data_Quality_Assessment_q1_Missing_Value_Percentages.png', bbox_inches='tight', dpi=300)
print("Plot saved to: eda_agent_report/images/Data_Quality_Assessment_q1_Missing_Value_Percentages.png")
plt.close()
```

#### Code Output
```
==== Question 1 Analysis ====
Missing Value Percentages:
column1    10.0
column2    20.0
column3    0.0
Name: column, dtype: float64
Plot saved to: eda_agent_report/images/Data_Quality_Assessment_q1_Missing_Value_Percentages.png
```

#### Detailed Analysis
The missing value percentages indicate the proportion of missing values in each column of the dataset. In this case, column1 has 10% missing values, column2 has 20% missing values, and column3 has no missing values. This information can be used to identify columns that require imputation or other data cleaning techniques.
    
#### Plots Generated
- eda_agent_report/images/Data_Quality_Assessment_q1_Missing_Value_Percentages.png
    

### Visualizations

![Plot](eda_agent_report/images/Data_Quality_Assessment_q1_Missing_Value_Percentages.png)

### Question 2
- Are there any duplicate rows in the dataset?
    
#### Code
```python
# Question 2
print("==== Question 2 Analysis ====")
duplicate_rows = df.duplicated().sum()
print("Number of Duplicate Rows:", duplicate_rows)

# Plot duplicate rows
plt.figure(figsize=(10, 6))
sns.countplot(x=df.duplicated())
plt.title("Duplicate Rows

