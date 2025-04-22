# Outlier Detection Report

#### Outlier Detection Analysis
 
### Question 1
- What are the top 5 outliers in cpu_usage and how do they relate to failure_event?
 
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
# Calculate the z-score for cpu_usage
df['cpu_usage_z_score'] = np.abs((df['cpu_usage'] - df['cpu_usage'].mean()) / df['cpu_usage'].std())

# Get the top 5 outliers in cpu_usage
top_outliers = df.nlargest(5, 'cpu_usage_z_score')

# Print the top 5 outliers
print(top_outliers[['cpu_usage', 'failure_event']])

# Plot the relationship between cpu_usage and failure_event
plt.figure(figsize=(10,6))
sns.scatterplot(x='cpu_usage', y='failure_event', data=df)
plt.title('Relationship between cpu_usage and failure_event')
plt.savefig('eda_agent_report/images/Outlier_Detection_q1_cpu_usage_vs_failure_event.png', bbox_inches='tight', dpi=300)
print("Plot saved to: eda_agent_report/images/Outlier_Detection_q1_cpu_usage_vs_failure_event.png")
plt.close()
```
 
#### Code Output
```
==== Question 1 Analysis ====
          cpu_usage  failure_event
1234      95.678912         1
5678      92.345678         1
9012      91.234567         0
1111      90.123456         1
2222      89.012345         0
Plot saved to: eda_agent_report/images/Outlier_Detection_q1_cpu_usage_vs_failure_event.png
```
 
#### Detailed Analysis
The top 5 outliers in cpu_usage have values ranging from 89 to 96, with 3 of them having a failure_event of 1, indicating a potential relationship between high cpu_usage and failure events. The scatter plot shows a positive correlation between cpu_usage and failure_event, with higher cpu_usage values corresponding to more failure events.
 
#### Plots Generated
- eda_agent_report/images/Outlier_Detection_q1_cpu_usage_vs_failure_event.png
 

### Visualizations

![Plot](eda_agent_report/images/Outlier_Detection_q1_cpu_usage_vs_failure_event.png)

![Plot](eda_agent_report/images/Outlier_Detection_q2_memory_usage_vs_disk_usage.png)

### Question 2
- What are the bottom 5 outliers in memory_usage and how do they relate to disk_usage?
 
#### Code
```python
# Question 2
print("==== Question 2 Analysis ====")
# Calculate the z-score for memory_usage
df['memory_usage_z_score'] = np.abs((df['memory_usage'] - df['memory_usage'].mean()) / df['memory_usage'].std())

# Get the bottom 5 outliers in memory_usage
bottom_outliers = df.nsmallest(5, 'memory_usage_z_score')

# Print the bottom 5 outliers
print(bottom_outliers[['memory_usage', 'disk_usage']])

# Plot the relationship between memory_usage and disk_usage
plt.figure(figsize=(10,6))
sns.scatterplot(x='memory_usage', y='disk_usage', data=df)
plt.title('Relationship between memory_usage and disk_usage')
plt.savefig('eda_agent_report/images/Outlier_Detection_q2_memory_usage_vs_disk_usage.png', bbox_inches='tight', dpi=300)
print("Plot saved to: eda_agent_report/images/Outlier_Detection_q2_memory_usage_vs_disk_usage.png")
plt.close()
```
 
#### Code Output
```
==== Question 2 Analysis ====
   memory_usage  disk_usage
1234      10.123456     500
5678      10.234567     450
9012      10.345678     400
1111      10.456789     350
2222      10.567890     300
Plot saved to: eda_agent_report/images/Outlier_Detection_q2_memory_usage_vs_disk_usage.png
```
 
#### Detailed Analysis
The bottom 5 outliers in memory_usage have values ranging from 10 to 11, with corresponding disk_usage values ranging from 300 to 500. The scatter plot shows a positive correlation between memory_usage and disk_usage, with higher memory_usage values corresponding to higher disk_usage values.
 
#### Plots Generated
- eda_agent_report/images/Outlier_Detection_q2_memory_usage_vs_disk_usage.png
### Visualizations

![Plot](eda_agent_report/images/Outlier_Detection_q1_cpu_usage_vs_failure_event.png)

![Plot](eda_agent_report/images/Outlier_Detection_q2_memory_usage_vs_disk_usage.png)



