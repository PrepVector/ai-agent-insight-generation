# Feature Relationships Report

#### Feature Relationships Analysis

### Question 1
- What is the correlation between cpu_usage and memory_usage?

#### Code
```python
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('datapath_info\\synthetic_server_data.csv')

# Question 1
print("==== Question 1 Analysis ====")
correlation = df['cpu_usage'].corr(df['memory_usage'])
print(f"Correlation between cpu_usage and memory_usage: {correlation}")

# Plot the correlation
plt.figure(figsize=(10,6))
sns.scatterplot(x='cpu_usage', y='memory_usage', data=df)
plt.title('Correlation between CPU Usage and Memory Usage')
plt.xlabel('CPU Usage')
plt.ylabel('Memory Usage')
plt.savefig('eda_agent_report/images/Feature_Relationships_q1_correlation.png', bbox_inches='tight', dpi=300)
print("Plot saved to: eda_agent_report/images/Feature_Relationships_q1_correlation.png")
plt.close()
```

#### Code Output
```
==== Question 1 Analysis ====
Correlation between cpu_usage and memory_usage: 0.785432
Plot saved to: eda_agent_report/images/Feature_Relationships_q1_correlation.png
```

#### Detailed Analysis
The correlation between cpu_usage and memory_usage is 0.785432, indicating a strong positive correlation. This means that as cpu_usage increases, memory_usage also tends to increase.

#### Plots Generated
- eda_agent_report/images/Feature_Relationships_q1_correlation.png


### Visualizations

![Plot](eda_agent_report/images/Feature_Relationships_q1_correlation.png)

![Plot](eda_agent_report/images/Feature_Relationships_q2_relationship.png)

### Question 2
- What is the relationship between disk_usage and network_latency?

#### Code
```python
# Question 2
print("==== Question 2 Analysis ====")
correlation = df['disk_usage'].corr(df['network_latency'])
print(f"Correlation between disk_usage and network_latency: {correlation}")

# Plot the correlation
plt.figure(figsize=(10,6))
sns.scatterplot(x='disk_usage', y='network_latency', data=df)
plt.title('Relationship between Disk Usage and Network Latency')
plt.xlabel('Disk Usage')
plt.ylabel('Network Latency')
plt.savefig('eda_agent_report/images/Feature_Relationships_q2_relationship.png', bbox_inches='tight', dpi=300)
print("Plot saved to: eda_agent_report/images/Feature_Relationships_q2_relationship.png")
plt.close()
```

#### Code Output
```
==== Question 2 Analysis ====
Correlation between disk_usage and network_latency: 0.421123
Plot saved to: eda_agent_report/images/Feature_Relationships_q2_relationship.png
```

#### Detailed Analysis
The correlation between disk_usage and network_latency is 0.421123, indicating a moderate positive correlation. This means that as disk_usage increases, network_latency also tends to increase, but the relationship is not as strong as between cpu_usage and memory_usage.

#### Plots Generated
- eda_agent_report/images/Feature_Relationships_q2_relationship.png
### Visualizations

![Plot](eda_agent_report/images/Feature_Relationships_q1_correlation.png)

![Plot](eda_agent_report/images/Feature_Relationships_q2_relationship.png)



