# InsightBot
An intelligent data analysis system combining Business Analytics and Exploratory Data Analysis capabilities. (# ai-agent-insight-generation ->
AI agent that generates data insights)

![InsightBot_Workflow drawio](https://github.com/user-attachments/assets/4acac482-eb8c-4f10-8362-5e66737f4769)

## Overview
InsightBot is a sophisticated analysis tool that leverages two specialized agents:
- **BA Agent**: Handles business analytics and insights generation
- **EDA Agent**: Performs exploratory data analysis and statistical computations

## Features
- Automated data analysis and visualization
- Business insights generation
- Statistical analysis and pattern detection
- Interactive query processing
- Comprehensive data exploration

## Getting Started
### Prerequisites
1. Create a virtual environment
2. Activate the virtual environment
3. Install the required dependencies
```bash
pip install -r requirements.txt
```
### Usage
1. Run the main application:
```bash
streamlit run app.py
```
2. Use the system through the provided interface to:
    - Load and analyze datasets (The uploaded dataset, metdata (optional) is stored in the datapath_info folder. It can be deleted if required (Delete Dataset button)
    - Generate business insights (Generate EDA Questions button)
    - Perform exploratory analysis and Visualize data patterns (Run EDA Analysis button)

## Components
### BA Agent
- Business analytics processing
- Insight generation
- Strategic recommendations
### EDA Agent
- Statistical analysis
- Data visualization
- Pattern detection
- Exploratory analysis

## Further Improvements
- Generate Business Report (Note: It will in bullet pointers -> Comprehensive report containing the summary of the EDA on dataset)
- Dockefile
- API Integration
- Deployment to Streamlit Cloud etc


## Limitations
- Rate Limit Error


## Contributing
Feel free to submit issues and enhancement requests.
## License
This project is licensed under the MIT License.

