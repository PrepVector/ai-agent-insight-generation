import os
import warnings
from typing import List, Dict
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
import pandas as pd
from dotenv import load_dotenv
import json

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Load environment variables from .env file
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize the language model
llm = ChatGroq(
    temperature=0,
    model_name="llama3-70b-8192",
)

# Format questions from dictionary to editable text format
def format_questions_to_text(questions_dict: dict) -> str:
    """Converts the nested questions dictionary to a formatted text for editing."""
    text = ""
    for key, value in questions_dict.items():
        category_name = value.get("category", key).strip()
        text += f"### {category_name}\n"
        for idx, question in enumerate(value.get("questions", []), 1):
            text += f"{idx}. {question}\n"
        text += "\n"
    return text

# Parse edited text back to a structured format
def parse_text_to_questions(text: str) -> dict:
    """Parses edited text back into the nested dictionary format."""
    lines = text.split('\n')
    result = {}
    current_key = None
    current_category = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith('###'):
            current_category = line[3:].strip()
            current_key = current_category.lower().replace(' ', '_')
            result[current_key] = {
                "category": current_category,
                "questions": []
            }
        elif line[0].isdigit() and '. ' in line and current_key:
            question = line.split('. ', 1)[1]
            result[current_key]["questions"].append(question)

    return result

def generate_summary_text(df):
    print(f"Successfully loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    print("\nGenerating dataset summary statistics...")
    # Basic summary statistics
    summary_stats = df.describe().to_dict()
    # Count missing values
    missing_values = df.isnull().sum().to_dict()
    # Count duplicate rows
    duplicate_count = df.duplicated().sum()
    # Get data types
    data_types = df.dtypes.apply(lambda x: str(x)).to_dict()
    """
    Advanced Statistical summary of the dataset
    Numerical features - mean, median, mode, standard deviation, variance, IQR, skewness, and kurtosis.
    Categorical features - counts and unique values.
    """
    numerical_stats = {}
    categorical_stats = {}

    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:
            # Numerical statistics
            numerical_stats[column] = {
                'mean': df[column].mean(),
                'median': df[column].median(),
                'mode': df[column].mode().iloc[0] if not df[column].mode().empty else None,
                'std': df[column].std(),
                'variance': df[column].var(),
                'iqr': df[column].quantile(0.75) - df[column].quantile(0.25),
                'skewness': df[column].skew(),
                'kurtosis': df[column].kurtosis()
            }
        else:
            # Categorical statistics
            categorical_stats[column] = {
                'unique_values': df[column].nunique()
            }

    # Generate summary text from the statistics
    summary_text_llm = f"""
    Dataset Summary:
    - {df.shape[0]} rows and {df.shape[1]} columns
    - Column names: {', '.join(df.columns.tolist())}
    - Missing values: {sum(missing_values.values())} in total
    - Duplicate rows: {duplicate_count}
    - Data types: {len([dt for dt in data_types.values() if 'float' in dt])} numerical columns, {len([dt for dt in data_types.values() if 'object' in dt])} categorical columns
    - Summary statistics: {summary_stats}
    - Advanced numerical statistics: {numerical_stats}
    - Categorical statistics: {categorical_stats}
    """
    return summary_text_llm

class EDAQuestions(BaseModel):
    """
    Pydantic model for holding EDA questions by category.
    """
    data_quality_assessment: List[str] = Field(..., description="Questions about data completeness, validity, consistency.")
    statistical_summary: List[str] = Field(..., description="Questions about distributions, central tendency, spread.")
    outlier_detection: List[str] = Field(..., description="Questions about detecting anomalies or extreme values.")
    feature_relationships: List[str] = Field(..., description="Questions about correlations or interactions between features.")
    pattern_trend_anomalies: List[str] = Field(..., description="Questions about trends, seasonality, and unexpected shifts.")

def extract_json_from_response(response_text):
    """Extract JSON from a text response that might contain additional text."""
    try:
        # Look for content between triple backticks with json
        import re
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            return json.loads(json_match.group(1))
        
        # Try to find content between curly braces
        json_match = re.search(r'(\{[\s\S]*\})', response_text)
        if json_match:
            return json.loads(json_match.group(1))
        
        # try to parse the entire response
        return json.loads(response_text)
    except Exception as e:
        print(f"JSON extraction failed: {e}")
        return None
    
def run_business_analysis(df:pd.DataFrame, metadata: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Generate EDA questions across five categories, based on metadata and summary.

    Returns:
        A nested dict with keys like:
        {
            "data_quality_assessment": {
                "category": "Data Quality Assessment",
                "questions": [...]
            },
            ...
        }
    """
    summary = generate_summary_text(df)
    has_metadata = metadata is not None and metadata.strip() != ""
    
    # Base template that's common for both scenarios
    base_template = """
IMPORTANT: Your response MUST be a valid JSON that can be parsed into a Python dictionary with exactly these keys:
- data_quality_assessment
- statistical_summary
- outlier_detection
- feature_relationships
- pattern_trend_anomalies

Each key should contain a list of 5 relevant questions as strings.

Example of the required format:
```json
{{
    "data_quality_assessment": [
        "Question 1?",
        "Question 2?",
        "Question 3?",
        "Question 4?",
        "Question 5?"
    ],
    "statistical_summary": [
        "Question 1?",
        "Question 2?",
        "Question 3?",
        "Question 4?",
        "Question 5?"
    ],
    "outlier_detection": [
        "Question 1?",
        "Question 2?",
        "Question 3?",
        "Question 4?",
        "Question 5?"
    ],
    "feature_relationships": [
        "Question 1?",
        "Question 2?",
        "Question 3?",
        "Question 4?",
        "Question 5?"
    ],
    "pattern_trend_anomalies": [
        "Question 1?",
        "Question 2?",
        "Question 3?",
        "Question 4?",
        "Question 5?"
    ]
}}
```

YOU MUST RETURN ONLY A VALID JSON OBJECT WITH THE EXACT STRUCTURE SHOWN ABOVE.
Do not include any explanations, markdown formatting, or additional text outside the JSON structure.
"""

    # Construct the appropriate prompt based on metadata availability
    if has_metadata:
        prompt_text = f"""
You are a business analyst tasked with generating exploratory data analysis (EDA) questions for a dataset.
Based on the following metadata and dataset summary, generate a valid JSON with questions for each category:

Metadata:
{metadata}

Here is the dataset summary:
{summary}

{base_template}

MAKE SURE TO INCORPORATE INSIGHTS FROM BOTH THE METADATA AND SUMMARY IN YOUR QUESTIONS.
"""
    # If no metadata is provided, use only the summary
    else:
        prompt_text = f"""
You are a business analyst tasked with generating exploratory data analysis (EDA) questions for a dataset.
Based on ONLY the dataset summary (no additional metadata was provided), generate a valid JSON with questions for each category:

Here is the dataset summary:
{summary}

{base_template}

YOUR QUESTIONS SHOULD RELY ONLY ON THE DATASET SUMMARY SINCE NO METADATA WAS PROVIDED.
"""
    
    # Get response directly using the formatted string
    response = llm.invoke(prompt_text.strip())
    print(f"Response from LLM: {response}")
    response_text = response.content if hasattr(response, 'content') else str(response)
    
    # Try to extract JSON from the response
    json_data = extract_json_from_response(response_text)
    
    if json_data:
        # Validate required keys
        required_keys = ["data_quality_assessment", "statistical_summary", 
                        "outlier_detection", "feature_relationships", 
                        "pattern_trend_anomalies"]
        
        if all(key in json_data for key in required_keys):
            # Convert to the required nested format
            nested_dict = {
                key: {
                    "category": key.replace("_", " ").title(),
                    "questions": questions
                }
                for key, questions in json_data.items()
                if key in required_keys
            }
            print(json.dumps(nested_dict, indent=4))
            return nested_dict