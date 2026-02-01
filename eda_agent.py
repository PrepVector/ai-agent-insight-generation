from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM
from pydantic import BaseModel
from typing import List
import warnings
import os
import re
import time
import random
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

warnings.filterwarnings('ignore')
load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

class RateLimitException(Exception):
    pass

class QuestionList(BaseModel):
    questions: List[str]

@retry(
    retry=retry_if_exception_type((RateLimitException, Exception)),
    wait=wait_exponential(multiplier=2, min=10, max=120),
    stop=stop_after_attempt(5)
)
def create_llm():
    """Create LLM with retry logic"""
    try:
        return LLM(
            model="azure/gpt-4o",
            api_key=AZURE_OPENAI_API_KEY,
            base_url=AZURE_OPENAI_ENDPOINT,  # Changed from api_base
            api_version=AZURE_OPENAI_API_VERSION,
            temperature=0.01,
            max_tokens=4000
        )
    except Exception as e:
        if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
            wait_time = 20 + random.uniform(2, 10)
            print(f"Rate limit hit. Waiting {wait_time:.2f}s...")
            time.sleep(wait_time)
            raise RateLimitException(f"Rate limit: {str(e)}")
        else:
            wait_time = 10 + random.uniform(1, 5)
            print(f"Error creating LLM. Waiting {wait_time:.2f}s...")
            time.sleep(wait_time)
            raise

def create_data_scientist(category_name):
    """Create data scientist agent with unique identity per category"""
    return Agent(
        role=f"Data Scientist - {category_name}",
        goal=f"Execute Python code for {category_name} analysis and provide actual results with visualizations",
        backstory=f"Expert data scientist specialized in {category_name} analysis. You MUST execute code using the PythonREPL tool and provide actual numerical results, never placeholders.",
        verbose=True,
        llm=create_llm(),
        max_retry_limit=2,
        memory=False,
        allow_delegation=False
    )

def create_eda_task(questions_list, category_name, dataset_path, image_dir):
    """Create EDA task for a category with improved code generation"""
    questions_str = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions_list)])
    
    timestamp = int(time.time())
    unique_id = f"{category_name}_{timestamp}"
    
    # Dynamically generate code template for all questions
    code_template_sections = []
    for i, question in enumerate(questions_list):
        code_template_sections.append(f"""
        # Question {i+1} Analysis: {question}
        print("\\n==== Question {i+1} Analysis ====")
        # Add specific analysis code here based on the question: {question}
        # Always include print statements for results.
        
        # Create visualization for Question {i+1}
        plt.figure(figsize=(10, 6))
        # Add plotting code here for Question {i+1}
        plot_path = '{image_dir}/{category_name.replace(" ", "_")}_q{i+1}_analysis.png'
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to: {{plot_path}}")
        plt.close()
        """)
    
    full_code_template = f"""
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import os
        
        # Ensure directory exists
        os.makedirs('{image_dir}', exist_ok=True)
        
        # Load the dataset
        df = pd.read_csv('{dataset_path}')
        print(f"Dataset loaded successfully. Shape: {{df.shape}}")
        
        {''.join(code_template_sections)}
    """

    return Task(
        description=f"""
        CRITICAL: You MUST execute Python code using the PythonREPL tool and provide ACTUAL results.
        
        Analysis ID: {unique_id}
        Dataset: {dataset_path}
        Category: {category_name}
        Save images to: {image_dir}
        
        Questions to analyze:
        {questions_str}
        
        EXECUTION REQUIREMENTS:
        1. Use the PythonREPL tool to execute ALL code
        2. Generate ONE complete Python script that answers ALL questions listed above.
        3. MUST include print statements for all calculated values.
        4. MUST save plots with exact naming convention: `{{category_name.replace(" ", "_")}}_q{{question_number}}_analysis.png`
        5. MUST provide actual numerical results, never use placeholders.
        
        Code Template (EXECUTE THIS and fill in the details for each question):
        ```python
        {full_code_template}
        ```
        
        IMPORTANT: 
        - Execute the code using PythonREPL tool.
        - Capture and report ALL printed output.
        - Provide real numerical results, not placeholders.
        - Each question must have both analysis and visualization.
        """,
        expected_output=f"""
        You must execute the Python code using the PythonREPL tool and provide the actual execution results.

        Format your response as:

        ### Question 1
        - **[Question text]**

        #### Code
        ```python
        [The actual code that was executed for Question 1]
        ```

        #### Code Output
        ```
        [Real output from PythonREPL execution for Question 1 - actual numbers, not placeholders]
        ```

        #### Detailed Analysis
        [Analysis based on the actual results obtained for Question 1]

        #### Plots Generated
        ![Plot](images/[actual_filename_q1].png)

        ### Question 2
        - **[Question text]**

        #### Code
        ```python
        [The actual code that was executed for Question 2]
        ```

        #### Code Output
        ```
        [Real output from PythonREPL execution for Question 2 - actual numbers, not placeholders]
        ```

        #### Detailed Analysis
        [Analysis based on the actual results obtained for Question 2]

        #### Plots Generated
        ![Plot](images/[actual_filename_q2].png)

        [Repeat for all questions with actual executed results. Ensure each question has its own section with Code, Code Output, Detailed Analysis, and Plots Generated.]
        
        CRITICAL: Never use [calculated_value] or similar placeholders. All results must be actual numbers from code execution.
        """,
        agent=create_data_scientist(category_name)
    )

def extract_plot_paths(text):
    """Extract plot paths from result text and normalize them to images/ format"""
    patterns = [
        r'Plot saved to: ([^\n\r]+\.png)',
        r'Saved to: ([^\n\r]+\.png)',
        r'(eda_agent_report/images/[^\s]+\.png)',
        r'(images/[^\s]+\.png)'
    ]
    
    paths = []
    for pattern in patterns:
        found_paths = re.findall(pattern, text, re.IGNORECASE)
        paths.extend(found_paths)
    
    # Normalize all paths to images/ format
    normalized_paths = []
    for path in paths:
        path = path.replace('\\', '/')
        # Extract just the filename from full paths
        if 'eda_agent_report/images/' in path:
            filename = path.split('eda_agent_report/images/')[-1]
            normalized_paths.append(f'images/{filename}')
        elif path.startswith('images/'):
            normalized_paths.append(path)
        else:
            # If it's just a filename, assume it goes in images/
            filename = os.path.basename(path)
            normalized_paths.append(f'images/{filename}')
    
    return list(set(normalized_paths))

def embed_plots_in_report(result_text, plot_paths):
    """Embed plots in report at proper locations with proper spacing"""
    if not plot_paths:
        return result_text
    
    # Sort plot paths by question number to ensure correct order
    def extract_question_num(path):
        match = re.search(r'_q(\d+)_', path)
        return int(match.group(1)) if match else 999
    
    sorted_plots = sorted(plot_paths, key=extract_question_num)
    
    # Split by questions and embed plots
    sections = result_text.split('### Question')
    updated_result = sections[0] if sections else ""
    
    for i, section in enumerate(sections[1:], 1):
        # Find the corresponding plot for this question
        question_plot = None
        for plot in sorted_plots:
            if f'_q{i}_' in plot:
                question_plot = plot
                break
        
        # If this section has "Plots Generated" and we have a plot, embed it
        if "#### Plots Generated" in section and question_plot:
            # Split the section at "#### Plots Generated"
            parts = section.split("#### Plots Generated")
            if len(parts) > 1:
                before_plot = parts[0]
                after_plot = parts[1] if len(parts) > 1 else ""
                
                # Clean up the after_plot section - remove any existing plot references
                after_plot = re.sub(r'!\[Plot\]\([^)]+\)', '', after_plot)
                after_plot = after_plot.strip()
                
                # Add the section with the correct plot
                plot_section = f"#### Plots Generated\n![Plot]({question_plot})\n\n"
                
                # If there's content after the plot, add it with proper spacing
                if after_plot:
                    if not after_plot.startswith('###'):
                        plot_section += after_plot
                    else:
                        plot_section += after_plot
                
                updated_result += f"### Question{before_plot}{plot_section}"
            else:
                updated_result += f"### Question{section}"
        else:
            updated_result += f"### Question{section}"
    
    return updated_result

def validate_execution_results(result_text):
    """Validate that the results contain actual executed output, not placeholders"""
    placeholder_indicators = [
        "[calculated_value]",
        "[real_value]",
        "[actual_value]",
        "Execution completed.",
        "Code executed successfully."
    ]
    
    has_placeholders = any(indicator in result_text for indicator in placeholder_indicators)
    has_actual_numbers = bool(re.search(r'\d+\.\d+|\d+', result_text))
    
    return not has_placeholders and has_actual_numbers

def run_eda_analysis(dataset_path, questions_data, image_dir):
    """Run EDA analysis with improved execution validation"""
    
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs("eda_agent_report", exist_ok=True)
    
    from tools.custom_tool import PythonREPLTool # Assuming this tool is correctly defined elsewhere
    
    all_results = {}
    category_reports = {}
    
    print("Starting EDA analysis...")
    
    for idx, (category, data) in enumerate(questions_data.items()):
        print(f"\n=== Processing {data['category']} ({idx+1}/{len(questions_data)}) ===")
        
        if idx > 0:
            time.sleep(60)  # Rate limit prevention
        
        retry_count = 0
        max_retries = 2
        
        while retry_count < max_retries:
            try:
                # Create fresh instances for each category
                python_tool = PythonREPLTool()
                task = create_eda_task(data["questions"], data['category'], dataset_path, image_dir)
                agent = create_data_scientist(data['category'])
                agent.tools = [python_tool]
                task.agent = agent
                
                print(f"Starting analysis for {data['category']} (attempt {retry_count + 1})...")
                
                # Execute analysis with fresh crew
                crew = Crew(
                    agents=[agent], 
                    tasks=[task], 
                    process=Process.sequential, 
                    verbose=True,
                    memory=False
                )
                
                result = crew.kickoff()
                result_text = str(result)
                
                # Validate execution results
                if validate_execution_results(result_text):
                    print(f"✓ Execution validation passed for {data['category']}")
                    break
                else:
                    print(f"⚠️  Execution validation failed for {data['category']} (attempt {retry_count + 1})")
                    if retry_count < max_retries - 1:
                        retry_count += 1
                        time.sleep(30)
                        continue
                    else:
                        print(f"⚠️  Max retries reached for {data['category']}. Using available results.")
                
            except Exception as e:
                print(f"✗ Error in attempt {retry_count + 1} for {data['category']}: {str(e)}")
                if retry_count < max_retries - 1:
                    retry_count += 1
                    time.sleep(30)
                    continue
                else:
                    result_text = f"Error processing {data['category']}: {str(e)}"
            
            break
        
        # Process results
        plot_paths = extract_plot_paths(result_text)
        embedded_result = embed_plots_in_report(result_text, plot_paths)
        
        # Store results
        all_results[category] = {
            "questions": data["questions"],
            "result": embedded_result,
            "plots": plot_paths
        }
        
        # Create category report
        category_report = f"# {data['category']} Report\n\n{embedded_result}\n"
        category_reports[category] = category_report
        
        # Save individual category report
        filename = f"eda_agent_report/{data['category'].lower().replace(' ', '_')}_report.md"
        with open(filename, "w", encoding='utf-8') as f:
            f.write(category_report)
        
        print(f"✓ Completed {data['category']} - Generated {len(plot_paths)} plots")
    
    # Generate technical report
    technical_report = "# Exploratory Data Analysis Technical Report\n\n"
    technical_report += "## Executive Summary\n\n"
    technical_report += "Comprehensive EDA with automated analysis and visualizations.\n\n"
    technical_report += "## Analysis Results\n\n"
    
    for category, data in questions_data.items():
        if category in category_reports:
            content = category_reports[category].split('\n', 1)[1] if '\n' in category_reports[category] else category_reports[category]
            technical_report += f"## {data['category']}\n{content}\n---\n\n"
    
    # Save technical report
    with open("eda_agent_report/technical_report.md", "w", encoding='utf-8') as f:
        f.write(technical_report)
    
    print("✓ Technical report generated: eda_agent_report/technical_report.md")
    return technical_report
