# from crewai import Agent, Task, Crew, Process
# from langchain_groq import ChatGroq
# import pandas as pd
# import os
# import matplotlib.pyplot as plt
# import seaborn as sns
# import base64
# from io import BytesIO
# from datetime import datetime
# import numpy as np
# import time
# import random
# from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# # Import LiteLLM for exception handling
# try:
#     import litellm.exceptions
#     HAS_LITELLM = True
# except ImportError:
#     # If litellm is not installed, create a mock exception class
#     HAS_LITELLM = False
#     class litellm:
#         class exceptions:
#             class RateLimitError(Exception):
#                 pass

# # Define a custom exception for rate limits that will be used consistently
# class RateLimitException(Exception):
#     pass

# # Decorator to retry on RateLimitError with exponential backoff
# @retry(
#     retry=retry_if_exception_type((RateLimitException, Exception)),
#     wait=wait_exponential(multiplier=2, min=10, max=120),  # Increased backoff times
#     stop=stop_after_attempt(8)  # Increased max attempts
# )
# def create_llm_with_retry():
#     """Create an LLM instance with retry logic for rate limits"""
#     try:
#         # Try primary model
#         llm = ChatGroq(temperature=0, model_name="groq/llama3-70b-8192")
#         # Test the connection with a simple prompt to ensure it's working
#         return llm
#     except Exception as e:
#         error_str = str(e).lower()
#         # Check if it's a rate limit error
#         if "rate limit" in error_str or "too many requests" in error_str:
#             print(f"Rate limit hit: {str(e)}")
#             # Add jitter to avoid synchronized retries
#             wait_time = 20 + random.uniform(2, 10)
#             print(f"Waiting for {wait_time:.2f} seconds before retry...")
#             time.sleep(wait_time)
#             raise RateLimitException(f"Rate limit exceeded: {str(e)}")
#         else:
#             # For other errors
#             print(f"Error creating LLM: {str(e)}")
#             wait_time = 10 + random.uniform(1, 5)
#             print(f"Waiting for {wait_time:.2f} seconds before retry...")
#             time.sleep(wait_time)
#             raise  # Re-raise to let tenacity handle the retry

# def encode_image_to_base64(fig):
#     """Convert matplotlib figure to base64 encoded string for markdown embedding"""
#     buffer = BytesIO()
#     fig.savefig(buffer, format='png', bbox_inches='tight')
#     buffer.seek(0)
#     image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
#     plt.close(fig)  # Close the figure to free memory
#     return image_base64

# def run_eda_analysis(df, questions_dict):
#     """
#     Run EDA analysis by category using CrewAI agents with rate limit handling
    
#     Args:
#         df: Pandas DataFrame containing the data
#         questions_dict: Dictionary of questions organized by category
    
#     Returns:
#         A combined report as a string (not a tuple)
#     """
#     output_dir = "eda_agent_report"
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Create images directory if it doesn't exist
#     images_dir = os.path.join(output_dir, "images")
#     os.makedirs(images_dir, exist_ok=True)
    
#     all_category_reports = []
    
#     # First attempt to establish a connection to verify API works
#     print("Verifying API connection before starting analysis...")
#     try:
#         # Initial connection test with forced delay to warm up
#         time.sleep(5)  # Initial cooldown
#         llm = create_llm_with_retry()
#         print("API connection established successfully!")
#         time.sleep(10)  # Additional cooldown after successful connection
#     except Exception as e:
#         print(f"Warning: Initial API connection test failed: {str(e)}")
#         print("Will attempt to proceed with analysis anyway...")
#         time.sleep(30)  # Extended cooldown after failure
    
#     # Process each category separately to avoid rate limits
#     for category_key, category_data in questions_dict.items():
#         category_name = category_data["category"]
#         category_questions = category_data["questions"]
        
#         print(f"\n{'='*80}\nProcessing category: {category_name}\n{'='*80}")
        
#         # Wait between categories to prevent rate limits
#         if len(all_category_reports) > 0:
#             wait_time = 90  # Increased wait time between categories
#             print(f"Waiting {wait_time} seconds before processing next category...")
#             time.sleep(wait_time)
        
#         # Multiple attempts for processing a category
#         max_category_attempts = 3
#         for attempt in range(1, max_category_attempts + 1):
#             try:
#                 # Create a fresh LLM instance for each category
#                 llm = create_llm_with_retry()
#                 category_report = process_category(df, category_name, category_questions, llm)
#                 all_category_reports.append(f"## {category_name}\n\n{category_report}")
                
#                 # Save intermediate report
#                 category_filename = f"{category_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
#                 with open(os.path.join(output_dir, category_filename), "w") as f:
#                     f.write(f"# {category_name}\n\n{category_report}")
                    
#                 print(f"Completed analysis for {category_name}")
#                 break  # Successfully processed this category
                
#             except Exception as e:
#                 error_msg = f"Error processing category {category_name} (attempt {attempt}/{max_category_attempts}): {str(e)}"
#                 print(error_msg)
                
#                 if attempt == max_category_attempts:
#                     # All attempts failed, add error message to report
#                     all_category_reports.append(f"## {category_name}\n\n**Error:** {error_msg}\n\n")
#                 else:
#                     # Wait before retrying this category
#                     retry_wait = 60 * attempt  # Increase wait time with each attempt
#                     print(f"Waiting {retry_wait} seconds before retrying category...")
#                     time.sleep(retry_wait)
    
#     # Combine all category reports
#     combined_report = f"# Exploratory Data Analysis Report\n\nGenerated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
#     combined_report += "\n\n".join(all_category_reports)
    
#     # Save the combined report
#     report_path = os.path.join(output_dir, f"full_eda_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
#     with open(report_path, "w") as f:
#         f.write(combined_report)
    
#     print(f"\nComplete EDA report generated at: {report_path}")
    
#     # Return only the combined report string, not a tuple
#     return combined_report

# def process_category(df, category_name, category_questions, llm):
#     """Process a single category of questions"""
#     print(f"Starting analysis for {category_name} with {len(category_questions)} questions")
    
#     # Create a sample dataset description and code for handling visualizations
#     dataset_description = f"""
#     Dataset information:
#     - Rows: {df.shape[0]}
#     - Columns: {df.shape[1]}
#     - Column names: {', '.join(df.columns)}
#     - Sample data (5 rows): 
#     {df.head(5).to_string()}
    
#     # Helper function for handling visualizations
#     ```python
#     def encode_image_to_base64(fig):
#         buffer = BytesIO()
#         fig.savefig(buffer, format='png', bbox_inches='tight')
#         buffer.seek(0)
#         image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
#         plt.close(fig)  # Close the figure to free memory
#         return image_base64
#     ```
#     """
    
#     # Add wait time before creating agents to help avoid rate limits
#     time.sleep(5)
    
#     # Define the agents for this category
#     feature_engineer = Agent(
#         role="Feature Engineer",
#         goal=f"Create necessary features for {category_name} analysis",
#         backstory="An expert data scientist specialized in feature engineering and data transformation",
#         llm=llm,
#         verbose=True
#     )
    
#     # Add delay between agent creations
#     time.sleep(2)
    
#     data_analyst = Agent(
#         role="Data Analyst",
#         goal=f"Perform {category_name} analysis on the dataset",
#         backstory="An expert data analyst skilled in uncovering insights from data",
#         llm=llm,
#         verbose=True
#     )
    
#     # Add delay between agent creations
#     time.sleep(2)
    
#     viz_expert = Agent(
#         role="Visualization Expert",
#         goal=f"Create informative visualizations for {category_name} analysis",
#         backstory="An expert in data visualization with years of experience in communicating insights through charts",
#         llm=llm,
#         verbose=True
#     )
    
#     # Add delay after creating agents
#     time.sleep(5)
    
#     # Create a feature engineering task
#     feature_engineering_task = Task(
#         description=f"""
#         You are provided with a dataset for EDA analysis.
        
#         {dataset_description}
        
#         You will be focusing on the '{category_name}' category with these questions:
#         {chr(10).join([f"{idx+1}. {q}" for idx, q in enumerate(category_questions)])}
        
#         Based on these questions, determine if any new features need to be created.
#         If so, create a Python function that will add these features to the dataset.
        
#         Your output should be the Python code to add these features, with detailed explanations.
#         """,
#         agent=feature_engineer,
#         expected_output="Python code to add necessary features to the dataset with explanations"
#     )
    
#     # Create an EDA task
#     eda_task = Task(
#         description=f"""
#         You will analyze the dataset focusing on the '{category_name}' category.
        
#         {dataset_description}
        
#         Please address the following specific questions:
#         {chr(10).join([f"{idx+1}. {q}" for idx, q in enumerate(category_questions)])}
        
#         For each question:
#         1. Generate and execute the appropriate Python code
#         2. Include detailed analysis and interpretations
        
#         Make sure to handle missing values and outliers appropriately.
#         Include the full Python code for each analysis.
#         Keep your responses focused specifically on these questions.
        
#         Use the feature engineering results from your colleague to enhance your analysis.
#         """,
#         agent=data_analyst,
#         expected_output=f"Detailed analysis of {category_name} questions with Python code and interpretations"
#     )
#     imagepath_dir = "eda_agent_report/images"
    
#     # Create a visualization task
#     viz_task = Task(
#         description=f"""
#         Based on the EDA results for the '{category_name}' category, create appropriate visualizations.
        
#         {dataset_description}
        
#         Focus on these questions:
#         {chr(10).join([f"{idx+1}. {q}" for idx, q in enumerate(category_questions)])}
        
#         For each visualization:
#         1. Generate the appropriate Python code using matplotlib and seaborn
#         2. Ensure each plot has a title, labeled axes, and appropriate annotations
#         3. Add a clear interpretation for each visualization
        
#         For each visualization, include placeholder text as follows:
#         ```
#         ![Visualization Title](data:image/png;base64,{{plot_base64}})
#         ```
        
#         This will be replaced with the actual encoded image in the final report.

#         - The plot must be saved in png format at the below directory. Add plt.savefig() with the location below after the plots is created.  
#         <imagepath> 
#           {imagepath_dir} 
#         </imagepath>
        
#         Example of how to create a visualization:
#         ```python
#         import matplotlib.pyplot as plt
#         import seaborn as sns
#         import base64
#         from io import BytesIO

#         fig, ax = plt.subplots(figsize=(10, 6))
#         sns.histplot(data=df, x='column_name', kde=True, ax=ax)
#         ax.set_title('Distribution of Column')
#         ax.set_xlabel('Values')
#         ax.set_ylabel('Frequency')
#         ax.annotate('Key observation about the data', xy=(0.5, 0.9), xycoords='axes fraction', ha='center')
#         plot_base64 = encode_image_to_base64(fig)
#         ```
        
#         For each visualization, provide a detailed interpretation of what the visualization shows and what insights can be drawn from it.
#         """,
#         agent=viz_expert,
#         expected_output=f"Visualizations for {category_name} analysis with interpretations"
#     )
    
#     # Assemble the crew for this category
#     crew = Crew(
#         agents=[feature_engineer, data_analyst, viz_expert],
#         tasks=[feature_engineering_task, eda_task, viz_task],
#         process=Process.sequential,
#         verbose=True
#     )
    
#     # Execute the crew with error handling and additional wait time between tasks
#     try:
#         # Add retry mechanism specifically for crew kickoff
#         max_crew_attempts = 3
#         for attempt in range(1, max_crew_attempts + 1):
#             try:
#                 print(f"Starting crew execution (attempt {attempt}/{max_crew_attempts})...")
#                 crew_output = crew.kickoff()
#                 return str(crew_output)
#             except Exception as e:
#                 error_str = str(e).lower()
#                 if attempt < max_crew_attempts:
#                     # Check if it's a rate limit error
#                     if "rate limit" in error_str or "too many requests" in error_str:
#                         wait_time = 120 * attempt  # Progressive backoff
#                         print(f"Rate limit hit. Waiting {wait_time} seconds before retry...")
#                     else:
#                         wait_time = 60 * attempt
#                         print(f"Error during crew execution: {str(e)}")
#                         print(f"Waiting {wait_time} seconds before retry...")
                    
#                     time.sleep(wait_time)
#                 else:
#                     # Last attempt failed, re-raise the exception
#                     raise
        
#         # Should not reach here due to raise in the loop
#         return "Error: Failed to execute crew tasks"
        
#     except Exception as e:
#         # If we encounter any error, capture it and return a helpful message
#         error_msg = str(e)
#         return f"""
#         **Error Encountered During Analysis**        
#         The analysis for this category was interrupted due to an error.        
#         Error details: {error_msg}        
#         Please check your dataset and try again.
#         """

from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from datetime import datetime
import numpy as np
import time
import random
import re
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import re
import os
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Import LiteLLM for exception handling
try:
    import litellm.exceptions
    HAS_LITELLM = True
except ImportError:
    # If litellm is not installed, create a mock exception class
    HAS_LITELLM = False
    class litellm:
        class exceptions:
            class RateLimitError(Exception):
                pass

# Define a custom exception for rate limits that will be used consistently
class RateLimitException(Exception):
    pass

# Decorator to retry on RateLimitError with exponential backoff
@retry(
    retry=retry_if_exception_type((RateLimitException, Exception)),
    wait=wait_exponential(multiplier=2, min=10, max=120),  # Increased backoff times
    stop=stop_after_attempt(8)  # Increased max attempts
)
def create_llm_with_retry():
    """Create an LLM instance with retry logic for rate limits"""
    try:
        # Try primary model
        llm = ChatGroq(temperature=0, model_name="groq/llama3-70b-8192")
        # Test the connection with a simple prompt to ensure it's working
        return llm
    except Exception as e:
        error_str = str(e).lower()
        # Check if it's a rate limit error
        if "rate limit" in error_str or "too many requests" in error_str:
            print(f"Rate limit hit: {str(e)}")
            # Add jitter to avoid synchronized retries
            wait_time = 20 + random.uniform(2, 10)
            print(f"Waiting for {wait_time:.2f} seconds before retry...")
            time.sleep(wait_time)
            raise RateLimitException(f"Rate limit exceeded: {str(e)}")
        else:
            # For other errors
            print(f"Error creating LLM: {str(e)}")
            wait_time = 10 + random.uniform(1, 5)
            print(f"Waiting for {wait_time:.2f} seconds before retry...")
            time.sleep(wait_time)
            raise  # Re-raise to let tenacity handle the retry

import re
import os

def fix_markdown_image_paths(markdown_content, base_path='eda_agent_report/images'):
    """
    Fix image paths in markdown content to ensure they are correctly formatted.
    
    Args:
        markdown_content (str): The markdown content to fix
        base_path (str): The base path where images should be stored
        
    Returns:
        str: Fixed markdown content
    """
    # Pattern to match markdown image references
    img_pattern = r'!\[(.*?)\]\((.*?)\)'
    
    def fix_path(match):
        alt_text = match.group(1)
        path = match.group(2)
        
        # Normalize path (convert backslashes to forward slashes)
        path = path.replace('\\', '/')
        
        # Add base path if not present
        if not path.startswith('eda_agent_report/'):
            if path.startswith('images/'):
                path = f"eda_agent_report/{path}"
            else:
                # Extract the filename if it's a direct reference
                filename = os.path.basename(path)
                path = f"{base_path}/{filename}"
        
        return f"![{alt_text}]({path})"
    
    # Fix all image references
    fixed_content = re.sub(img_pattern, fix_path, markdown_content)
    
    # Remove duplicate image references (when there are two consecutive references to the same image)
    duplicate_pattern = r'(!\[.*?\]\(.*?\))\s*\n+\s*(!\[.*?\]\(.*?\))'
    
    while re.search(duplicate_pattern, fixed_content):
        fixed_content = re.sub(duplicate_pattern, r'\1', fixed_content)
    
    return fixed_content

def process_report_file(input_file, output_file=None):
    """
    Process a markdown report file to fix image paths
    
    Args:
        input_file (str): Path to the input markdown file
        output_file (str, optional): Path to save the fixed file. If None, will create a new file with '_fixed' suffix
    
    Returns:
        str: Path to the fixed file
    """
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_fixed{ext}"
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    fixed_content = fix_markdown_image_paths(content)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print(f"Fixed report saved to: {output_file}")
    return output_file

# def encode_image_to_base64(fig):
#     """Convert matplotlib figure to base64 encoded string for markdown embedding"""
#     buffer = BytesIO()
#     fig.savefig(buffer, format='png', bbox_inches='tight')
#     buffer.seek(0)
#     image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
#     plt.close(fig)  # Close the figure to free memory
#     return image_base64
def encode_image_to_base64(fig):
    """Convert matplotlib figure to base64 encoded string for markdown embedding"""
    buffer = BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)  # Close the figure to free memory
    return image_base64

# def execute_plot_code(plot_code, plot_idx, category_name, images_dir, df):
#     """
#     Execute the plot code and save the resulting image to the specified directory
#     Returns the path to the saved image
#     """
#     # Make a clean namespace to prevent variable conflicts
#     exec_namespace = {
#         'pd': pd,
#         'plt': plt,
#         'sns': sns,
#         'np': np,
#         'df': df,  # Pass the actual dataframe to the namespace
#         'encode_image_to_base64': encode_image_to_base64,
#         'BytesIO': BytesIO,
#         'base64': base64
#     }
    
#     # Replace the base64 placeholder with actual file saving
#     safe_category = category_name.replace(' ', '_').lower()
#     filename = f"{safe_category}_plot_{plot_idx}.png"
#     filepath = os.path.join(images_dir, filename)
    
#     # FIX: Replace generic df loading with using the passed dataframe
#     plot_code = re.sub(r"df\s*=\s*pd\.read_csv\(['\"]dataset\.csv['\"]\)", "", plot_code)
    
#     # Add code to save the figure
#     if 'plt.savefig' not in plot_code:
#         # Add code to save the figure before any plt.close() calls
#         save_code = f"\nplt.savefig('{filepath}', bbox_inches='tight', dpi=300)\n"
#         if 'plt.close' in plot_code:
#             plot_code = plot_code.replace('plt.close', f"\nplt.savefig('{filepath}', bbox_inches='tight', dpi=300)\nplt.close")
#         else:
#             plot_code += save_code
    
#     try:
#         print(f"Executing plot code for {category_name}, plot #{plot_idx}")
#         print(f"Saving plot to: {filepath}")
#         # FIX: Ensure directory exists
#         os.makedirs(os.path.dirname(filepath), exist_ok=True)
#         exec(plot_code, exec_namespace)
#         # FIX: Verify file was created
#         if os.path.exists(filepath):
#             print(f"Successfully saved plot to {filepath}")
#         else:
#             print(f"WARNING: Plot file was not created at {filepath}")
#         return filepath
#     except Exception as e:
#         print(f"Error executing plot code: {str(e)}")
#         print(f"Problematic code:\n{plot_code}")
#         return None
def execute_plot_code(plot_code, plot_idx, category_name, images_dir, df):
    """
    Execute the plot code and save the resulting image to the specified directory
    Returns the path to the saved image
    """
    # Make a clean namespace to prevent variable conflicts
    exec_namespace = {
        'pd': pd,
        'plt': plt,
        'sns': sns,
        'np': np,
        'df': df,  # Pass the actual dataframe to the namespace
        'encode_image_to_base64': encode_image_to_base64,
        'BytesIO': BytesIO,
        'base64': base64
    }
    
    # Replace the base64 placeholder with actual file saving
    safe_category = category_name.replace(' ', '_').lower()
    filename = f"{safe_category}_plot_{plot_idx}.png"
    filepath = os.path.join(images_dir, filename)
    
    # Ensure consistent forward slash path format
    filepath = filepath.replace('\\', '/')
    
    # FIX: Replace generic df loading with using the passed dataframe
    plot_code = re.sub(r"df\s*=\s*pd\.read_csv\(['\"]dataset\.csv['\"]\)", "", plot_code)
    plot_code = re.sub(r"df\s*=\s*pd\.read_csv\(['\"]eda_agent_report\\temp_dataset\.csv['\"]\)", "", plot_code)
    
    # Add code to save the figure
    if 'plt.savefig' not in plot_code:
        # Add code to save the figure before any plt.close() calls
        save_code = f"\nplt.savefig('{filepath}', bbox_inches='tight', dpi=300)\n"
        if 'plt.close' in plot_code:
            plot_code = plot_code.replace('plt.close', f"\nplt.savefig('{filepath}', bbox_inches='tight', dpi=300)\nplt.close")
        else:
            plot_code += save_code
    else:
        # Fix any existing savefig paths to use consistent format
        plot_code = re.sub(
            r"plt\.savefig\(['\"].*?['\"]", 
            f"plt.savefig('{filepath}'", 
            plot_code
        )
    
    try:
        print(f"Executing plot code for {category_name}, plot #{plot_idx}")
        print(f"Saving plot to: {filepath}")
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        exec(plot_code, exec_namespace)
        # Verify file was created
        if os.path.exists(filepath):
            print(f"Successfully saved plot to {filepath}")
        else:
            print(f"WARNING: Plot file was not created at {filepath}")
        return filepath
    except Exception as e:
        print(f"Error executing plot code: {str(e)}")
        print(f"Problematic code:\n{plot_code}")
        return None

def enhanced_process_report_for_images(report_text, images_dir, category_name, df):
    """
    Enhanced version of process_report_for_images that fixes path consistency issues
    and properly handles image references
    """
    # Updated pattern to match Python code blocks
    code_block_pattern = r'```python(.*?)```'
    plot_blocks = re.findall(code_block_pattern, report_text, re.DOTALL)
    
    updated_report = report_text
    
    # Track which code blocks have been processed
    processed_blocks = []
    
    for i, code_block in enumerate(plot_blocks):
        # Skip code blocks that don't appear to be plot-related
        if not any(plot_term in code_block.lower() for plot_term in ['plt', 'plot', 'figure', 'ax', 'sns.']):
            continue
            
        try:
            # Execute the code and get the image filepath
            filepath = execute_plot_code(code_block, i+1, category_name, images_dir, df)
            
            if filepath:
                # Ensure path uses forward slashes for markdown
                filepath = filepath.replace('\\', '/')
                
                # Get relative path for the markdown file
                relative_path = os.path.relpath(filepath, start=os.path.dirname(images_dir))
                relative_path = relative_path.replace('\\', '/')
                
                # Create the image reference
                image_ref = f"\n\n![{category_name} Plot {i+1}]({relative_path})\n"
                
                # Find if there's already an image reference after this code block
                full_code_block = f"```python{code_block}```"
                
                # First check if there's already an image reference with this plot number
                existing_ref_pattern = re.escape(full_code_block) + r"\s*\n+\s*!\[.*?Plot\s+" + str(i+1) + r"\]"
                if re.search(existing_ref_pattern, updated_report):
                    # Already has a reference, need to fix it if needed
                    old_ref_pattern = re.escape(full_code_block) + r"\s*\n+\s*(!\[.*?Plot\s+" + str(i+1) + r"\]\(.*?\))"
                    if re.search(old_ref_pattern, updated_report):
                        # Replace the old reference with the new one
                        updated_report = re.sub(
                            old_ref_pattern,
                            f"{full_code_block}\n\n![{category_name} Plot {i+1}]({relative_path})",
                            updated_report
                        )
                else:
                    # No existing reference, add a new one
                    code_with_image = f"{full_code_block}{image_ref}"
                    updated_report = updated_report.replace(full_code_block, code_with_image)
                
                # Mark this block as processed
                processed_blocks.append(code_block)
        
        except Exception as e:
            print(f"Error processing plot #{i+1}: {str(e)}")
    
    # Clean up any duplicate or wrong image references
    img_pattern = r'!\[(.*?)\]\((.*?)\)'
    matches = re.findall(img_pattern, updated_report)
    
    # Create a set of already processed images
    processed_images = set()
    
    for alt_text, path in matches:
        normalized_path = path.replace('\\', '/')
        
        # If this is a duplicate path or has backslashes, fix it
        if normalized_path in processed_images or '\\' in path:
            # Replace incorrect reference with correct one
            updated_report = updated_report.replace(
                f"![{alt_text}]({path})",
                f"![{alt_text}]({normalized_path})"
            )
        
        processed_images.add(normalized_path)
    
    return updated_report

def process_report_for_images(report_text, images_dir, category_name, df):
    """
    Process the report text to identify plot code blocks, execute them,
    and update the markdown to include the images.
    """
    # Updated pattern to match Python code blocks
    code_block_pattern = r'```python(.*?)```'
    plot_blocks = re.findall(code_block_pattern, report_text, re.DOTALL)
    
    updated_report = report_text
    
    for i, code_block in enumerate(plot_blocks):
        # Skip code blocks that don't appear to be plot-related
        if not any(plot_term in code_block.lower() for plot_term in ['plt', 'plot', 'figure', 'ax']):
            continue
            
        try:
            # Execute the code and get the image filepath
            filepath = execute_plot_code(code_block, i+1, category_name, images_dir, df)
            
            if filepath:
                # Replace the base64 placeholder pattern if it exists
                placeholder_pattern = r'!\[.*?\]\(data:image/png;base64,\{plot_base64\}\)'
                if re.search(placeholder_pattern, updated_report):
                    relative_path = os.path.relpath(filepath, start=os.path.dirname(images_dir))
                    updated_report = re.sub(
                        placeholder_pattern,
                        f"![{category_name} Plot {i+1}]({relative_path})",
                        updated_report,
                        count=1
                    )
                
                # If we didn't find a placeholder, add the image reference after the code block
                else:
                    relative_path = os.path.relpath(filepath, start=os.path.dirname(images_dir))
                    image_ref = f"\n\n![{category_name} Plot {i+1}]({relative_path})\n"
                    code_with_image = f"```python{code_block}```{image_ref}"
                    updated_report = updated_report.replace(f"```python{code_block}```", code_with_image)
        
        except Exception as e:
            print(f"Error processing plot #{i+1}: {str(e)}")
    
    return updated_report

def run_eda_analysis(df, questions_dict):
    """
    Run EDA analysis by category using CrewAI agents with rate limit handling
    
    Args:
        df: Pandas DataFrame containing the data
        questions_dict: Dictionary of questions organized by category
    
    Returns:
        A combined report as a string (not a tuple)
    """
    import os
    from datetime import datetime
    import time
    
    output_dir = "eda_agent_report"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create images directory if it doesn't exist
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    all_category_reports = []
    
    # First attempt to establish a connection to verify API works
    print("Verifying API connection before starting analysis...")
    try:
        # Initial connection test with forced delay to warm up
        time.sleep(5)  # Initial cooldown
        llm = create_llm_with_retry()
        print("API connection established successfully!")
        time.sleep(10)  # Additional cooldown after successful connection
    except Exception as e:
        print(f"Warning: Initial API connection test failed: {str(e)}")
        print("Will attempt to proceed with analysis anyway...")
        time.sleep(30)  # Extended cooldown after failure
    
    # Save dataset to a temp file for agent access - using forward slashes consistently
    dataset_path = os.path.join(output_dir, "temp_dataset.csv").replace('\\', '/')
    df.to_csv(dataset_path, index=False)
    
    # Process each category separately to avoid rate limits
    for category_key, category_data in questions_dict.items():
        category_name = category_data["category"]
        category_questions = category_data["questions"]
        
        print(f"\n{'='*80}\nProcessing category: {category_name}\n{'='*80}")
        
        # Wait between categories to prevent rate limits
        if len(all_category_reports) > 0:
            wait_time = 90  # Increased wait time between categories
            print(f"Waiting {wait_time} seconds before processing next category...")
            time.sleep(wait_time)
        
        # Multiple attempts for processing a category
        max_category_attempts = 3
        for attempt in range(1, max_category_attempts + 1):
            try:
                # Create a fresh LLM instance for each category
                llm = create_llm_with_retry()
                # Pass the dataset path to the process_category function
                category_report = process_category(df, category_name, category_questions, llm, images_dir, dataset_path)
                
                # Process the report to extract and execute plot code, and update image references
                # Use our enhanced version of the function
                processed_report = enhanced_process_report_for_images(category_report, images_dir, category_name, df)
                
                all_category_reports.append(f"## {category_name}\n\n{processed_report}")
                
                # Save intermediate report with consistent path format
                category_filename = f"{category_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                with open(os.path.join(output_dir, category_filename), "w") as f:
                    f.write(f"# {category_name}\n\n{processed_report}")
                    
                print(f"Completed analysis for {category_name}")
                break  # Successfully processed this category
                
            except Exception as e:
                error_msg = f"Error processing category {category_name} (attempt {attempt}/{max_category_attempts}): {str(e)}"
                print(error_msg)
                
                if attempt == max_category_attempts:
                    # All attempts failed, add error message to report
                    all_category_reports.append(f"## {category_name}\n\n**Error:** {error_msg}\n\n")
                else:
                    # Wait before retrying this category
                    retry_wait = 60 * attempt  # Increase wait time with each attempt
                    print(f"Waiting {retry_wait} seconds before retrying category...")
                    time.sleep(retry_wait)
    
    # Combine all category reports
    combined_report = f"# Exploratory Data Analysis Report\n\nGenerated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    combined_report += "\n\n".join(all_category_reports)
    
    # Fix any remaining path issues in the combined report
    combined_report = fix_markdown_image_paths(combined_report)
    
    # Save the combined report with consistent path format
    report_path = os.path.join(output_dir, f"full_eda_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md").replace('\\', '/')
    with open(report_path, "w") as f:
        f.write(combined_report)
    
    print(f"\nComplete EDA report generated at: {report_path}")
    
    # Clean up temp dataset file
    try:
        os.remove(dataset_path)
    except:
        pass
    
    # Return only the combined report string
    return combined_report

def process_category(df, category_name, category_questions, llm, images_dir, dataset_path):
    """Process a single category of questions"""
    print(f"Starting analysis for {category_name} with {len(category_questions)} questions")
    
    # Create a sample dataset description and code for handling visualizations
    dataset_description = f"""
    Dataset information:
    - Rows: {df.shape[0]}
    - Columns: {df.shape[1]}
    - Column names: {', '.join(df.columns)}
    - Sample data (5 rows): 
    {df.head(5).to_string()}
    
    # FIX: Add actual dataset path for the agents to use
    The dataset is available at: '{dataset_path}'
    
    # Helper function for handling visualizations
    ```python
    def encode_image_to_base64(fig):
        buffer = BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)  # Close the figure to free memory
        return image_base64
    ```
    """
    
    # Add wait time before creating agents to help avoid rate limits
    time.sleep(5)
    
    # Define the agents for this category
    feature_engineer = Agent(
        role="Feature Engineer",
        goal=f"Create necessary features for {category_name} analysis",
        backstory="An expert data scientist specialized in feature engineering and data transformation",
        llm=llm,
        verbose=True
    )
    
    # Add delay between agent creations
    time.sleep(2)
    
    data_analyst = Agent(
        role="Data Analyst",
        goal=f"Perform {category_name} analysis on the dataset",
        backstory="An expert data analyst skilled in uncovering insights from data",
        llm=llm,
        verbose=True
    )
    
    # Add delay between agent creations
    time.sleep(2)
    
    viz_expert = Agent(
        role="Visualization Expert",
        goal=f"Create informative visualizations for {category_name} analysis",
        backstory="An expert in data visualization with years of experience in communicating insights through charts",
        llm=llm,
        verbose=True
    )
    
    # Add delay after creating agents
    time.sleep(5)
    
    # Create a feature engineering task
    feature_engineering_task = Task(
        description=f"""
        You are provided with a dataset for EDA analysis.
        
        {dataset_description}
        
        You will be focusing on the '{category_name}' category with these questions:
        {chr(10).join([f"{idx+1}. {q}" for idx, q in enumerate(category_questions)])}
        
        Based on these questions, determine if any new features need to be created.
        If so, create a Python function that will add these features to the dataset.
        
        IMPORTANT: When loading the dataset, use this code:
        ```python
        df = pd.read_csv('{dataset_path}')
        ```
        
        Your output should be the Python code to add these features, with detailed explanations.
        """,
        agent=feature_engineer,
        expected_output="Python code to add necessary features to the dataset with explanations"
    )
    
    # Create an EDA task
    eda_task = Task(
        description=f"""
        You will analyze the dataset focusing on the '{category_name}' category.
        
        {dataset_description}
        
        Please address the following specific questions:
        {chr(10).join([f"{idx+1}. {q}" for idx, q in enumerate(category_questions)])}
        
        For each question:
        1. Generate and execute the appropriate Python code
        2. Include detailed analysis and interpretations
        
        Make sure to handle missing values and outliers appropriately.
        Include the full Python code for each analysis.
        Keep your responses focused specifically on these questions.
        
        IMPORTANT: When loading the dataset, use this code:
        ```python
        df = pd.read_csv('{dataset_path}')
        ```
        
        Use the feature engineering results from your colleague to enhance your analysis.
        """,
        agent=data_analyst,
        expected_output=f"Detailed analysis of {category_name} questions with Python code and interpretations"
    )
    
    # Create a visualization task with updated instructions
    viz_task = Task(
        description=f"""
        Based on the EDA results for the '{category_name}' category, create appropriate visualizations.
        
        {dataset_description}
        
        Focus on these questions:
        {chr(10).join([f"{idx+1}. {q}" for idx, q in enumerate(category_questions)])}
        
        For each visualization:
        1. Generate the appropriate Python code using matplotlib and seaborn
        2. Ensure each plot has a title, labeled axes, and appropriate annotations
        3. Add a clear interpretation for each visualization
        
        IMPORTANT: When loading the dataset, use this code:
        ```python
        df = pd.read_csv('{dataset_path}')
        ```
        
        IMPORTANT: Each plot must be saved to a file. Make sure to include this line in your code:
        plt.savefig('eda_agent_report/images/{category_name.replace(' ', '_').lower()}_plot_X.png', bbox_inches='tight', dpi=300)
        (Replace X with the plot number)
        
        For each visualization, include the following line after the plot code:
        ![{category_name} Plot X](eda_agent_report/images/{category_name.replace(' ', '_').lower()}_plot_X.png)
        (Replace X with the plot number)
        
        Example of how to create a visualization:
        ```python
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        # Load the dataset
        df = pd.read_csv('{dataset_path}')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=df, x='column_name', kde=True, ax=ax)
        ax.set_title('Distribution of Column')
        ax.set_xlabel('Values')
        ax.set_ylabel('Frequency')
        ax.annotate('Key observation about the data', xy=(0.5, 0.9), xycoords='axes fraction', ha='center')
        plt.savefig('eda_agent_report/images/{category_name.replace(' ', '_').lower()}_plot_1.png', bbox_inches='tight', dpi=300)
        plt.close(fig)
        ```
        
        For each visualization, provide a detailed interpretation of what the visualization shows and what insights can be drawn from it.
        """,
        agent=viz_expert,
        expected_output=f"Visualizations for {category_name} analysis with interpretations"
    )
    
    # Assemble the crew for this category
    crew = Crew(
        agents=[feature_engineer, data_analyst, viz_expert],
        tasks=[feature_engineering_task, eda_task, viz_task],
        process=Process.sequential,
        verbose=True
    )
    
    # Execute the crew with error handling and additional wait time between tasks
    try:
        # Add retry mechanism specifically for crew kickoff
        max_crew_attempts = 3
        for attempt in range(1, max_crew_attempts + 1):
            try:
                print(f"Starting crew execution (attempt {attempt}/{max_crew_attempts})...")
                crew_output = crew.kickoff()
                return str(crew_output)
            except Exception as e:
                error_str = str(e).lower()
                if attempt < max_crew_attempts:
                    # Check if it's a rate limit error
                    if "rate limit" in error_str or "too many requests" in error_str:
                        wait_time = 120 * attempt  # Progressive backoff
                        print(f"Rate limit hit. Waiting {wait_time} seconds before retry...")
                    else:
                        wait_time = 60 * attempt
                        print(f"Error during crew execution: {str(e)}")
                        print(f"Waiting {wait_time} seconds before retry...")
                    
                    time.sleep(wait_time)
                else:
                    # Last attempt failed, re-raise the exception
                    raise
        
        # Should not reach here due to raise in the loop
        return "Error: Failed to execute crew tasks"
        
    except Exception as e:
        # If we encounter any error, capture it and return a helpful message
        error_msg = str(e)
        return f"""
        **Error Encountered During Analysis**        
        The analysis for this category was interrupted due to an error.        
        Error details: {error_msg}        
        Please check your dataset and try again.
        """

# Function to extract and execute plot code from existing reports
def process_existing_reports(output_dir="eda_agent_report", df=None):
    """
    Process existing report files to find plot code, execute it,
    and update the reports with proper image references
    """
    # FIX: Verify df is provided or load it if not
    if df is None:
        print("Error: DataFrame must be provided for processing existing reports")
        return
        
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Find all markdown files in the output directory
    md_files = [f for f in os.listdir(output_dir) if f.endswith('.md')]
    
    for file in md_files:
        file_path = os.path.join(output_dir, file)
        
        # Extract category name from filename
        category_name = file.split('_')[0].replace('_', ' ').title()
        
        with open(file_path, 'r') as f:
            report_content = f.read()
        
        # Process the report to extract plots, execute code, and update references
        updated_report = process_report_for_images(report_content, images_dir, category_name, df)
        
        # Save the updated report
        updated_file_path = os.path.join(output_dir, f"updated_{file}")
        with open(updated_file_path, 'w') as f:
            f.write(updated_report)
        
        print(f"Updated report saved to {updated_file_path}")