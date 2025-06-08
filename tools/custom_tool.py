from crewai.tools import BaseTool
import sys
from io import StringIO
import contextlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats


import time

class PythonREPLTool(BaseTool):
    """
    Custom tool to execute Python code and return the result.
    """
    name: str = "PythonREPL"
    description: str = (
        "Executes Python code and returns output including print statements and results."
    )
    
    def _run(self, code: str) -> str:
        """
        Runs the provided Python code and returns the output.
        
        Args:
        code (str): Python code to execute.

        Returns:
        str: Output of the executed code or error message.
        """
        try:
            # Capture stdout to get print statements
            captured_output = StringIO()
            
            # Create execution environment with all necessary imports
            exec_globals = {
                '__builtins__': __builtins__,
                'pd': pd,
                'pandas': pd,
                'np': np,
                'numpy': np,
                'plt': plt,
                'matplotlib': plt,
                'sns': sns,
                'seaborn': sns,
                'sm': sm,
                'statsmodels': sm,
                'stats': stats,
                'scipy': stats,
                'time': time
            }
            exec_locals = {}
            
            # Redirect stdout to capture print statements
            with contextlib.redirect_stdout(captured_output):
                exec(code, exec_globals, exec_locals)
            
            # Get the captured output
            output = captured_output.getvalue()
            
            # If there's no print output but there might be a result variable, check for it
            if not output.strip() and 'result' in exec_locals:
                output = str(exec_locals['result'])
            elif not output.strip():
                output = "Code executed successfully."
            
            return output
            
        except Exception as e:
            return f"Error executing code: {str(e)}"
