#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modify the final_sp500_strategy_no_seasonality.py file to exclude seasonality scoring
"""

def modify_no_seasonality():
    # Read the file
    with open('final_sp500_strategy_no_seasonality.py', 'r') as f:
        content = f.read()
    
    # Find the _get_seasonality_score_from_data method
    seasonality_method_start = content.find("def _get_seasonality_score_from_data(self, row, seasonality_data):")
    if seasonality_method_start == -1:
        print("Could not find the _get_seasonality_score_from_data method")
        return
    
    # Find the end of the method
    method_body_start = content.find(":", seasonality_method_start) + 1
    indentation = "        "  # Assuming 8 spaces indentation for method body
    
    # Replace the method body with a simplified version that always returns 0.5 (neutral)
    new_method_body = f"""
{indentation}\"\"\"
{indentation}Modified version that excludes seasonality scoring.
{indentation}Always returns a neutral score of 0.5.
{indentation}\"\"\"
{indentation}return 0.5  # Neutral score, effectively disabling seasonality influence
"""
    
    # Find the end of the method by looking for the next method or end of class
    next_def = content.find("\n    def ", seasonality_method_start + 1)
    if next_def == -1:
        print("Could not find the end of the _get_seasonality_score_from_data method")
        return
    
    # Replace the method body
    modified_content = content[:method_body_start] + new_method_body + content[next_def:]
    
    # Update the run_backtest function to indicate no seasonality
    run_backtest_start = modified_content.find("def run_backtest(")
    if run_backtest_start == -1:
        print("Could not find the run_backtest function")
        return
    
    # Add a comment to indicate no seasonality
    run_backtest_line_end = modified_content.find("\n", run_backtest_start)
    modified_content = modified_content[:run_backtest_line_end] + " # No seasonality version" + modified_content[run_backtest_line_end:]
    
    # Add a log message to indicate no seasonality
    setup_pattern = "strategy = SP500Strategy(api=None, config=config, mode=mode, backtest_mode=True, backtest_start_date=start_date, backtest_end_date=end_date)"
    replacement = "strategy = SP500Strategy(api=None, config=config, mode=mode, backtest_mode=True, backtest_start_date=start_date, backtest_end_date=end_date)\n    logger.info(\"Running backtest WITHOUT seasonality scoring\")"
    modified_content = modified_content.replace(setup_pattern, replacement)
    
    # Write the modified content back to the file
    with open('final_sp500_strategy_no_seasonality.py', 'w') as f:
        f.write(modified_content)
    
    print("Modified final_sp500_strategy_no_seasonality.py to exclude seasonality scoring")

if __name__ == "__main__":
    modify_no_seasonality()
