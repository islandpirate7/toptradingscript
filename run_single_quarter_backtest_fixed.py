#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run a backtest for a single quarter and check the results.
This script runs a backtest for Q1 2023 and checks if the continuous capital is working correctly.
"""

import os
import json
import time
import requests
import logging
import glob
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_backtest(quarter, initial_capital=1000, max_signals=100, weekly_selection=True):
    """Run a backtest for a specific quarter."""
    url = "http://localhost:5000/run_comprehensive_backtest"
    payload = {
        "quarters": [quarter],
        "max_signals": max_signals,
        "initial_capital": initial_capital,
        "weekly_selection": weekly_selection,
        "continuous_capital": False  # We're manually handling the continuous capital
    }
    
    logger.info(f"Running backtest for {quarter} with initial capital ${initial_capital}")
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        logger.info(f"Backtest for {quarter} started successfully: {result}")
        return result
    else:
        logger.error(f"Failed to start backtest for {quarter}: {response.text}")
        return None

def wait_for_backtest_completion(run_id, timeout=600):  # Increased timeout to 10 minutes
    """Wait for the backtest to complete."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        # Check if result files with the run_id exist
        files = glob.glob(f"backtest_results/*_{run_id}.json")
        if files:
            logger.info(f"Backtest completed, found result files: {files}")
            return files[0]  # Return the first result file
        
        # Also check for combined result file
        combined_file = f"backtest_results/combined_backtest_results_{run_id}.json"
        if os.path.exists(combined_file):
            logger.info(f"Found combined result file: {combined_file}")
            return combined_file
        
        # Check web_interface directory as well
        web_files = glob.glob(f"web_interface/backtest_results/*_{run_id}.json")
        if web_files:
            logger.info(f"Backtest completed, found result files in web_interface: {web_files}")
            return web_files[0]
        
        web_combined = f"web_interface/backtest_results/combined_backtest_results_{run_id}.json"
        if os.path.exists(web_combined):
            logger.info(f"Found combined result file in web_interface: {web_combined}")
            return web_combined
        
        # Check for backtest log file to see if it's still running
        log_file = f"logs/test_comprehensive_backtest_{run_id}.log"
        if os.path.exists(log_file):
            # Get the last modified time of the log file
            last_modified = os.path.getmtime(log_file)
            # If the log file was modified in the last 30 seconds, the backtest is still running
            if time.time() - last_modified < 30:
                logger.info(f"Backtest is still running, log file was modified {time.time() - last_modified:.1f}s ago")
            else:
                logger.info(f"Log file hasn't been modified in {time.time() - last_modified:.1f}s, backtest may be stuck")
        
        # Only log every 30 seconds to reduce output
        if int((time.time() - start_time) / 30) != int((time.time() - start_time - 5) / 30):
            logger.info(f"Waiting for backtest to complete... (elapsed: {time.time() - start_time:.1f}s)")
        time.sleep(5)
    
    logger.error(f"Timeout waiting for backtest to complete after {timeout} seconds")
    return None

def check_log_file(run_id):
    """Check the backtest log file for errors."""
    log_file = f"logs/test_comprehensive_backtest_{run_id}.log"
    if os.path.exists(log_file):
        logger.info(f"Checking log file: {log_file}")
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        # Look for error messages
        error_lines = [line for line in log_content.split('\n') if 'ERROR' in line]
        if error_lines:
            logger.error(f"Found {len(error_lines)} error lines in log file:")
            for line in error_lines[:10]:  # Show first 10 errors
                logger.error(f"  {line}")
        
        # Look for the last few lines to see what's happening
        last_lines = log_content.split('\n')[-20:]
        logger.info(f"Last few lines of log file:")
        for line in last_lines:
            logger.info(f"  {line}")
        
        return log_content
    else:
        logger.error(f"Log file not found: {log_file}")
        return None

def analyze_result_file(result_file):
    """Analyze the backtest result file."""
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Analyzing result file: {result_file}")
        
        # Check if it's a combined result file
        if 'quarters' in data:
            logger.info("This is a combined result file")
            for quarter, quarter_data in data['quarters'].items():
                logger.info(f"Quarter: {quarter}")
                analyze_quarter_data(quarter_data)
        else:
            # Check for quarter info in the summary
            quarter = data.get('summary', {}).get('quarter', 'Unknown')
            logger.info(f"Quarter: {quarter}")
            analyze_quarter_data(data)
        
        return data
    except Exception as e:
        logger.error(f"Error analyzing result file {result_file}: {str(e)}")
        return None

def analyze_quarter_data(data):
    """Analyze quarter data for capital continuity."""
    summary = data.get('summary', {})
    
    # Extract capital values
    initial_capital = summary.get('initial_capital')
    final_capital = summary.get('final_capital')
    
    if initial_capital is not None:
        logger.info(f"Initial Capital: ${initial_capital:.2f}")
    else:
        logger.info("Initial Capital: Not found")
    
    if final_capital is not None:
        logger.info(f"Final Capital: ${final_capital:.2f}")
    else:
        logger.info("Final Capital: Not found")
    
    # Calculate profit/loss
    if initial_capital is not None and final_capital is not None:
        profit_loss = final_capital - initial_capital
        profit_loss_pct = (profit_loss / initial_capital) * 100
        logger.info(f"Profit/Loss: ${profit_loss:.2f} ({profit_loss_pct:.2f}%)")
    
    # Check for other relevant metrics
    metrics = summary.get('metrics', {})
    if metrics:
        logger.info("Metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")

def main():
    """Run a backtest for Q1 2023 and check the results."""
    quarter = "Q1_2023"
    initial_capital = 1000
    
    # Run backtest for Q1 2023
    result = run_backtest(quarter, initial_capital)
    if not result:
        logger.error("Failed to start backtest")
        return
    
    run_id = result.get('run_id')
    if not run_id:
        logger.error("No run_id in the response")
        return
    
    # Wait for the backtest to complete
    logger.info(f"Waiting for the backtest (run_id: {run_id}) to complete...")
    result_file = wait_for_backtest_completion(run_id)
    
    # If no result file was found, check the log file for errors
    if not result_file:
        logger.error("No result file found, checking log file for errors")
        log_content = check_log_file(run_id)
        if not log_content:
            logger.error("No log file found either, backtest may have failed to start")
        return
    
    # Analyze the result file
    data = analyze_result_file(result_file)
    if not data:
        logger.error("Failed to analyze result file")
        return
    
    logger.info("Backtest completed successfully")

if __name__ == "__main__":
    logger.info("Starting single quarter backtest...")
    main()
    logger.info("Done")
