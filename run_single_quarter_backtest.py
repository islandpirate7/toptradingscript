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

def wait_for_backtest_completion(run_id, timeout=300):
    """Wait for the backtest to complete."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        # Check if result files with the run_id exist
        files = glob.glob(f"backtest_results_*_{run_id}.json")
        if files:
            logger.info(f"Backtest completed, found result files: {files}")
            return files[0]  # Return the first result file
        
        # Also check for combined result file
        combined_file = f"combined_backtest_results_{run_id}.json"
        if os.path.exists(combined_file):
            logger.info(f"Found combined result file: {combined_file}")
            return combined_file
        
        # Check web_interface directory as well
        web_files = glob.glob(f"web_interface/backtest_results_*_{run_id}.json")
        if web_files:
            logger.info(f"Backtest completed, found result files in web_interface: {web_files}")
            return web_files[0]
        
        web_combined = f"web_interface/combined_backtest_results_{run_id}.json"
        if os.path.exists(web_combined):
            logger.info(f"Found combined result file in web_interface: {web_combined}")
            return web_combined
        
        logger.info(f"Waiting for backtest to complete... (elapsed: {time.time() - start_time:.1f}s)")
        time.sleep(5)
    
    logger.error(f"Timeout waiting for backtest to complete after {timeout} seconds")
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
    if not result_file:
        logger.error("No result file found")
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
