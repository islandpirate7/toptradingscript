#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for continuous capital functionality with custom date ranges.
This script runs backtests for multiple custom date ranges with continuous capital enabled.
"""

import os
import json
import time
import requests
import logging
import glob
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_custom_date_range_backtest(start_date, end_date, initial_capital=1000, max_signals=100, weekly_selection=True, continuous_capital=False):
    """Run a backtest for a custom date range."""
    url = "http://localhost:5000/run_comprehensive_backtest"
    payload = {
        "quarters": ["custom_range"],
        "custom_start_date": start_date,
        "custom_end_date": end_date,
        "max_signals": max_signals,
        "initial_capital": initial_capital,
        "weekly_selection": weekly_selection,
        "continuous_capital": continuous_capital
    }
    
    logger.info(f"Running backtest for custom date range {start_date} to {end_date} with initial capital ${initial_capital}")
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        logger.info(f"Custom date range backtest started successfully: {result}")
        return result
    else:
        logger.error(f"Failed to start custom date range backtest: {response.text}")
        return None

def wait_for_backtest_completion(run_id, timeout=600):  # 10 minutes timeout
    """Wait for the backtest to complete."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        # Check if result files with the run_id exist
        files = glob.glob(f"backtest_results/backtest_custom_*_{run_id}.json")
        if files:
            logger.info(f"Backtest completed, found result files: {files}")
            return files[0]  # Return the first result file
        
        # Also check for combined result file
        combined_file = f"backtest_results/combined_backtest_results_{run_id}.json"
        if os.path.exists(combined_file):
            logger.info(f"Found combined result file: {combined_file}")
            return combined_file
        
        # Check web_interface directory as well
        web_files = glob.glob(f"web_interface/backtest_results/backtest_custom_*_{run_id}.json")
        if web_files:
            logger.info(f"Backtest completed, found result files in web_interface: {web_files}")
            return web_files[0]
        
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

def analyze_result_file(result_file):
    """Analyze the backtest result file."""
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Analyzing result file: {result_file}")
        
        # Extract summary information
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
        win_rate = summary.get('win_rate')
        if win_rate is not None:
            logger.info(f"Win Rate: {win_rate:.2f}%")
        
        total_trades = summary.get('total_trades')
        if total_trades is not None:
            logger.info(f"Total Trades: {total_trades}")
        
        return data, final_capital if final_capital is not None else initial_capital
    except Exception as e:
        logger.error(f"Error analyzing result file {result_file}: {str(e)}")
        return None, None

def main():
    """Run backtests for multiple custom date ranges with continuous capital."""
    # Define custom date ranges (3 consecutive months in 2023)
    date_ranges = [
        ("2023-01-01", "2023-01-31"),  # January 2023
        ("2023-02-01", "2023-02-28"),  # February 2023
        ("2023-03-01", "2023-03-31")   # March 2023
    ]
    
    initial_capital = 1000
    current_capital = initial_capital
    
    # Run backtests for each date range with continuous capital
    for i, (start_date, end_date) in enumerate(date_ranges):
        logger.info(f"Running backtest {i+1}/{len(date_ranges)}: {start_date} to {end_date} with capital ${current_capital:.2f}")
        
        # Run backtest with current capital
        result = run_custom_date_range_backtest(start_date, end_date, current_capital)
        if not result:
            logger.error(f"Failed to start backtest for date range {start_date} to {end_date}")
            continue
        
        run_id = result.get('run_id')
        if not run_id:
            logger.error("No run_id in the response")
            continue
        
        # Wait for the backtest to complete
        logger.info(f"Waiting for the backtest (run_id: {run_id}) to complete...")
        result_file = wait_for_backtest_completion(run_id)
        
        # If no result file was found, continue to the next date range
        if not result_file:
            logger.error(f"No result file found for date range {start_date} to {end_date}")
            continue
        
        # Analyze the result file and get the final capital
        data, final_capital = analyze_result_file(result_file)
        if not data:
            logger.error(f"Failed to analyze result file for date range {start_date} to {end_date}")
            continue
        
        # Update current capital for the next date range
        if final_capital is not None:
            logger.info(f"Updating capital from ${current_capital:.2f} to ${final_capital:.2f} for next date range")
            current_capital = final_capital
        else:
            logger.warning(f"No final capital found for date range {start_date} to {end_date}, keeping capital at ${current_capital:.2f}")
    
    # Calculate overall performance
    overall_return = ((current_capital / initial_capital) - 1) * 100
    logger.info(f"Overall performance across all date ranges:")
    logger.info(f"Initial Capital: ${initial_capital:.2f}")
    logger.info(f"Final Capital: ${current_capital:.2f}")
    logger.info(f"Overall Return: {overall_return:.2f}%")

def run_single_continuous_backtest():
    """Run a single backtest with multiple quarters and continuous capital enabled."""
    # Use just two quarters for faster testing
    quarters = ['Q1_2023', 'Q2_2023']
    initial_capital = 1000
    
    logger.info(f"Running continuous backtest for quarters: {quarters} with initial capital ${initial_capital}")
    
    # Prepare the request parameters
    params = {
        'quarters': quarters,
        'initial_capital': initial_capital,
        'continuous_capital': True
    }
    
    # Send the request to start the backtest
    try:
        response = requests.post('http://localhost:5000/run_comprehensive_backtest', json=params)
        response.raise_for_status()
        
        result = response.json()
        if result.get('success'):
            run_id = result.get('run_id')
            logger.info(f"Continuous backtest started successfully: {result}")
            
            # Wait for the backtest to complete
            result_file = wait_for_backtest_completion(run_id, timeout=300)  # Reduced timeout to 5 minutes
            
            # If no result file was found, continue
            if not result_file:
                logger.error(f"No result file found for continuous backtest")
                return
            
            # Analyze the result file
            data, final_capital = analyze_result_file(result_file)
            if not data:
                logger.error(f"Failed to analyze result file for continuous backtest")
                return
            
            # Calculate overall performance
            overall_return = ((final_capital / initial_capital) - 1) * 100
            logger.info(f"Overall performance for continuous backtest:")
            logger.info(f"Initial Capital: ${initial_capital:.2f}")
            logger.info(f"Final Capital: ${final_capital:.2f}")
            logger.info(f"Overall Return: {overall_return:.2f}%")
        else:
            logger.error(f"Failed to start continuous backtest: {response.text}")
    except Exception as e:
        logger.error(f"Failed to start continuous backtest: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting test for continuous capital with multiple quarters...")
    
    # Comment out the manual continuous capital test and run the built-in one
    # main()  # Run multiple custom date range backtests with manual continuous capital
    run_single_continuous_backtest()  # Run a single backtest with multiple quarters and continuous capital
    
    logger.info("Test completed")
