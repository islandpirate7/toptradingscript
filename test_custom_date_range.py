#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for custom date range functionality in the web interface.
This script runs a backtest with a custom date range and checks the results.
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
        
        # Check if trades data exists
        trades = data.get('trades', [])
        logger.info(f"Number of trades: {len(trades)}")
        
        # Show a few sample trades if available
        if trades:
            logger.info("Sample trades:")
            for i, trade in enumerate(trades[:5]):  # Show first 5 trades
                logger.info(f"  Trade {i+1}: {trade.get('symbol')} - Entry: {trade.get('entry_date')} - Exit: {trade.get('exit_date')} - PnL: ${trade.get('profit_loss', 0):.2f}")
        
        return data
    except Exception as e:
        logger.error(f"Error analyzing result file {result_file}: {str(e)}")
        return None

def main():
    """Run a backtest with a custom date range and check the results."""
    # Define custom date range (2 months in 2023)
    start_date = "2023-03-01"
    end_date = "2023-04-30"
    initial_capital = 1000
    
    # Run backtest with custom date range
    result = run_custom_date_range_backtest(start_date, end_date, initial_capital)
    if not result:
        logger.error("Failed to start custom date range backtest")
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
    
    logger.info("Custom date range backtest completed successfully")

if __name__ == "__main__":
    logger.info("Starting test for custom date range functionality...")
    main()
    logger.info("Test completed")
