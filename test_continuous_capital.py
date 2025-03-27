#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for continuous capital functionality.
This script runs individual backtests for each quarter and checks if the continuous capital is working correctly.
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

def run_backtest_for_quarter(quarter, initial_capital=1000, max_signals=100, weekly_selection=True):
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

def find_latest_backtest_file(quarter):
    """Find the most recent backtest result file for a specific quarter."""
    # Get all backtest result files for the quarter
    pattern = f"backtest_results_{quarter}_*.json"
    files = glob.glob(pattern)
    
    if not files:
        logger.error(f"No result file found for {quarter}")
        return None
    
    # Sort by modification time (newest first)
    latest_file = max(files, key=os.path.getmtime)
    logger.info(f"Found latest result file for {quarter}: {latest_file}")
    return latest_file

def get_backtest_result(result_file):
    """Get the backtest result from the result file."""
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error reading result file {result_file}: {str(e)}")
        return None

def wait_for_backtest_completion(run_id, timeout=180):
    """Wait for the backtest to complete."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        # Check if result files with the run_id exist
        files = glob.glob(f"backtest_results_*_{run_id}.json")
        if files:
            logger.info(f"Backtest completed, found result files: {files}")
            return True
        
        # Also check for combined result file
        combined_file = f"combined_backtest_results_{run_id}.json"
        if os.path.exists(combined_file):
            logger.info(f"Found combined result file: {combined_file}")
            return True
        
        logger.info(f"Waiting for backtest to complete... (elapsed: {time.time() - start_time:.1f}s)")
        time.sleep(5)
    
    logger.error(f"Timeout waiting for backtest to complete after {timeout} seconds")
    return False

def test_continuous_capital():
    """Test the continuous capital functionality."""
    quarters = ["Q1_2023", "Q2_2023", "Q3_2023"]
    initial_capital = 1000
    
    # Run backtest for the first quarter
    result = run_backtest_for_quarter(quarters[0], initial_capital)
    if not result:
        logger.error("Failed to start backtest for the first quarter")
        return
    
    run_id = result.get('run_id')
    if not run_id:
        logger.error("No run_id in the response")
        return
    
    # Wait for the backtest to complete
    logger.info(f"Waiting for the first backtest (run_id: {run_id}) to complete...")
    if not wait_for_backtest_completion(run_id):
        return
    
    # Find the latest result file for the first quarter
    result_file = find_latest_backtest_file(quarters[0])
    if not result_file:
        return
    
    # Get the final capital from the first quarter
    data = get_backtest_result(result_file)
    if not data or 'summary' not in data or 'final_capital' not in data['summary']:
        logger.error(f"No final capital found in the result file for {quarters[0]}")
        return
    
    final_capital = data['summary']['final_capital']
    logger.info(f"{quarters[0]} final capital: ${final_capital:.2f}")
    
    # Run backtest for the second quarter with the final capital from the first quarter
    result = run_backtest_for_quarter(quarters[1], final_capital)
    if not result:
        logger.error("Failed to start backtest for the second quarter")
        return
    
    run_id = result.get('run_id')
    if not run_id:
        logger.error("No run_id in the response")
        return
    
    # Wait for the backtest to complete
    logger.info(f"Waiting for the second backtest (run_id: {run_id}) to complete...")
    if not wait_for_backtest_completion(run_id):
        return
    
    # Find the latest result file for the second quarter
    result_file = find_latest_backtest_file(quarters[1])
    if not result_file:
        return
    
    # Get the initial and final capital from the second quarter
    data = get_backtest_result(result_file)
    if not data or 'summary' not in data:
        logger.error(f"No summary found in the result file for {quarters[1]}")
        return
    
    initial_capital_q2 = data['summary'].get('initial_capital')
    final_capital_q2 = data['summary'].get('final_capital')
    
    if initial_capital_q2 is None:
        logger.error(f"No initial capital found in the result file for {quarters[1]}")
        return
    
    if final_capital_q2 is None:
        logger.error(f"No final capital found in the result file for {quarters[1]}")
        return
    
    logger.info(f"{quarters[1]} initial capital: ${initial_capital_q2:.2f}")
    logger.info(f"{quarters[1]} final capital: ${final_capital_q2:.2f}")
    
    # Check if the initial capital of the second quarter matches the final capital of the first quarter
    if round(initial_capital_q2, 2) == round(final_capital, 2):
        logger.info(f"PASS: {quarters[0]} final capital matches {quarters[1]} initial capital")
    else:
        logger.error(f"FAIL: {quarters[0]} final capital does NOT match {quarters[1]} initial capital")
        logger.error(f"  {quarters[0]} final: ${final_capital:.2f}, {quarters[1]} initial: ${initial_capital_q2:.2f}")
    
    # Run backtest for the third quarter with the final capital from the second quarter
    result = run_backtest_for_quarter(quarters[2], final_capital_q2)
    if not result:
        logger.error("Failed to start backtest for the third quarter")
        return
    
    run_id = result.get('run_id')
    if not run_id:
        logger.error("No run_id in the response")
        return
    
    # Wait for the backtest to complete
    logger.info(f"Waiting for the third backtest (run_id: {run_id}) to complete...")
    if not wait_for_backtest_completion(run_id):
        return
    
    # Find the latest result file for the third quarter
    result_file = find_latest_backtest_file(quarters[2])
    if not result_file:
        return
    
    # Get the initial capital from the third quarter
    data = get_backtest_result(result_file)
    if not data or 'summary' not in data or 'initial_capital' not in data['summary']:
        logger.error(f"No initial capital found in the result file for {quarters[2]}")
        return
    
    initial_capital_q3 = data['summary']['initial_capital']
    logger.info(f"{quarters[2]} initial capital: ${initial_capital_q3:.2f}")
    
    # Check if the initial capital of the third quarter matches the final capital of the second quarter
    if round(initial_capital_q3, 2) == round(final_capital_q2, 2):
        logger.info(f"PASS: {quarters[1]} final capital matches {quarters[2]} initial capital")
    else:
        logger.error(f"FAIL: {quarters[1]} final capital does NOT match {quarters[2]} initial capital")
        logger.error(f"  {quarters[1]} final: ${final_capital_q2:.2f}, {quarters[2]} initial: ${initial_capital_q3:.2f}")

if __name__ == "__main__":
    logger.info("Starting test for continuous capital functionality...")
    test_continuous_capital()
    logger.info("Test completed")
