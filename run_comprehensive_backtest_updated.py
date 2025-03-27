#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive Backtest Runner (Updated)
---------------------------------------
This module implements a comprehensive backtest runner that doesn't rely on the alpaca_trade_api library.
It uses direct HTTP requests to simulate the functionality needed for backtesting.
"""

import os
import pandas as pd
import json
from datetime import datetime
import yaml
import logging
import argparse
import sys
import traceback
import multiprocessing
import numpy as np
import time

# Import our updated backtest implementation
from backtest_engine_updated import run_backtest

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', f"comprehensive_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"), mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add a console handler to ensure output is visible in web interface
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

def display_performance_metrics(summary):
    """Display performance metrics from a backtest summary"""
    if not summary:
        print("No performance metrics available")
        return
    
    print("\n===== PERFORMANCE METRICS =====")
    print(f"Win Rate: {summary.get('win_rate', 0):.2f}%")
    print(f"Profit Factor: {summary.get('profit_factor', 0):.2f}")
    print(f"Total Return: {summary.get('total_return', 0):.2f}%")
    print(f"Net Profit: ${summary.get('net_profit', 0):.2f}")
    print(f"Average Win: ${summary.get('avg_profit_per_winner', 0):.2f}")
    print(f"Average Loss: ${summary.get('avg_loss_per_loser', 0):.2f}")
    print(f"Average Holding Period: {summary.get('avg_holding_period', 0):.1f} days")
    print(f"Number of Trades: {summary.get('num_trades', 0)}")

def run_quarter_backtest(quarter, start_date, end_date, max_signals, initial_capital, weekly_selection=False):
    """Run backtest for a specific quarter in a separate process"""
    print(f"\n{'=' * 50}")
    print(f"Running backtest for {quarter}: {start_date} to {end_date}")
    print(f"{'=' * 50}")
    
    try:
        # Log the parameters being used
        logger.info(f"Running backtest with parameters: quarter={quarter}, start_date={start_date}, end_date={end_date}, max_signals={max_signals}, initial_capital={initial_capital}, weekly_selection={weekly_selection}")
        
        # Run backtest for this quarter
        summary, signals = run_backtest(
            start_date, 
            end_date, 
            mode='backtest', 
            max_signals=max_signals, 
            initial_capital=initial_capital,
            weekly_selection=weekly_selection
        )
        
        # Check if summary is None and log an error if it is
        if summary is None:
            logger.error(f"Backtest for {quarter} returned None summary. This indicates an error in the backtest execution.")
            # Create a minimal summary to avoid null values in the results file
            summary = {
                'quarter': quarter,
                'start_date': start_date,
                'end_date': end_date,
                'error': 'Backtest execution failed to return valid results',
                'win_rate': 0,
                'profit_factor': 0,
                'total_return': 0,
                'initial_capital': initial_capital,
                'final_capital': initial_capital
            }
        else:
            logger.info(f"Backtest for {quarter} completed successfully with win_rate: {summary.get('win_rate', 'N/A')}%, profit_factor: {summary.get('profit_factor', 'N/A')}")
        
        # Create a unique filename for this quarter's results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure the backtest_results directory exists at the root level (where the web interface expects it)
        backtest_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_results")
        os.makedirs(backtest_dir, exist_ok=True)
        
        # Create a standardized filename format that the web interface can recognize
        results_file = os.path.join(backtest_dir, f"backtest_{quarter}_{start_date}_to_{end_date}_{timestamp}.json")
        
        # Save results to a file
        with open(results_file, 'w') as f:
            json.dump({
                'summary': summary,
                'signals': signals if signals else []
            }, f, default=str)
            
        logger.info(f"Backtest results saved to {results_file}")
        
        return results_file
    except Exception as e:
        logger.error(f"Error in run_quarter_backtest: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Create a minimal summary with error information
        error_summary = {
            'quarter': quarter,
            'start_date': start_date,
            'end_date': end_date,
            'error': str(e),
            'win_rate': 0,
            'profit_factor': 0,
            'total_return': 0,
            'initial_capital': initial_capital,
            'final_capital': initial_capital
        }
        
        # Create a unique filename for this quarter's results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure the backtest_results directory exists
        backtest_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_results")
        os.makedirs(backtest_dir, exist_ok=True)
        
        # Create a standardized filename format
        results_file = os.path.join(backtest_dir, f"backtest_{quarter}_{start_date}_to_{end_date}_error_{timestamp}.json")
        
        # Save error results
        with open(results_file, 'w') as f:
            json.dump({
                'summary': error_summary,
                'signals': [],
                'error': str(e),
                'traceback': traceback.format_exc()
            }, f, default=str)
            
        logger.info(f"Error results saved to {results_file}")
        
        return results_file

def run_multiple_backtests(quarter, start_date, end_date, max_signals=100, initial_capital=10000, num_runs=5, random_seed=42, continuous_capital=False, previous_capital=None, weekly_selection=False):
    """
    Run multiple backtests and average the results to get a more stable assessment
    
    Args:
        quarter (str): Quarter identifier (e.g., 'Q1_2023')
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        max_signals (int): Maximum number of signals to use
        initial_capital (float): Initial capital for the backtest
        num_runs (int): Number of backtest runs to perform
        random_seed (int): Base random seed for reproducibility
        continuous_capital (bool): Whether to use continuous capital across quarters
        previous_capital (float): Previous ending capital to use as initial capital (if continuous_capital is True)
        weekly_selection (bool): Whether to enable weekly stock selection refresh
        
    Returns:
        dict: Averaged backtest results and final capital
    """
    print(f"\n{'=' * 50}")
    print(f"Running {num_runs} backtests for {quarter}: {start_date} to {end_date}")
    if continuous_capital and previous_capital is not None:
        print(f"Using continuous capital: Starting with ${previous_capital:.2f} from previous quarter")
    print(f"{'=' * 50}")
    
    # Set the initial capital based on continuous capital setting
    if continuous_capital and previous_capital is not None:
        current_initial_capital = previous_capital
    else:
        current_initial_capital = initial_capital
    
    # Lists to store results from each run
    win_rates = []
    profit_factors = []
    total_returns = []
    net_profits = []
    final_capitals = []
    
    # Run multiple backtests with different random seeds
    for run in range(1, num_runs + 1):
        print(f"\nRun {run}/{num_runs} for {quarter}")
        
        # Calculate seed for this run
        run_seed = random_seed + run
        
        try:
            # Run backtest with this seed
            summary, _ = run_backtest(
                start_date=start_date,
                end_date=end_date,
                mode='backtest',
                max_signals=max_signals,
                initial_capital=current_initial_capital,
                random_seed=run_seed,
                weekly_selection=weekly_selection
            )
            
            if summary is None:
                logger.error(f"Run {run}/{num_runs} for {quarter} returned None summary")
                continue
            
            # Extract metrics
            win_rate = summary.get('win_rate', 0)
            profit_factor = summary.get('profit_factor', 0)
            total_return = summary.get('total_return', 0)
            net_profit = summary.get('net_profit', 0)
            final_capital = summary.get('final_capital', current_initial_capital)
            
            # Store metrics
            win_rates.append(win_rate)
            profit_factors.append(profit_factor)
            total_returns.append(total_return)
            net_profits.append(net_profit)
            final_capitals.append(final_capital)
            
            print(f"Run {run} Results:")
            print(f"  Win Rate: {win_rate:.2f}%")
            print(f"  Profit Factor: {profit_factor:.2f}")
            print(f"  Total Return: {total_return:.2f}%")
            print(f"  Net Profit: ${net_profit:.2f}")
            print(f"  Final Capital: ${final_capital:.2f}")
            
        except Exception as e:
            logger.error(f"Error in run {run}/{num_runs} for {quarter}: {str(e)}")
            logger.error(traceback.format_exc())
    
    # Calculate averages
    if len(win_rates) > 0:
        avg_win_rate = np.mean(win_rates)
        avg_profit_factor = np.mean(profit_factors)
        avg_total_return = np.mean(total_returns)
        avg_net_profit = np.mean(net_profits)
        avg_final_capital = np.mean(final_capitals)
        
        # Calculate standard deviations
        std_win_rate = np.std(win_rates)
        std_profit_factor = np.std(profit_factors)
        std_total_return = np.std(total_returns)
        std_net_profit = np.std(net_profits)
        
        print(f"\nAverage Results for {quarter} ({len(win_rates)}/{num_runs} successful runs):")
        print(f"  Win Rate: {avg_win_rate:.2f}% (±{std_win_rate:.2f}%)")
        print(f"  Profit Factor: {avg_profit_factor:.2f} (±{std_profit_factor:.2f})")
        print(f"  Total Return: {avg_total_return:.2f}% (±{std_total_return:.2f}%)")
        print(f"  Net Profit: ${avg_net_profit:.2f} (±${std_net_profit:.2f})")
        print(f"  Final Capital: ${avg_final_capital:.2f}")
        
        # Create averaged summary
        averaged_summary = {
            'quarter': quarter,
            'start_date': start_date,
            'end_date': end_date,
            'win_rate': avg_win_rate,
            'profit_factor': avg_profit_factor,
            'total_return': avg_total_return,
            'net_profit': avg_net_profit,
            'initial_capital': current_initial_capital,
            'final_capital': avg_final_capital,
            'num_runs': len(win_rates),
            'std_win_rate': std_win_rate,
            'std_profit_factor': std_profit_factor,
            'std_total_return': std_total_return,
            'std_net_profit': std_net_profit
        }
        
        return averaged_summary, avg_final_capital
    else:
        logger.error(f"No successful runs for {quarter}")
        
        # Create error summary
        error_summary = {
            'quarter': quarter,
            'start_date': start_date,
            'end_date': end_date,
            'error': 'No successful backtest runs',
            'win_rate': 0,
            'profit_factor': 0,
            'total_return': 0,
            'net_profit': 0,
            'initial_capital': current_initial_capital,
            'final_capital': current_initial_capital,
            'num_runs': 0
        }
        
        return error_summary, current_initial_capital

def run_comprehensive_backtest(quarter, max_signals=100, initial_capital=10000, multiple_runs=False, num_runs=5, continuous_capital=False, weekly_selection=False):
    """
    Run a comprehensive backtest for a specific quarter with detailed signal analysis
    
    Args:
        quarter (str): Quarter to run backtest for ('Q1_2023', 'Q2_2023', etc. or 'all')
        max_signals (int): Maximum number of signals to use
        initial_capital (float): Initial capital for the backtest
        multiple_runs (bool): Whether to run multiple backtests and average results
        num_runs (int): Number of backtest runs to perform if multiple_runs is True
        continuous_capital (bool): Whether to use continuous capital across quarters
        weekly_selection (bool): Whether to enable weekly stock selection refresh
    """
    # Define quarters and their date ranges
    quarters = {
        'Q1_2023': ('2023-01-01', '2023-03-31'),
        'Q2_2023': ('2023-04-01', '2023-06-30'),
        'Q3_2023': ('2023-07-01', '2023-09-30'),
        'Q4_2023': ('2023-10-01', '2023-12-31'),
        'Q1_2024': ('2024-01-01', '2024-03-31')
    }
    
    # Check if the requested quarter exists
    if quarter != 'all' and quarter not in quarters:
        print(f"Error: Quarter '{quarter}' not found. Available quarters: {', '.join(quarters.keys())} or 'all'")
        return
    
    # If 'all' is specified, run for all quarters
    if quarter == 'all':
        run_all_quarters_backtest(
            max_signals=max_signals,
            initial_capital=initial_capital,
            multiple_runs=multiple_runs,
            num_runs=num_runs,
            continuous_capital=continuous_capital,
            weekly_selection=weekly_selection
        )
        return
    
    # Get date range for the specified quarter
    start_date, end_date = quarters[quarter]
    
    # Run backtest for the specified quarter
    if multiple_runs:
        # Run multiple backtests and average results
        summary, _ = run_multiple_backtests(
            quarter=quarter,
            start_date=start_date,
            end_date=end_date,
            max_signals=max_signals,
            initial_capital=initial_capital,
            num_runs=num_runs,
            weekly_selection=weekly_selection
        )
        
        # Display performance metrics
        if summary:
            print(f"\nAverage Performance Metrics for {quarter}:")
            display_performance_metrics(summary)
    else:
        # Run a single backtest
        results_file = run_quarter_backtest(
            quarter=quarter,
            start_date=start_date,
            end_date=end_date,
            max_signals=max_signals,
            initial_capital=initial_capital,
            weekly_selection=weekly_selection
        )
        
        # Load results from file
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Display performance metrics
        summary = results.get('summary', {})
        if summary:
            print(f"\nPerformance Metrics for {quarter}:")
            display_performance_metrics(summary)

def run_all_quarters_backtest(max_signals=100, initial_capital=10000, multiple_runs=False, num_runs=5, continuous_capital=False, weekly_selection=False):
    """
    Run comprehensive backtests for all quarters
    
    Args:
        max_signals (int): Maximum number of signals to use
        initial_capital (float): Initial capital for the backtest
        multiple_runs (bool): Whether to run multiple backtests and average results
        num_runs (int): Number of backtest runs to perform if multiple_runs is True
        continuous_capital (bool): Whether to use continuous capital across quarters
        weekly_selection (bool): Whether to enable weekly stock selection refresh
    """
    # Define quarters and their date ranges
    quarters = {
        'Q1_2023': ('2023-01-01', '2023-03-31'),
        'Q2_2023': ('2023-04-01', '2023-06-30'),
        'Q3_2023': ('2023-07-01', '2023-09-30'),
        'Q4_2023': ('2023-10-01', '2023-12-31'),
        'Q1_2024': ('2024-01-01', '2024-03-31')
    }
    
    # Track current capital for continuous capital mode
    current_capital = initial_capital
    
    # Store quarterly results
    quarterly_results = {}
    
    # Run backtest for each quarter
    for quarter, (start_date, end_date) in quarters.items():
        print(f"\n{'=' * 80}")
        print(f"Running backtest for {quarter}: {start_date} to {end_date}")
        if continuous_capital:
            print(f"Starting capital: ${current_capital:.2f}")
        print(f"{'=' * 80}")
        
        if multiple_runs:
            # Run multiple backtests and average results
            summary, final_capital = run_multiple_backtests(
                quarter=quarter,
                start_date=start_date,
                end_date=end_date,
                max_signals=max_signals,
                initial_capital=current_capital if continuous_capital else initial_capital,
                num_runs=num_runs,
                continuous_capital=continuous_capital,
                previous_capital=current_capital if continuous_capital else None,
                weekly_selection=weekly_selection
            )
            
            # Update current capital for next quarter if using continuous capital
            if continuous_capital:
                current_capital = final_capital
            
            # Store results
            quarterly_results[quarter] = summary
            
            # Display performance metrics
            if summary:
                print(f"\nAverage Performance Metrics for {quarter}:")
                display_performance_metrics(summary)
        else:
            # Run a single backtest
            results_file = run_quarter_backtest(
                quarter=quarter,
                start_date=start_date,
                end_date=end_date,
                max_signals=max_signals,
                initial_capital=current_capital if continuous_capital else initial_capital,
                weekly_selection=weekly_selection
            )
            
            # Load results from file
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            # Extract summary
            summary = results.get('summary', {})
            
            # Update current capital for next quarter if using continuous capital
            if continuous_capital and 'final_capital' in summary:
                current_capital = summary['final_capital']
            
            # Store results
            quarterly_results[quarter] = summary
            
            # Display performance metrics
            if summary:
                print(f"\nPerformance Metrics for {quarter}:")
                display_performance_metrics(summary)
    
    # Display summary of all quarters
    print(f"\n{'=' * 80}")
    print("Summary of All Quarters:")
    print(f"{'=' * 80}")
    
    for quarter, summary in quarterly_results.items():
        print(f"\n{quarter}:")
        print(f"  Win Rate: {summary.get('win_rate', 0):.2f}%")
        print(f"  Profit Factor: {summary.get('profit_factor', 0):.2f}")
        print(f"  Total Return: {summary.get('total_return', 0):.2f}%")
        print(f"  Net Profit: ${summary.get('net_profit', 0):.2f}")
        
        if continuous_capital:
            print(f"  Initial Capital: ${summary.get('initial_capital', 0):.2f}")
            print(f"  Final Capital: ${summary.get('final_capital', 0):.2f}")

def run_backtest_for_web(start_date, end_date, max_signals=100, initial_capital=10000, continuous_capital=False, weekly_selection=False):
    """
    Run a backtest for a specific date range using real data - designed for web interface integration
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        max_signals (int): Maximum number of signals to generate
        initial_capital (float): Initial capital for the backtest
        continuous_capital (bool): Whether to use continuous capital
        weekly_selection (bool): Whether to use weekly selection
        
    Returns:
        dict: Backtest results in the format expected by the web interface
    """
    logger.info(f"[DEBUG] Starting run_backtest_for_web in run_comprehensive_backtest_updated.py")
    logger.info(f"[DEBUG] Running backtest from {start_date} to {end_date}")
    logger.info(f"[DEBUG] Parameters: max_signals={max_signals}, initial_capital={initial_capital}, continuous_capital={continuous_capital}, weekly_selection={weekly_selection}")
    
    # Extract quarter information from dates
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    
    year = start_dt.year
    quarter = (start_dt.month - 1) // 3 + 1
    quarter_name = f"Q{quarter}_{year}"
    
    try:
        # Run the backtest
        logger.info(f"[DEBUG] Calling run_backtest from backtest_engine_updated")
        start_time = time.time()
        
        summary, signals = run_backtest(
            start_date, 
            end_date, 
            mode='backtest', 
            max_signals=max_signals, 
            initial_capital=initial_capital,
            weekly_selection=weekly_selection)
        
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"[DEBUG] run_backtest execution time: {execution_time:.2f} seconds")
        
        # Check if summary is None
        if summary is None:
            logger.error("[DEBUG] Backtest returned None summary. This indicates an error in the backtest execution.")
            raise Exception("Backtest execution failed to return valid results")
        
        # Log some details about the results to verify they're real
        logger.info(f"[DEBUG] Backtest returned summary with {len(signals) if signals else 0} signals")
        if signals and len(signals) > 0:
            logger.info(f"[DEBUG] First few signals: {signals[:3]}")
        
        # Create result object in the format expected by the web interface
        result = {
            'summary': summary,
            'trades': signals if signals else [],
            'parameters': {
                'max_signals': max_signals,
                'initial_capital': initial_capital,
                'continuous_capital': continuous_capital,
                'weekly_selection': weekly_selection
            }
        }
        
        logger.info(f"[DEBUG] Backtest completed with {summary.get('num_trades', 0)} trades")
        logger.info(f"[DEBUG] Win rate: {summary.get('win_rate', 0)}%, Profit factor: {summary.get('profit_factor', 0)}, Total return: {summary.get('total_return', 0)}%")
        
        return result
    
    except Exception as e:
        logger.error(f"[DEBUG] Error running backtest: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Raise the exception to be handled by the caller
        raise Exception(f"Error running backtest: {str(e)}")

def main():
    """Main function to run the comprehensive backtest"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run a comprehensive backtest for a specific quarter')
    parser.add_argument('--quarter', type=str, default='Q1_2023', help='Quarter to run backtest for (Q1_2023, Q2_2023, etc. or "all")')
    parser.add_argument('--max-signals', type=int, default=100, help='Maximum number of signals to use')
    parser.add_argument('--initial-capital', type=float, default=10000, help='Initial capital for the backtest')
    parser.add_argument('--multiple-runs', action='store_true', help='Run multiple backtests and average results')
    parser.add_argument('--num-runs', type=int, default=5, help='Number of backtest runs to perform if multiple-runs is True')
    parser.add_argument('--continuous-capital', action='store_true', help='Use continuous capital across quarters')
    parser.add_argument('--weekly-selection', action='store_true', help='Enable weekly stock selection refresh')
    
    args = parser.parse_args()
    
    # Run the comprehensive backtest
    run_comprehensive_backtest(
        quarter=args.quarter,
        max_signals=args.max_signals,
        initial_capital=args.initial_capital,
        multiple_runs=args.multiple_runs,
        num_runs=args.num_runs,
        continuous_capital=args.continuous_capital,
        weekly_selection=args.weekly_selection
    )

if __name__ == "__main__":
    main()
