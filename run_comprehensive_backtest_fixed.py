#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import alpaca_trade_api as tradeapi
from final_sp500_strategy import run_backtest
from datetime import datetime
import yaml
import logging
import argparse
import sys
import traceback
import json
import multiprocessing
import numpy as np

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def display_performance_metrics(summary):
    """Display performance metrics from a backtest summary"""
    if not summary:
        print("No performance metrics available")
        return
    
    # Convert tuple to dictionary if necessary
    if isinstance(summary, tuple):
        # The summary tuple contains the dictionary as the first element
        summary_dict = summary[0] if len(summary) > 0 else {}
    else:
        summary_dict = summary
    
    print("\n===== PERFORMANCE METRICS =====")
    print(f"Win Rate: {summary_dict.get('win_rate', 0):.2f}%")
    print(f"Profit Factor: {summary_dict.get('profit_factor', 0):.2f}")
    print(f"Average Win: ${summary_dict.get('avg_win', 0):.2f}")
    print(f"Average Loss: ${summary_dict.get('avg_loss', 0):.2f}")
    print(f"Average Holding Period: {summary_dict.get('avg_holding_period', 0):.1f} days")
    
    # Check if tier_metrics is available and has long/short win rates
    if 'tier_metrics' in summary_dict:
        tier_metrics = summary_dict['tier_metrics']
        if isinstance(tier_metrics, dict):
            # Check for long_win_rate directly in the tier_metrics dictionary
            for tier_name, tier_data in tier_metrics.items():
                if isinstance(tier_data, dict) and 'long_win_rate' in tier_data:
                    print(f"LONG Win Rate: {tier_data['long_win_rate']:.2f}%")
                    break
    
    # Display max drawdown if available
    if 'max_drawdown' in summary_dict:
        print(f"Max Drawdown: {summary_dict.get('max_drawdown', 0):.2f}%")
    
    # Display Sharpe and Sortino ratios if available
    if 'sharpe_ratio' in summary_dict:
        print(f"Sharpe Ratio: {summary_dict.get('sharpe_ratio', 0):.2f}")
    if 'sortino_ratio' in summary_dict:
        print(f"Sortino Ratio: {summary_dict.get('sortino_ratio', 0):.2f}")
    
    # Display final capital and total return if available
    if 'final_capital' in summary_dict and 'initial_capital' in summary_dict:
        initial_capital = summary_dict.get('initial_capital', 0)
        final_capital = summary_dict.get('final_capital', 0)
        total_return = ((final_capital / initial_capital) - 1) * 100 if initial_capital > 0 else 0
        print(f"Initial Capital: ${initial_capital:.2f}")
        print(f"Final Capital: ${final_capital:.2f}")
        print(f"Total Return: {total_return:.2f}%")
    
    print("===== END METRICS =====\n")

def run_quarter_backtest(quarter, start_date, end_date, max_signals, initial_capital, weekly_selection=False):
    """Run backtest for a specific quarter in a separate process"""
    try:
        print(f"\n==================================================")
        print(f"Running backtest for {quarter}: {start_date} to {end_date}")
        print(f"==================================================")
        
        # Run backtest
        random_seed = 42  # Fixed seed for reproducibility
        
        # Run backtest with weekly selection parameter
        summary = run_backtest(
            start_date, 
            end_date, 
            mode='backtest', 
            max_signals=max_signals, 
            initial_capital=initial_capital, 
            random_seed=random_seed,
            weekly_selection=weekly_selection
        )
        
        # Display performance metrics
        display_performance_metrics(summary)
        
        return summary
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        traceback.print_exc()
        return None

def run_multiple_backtests(quarter, start_date, end_date, max_signals=100, initial_capital=300, num_runs=5, random_seed=42, continuous_capital=False, previous_capital=None, weekly_selection=False):
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
    try:
        print(f"\n==================================================")
        print(f"Running {num_runs} backtests for {quarter}: {start_date} to {end_date}")
        print(f"==================================================")
        
        # Initialize metrics storage
        all_metrics = {
            'win_rate': [],
            'profit_factor': [],
            'avg_win': [],
            'avg_loss': [],
            'avg_holding_period': [],
            'max_drawdown': [],
            'sharpe_ratio': [],
            'sortino_ratio': [],
            'final_capital': []
        }
        
        # Use previous capital if continuous mode is enabled and previous capital is provided
        if continuous_capital and previous_capital is not None:
            initial_capital = previous_capital
            print(f"Using continuous capital: ${initial_capital:.2f}")
        
        # Run multiple backtests
        for i in range(num_runs):
            print(f"\nRun {i+1}/{num_runs}:")
            current_seed = random_seed + i
            
            # Run backtest with weekly selection parameter
            summary = run_backtest(
                start_date, 
                end_date, 
                mode='backtest', 
                max_signals=max_signals, 
                initial_capital=initial_capital, 
                random_seed=current_seed,
                weekly_selection=weekly_selection
            )
            
            if summary:
                # Store metrics
                all_metrics['win_rate'].append(summary.get('win_rate', 0))
                all_metrics['profit_factor'].append(summary.get('profit_factor', 0))
                all_metrics['avg_win'].append(summary.get('avg_win', 0))
                all_metrics['avg_loss'].append(summary.get('avg_loss', 0))
                all_metrics['avg_holding_period'].append(summary.get('avg_holding_period', 0))
                all_metrics['max_drawdown'].append(summary.get('max_drawdown', 0))
                all_metrics['sharpe_ratio'].append(summary.get('sharpe_ratio', 0))
                all_metrics['sortino_ratio'].append(summary.get('sortino_ratio', 0))
                all_metrics['final_capital'].append(summary.get('final_capital', initial_capital))
        
        # Calculate averages
        avg_metrics = {
            'win_rate': np.mean(all_metrics['win_rate']) if all_metrics['win_rate'] else 0,
            'profit_factor': np.mean(all_metrics['profit_factor']) if all_metrics['profit_factor'] else 0,
            'avg_win': np.mean(all_metrics['avg_win']) if all_metrics['avg_win'] else 0,
            'avg_loss': np.mean(all_metrics['avg_loss']) if all_metrics['avg_loss'] else 0,
            'avg_holding_period': np.mean(all_metrics['avg_holding_period']) if all_metrics['avg_holding_period'] else 0,
            'max_drawdown': np.mean(all_metrics['max_drawdown']) if all_metrics['max_drawdown'] else 0,
            'sharpe_ratio': np.mean(all_metrics['sharpe_ratio']) if all_metrics['sharpe_ratio'] else 0,
            'sortino_ratio': np.mean(all_metrics['sortino_ratio']) if all_metrics['sortino_ratio'] else 0,
            'final_capital': np.mean(all_metrics['final_capital']) if all_metrics['final_capital'] else initial_capital
        }
        
        # Display averaged metrics
        print("\n===== AVERAGED PERFORMANCE METRICS =====")
        print(f"Win Rate: {avg_metrics['win_rate']:.2f}%")
        print(f"Profit Factor: {avg_metrics['profit_factor']:.2f}")
        print(f"Average Win: ${avg_metrics['avg_win']:.2f}")
        print(f"Average Loss: ${avg_metrics['avg_loss']:.2f}")
        print(f"Average Holding Period: {avg_metrics['avg_holding_period']:.1f} days")
        print(f"Max Drawdown: {avg_metrics['max_drawdown']:.2f}%")
        print(f"Sharpe Ratio: {avg_metrics['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio: {avg_metrics['sortino_ratio']:.2f}")
        print(f"Final Capital: ${avg_metrics['final_capital']:.2f}")
        print("===== END AVERAGED METRICS =====\n")
        
        return {
            'metrics': avg_metrics,
            'final_capital': avg_metrics['final_capital']
        }
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        traceback.print_exc()
        return {
            'metrics': {},
            'final_capital': initial_capital
        }

def run_comprehensive_backtest(quarter, max_signals=100, initial_capital=300, multiple_runs=False, num_runs=5, continuous_capital=False, weekly_selection=False):
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
    try:
        # Define quarters mapping
        quarters_map = {
            'Q1_2023': ('2023-01-01', '2023-03-31'),
            'Q2_2023': ('2023-04-01', '2023-06-30'),
            'Q3_2023': ('2023-07-01', '2023-09-30'),
            'Q4_2023': ('2023-10-01', '2023-12-31'),
            'Q1_2024': ('2024-01-01', '2024-03-31')
        }
        
        # Check if quarter is valid
        if quarter not in quarters_map:
            logger.error(f"Invalid quarter: {quarter}")
            print(f"Invalid quarter: {quarter}")
            print(f"Valid quarters: {', '.join(quarters_map.keys())}")
            return
        
        # Get start and end dates for the quarter
        start_date, end_date = quarters_map[quarter]
        
        # Run backtest(s)
        if multiple_runs:
            # Run multiple backtests and average results
            run_multiple_backtests(
                quarter, 
                start_date, 
                end_date, 
                max_signals=max_signals, 
                initial_capital=initial_capital,
                num_runs=num_runs,
                continuous_capital=continuous_capital,
                weekly_selection=weekly_selection
            )
        else:
            # Run a single backtest
            run_quarter_backtest(
                quarter, 
                start_date, 
                end_date, 
                max_signals=max_signals, 
                initial_capital=initial_capital,
                weekly_selection=weekly_selection
            )
    except Exception as e:
        logger.error(f"Error running comprehensive backtest: {str(e)}")
        traceback.print_exc()

def run_all_quarters_backtest(max_signals=100, initial_capital=300, multiple_runs=False, num_runs=5, continuous_capital=False, weekly_selection=False):
    """Run comprehensive backtests for all quarters"""
    try:
        # Define quarters mapping
        quarters_map = {
            'Q1_2023': ('2023-01-01', '2023-03-31'),
            'Q2_2023': ('2023-04-01', '2023-06-30'),
            'Q3_2023': ('2023-07-01', '2023-09-30'),
            'Q4_2023': ('2023-10-01', '2023-12-31'),
            'Q1_2024': ('2024-01-01', '2024-03-31')
        }
        
        # Track capital for continuous mode
        current_capital = initial_capital
        
        # Run backtest for each quarter
        for quarter, (start_date, end_date) in quarters_map.items():
            print(f"\n==================================================")
            print(f"Running backtest for {quarter}: {start_date} to {end_date}")
            print(f"==================================================")
            
            if multiple_runs:
                # Run multiple backtests and average results
                result = run_multiple_backtests(
                    quarter, 
                    start_date, 
                    end_date, 
                    max_signals=max_signals, 
                    initial_capital=current_capital if continuous_capital else initial_capital,
                    num_runs=num_runs,
                    continuous_capital=continuous_capital,
                    weekly_selection=weekly_selection
                )
                
                # Update capital for continuous mode
                if continuous_capital and result and 'final_capital' in result:
                    current_capital = result['final_capital']
                    print(f"Updated capital: ${current_capital:.2f}")
            else:
                # Run a single backtest
                summary = run_quarter_backtest(
                    quarter, 
                    start_date, 
                    end_date, 
                    max_signals=max_signals, 
                    initial_capital=current_capital if continuous_capital else initial_capital,
                    weekly_selection=weekly_selection
                )
                
                # Update capital for continuous mode
                if continuous_capital and summary and 'final_capital' in summary:
                    current_capital = summary['final_capital']
                    print(f"Updated capital: ${current_capital:.2f}")
    except Exception as e:
        logger.error(f"Error running all quarters backtest: {str(e)}")
        traceback.print_exc()

def main():
    """Main function to run the comprehensive backtest"""
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Run comprehensive backtest for specified quarters')
        parser.add_argument('quarters', nargs='+', help='Quarters to run backtest for (e.g., Q1_2023 Q2_2023)')
        parser.add_argument('--max_signals', type=int, default=100, help='Maximum number of signals to use')
        parser.add_argument('--initial_capital', type=float, default=300, help='Initial capital for the backtest')
        parser.add_argument('--multiple_runs', action='store_true', help='Run multiple backtests and average results')
        parser.add_argument('--num_runs', type=int, default=5, help='Number of backtest runs to perform when using --multiple_runs')
        parser.add_argument('--random_seed', type=int, default=42, help='Base random seed for reproducibility')
        parser.add_argument('--continuous_capital', action='store_true', help='Use continuous capital across quarters')
        parser.add_argument('--weekly_selection', action='store_true', help='Enable weekly stock selection refresh')
        args = parser.parse_args()
        
        # Define quarters mapping
        quarters_map = {
            'Q1_2023': ('2023-01-01', '2023-03-31'),
            'Q2_2023': ('2023-04-01', '2023-06-30'),
            'Q3_2023': ('2023-07-01', '2023-09-30'),
            'Q4_2023': ('2023-10-01', '2023-12-31'),
            'Q1_2024': ('2024-01-01', '2024-03-31')
        }
        
        # Check if 'all' is specified
        if 'all' in args.quarters:
            print("Running backtest for all quarters")
            run_all_quarters_backtest(
                max_signals=args.max_signals, 
                initial_capital=args.initial_capital,
                multiple_runs=args.multiple_runs,
                num_runs=args.num_runs,
                continuous_capital=args.continuous_capital,
                weekly_selection=args.weekly_selection
            )
            return
        
        # Run backtest for each quarter
        for quarter in args.quarters:
            run_comprehensive_backtest(
                quarter, 
                max_signals=args.max_signals, 
                initial_capital=args.initial_capital,
                multiple_runs=args.multiple_runs,
                num_runs=args.num_runs,
                continuous_capital=args.continuous_capital,
                weekly_selection=args.weekly_selection
            )
    except Exception as e:
        logger.error(f"Error running comprehensive backtest: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
