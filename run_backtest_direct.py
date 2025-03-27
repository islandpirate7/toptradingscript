#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Direct Backtest Runner

This script directly runs a backtest without going through the web interface.
It imports the run_backtest function from final_sp500_strategy.py and runs it with specified parameters.
"""

import os
import sys
import json
import yaml
import logging
import datetime
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/backtest_direct_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

def main():
    """Main function"""
    logger.info("Starting direct backtest")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run a backtest directly')
    parser.add_argument('--start-date', type=str, default='2023-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2023-03-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--initial-capital', type=float, default=300, help='Initial capital')
    parser.add_argument('--max-signals', type=int, default=40, help='Maximum signals per day')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed')
    parser.add_argument('--weekly-selection', action='store_true', help='Use weekly selection')
    parser.add_argument('--continuous-capital', action='store_true', help='Use continuous capital')
    parser.add_argument('--tier1-threshold', type=float, default=0.8, help='Tier 1 threshold')
    parser.add_argument('--tier2-threshold', type=float, default=0.7, help='Tier 2 threshold')
    parser.add_argument('--tier3-threshold', type=float, default=0.6, help='Tier 3 threshold')
    parser.add_argument('--quarter', type=str, help='Quarter to run (e.g., Q1_2023)')
    
    args = parser.parse_args()
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Add the parent directory to the path
    sys.path.insert(0, script_dir)
    
    # Check if quarter is specified
    if args.quarter:
        # Define quarter date ranges
        quarter_dates = {
            'Q1_2023': ('2023-01-01', '2023-03-31'),
            'Q2_2023': ('2023-04-01', '2023-06-30'),
            'Q3_2023': ('2023-07-01', '2023-09-30'),
            'Q4_2023': ('2023-10-01', '2023-12-31'),
            'Q1_2024': ('2024-01-01', '2024-03-31'),
            'Q2_2024': ('2024-04-01', '2024-06-30'),
            'Q3_2024': ('2024-07-01', '2024-09-30'),
            'Q4_2024': ('2024-10-01', '2024-12-31')
        }
        
        if args.quarter in quarter_dates:
            args.start_date, args.end_date = quarter_dates[args.quarter]
            logger.info(f"Using date range for {args.quarter}: {args.start_date} to {args.end_date}")
        else:
            logger.error(f"Invalid quarter: {args.quarter}")
            sys.exit(1)
    
    # Check if sp500_config.yaml exists
    config_path = os.path.join(script_dir, 'sp500_config.yaml')
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    # Check if alpaca_credentials.json exists
    credentials_path = os.path.join(script_dir, 'alpaca_credentials.json')
    if not os.path.exists(credentials_path):
        logger.error(f"Credentials file not found: {credentials_path}")
        sys.exit(1)
    
    # Create required directories
    required_dirs = [
        'backtest_results',
        'data',
        'logs',
        'models',
        'plots',
        'results',
        'trades',
        os.path.join('performance', 'SP500Strategy')
    ]
    
    for directory in required_dirs:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")
    
    # Import the backtest function
    try:
        logger.info("Importing run_backtest from final_sp500_strategy")
        from final_sp500_strategy import run_backtest as strategy_run_backtest
    except ImportError as e:
        logger.error(f"Error importing run_backtest: {str(e)}")
        sys.exit(1)
    
    # Run the backtest
    logger.info(f"Running backtest with parameters:")
    logger.info(f"  - Start date: {args.start_date}")
    logger.info(f"  - End date: {args.end_date}")
    logger.info(f"  - Initial capital: {args.initial_capital}")
    logger.info(f"  - Max signals: {args.max_signals}")
    logger.info(f"  - Random seed: {args.random_seed}")
    logger.info(f"  - Weekly selection: {args.weekly_selection}")
    logger.info(f"  - Continuous capital: {args.continuous_capital}")
    logger.info(f"  - Tier 1 threshold: {args.tier1_threshold}")
    logger.info(f"  - Tier 2 threshold: {args.tier2_threshold}")
    logger.info(f"  - Tier 3 threshold: {args.tier3_threshold}")
    
    try:
        result = strategy_run_backtest(
            start_date=args.start_date,
            end_date=args.end_date,
            mode='backtest',
            max_signals=args.max_signals,
            initial_capital=args.initial_capital,
            random_seed=args.random_seed,
            weekly_selection=args.weekly_selection,
            continuous_capital=args.continuous_capital,
            tier1_threshold=args.tier1_threshold,
            tier2_threshold=args.tier2_threshold,
            tier3_threshold=args.tier3_threshold
        )
        
        logger.info("Backtest completed successfully")
        
        # Print summary of results
        if isinstance(result, dict):
            logger.info("Backtest Results Summary:")
            
            if 'performance' in result:
                perf = result['performance']
                logger.info(f"  - Initial Capital: ${perf.get('initial_capital', 0):.2f}")
                logger.info(f"  - Final Value: ${perf.get('final_value', 0):.2f}")
                logger.info(f"  - Return: {perf.get('return', 0) * 100:.2f}%")
                logger.info(f"  - Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}")
                logger.info(f"  - Max Drawdown: {perf.get('max_drawdown', 0) * 100:.2f}%")
                logger.info(f"  - Win Rate: {perf.get('win_rate', 0) * 100:.2f}%")
            
            if 'trades' in result:
                logger.info(f"  - Total Trades: {len(result['trades'])}")
            
            if 'result_file' in result:
                logger.info(f"  - Result File: {result['result_file']}")
        else:
            logger.info(f"Backtest result: {result}")
        
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
