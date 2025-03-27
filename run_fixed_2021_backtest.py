#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run Fixed Backtest for 2021
--------------------------
This script runs the fixed backtest function for the year 2021.
"""

import os
import sys
import yaml
import logging
import datetime
from datetime import datetime
from tqdm import tqdm

# Import our modules
from alpaca_api import AlpacaAPI
from portfolio import Portfolio
from signal_generator import generate_signals
from fixed_backtest import fixed_run_backtest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    # Load configuration
    try:
        with open('sp500_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        logger.info("Successfully loaded configuration from sp500_config.yaml")
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        return
    
    # Set up backtest parameters
    start_date = '2021-01-01'
    end_date = '2021-12-31'
    initial_capital = 100000
    
    logger.info(f"Starting backtest from {start_date} to {end_date}")
    
    # Create log directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create log file
    log_filename = f"logs/fixed_strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file_handle = open(log_filename, 'w')
    log_file_handle.write(f"{datetime.now()} - INFO - Starting backtest from {start_date} to {end_date}\n")
    log_file_handle.write(f"{datetime.now()} - INFO - Backtest log file created: {log_filename}\n")
    log_file_handle.write(f"{datetime.now()} - INFO - Initial capital: ${initial_capital}\n")
    
    # Initialize API
    api = AlpacaAPI(
        api_key=config['alpaca']['api_key'],
        api_secret=config['alpaca']['api_secret'],
        base_url=config['alpaca']['base_url'],
        data_url=config['alpaca']['data_url']
    )
    
    # Generate signals
    signals = generate_signals(
        start_date=start_date,
        end_date=end_date,
        config=config,
        alpaca=api
    )
    
    # Initialize portfolio
    portfolio = Portfolio(
        initial_capital=initial_capital,
        cash_allocation=0.9,
        max_positions=20,
        position_size=0.1,
        stop_loss=0.05,
        take_profit=0.15
    )
    
    # Run backtest
    result = fixed_run_backtest(
        api=api,
        portfolio=portfolio,
        start_date=start_date,
        end_date=end_date,
        signals=signals,
        tier1_threshold=0.8,
        tier2_threshold=0.6,
        log_file_handle=log_file_handle
    )
    
    # Check if backtest was successful
    if result['success']:
        performance = result['performance']
        
        # Print performance metrics
        logger.info("Backtest completed successfully")
        logger.info(f"Final portfolio value: ${performance['final_value']:.2f}")
        logger.info(f"Return: {performance['return']:.2f}%")
        logger.info(f"Annualized return: {performance['annualized_return']:.2f}%")
        logger.info(f"Sharpe ratio: {performance['sharpe_ratio']:.2f}")
        logger.info(f"Max drawdown: {performance['max_drawdown']:.2f}%")
        logger.info(f"Win rate: {performance['win_rate']:.2f}%")
        logger.info(f"Log file: {log_filename}")
        
        # Print equity curve summary
        equity_curve = result.get('equity_curve', [])
        logger.info(f"Equity curve points: {len(equity_curve)}")
        if equity_curve:
            logger.info(f"First equity point: {equity_curve[0]['timestamp']} - ${equity_curve[0]['equity']:.2f}")
            if len(equity_curve) > 1:
                logger.info(f"Last equity point: {equity_curve[-1]['timestamp']} - ${equity_curve[-1]['equity']:.2f}")
    else:
        logger.error(f"Backtest failed: {result.get('error', 'Unknown error')}")
    
    # Close log file
    log_file_handle.close()

if __name__ == "__main__":
    main()
