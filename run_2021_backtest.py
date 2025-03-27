#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run a backtest for 2021
"""

import os
import sys
import yaml
import logging
from datetime import datetime
from final_sp500_strategy import run_backtest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def main():
    # Load configuration
    with open('sp500_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Set backtest parameters
    start_date = '2021-01-01'
    end_date = '2021-12-31'
    initial_capital = 100000
    max_signals = 20  # Limit to top 20 signals
    min_score = 0.6   # Minimum signal score threshold
    
    # Run the backtest
    logger.info(f"Starting backtest from {start_date} to {end_date}")
    
    try:
        results = run_backtest(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            max_signals=max_signals,
            min_score=min_score,
            config_path='sp500_config.yaml'
        )
        
        # Check if results contains the expected keys
        if isinstance(results, dict) and 'performance' in results:
            performance = results['performance']
            
            # Print performance metrics
            logger.info("Backtest completed successfully")
            logger.info(f"Final portfolio value: ${performance['final_value']:.2f}")
            logger.info(f"Return: {performance['return']:.2f}%")
            logger.info(f"Annualized return: {performance['annualized_return']:.2f}%")
            logger.info(f"Sharpe ratio: {performance['sharpe_ratio']:.2f}")
            logger.info(f"Max drawdown: {performance['max_drawdown']:.2f}%")
            logger.info(f"Win rate: {performance['win_rate']:.2f}%")
            
            # Print log file location
            if 'log_file' in results:
                logger.info(f"Log file: {results['log_file']}")
                
            # Print portfolio equity curve points (first, middle, last)
            if 'portfolio' in results:
                portfolio = results['portfolio']
                if hasattr(portfolio, 'equity_curve') and len(portfolio.equity_curve) > 0:
                    logger.info(f"Equity curve points: {len(portfolio.equity_curve)}")
                    
                    if len(portfolio.equity_curve) > 0:
                        first_point = portfolio.equity_curve[0]
                        logger.info(f"First equity point: {first_point['timestamp']} - ${first_point['equity']:.2f}")
                    
                    if len(portfolio.equity_curve) > 2:
                        middle_idx = len(portfolio.equity_curve) // 2
                        middle_point = portfolio.equity_curve[middle_idx]
                        logger.info(f"Middle equity point: {middle_point['timestamp']} - ${middle_point['equity']:.2f}")
                    
                    if len(portfolio.equity_curve) > 1:
                        last_point = portfolio.equity_curve[-1]
                        logger.info(f"Last equity point: {last_point['timestamp']} - ${last_point['equity']:.2f}")
        else:
            logger.error("Backtest results are not in the expected format")
            if isinstance(results, dict) and 'error' in results:
                logger.error(f"Error: {results['error']}")
    
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
