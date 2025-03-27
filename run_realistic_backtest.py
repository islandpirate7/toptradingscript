#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run a realistic historical backtest with real market data.
This script uses the fixed_backtest_v2 module to run a backtest with real historical data
from Alpaca, analyzing all S&P 500 and mid-cap stocks.
"""

import os
import sys
import yaml
import logging
import argparse
from datetime import datetime, timedelta

# Import our modules
from fixed_backtest_v2 import run_backtest
from fixed_signal_generator import generate_signals
from portfolio import Portfolio
from alpaca_api import AlpacaAPI

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run a realistic historical backtest')
    
    # Date range
    parser.add_argument('--start-date', type=str, default='2023-01-01',
                        help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end-date', type=str, default='2023-03-31',
                        help='End date in YYYY-MM-DD format')
    
    # Capital
    parser.add_argument('--initial-capital', type=float, default=100000,
                        help='Initial capital for the backtest')
    
    # Signal parameters
    parser.add_argument('--min-score', type=float, default=0.7,
                        help='Minimum score for a signal to be considered')
    parser.add_argument('--max-signals', type=int, default=30,
                        help='Maximum number of signals to use')
    
    # Tier thresholds
    parser.add_argument('--tier1-threshold', type=float, default=0.8,
                        help='Threshold for Tier 1 signals')
    parser.add_argument('--tier2-threshold', type=float, default=0.7,
                        help='Threshold for Tier 2 signals')
    
    # Allocation
    parser.add_argument('--largecap-allocation', type=float, default=0.7,
                        help='Allocation for large-cap stocks (0-1)')
    parser.add_argument('--midcap-allocation', type=float, default=0.3,
                        help='Allocation for mid-cap stocks (0-1)')
    
    # Config
    parser.add_argument('--config-path', type=str, default='sp500_config.yaml',
                        help='Path to the configuration file')
    
    return parser.parse_args()

def run_parameter_optimization(start_date, end_date, config_path):
    """
    Run a parameter optimization to find the best parameters.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Best parameters and their performance
    """
    logger.info("Starting parameter optimization...")
    
    # Define parameter ranges to test
    param_grid = {
        'min_score': [0.65, 0.7, 0.75, 0.8],
        'max_signals': [20, 30, 40, 50],
        'position_size': [0.03, 0.05, 0.07, 0.1],
        'stop_loss': [0.03, 0.05, 0.07, 0.1],
        'take_profit': [0.1, 0.15, 0.2, 0.25]
    }
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize best parameters and performance
    best_params = None
    best_performance = None
    best_sharpe = -float('inf')
    
    # Track all results
    all_results = []
    
    # Generate all parameter combinations (simplified approach)
    # In a real implementation, you might want to use a more sophisticated approach
    # like grid search or Bayesian optimization
    
    # For this example, we'll just test a few key combinations
    test_combinations = [
        {'min_score': 0.7, 'max_signals': 30, 'position_size': 0.05, 'stop_loss': 0.05, 'take_profit': 0.15},
        {'min_score': 0.75, 'max_signals': 30, 'position_size': 0.05, 'stop_loss': 0.05, 'take_profit': 0.15},
        {'min_score': 0.7, 'max_signals': 40, 'position_size': 0.05, 'stop_loss': 0.05, 'take_profit': 0.15},
        {'min_score': 0.7, 'max_signals': 30, 'position_size': 0.07, 'stop_loss': 0.05, 'take_profit': 0.15},
        {'min_score': 0.7, 'max_signals': 30, 'position_size': 0.05, 'stop_loss': 0.07, 'stop_loss': 0.05, 'take_profit': 0.15},
        {'min_score': 0.7, 'max_signals': 30, 'position_size': 0.05, 'stop_loss': 0.05, 'take_profit': 0.2},
    ]
    
    for params in test_combinations:
        logger.info(f"Testing parameters: {params}")
        
        # Update configuration with current parameters
        if 'strategy' not in config:
            config['strategy'] = {}
        if 'position_sizing' not in config['strategy']:
            config['strategy']['position_sizing'] = {}
        if 'risk_management' not in config['strategy']:
            config['strategy']['risk_management'] = {}
        
        config['strategy']['min_signal_score'] = params['min_score']
        config['strategy']['max_top_signals'] = params['max_signals']
        config['strategy']['position_sizing']['base_position_pct'] = params['position_size'] * 100
        config['strategy']['risk_management']['stop_loss'] = params['stop_loss']
        config['strategy']['risk_management']['take_profit'] = params['take_profit']
        
        # Save updated configuration
        temp_config_path = f"temp_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)
        
        try:
            # Run backtest with current parameters
            results = run_backtest(
                start_date=start_date,
                end_date=end_date,
                initial_capital=100000,
                config_path=temp_config_path,
                min_score=params['min_score'],
                max_signals=params['max_signals']
            )
            
            # Extract performance metrics
            performance = results['performance']
            
            # Calculate overall score (you can adjust the weights)
            # We prioritize Sharpe ratio, but also consider return and drawdown
            sharpe_ratio = performance.get('sharpe_ratio', 0)
            total_return = performance.get('return', 0)
            max_drawdown = performance.get('max_drawdown', 0)
            win_rate = performance.get('win_rate', 0)
            
            # Record results
            result_entry = {
                'params': params,
                'performance': performance,
                'sharpe_ratio': sharpe_ratio,
                'total_return': total_return,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate
            }
            all_results.append(result_entry)
            
            # Update best parameters if current is better
            if sharpe_ratio > best_sharpe:
                best_sharpe = sharpe_ratio
                best_params = params
                best_performance = performance
                
            logger.info(f"Results for parameters {params}:")
            logger.info(f"  - Sharpe Ratio: {sharpe_ratio:.2f}")
            logger.info(f"  - Total Return: {total_return:.2f}%")
            logger.info(f"  - Max Drawdown: {max_drawdown:.2f}%")
            logger.info(f"  - Win Rate: {win_rate:.2f}%")
            
        except Exception as e:
            logger.error(f"Error running backtest with parameters {params}: {str(e)}")
        
        # Clean up temporary config file
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
    
    # Log best parameters
    if best_params:
        logger.info(f"Best parameters found: {best_params}")
        logger.info(f"Best performance:")
        logger.info(f"  - Sharpe Ratio: {best_performance.get('sharpe_ratio', 0):.2f}")
        logger.info(f"  - Total Return: {best_performance.get('return', 0):.2f}%")
        logger.info(f"  - Max Drawdown: {best_performance.get('max_drawdown', 0):.2f}%")
        logger.info(f"  - Win Rate: {best_performance.get('win_rate', 0):.2f}%")
    
    # Return best parameters and all results
    return {
        'best_params': best_params,
        'best_performance': best_performance,
        'all_results': all_results
    }

def main():
    """Run the main backtest."""
    # Parse arguments
    args = parse_arguments()
    
    # Log start
    logger.info("Starting realistic historical backtest")
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    logger.info(f"Initial capital: ${args.initial_capital:,.2f}")
    logger.info(f"Signal parameters: min_score={args.min_score}, max_signals={args.max_signals}")
    logger.info(f"Tier thresholds: tier1={args.tier1_threshold}, tier2={args.tier2_threshold}")
    
    # Run backtest
    results = run_backtest(
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        config_path=args.config_path,
        min_score=args.min_score,
        max_signals=args.max_signals,
        tier1_threshold=args.tier1_threshold,
        tier2_threshold=args.tier2_threshold,
        largecap_allocation=args.largecap_allocation,
        midcap_allocation=args.midcap_allocation
    )
    
    # Check if backtest was successful
    if not results.get('success', True):
        logger.error(f"Backtest failed: {results.get('error', 'Unknown error')}")
        return
    
    # Extract performance metrics
    performance = results['performance']
    portfolio = results['portfolio']
    signals = results['signals']
    log_file = results['log_file']
    
    # Print performance metrics
    logger.info("Backtest completed successfully")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Final portfolio value: ${performance.get('final_value', 0):,.2f}")
    logger.info(f"Total return: {performance.get('return', 0):.2f}%")
    logger.info(f"Annualized return: {performance.get('annualized_return', 0):.2f}%")
    logger.info(f"Sharpe ratio: {performance.get('sharpe_ratio', 0):.2f}")
    logger.info(f"Max drawdown: {performance.get('max_drawdown', 0):.2f}%")
    logger.info(f"Win rate: {performance.get('win_rate', 0):.2f}%")
    
    # Automatically run parameter optimization
    logger.info("\nRunning parameter optimization...")
    optimization_results = run_parameter_optimization(
        start_date=args.start_date,
        end_date=args.end_date,
        config_path=args.config_path
    )
    
    # Print optimization results
    logger.info("Parameter optimization completed")
    logger.info(f"Best parameters: {optimization_results['best_params']}")
    logger.info(f"Best performance:")
    logger.info(f"  - Sharpe Ratio: {optimization_results['best_performance'].get('sharpe_ratio', 0):.2f}")
    logger.info(f"  - Total Return: {optimization_results['best_performance'].get('return', 0):.2f}%")
    logger.info(f"  - Max Drawdown: {optimization_results['best_performance'].get('max_drawdown', 0):.2f}%")
    logger.info(f"  - Win Rate: {optimization_results['best_performance'].get('win_rate', 0):.2f}%")
    
    logger.info("Backtest process completed")

if __name__ == "__main__":
    main()
