#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for running a backtest using the final_sp500_strategy module
"""

import os
import sys
import yaml
import logging
import datetime
import pandas as pd
import numpy as np
from final_sp500_strategy import run_backtest
from backtest_data_provider import BacktestDataProvider

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def load_or_create_config():
    """Load the configuration file or create it with default values"""
    config_path = 'sp500_config.yaml'
    
    # Check if config file exists
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    else:
        logger.error(f"Configuration file {config_path} not found")
        return None
    
    # Check if Alpaca API credentials are set
    if 'alpaca' not in config or not config['alpaca']['api_key'] or config['alpaca']['api_key'] == 'YOUR_API_KEY':
        # Try to get API keys from environment variables
        api_key = os.environ.get('ALPACA_API_KEY')
        api_secret = os.environ.get('ALPACA_API_SECRET')
        
        if api_key and api_secret:
            logger.info("Using Alpaca API credentials from environment variables")
            config['alpaca'] = {
                'api_key': api_key,
                'api_secret': api_secret,
                'base_url': 'https://paper-api.alpaca.markets'
            }
            
            # Save the updated config
            with open(config_path, 'w') as file:
                yaml.dump(config, file, default_flow_style=False)
        else:
            logger.warning("Alpaca API credentials not found in config or environment variables")
            logger.warning("Using placeholder values for testing purposes")
            # Use placeholder values for testing
            config['alpaca'] = {
                'api_key': 'PLACEHOLDER_KEY',
                'api_secret': 'PLACEHOLDER_SECRET',
                'base_url': 'https://paper-api.alpaca.markets'
            }
            
            # Save the updated config
            with open(config_path, 'w') as file:
                yaml.dump(config, file, default_flow_style=False)
    
    return config

def main():
    """Run a test backtest"""
    try:
        # Load or create configuration
        config = load_or_create_config()
        if not config:
            logger.error("Failed to load or create configuration")
            return
        
        # Set up test parameters
        end_date = '2023-03-22'  # Use a date in the past
        start_date = '2023-02-20'  # 30 days before end_date
        
        logger.info(f"Running backtest from {start_date} to {end_date}")
        
        # Create data directory if it doesn't exist
        data_dir = config.get('paths', {}).get('data', './data')
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize the backtest data provider
        data_provider = BacktestDataProvider(data_dir=data_dir, use_local_data=True)
        
        # Check if we have any local data
        available_symbols = data_provider.list_available_symbols()
        if not available_symbols:
            logger.info("No local data found. Will generate synthetic data for backtesting.")
        else:
            logger.info(f"Found local data for {len(available_symbols)} symbols.")
        
        # Run the backtest
        results = run_backtest(
            start_date=start_date,
            end_date=end_date,
            mode='backtest',
            initial_capital=10000,
            random_seed=42,
            config_path='sp500_config.yaml',
            max_signals=10,
            min_score=0.6,
            tier1_threshold=0.8,
            tier2_threshold=0.7,
            tier3_threshold=0.6,
            data_provider=data_provider  # Pass the data provider to the backtest
        )
        
        # Check if the backtest was successful
        if isinstance(results, dict) and 'success' in results and results['success'] == False:
            logger.error(f"Backtest failed: {results['error']}")
            return
        
        # Display results
        logger.info("Backtest completed successfully")
        
        if 'performance' in results:
            performance = results['performance']
            logger.info(f"Final portfolio value: ${performance['final_value']:.2f}")
            logger.info(f"Return: {performance['return']:.2f}%")
            logger.info(f"Annualized return: {performance['annualized_return']:.2f}%")
            logger.info(f"Sharpe ratio: {performance['sharpe_ratio']:.2f}")
            logger.info(f"Max drawdown: {performance['max_drawdown']:.2f}%")
            logger.info(f"Win rate: {performance['win_rate']:.2f}%")
        
        # Check if log file was created
        if 'log_file' in results:
            log_file = results['log_file']
            logger.info(f"Log file created: {log_file}")
            
            # Display the first few lines of the log file
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()[:10]
                    logger.info("First few lines of log file:")
                    for line in lines:
                        print(line.strip())
            except Exception as e:
                logger.error(f"Error reading log file: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        import traceback
        traceback.print_exc()

def test_backtest():
    """Run a test backtest to verify implementation"""
    # Load configuration
    config = load_or_create_config()
    
    # Set test parameters
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    initial_capital = 100000
    
    logger.info(f"Running test backtest from {start_date} to {end_date} with ${initial_capital} initial capital")
    
    # Generate synthetic signals for testing
    def generate_test_signals(start_date, end_date, config, alpaca=None):
        """Generate synthetic signals for testing"""
        signals = []
        
        # Convert dates to datetime
        start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        
        # Generate dates
        dates = []
        current_dt = start_dt
        while current_dt <= end_dt:
            if current_dt.weekday() < 5:  # Weekdays only
                dates.append(current_dt.strftime('%Y-%m-%d'))
            current_dt += datetime.timedelta(days=1)
        
        # Generate signals for each date
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'AMD', 'INTC', 'IBM']
        
        for date in dates:
            # Generate 2-3 signals per day
            num_signals = np.random.randint(2, 4)
            selected_symbols = np.random.choice(symbols, num_signals, replace=False)
            
            for symbol in selected_symbols:
                # Generate random score between 0.6 and 0.9
                score = np.random.uniform(0.6, 0.9)
                
                # Create signal
                signal = {
                    'symbol': symbol,
                    'date': date,
                    'score': score,
                    'direction': 'LONG',
                    'price': np.random.uniform(50, 200)
                }
                
                signals.append(signal)
        
        logger.info(f"Generated {len(signals)} test signals")
        return signals
    
    # Run backtest with test signals
    results = run_backtest(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        config=config,
        generate_signals=generate_test_signals,
        min_score=0.6,
        max_signals=20
    )
    
    # Display results
    if results and 'performance' in results:
        perf = results['performance']
        print("\n===== BACKTEST RESULTS =====")
        print(f"Initial Capital: ${perf.get('initial_capital', 0):.2f}")
        print(f"Final Value: ${perf.get('final_value', 0):.2f}")
        print(f"Return: {perf.get('return', 0):.2f}%")
        print(f"Annualized Return: {perf.get('annualized_return', 0):.2f}%")
        print(f"Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown: {perf.get('max_drawdown', 0):.2f}%")
        print(f"Win Rate: {perf.get('win_rate', 0):.2f}%")
        print(f"Total Trades: {perf.get('total_trades', 0)}")
        print(f"Winning Trades: {perf.get('winning_trades', 0)}")
        print(f"Losing Trades: {perf.get('losing_trades', 0)}")
        print("============================\n")
        
        # Check if portfolio has open positions
        if 'portfolio' in results:
            portfolio = results['portfolio']
            print(f"Open Positions: {len(portfolio.open_positions)}")
            print(f"Closed Positions: {len(portfolio.closed_positions)}")
            print(f"Final Cash: ${portfolio.cash:.2f}")
            
            # Print sample of trades if available
            if hasattr(portfolio, 'trade_history') and portfolio.trade_history:
                print("\n===== SAMPLE TRADES =====")
                for i, trade in enumerate(portfolio.trade_history[:5]):
                    print(f"Trade {i+1}: {trade['action']} {trade['symbol']} - {trade['shares']} shares @ ${trade['price']:.2f}")
                print("=========================\n")
    else:
        print("Backtest failed or returned no results")

if __name__ == "__main__":
    main()
    test_backtest()
