#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run Alpaca Backtest
-------------------
This script runs a backtest using real market data from Alpaca API.
"""

import os
import sys
import yaml
import logging
from datetime import datetime, timedelta
import pandas as pd
import alpaca_trade_api as tradeapi

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/alpaca_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Import the backtest function
from fixed_backtest_v2 import run_backtest
from backtest_data_provider import BacktestDataProvider

def load_config():
    """Load the configuration file"""
    config_path = 'sp500_config.yaml'
    
    # Check if config file exists
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    else:
        logger.error(f"Configuration file {config_path} not found")
        return None

def initialize_alpaca_api(config):
    """Initialize Alpaca API with credentials from config"""
    if 'alpaca' not in config or not config['alpaca']['api_key'] or config['alpaca']['api_key'] == 'YOUR_API_KEY':
        # Try to get API keys from environment variables
        api_key = os.environ.get('ALPACA_API_KEY')
        api_secret = os.environ.get('ALPACA_API_SECRET')
        
        if not api_key or not api_secret:
            logger.error("Alpaca API credentials not found in config or environment variables")
            return None
            
        base_url = config.get('alpaca', {}).get('base_url', 'https://paper-api.alpaca.markets')
    else:
        api_key = config['alpaca']['api_key']
        api_secret = config['alpaca']['api_secret']
        base_url = config['alpaca']['base_url']
    
    logger.info(f"Initializing Alpaca API with base URL: {base_url}")
    
    try:
        api = tradeapi.REST(
            api_key,
            api_secret,
            base_url=base_url,
            api_version='v2'
        )
        # Test the API connection
        account = api.get_account()
        logger.info(f"Connected to Alpaca API. Account status: {account.status}")
        return api
    except Exception as e:
        logger.error(f"Error connecting to Alpaca API: {str(e)}")
        return None

def fix_symbol_format(symbols):
    """Fix known symbol format issues for Alpaca API"""
    replacements = {
        'BRK-B': 'BRK.B',
        'BF-B': 'BF.B'
    }
    
    fixed_symbols = []
    for symbol in symbols:
        if symbol in replacements:
            fixed_symbols.append(replacements[symbol])
            logger.info(f"Fixed symbol format: {symbol} -> {replacements[symbol]}")
        else:
            fixed_symbols.append(symbol)
    
    return fixed_symbols

def main():
    """Run a backtest using real Alpaca data"""
    try:
        # Load configuration
        config = load_config()
        if not config:
            logger.error("Failed to load configuration")
            return
        
        # Initialize Alpaca API
        api = initialize_alpaca_api(config)
        if not api:
            logger.error("Failed to initialize Alpaca API")
            return
        
        # Set up backtest parameters
        end_date = datetime.now() - timedelta(days=1)  # Yesterday
        start_date = end_date - timedelta(days=90)  # 90 days before end_date (3 months)
        
        # Format dates as strings
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        logger.info(f"Running backtest from {start_date_str} to {end_date_str}")
        
        # Create data directory if it doesn't exist
        data_dir = config.get('paths', {}).get('data', './data')
        os.makedirs(data_dir, exist_ok=True)
        
        # Create logs directory if it doesn't exist
        logs_dir = './logs'
        os.makedirs(logs_dir, exist_ok=True)
        
        # Initialize the backtest data provider with Alpaca API
        data_provider = BacktestDataProvider(data_dir=data_dir, use_local_data=False)
        data_provider.alpaca_api = api
        
        # Apply symbol format fixes
        if 'symbols' in config and 'sp500' in config['symbols']:
            config['symbols']['sp500'] = fix_symbol_format(config['symbols']['sp500'])
        
        # Get backtest parameters from config
        backtest_config = config.get('backtest', {})
        max_signals = backtest_config.get('max_signals', 10)
        min_score = backtest_config.get('min_score', 0.3)  # Lower the min score to match signal generator
        tier1_threshold = backtest_config.get('tier1_threshold', 0.7)  # Adjusted tier thresholds
        tier2_threshold = backtest_config.get('tier2_threshold', 0.6)
        tier3_threshold = backtest_config.get('tier3_threshold', 0.5)
        initial_capital = backtest_config.get('initial_capital', 20000)  # Increase initial capital
        
        logger.info(f"Running backtest with parameters:")
        logger.info(f"- Initial capital: ${initial_capital}")
        logger.info(f"- Max signals: {max_signals}")
        logger.info(f"- Min score: {min_score}")
        logger.info(f"- Tier thresholds: Tier 1: {tier1_threshold}, Tier 2: {tier2_threshold}, Tier 3: {tier3_threshold}")
        
        # Run the backtest
        results = run_backtest(
            start_date=start_date_str,
            end_date=end_date_str,
            mode='backtest',
            initial_capital=initial_capital,
            random_seed=42,
            config_path='sp500_config.yaml',
            max_signals=max_signals,
            min_score=min_score,
            tier1_threshold=tier1_threshold,
            tier2_threshold=tier2_threshold,
            tier3_threshold=tier3_threshold,
            data_provider=data_provider  # Pass the data provider to the backtest
        )
        
        # Check if the backtest was successful
        if isinstance(results, dict) and 'success' in results and results['success'] == False:
            logger.error(f"Backtest failed: {results.get('error', 'Unknown error')}")
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
            
            # Log equity curve details
            if 'equity_curve' in performance and performance['equity_curve']:
                equity_curve = performance['equity_curve']
                logger.info(f"Equity curve points: {len(equity_curve)}")
                
                # Log first 3 and last 3 points
                for i, point in enumerate(equity_curve[:3]):
                    logger.info(f"Equity point {i}: {point['timestamp']} - ${point['equity']:.2f}")
                
                if len(equity_curve) > 6:
                    logger.info("...")
                    
                for i, point in enumerate(equity_curve[-3:]):
                    idx = len(equity_curve) - 3 + i
                    logger.info(f"Equity point {idx}: {point['timestamp']} - ${point['equity']:.2f}")
        
        # Log open positions
        if 'portfolio' in results:
            portfolio = results['portfolio']
            logger.info(f"Open positions: {len(portfolio.open_positions)}")
            
            for symbol, position in portfolio.open_positions.items():
                logger.info(f"Open position: {symbol} - {position.direction} - {position.position_size} shares at ${position.entry_price:.2f}")
                
                # Calculate unrealized P&L if we have current price data
                if hasattr(position, 'current_price') and position.current_price:
                    unrealized_pnl, unrealized_pnl_pct = position.get_unrealized_pnl(position.current_price)
                    logger.info(f"  Unrealized P&L: ${unrealized_pnl:.2f} ({unrealized_pnl_pct:.2f}%)")
            
            # Log closed positions
            logger.info(f"Closed positions: {len(portfolio.closed_positions)}")
            
            for i, position in enumerate(portfolio.closed_positions):
                logger.info(f"Closed position {i+1}: {position.symbol} - {position.direction} - Entry: ${position.entry_price:.2f}, Exit: ${position.exit_price:.2f}")
                logger.info(f"  P&L: ${position.pnl:.2f} ({position.pnl_pct:.2f}%) - Reason: {position.exit_reason}")
        
        # Check if trades file was created
        if 'trades_file' in results:
            trades_file = results['trades_file']
            logger.info(f"Trades file created: {trades_file}")
            
            # Display the first few trades
            try:
                trades_df = pd.read_csv(trades_file)
                logger.info(f"Total trades: {len(trades_df)}")
                logger.info("First few trades:")
                logger.info(trades_df.head().to_string())
            except Exception as e:
                logger.error(f"Error reading trades file: {str(e)}")
        
        # Create a detailed performance report
        if 'log_file' in results:
            log_file = results['log_file']
            logger.info(f"Log file created: {log_file}")
            
            # Create a performance report file
            report_file = f"logs/performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
            with open(report_file, 'w') as f:
                f.write("=== BACKTEST PERFORMANCE REPORT ===\n\n")
                f.write(f"Backtest period: {start_date_str} to {end_date_str}\n")
                f.write(f"Initial capital: ${initial_capital:.2f}\n")
                
                if 'performance' in results:
                    perf = results['performance']
                    f.write(f"Final portfolio value: ${perf['final_value']:.2f}\n")
                    f.write(f"Total return: {perf['return']:.2f}%\n")
                    f.write(f"Annualized return: {perf['annualized_return']:.2f}%\n")
                    f.write(f"Sharpe ratio: {perf['sharpe_ratio']:.2f}\n")
                    f.write(f"Max drawdown: {perf['max_drawdown']:.2f}%\n")
                    f.write(f"Win rate: {perf['win_rate']:.2f}%\n\n")
                
                if 'portfolio' in results:
                    port = results['portfolio']
                    f.write(f"=== POSITION SUMMARY ===\n")
                    f.write(f"Open positions: {len(port.open_positions)}\n")
                    f.write(f"Closed positions: {len(port.closed_positions)}\n\n")
                    
                    if port.closed_positions:
                        f.write("=== CLOSED POSITIONS ===\n")
                        for pos in port.closed_positions:
                            f.write(f"{pos.symbol} ({pos.direction}):\n")
                            f.write(f"  Entry: ${pos.entry_price:.2f} on {pos.entry_time}\n")
                            f.write(f"  Exit: ${pos.exit_price:.2f} on {pos.exit_time}\n")
                            f.write(f"  P&L: ${pos.pnl:.2f} ({pos.pnl_pct:.2f}%)\n")
                            f.write(f"  Reason: {pos.exit_reason}\n\n")
            
            logger.info(f"Performance report created: {report_file}")
        
        # Return results for further analysis
        return results
        
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
