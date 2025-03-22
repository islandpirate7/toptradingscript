#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Expanded Universe Test for Original Trading Model
------------------------------------------------
This script runs the original trading model (which showed the best performance)
with an expanded universe of stocks from Alpaca.
"""

import os
import sys
import logging
import datetime as dt
import yaml
import json
import time
import argparse
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import the multi-strategy system
from multi_strategy_system import (
    MultiStrategySystem, SystemConfig, StockConfig, BacktestResult, 
    MarketRegime, MarketState, CandleData
)

# Import Alpaca integration
from alpaca_integration import AlpacaIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('expanded_universe_test.log')
    ]
)

logger = logging.getLogger("ExpandedUniverseTest")

def load_alpaca_credentials():
    """Load Alpaca API credentials from environment variables or config file"""
    # Try environment variables first
    api_key = os.environ.get('ALPACA_API_KEY')
    api_secret = os.environ.get('ALPACA_API_SECRET')
    
    # If not found, try config file
    if not api_key or not api_secret:
        try:
            with open('alpaca_credentials.json', 'r') as f:
                creds = json.load(f)
                api_key = creds.get('api_key')
                api_secret = creds.get('api_secret')
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    
    return api_key, api_secret

def create_expanded_config(original_config_file: str, alpaca_assets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create an expanded configuration with more stocks from Alpaca"""
    # Load original configuration
    with open(original_config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get a sample stock config to use as template
    sample_stock_config = config['stocks'][0] if config.get('stocks') else {}
    
    # Create new stock configurations
    new_stocks = []
    for asset in alpaca_assets:
        # Skip if already in original config
        if any(s.get('symbol') == asset['symbol'] for s in config.get('stocks', [])):
            continue
            
        # Create new stock config based on template
        stock_config = {
            'symbol': asset['symbol'],
            'max_position_size': min(int(1000000 / asset['price']), 1000),  # Limit to 1000 shares
            'min_position_size': 10,
            'max_risk_per_trade_pct': 0.5,
            'min_volume': int(asset['volume'] * 0.1),  # 10% of average volume
            'beta': 1.0,  # Default beta
            'sector': '',  # We don't have sector info from Alpaca
            'industry': '',
        }
        
        # Copy strategy parameters from template
        for param_key in ['mean_reversion_params', 'trend_following_params', 
                         'volatility_breakout_params', 'gap_trading_params']:
            if param_key in sample_stock_config:
                stock_config[param_key] = sample_stock_config[param_key]
        
        new_stocks.append(stock_config)
    
    # Add new stocks to config
    config['stocks'].extend(new_stocks)
    
    # Update other settings
    config['max_open_positions'] = min(50, len(config['stocks']) // 2)  # Allow more open positions
    config['data_source'] = 'ALPACA'  # Use Alpaca as data source
    
    logger.info(f"Created expanded configuration with {len(config['stocks'])} stocks")
    return config

def save_config(config: Dict[str, Any], output_file: str):
    """Save configuration to a YAML file"""
    with open(output_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Saved expanded configuration to {output_file}")

def run_expanded_universe_test(config_file: str, start_date: str, end_date: str, 
                              api_key: str, api_secret: str, output_file: str = None):
    """Run the trading model with an expanded universe of stocks"""
    # Parse dates
    start_date_obj = dt.datetime.strptime(start_date, '%Y-%m-%d').date()
    end_date_obj = dt.datetime.strptime(end_date, '%Y-%m-%d').date()
    
    # Initialize Alpaca integration
    try:
        alpaca = AlpacaIntegration(api_key, api_secret)
        logger.info("Alpaca integration initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Alpaca integration: {str(e)}")
        return None
    
    # Get tradable assets from Alpaca
    assets = alpaca.get_tradable_assets(
        min_price=5.0,          # Minimum price $5
        min_volume=500000,      # Minimum average volume 500K
        max_stocks=100          # Limit to 100 stocks for now
    )
    
    if not assets:
        logger.error("No tradable assets found from Alpaca")
        return None
    
    logger.info(f"Found {len(assets)} tradable assets from Alpaca")
    
    # Create expanded configuration
    expanded_config = create_expanded_config(config_file, assets)
    
    # Save expanded configuration
    expanded_config_file = 'expanded_universe_config.yaml'
    save_config(expanded_config, expanded_config_file)
    
    # Set Alpaca credentials in config
    expanded_config['api_key'] = api_key
    expanded_config['api_secret'] = api_secret
    
    # Create system configuration
    stock_configs = []
    for stock_data in expanded_config.get('stocks', []):
        stock_config = StockConfig(
            symbol=stock_data['symbol'],
            max_position_size=stock_data.get('max_position_size', 1000),
            min_position_size=stock_data.get('min_position_size', 10),
            max_risk_per_trade_pct=stock_data.get('max_risk_per_trade_pct', 1.0),
            min_volume=stock_data.get('min_volume', 5000),
            avg_daily_volume=stock_data.get('avg_daily_volume', 0),
            beta=stock_data.get('beta', 1.0),
            sector=stock_data.get('sector', ''),
            industry=stock_data.get('industry', ''),
            mean_reversion_params=stock_data.get('mean_reversion_params', {}),
            trend_following_params=stock_data.get('trend_following_params', {}),
            volatility_breakout_params=stock_data.get('volatility_breakout_params', {}),
            gap_trading_params=stock_data.get('gap_trading_params', {})
        )
        stock_configs.append(stock_config)
    
    # Parse market hours
    market_open_str = expanded_config.get('market_hours_start', '09:30')
    market_close_str = expanded_config.get('market_hours_end', '16:00')
    
    market_open = dt.datetime.strptime(market_open_str, '%H:%M').time()
    market_close = dt.datetime.strptime(market_close_str, '%H:%M').time()
    
    # Parse strategy weights
    strategy_weights = expanded_config.get('strategy_weights', {
        "MeanReversion": 0.25,
        "TrendFollowing": 0.25,
        "VolatilityBreakout": 0.25,
        "GapTrading": 0.25
    })
    
    # Parse rebalance interval
    rebalance_str = expanded_config.get('rebalance_interval', '1d')
    rebalance_unit = rebalance_str[-1]
    rebalance_value = int(rebalance_str[:-1])
    
    if rebalance_unit == 'd':
        rebalance_interval = dt.timedelta(days=rebalance_value)
    elif rebalance_unit == 'h':
        rebalance_interval = dt.timedelta(hours=rebalance_value)
    else:
        rebalance_interval = dt.timedelta(days=1)
    
    # Create system configuration
    system_config = SystemConfig(
        stocks=stock_configs,
        initial_capital=expanded_config.get('initial_capital', 100000.0),
        max_open_positions=expanded_config.get('max_open_positions', 10),
        max_positions_per_symbol=expanded_config.get('max_positions_per_symbol', 2),
        max_correlated_positions=expanded_config.get('max_correlated_positions', 5),
        max_sector_exposure_pct=expanded_config.get('max_sector_exposure_pct', 30.0),
        max_portfolio_risk_daily_pct=expanded_config.get('max_portfolio_risk_daily_pct', 2.0),
        strategy_weights=strategy_weights,
        rebalance_interval=rebalance_interval,
        data_lookback_days=expanded_config.get('data_lookback_days', 30),
        market_hours_start=market_open,
        market_hours_end=market_close,
        enable_auto_trading=expanded_config.get('enable_auto_trading', False),
        backtesting_mode=True,  # Always use backtesting mode for testing
        data_source='ALPACA',   # Use Alpaca as data source
        api_key=api_key,
        api_secret=api_secret
    )
    
    # Initialize trading system
    system = MultiStrategySystem(system_config)
    logger.info("Trading system initialized with expanded universe configuration")
    
    # Run backtest
    logger.info(f"Running backtest from {start_date} to {end_date}")
    result = system.run_backtest(start_date_obj, end_date_obj)
    
    if not result:
        logger.error("Backtest failed")
        return None
    
    # Print results
    print("\n===== EXPANDED UNIVERSE BACKTEST RESULTS =====")
    print(f"Period: {result.start_date} to {result.end_date}")
    print(f"Initial Capital: ${result.initial_capital:.2f}")
    print(f"Final Capital: ${result.final_capital:.2f}")
    print(f"Total Return: {result.total_return_pct:.2f}%")
    print(f"Annualized Return: {result.annualized_return_pct:.2f}%")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {result.max_drawdown_pct:.2f}%")
    print(f"Win Rate: {result.win_rate:.2%}")
    print(f"Profit Factor: {result.profit_factor:.2f}")
    print(f"Total Trades: {result.total_trades}")
    print("\nStrategy Performance:")
    for strategy, performance in result.strategy_performance.items():
        print(f"  {strategy}: Win Rate={performance.win_rate:.2%}, Profit Factor={performance.profit_factor:.2f}, Trades={performance.total_trades}")
    print("==============================================\n")
    
    # Save results if output file specified
    if output_file:
        # Convert to serializable format
        output_data = result.to_dict()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Write to file
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Backtest results saved to {output_file}")
    
    # Plot equity curve
    plot_file = 'expanded_universe_equity.png'
    plot_equity_curve(result, plot_file)
    
    return result

def plot_equity_curve(result: BacktestResult, output_file: str = None):
    """Plot equity curve from backtest results"""
    try:
        if not result or not result.equity_curve:
            logger.error("Cannot plot equity curve: No data available")
            return False
        
        # Convert equity curve to dataframe
        df = pd.DataFrame(result.equity_curve, columns=['date', 'equity'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Convert drawdown curve to dataframe
        dd_df = pd.DataFrame(result.drawdown_curve, columns=['date', 'drawdown'])
        dd_df['date'] = pd.to_datetime(dd_df['date'])
        dd_df.set_index('date', inplace=True)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot equity curve
        ax1.plot(df.index, df['equity'], label='Portfolio Value')
        ax1.set_title('Expanded Universe - Equity Curve')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True)
        ax1.legend()
        
        # Plot drawdown
        ax2.fill_between(dd_df.index, 0, dd_df['drawdown'], color='red', alpha=0.3)
        ax2.set_title('Drawdown')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Date')
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save or show
        if output_file:
            plt.savefig(output_file)
            logger.info(f"Equity curve saved to {output_file}")
        else:
            plt.show()
        
        plt.close()
        return True
        
    except Exception as e:
        logger.error(f"Error plotting equity curve: {str(e)}")
        return False

def main():
    """Main function to run the expanded universe test"""
    parser = argparse.ArgumentParser(description='Run the trading model with an expanded universe of stocks')
    parser.add_argument('--config', type=str, default='multi_strategy_config.yaml', help='Path to original configuration file')
    parser.add_argument('--start', type=str, default=(dt.date.today() - dt.timedelta(days=30)).strftime('%Y-%m-%d'), help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=dt.date.today().strftime('%Y-%m-%d'), help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default='expanded_universe_results.json', help='Output file for results')
    
    args = parser.parse_args()
    
    # Load Alpaca credentials
    api_key, api_secret = load_alpaca_credentials()
    
    if not api_key or not api_secret:
        logger.error("Alpaca API credentials not found. Please set ALPACA_API_KEY and ALPACA_API_SECRET environment variables or create an alpaca_credentials.json file.")
        return
    
    # Run test
    run_expanded_universe_test(
        args.config,
        args.start,
        args.end,
        api_key,
        api_secret,
        args.output
    )
    
    logger.info("Expanded universe test completed")

if __name__ == "__main__":
    main()
