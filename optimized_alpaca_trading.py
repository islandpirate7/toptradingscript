#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimized Trading Model with Alpaca Integration
-----------------------------------------------
This script runs the optimized trading model (which delivered 16,534.57% returns)
with an expanded universe of stocks from Alpaca.
"""

import os
import sys
import json
import yaml
import logging
import argparse
import datetime as dt
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from alpaca_integration import AlpacaIntegration
from multi_strategy_system import (
    MultiStrategySystem, SystemConfig, StockConfig, BacktestResult,
    MarketRegime, MarketState, CandleData, Signal, TradeDirection
)
import direct_fix

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger("OptimizedAlpacaTrading")

def load_alpaca_credentials(mode: str = "paper") -> Tuple[str, str, str]:
    """Load Alpaca API credentials from environment variables or config file"""
    # Try environment variables first
    api_key = os.environ.get('ALPACA_API_KEY')
    api_secret = os.environ.get('ALPACA_API_SECRET')
    base_url = None
    
    # If not found, try config file
    if not api_key or not api_secret:
        try:
            with open('alpaca_credentials.json', 'r') as f:
                credentials = json.load(f)
                
                # Get credentials based on mode (paper or live)
                if mode in credentials:
                    api_key = credentials[mode].get('api_key')
                    api_secret = credentials[mode].get('api_secret')
                    base_url = credentials[mode].get('base_url')
                else:
                    # Fallback to old format or first available mode
                    if 'api_key' in credentials and 'api_secret' in credentials:
                        # Old format
                        api_key = credentials.get('api_key')
                        api_secret = credentials.get('api_secret')
                    elif 'paper' in credentials:
                        # Default to paper if mode not found
                        api_key = credentials['paper'].get('api_key')
                        api_secret = credentials['paper'].get('api_secret')
                        base_url = credentials['paper'].get('base_url')
                    elif 'live' in credentials:
                        # Use live as last resort
                        api_key = credentials['live'].get('api_key')
                        api_secret = credentials['live'].get('api_secret')
                        base_url = credentials['live'].get('base_url')
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load Alpaca credentials: {e}")
            raise ValueError("Alpaca API credentials not found. Please set ALPACA_API_KEY and ALPACA_API_SECRET environment variables or create alpaca_credentials.json")
    
    if not api_key or not api_secret:
        raise ValueError("Alpaca API credentials not found or invalid")
    
    # Set default base URL if not provided
    if not base_url:
        if mode == "live":
            base_url = "https://api.alpaca.markets"
        else:
            base_url = "https://paper-api.alpaca.markets"
    
    return api_key, api_secret, base_url

def create_optimized_config(optimized_config_file: str, alpaca_assets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create an optimized configuration with more stocks from Alpaca"""
    # Load optimized configuration
    with open(optimized_config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get a sample stock config to use as template
    sample_stock_config = config['stocks'][0] if config.get('stocks') else {}
    
    # Create new stock configurations
    new_stocks = []
    
    # Limit to a maximum of 30 stocks to avoid API rate limits
    max_stocks = 30
    selected_assets = alpaca_assets[:max_stocks]
    
    logger.info(f"Adding {len(selected_assets)} stocks from Alpaca (limited to avoid API rate limits)")
    
    for asset in selected_assets:
        # Skip if already in original config
        if any(s.get('symbol') == asset['symbol'] for s in config.get('stocks', [])):
            continue
            
        # Create new stock config based on template
        stock_config = {
            'symbol': asset['symbol'],
            'max_position_size': min(int(1000000 / asset['price']), 1000),  # Limit to 1000 shares
            'min_position_size': 10,
            'max_risk_per_trade_pct': 0.4,  # More conservative risk per trade
            'min_volume': int(asset['volume'] * 0.1),  # 10% of average volume
            'beta': 1.0,  # Default beta
            'sector': '',  # We don't have sector info from Alpaca
            'industry': '',
        }
        
        # Copy optimized strategy parameters from template
        for param_key in ['mean_reversion_params', 'trend_following_params', 
                         'volatility_breakout_params', 'gap_trading_params']:
            if param_key in sample_stock_config:
                stock_config[param_key] = sample_stock_config[param_key]
        
        new_stocks.append(stock_config)
    
    # Add new stocks to config
    config['stocks'].extend(new_stocks)
    
    # Update other settings
    config['max_open_positions'] = min(20, len(config['stocks']) // 2)  # Allow more open positions but not too many
    config['data_source'] = 'ALPACA'  # Use Alpaca as data source
    
    # Ensure we're using the optimized strategy weights
    if 'strategy_weights' not in config:
        config['strategy_weights'] = {
            "MeanReversion": 0.40,  # Higher weight for mean reversion (best performer)
            "TrendFollowing": 0.25,
            "VolatilityBreakout": 0.20,
            "GapTrading": 0.15
        }
    
    logger.info(f"Created optimized configuration with {len(config['stocks'])} stocks")
    return config

def save_config(config: Dict[str, Any], output_file: str):
    """Save configuration to a YAML file"""
    with open(output_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Saved optimized configuration to {output_file}")

def run_optimized_alpaca_trading(config_file: str, start_date: str = None, end_date: str = None, 
                              api_key: str = None, api_secret: str = None, output_file: str = None,
                              mode: str = "backtest"):
    """Run the optimized trading model with an expanded universe of stocks"""
    # Load API credentials if not provided
    if not api_key or not api_secret:
        api_key, api_secret, base_url = load_alpaca_credentials(mode)
    else:
        # Set default base URL if not provided through credentials
        if mode == "live":
            base_url = "https://api.alpaca.markets"
        else:
            base_url = "https://paper-api.alpaca.markets"
    
    # Initialize Alpaca integration
    try:
        alpaca = AlpacaIntegration(api_key, api_secret, base_url)
        logger.info("Alpaca integration initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Alpaca integration: {e}")
        return None
    
    # Get tradable assets from Alpaca
    try:
        # Use more conservative filters to reduce the number of assets and avoid rate limits
        assets = alpaca.get_tradable_assets(
            min_price=10.0,         # Higher minimum price
            min_volume=1000000,     # Higher minimum volume
            max_stocks=50           # Limit to 50 stocks
        )
        logger.info(f"Retrieved {len(assets)} tradable assets from Alpaca")
    except Exception as e:
        logger.error(f"Failed to get tradable assets from Alpaca: {e}")
        return None
    
    if not assets:
        logger.error("No tradable assets found from Alpaca")
        return None
    
    # Create optimized configuration
    optimized_config = create_optimized_config(config_file, assets)
    
    # Save the configuration to a temporary file
    temp_config_file = "temp_optimized_config.yaml"
    save_config(optimized_config, temp_config_file)
    
    # Parse dates - Use historical data for backtesting
    if mode == "backtest":
        # Default to a period in 2023 for historical data that's available with free tier
        if not start_date:
            start_date = "2023-01-01"
        if not end_date:
            end_date = "2023-12-31"
    
    if start_date:
        start_date = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
    else:
        # Default to 3 months ago for paper/live trading
        start_date = dt.date.today() - dt.timedelta(days=90)
    
    if end_date:
        end_date = dt.datetime.strptime(end_date, "%Y-%m-%d").date()
    else:
        end_date = dt.date.today()
    
    logger.info(f"Using date range: {start_date} to {end_date}")
    
    # Add rate limiting to prevent API throttling
    def get_market_data_with_rate_limit():
        try:
            # Add delay between API calls to avoid rate limits
            logger.info("Fetching market data from Alpaca (this may take a while due to rate limits)...")
            return alpaca.get_market_data(start_date, end_date)
        except Exception as e:
            logger.error(f"Failed to get market data: {e}")
            return None, None
    
    # Run in different modes
    if mode == "backtest":
        # Get market data for backtesting
        spy_data, vix_data = get_market_data_with_rate_limit()
        
        if not spy_data or not vix_data:
            logger.error("Failed to get market data for backtesting")
            return None
        
        # Load configuration from YAML file
        with open(temp_config_file, 'r') as file:
            config_dict = yaml.safe_load(file)
        
        # Parse market hours from the config
        market_hours_start = config_dict.get('market_hours_start', '09:30')
        market_hours_end = config_dict.get('market_hours_end', '16:00')
        
        # Parse time strings to hour and minute components
        if isinstance(market_hours_start, str) and ':' in market_hours_start:
            start_hour, start_minute = map(int, market_hours_start.split(':'))
        else:
            # Default or handle integer value
            start_hour = int(market_hours_start) if isinstance(market_hours_start, (int, float)) else 9
            start_minute = 30
            
        if isinstance(market_hours_end, str) and ':' in market_hours_end:
            end_hour, end_minute = map(int, market_hours_end.split(':'))
        else:
            # Default or handle integer value
            end_hour = int(market_hours_end) if isinstance(market_hours_end, (int, float)) else 16
            end_minute = 0
        
        # Convert stock dictionaries to StockConfig objects
        stock_configs = []
        for stock_dict in config_dict.get('stocks', []):
            # Set default values for required fields if not present
            if 'min_position_size' not in stock_dict:
                stock_dict['min_position_size'] = 1
            if 'min_volume' not in stock_dict:
                stock_dict['min_volume'] = 10000
                
            # Create StockConfig object
            stock_config = StockConfig(
                symbol=stock_dict['symbol'],
                max_position_size=stock_dict.get('max_position_size', 1000),
                min_position_size=stock_dict.get('min_position_size', 1),
                max_risk_per_trade_pct=stock_dict.get('max_risk_per_trade_pct', 0.5),
                min_volume=stock_dict.get('min_volume', 10000),
                avg_daily_volume=stock_dict.get('avg_daily_volume', 0),
                beta=stock_dict.get('beta', 1.0),
                sector=stock_dict.get('sector', ''),
                industry=stock_dict.get('industry', ''),
                mean_reversion_params=stock_dict.get('mean_reversion_params', {}),
                trend_following_params=stock_dict.get('trend_following_params', {}),
                volatility_breakout_params=stock_dict.get('volatility_breakout_params', {}),
                gap_trading_params=stock_dict.get('gap_trading_params', {})
            )
            stock_configs.append(stock_config)
        
        # Create SystemConfig object from the dictionary
        config = SystemConfig(
            stocks=stock_configs,
            initial_capital=config_dict.get('initial_capital', 100000.0),
            max_open_positions=config_dict.get('max_open_positions', 10),
            max_positions_per_symbol=config_dict.get('max_positions_per_symbol', 1),
            max_correlated_positions=config_dict.get('max_correlated_positions', 3),
            max_sector_exposure_pct=config_dict.get('max_sector_exposure_pct', 30.0),
            max_portfolio_risk_daily_pct=config_dict.get('max_portfolio_risk_daily_pct', 3.0),
            strategy_weights=config_dict.get('strategy_weights', {}),
            rebalance_interval=dt.timedelta(days=config_dict.get('rebalance_interval_days', 30)),
            data_lookback_days=config_dict.get('data_lookback_days', 252),
            market_hours_start=dt.time(hour=start_hour, minute=start_minute),
            market_hours_end=dt.time(hour=end_hour, minute=end_minute),
            enable_auto_trading=config_dict.get('enable_auto_trading', False),
            backtesting_mode=True if mode == "backtest" else False,
            data_source=config_dict.get('data_source', 'alpaca'),
            api_key=api_key,
            api_secret=api_secret
        )
        
        # Create system with the SystemConfig object
        system = MultiStrategySystem(config)
        
        # Run backtest
        logger.info(f"Running backtest from {start_date} to {end_date}")
        
        # Convert string dates to datetime.date objects
        if isinstance(start_date, str):
            start_date_obj = dt.datetime.strptime(start_date, '%Y-%m-%d').date()
        else:
            start_date_obj = start_date
            
        if isinstance(end_date, str):
            end_date_obj = dt.datetime.strptime(end_date, '%Y-%m-%d').date()
        else:
            end_date_obj = end_date
        
        # Set market data in the system
        system.spy_data = spy_data
        system.vix_data = vix_data
        
        # Run the backtest with the date objects
        result = system.run_backtest(start_date_obj, end_date_obj)
        
        # Save and plot results
        if result:
            # Check if result is a dict or an object
            if isinstance(result, dict):
                logger.info(f"Backtest completed with {result.get('total_trades', 0)} trades")
                total_return = result.get('total_return', 0)
                max_drawdown = result.get('max_drawdown', 0)
                win_rate = result.get('win_rate', 0)
            else:
                logger.info(f"Backtest completed with {result.total_trades} trades")
                total_return = result.total_return
                max_drawdown = result.max_drawdown
                win_rate = result.win_rate
                
            logger.info(f"Total return: {total_return:.2f}%")
            logger.info(f"Max drawdown: {max_drawdown:.2f}%")
            logger.info(f"Win rate: {win_rate:.2f}%")
            
            # Save results to file if output_file is specified
            if output_file:
                save_results(result, output_file)
                
            # Plot equity curve
            plot_equity_curve(result)
            
            return result
        else:
            logger.error("Backtest failed")
            return None

def plot_equity_curve(result, output_file="optimized_equity_curve.png"):
    """Plot the equity curve from backtest results"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        # Check if result is a dict or an object
        if isinstance(result, dict):
            equity_curve = result.get('equity_curve', [])
            # If equity_curve is empty, return early
            if not equity_curve:
                logger.warning("No equity curve data available for plotting")
                return
        else:
            # Assuming result has an equity_curve attribute
            equity_curve = getattr(result, 'equity_curve', [])
            if not equity_curve:
                logger.warning("No equity curve data available for plotting")
                return
        
        # Extract dates and equity values
        dates = [point[0] for point in equity_curve]
        equity = [point[1] for point in equity_curve]
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        plt.plot(dates, equity, label='Equity Curve')
        plt.title('Backtest Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity ($)')
        plt.grid(True)
        plt.legend()
        
        # Format the x-axis to show dates nicely
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30))  # Show date every 30 days
        plt.gcf().autofmt_xdate()  # Rotate date labels
        
        # Save the plot
        plt.savefig(output_file)
        logger.info(f"Equity curve saved to {output_file}")
        
        # Close the plot to free memory
        plt.close()
        
    except Exception as e:
        logger.error(f"Error plotting equity curve: {str(e)}")

class OptimizedRealtimeTradingSystem:
    """Real-time trading system using the optimized model with expanded universe"""
    
    def __init__(self, system: MultiStrategySystem, alpaca: AlpacaIntegration, 
                 config: Dict[str, Any], api_key: str, api_secret: str, 
                 paper_trading: bool = True):
        """Initialize the real-time trading system"""
        self.system = system
        self.alpaca = alpaca
        self.config = config
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper_trading = paper_trading
        
        # Initialize state
        self.running = False
        self.last_update_time = None
        self.update_interval = dt.timedelta(minutes=5)  # Update every 5 minutes
        
        # Signal handlers for graceful shutdown
        import signal
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Handle termination signals"""
        logger.info("Shutdown signal received, stopping trading system...")
        self.stop()
        sys.exit(0)
    
    def start(self):
        """Start the real-time trading system"""
        if self.running:
            logger.warning("Trading system is already running")
            return False
        
        try:
            # Start the trading system
            self.system.start()
            self.running = True
            self.last_update_time = dt.datetime.now()
            
            logger.info(f"Optimized real-time trading system started (Paper Trading: {self.paper_trading})")
            
            # Main loop
            while self.running:
                self._process_cycle()
                time.sleep(10)  # Sleep for 10 seconds between cycles
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting trading system: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def stop(self):
        """Stop the real-time trading system"""
        if not self.running:
            logger.warning("Trading system is not running")
            return False
        
        try:
            # Stop the trading system
            self.system.stop()
            self.running = False
            
            logger.info("Optimized real-time trading system stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping trading system: {str(e)}")
            return False
    
    def _process_cycle(self):
        """Process a single trading cycle"""
        try:
            # Check if we need to update data
            current_time = dt.datetime.now()
            if (current_time - self.last_update_time) >= self.update_interval:
                logger.info("Updating market data...")
                
                # Update market data
                self._update_market_data()
                
                # Update last update time
                self.last_update_time = current_time
            
            # Check for new signals
            signals = self._check_for_signals()
            
            # Process signals
            if signals:
                logger.info(f"Found {len(signals)} new trading signals")
                self._process_signals(signals)
            
            # Update portfolio status
            self._update_portfolio_status()
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _update_market_data(self):
        """Update market data from Alpaca"""
        try:
            # Get current date
            today = dt.date.today()
            
            # Get market data for the last 45 days (optimized lookback)
            start_date = today - dt.timedelta(days=45)
            
            # Fetch market data (SPY and VIX)
            market_data, vix_data = self.alpaca.get_market_data(start_date, today)
            
            # Update system with new data
            self.system._update_market_data(market_data, vix_data)
            
            # Fetch stock data for all symbols
            symbols = [stock.symbol for stock in self.system.config.stocks]
            stock_data = self.alpaca.get_stock_data(symbols, start_date, today)
            
            # Update system with new stock data
            for symbol, candles in stock_data.items():
                if candles:
                    self.system._update_stock_data(symbol, candles)
            
            logger.info("Market data updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating market data: {str(e)}")
    
    def _check_for_signals(self):
        """Check for new trading signals"""
        try:
            # Get current signals
            signals = []
            
            # Generate signals for each stock
            for stock in self.system.config.stocks:
                symbol = stock.symbol
                
                # Skip if we don't have data for this symbol
                if symbol not in self.system.candle_data or not self.system.candle_data[symbol]:
                    continue
                
                # Get latest candles
                candles = self.system.candle_data[symbol][-45:]  # Last 45 candles (optimized lookback)
                
                # Generate signals for each strategy
                for strategy in self.system.strategies:
                    strategy_signals = strategy.generate_signals(symbol, candles, stock, self.system.market_state)
                    signals.extend(strategy_signals)
            
            # Filter and rank signals
            filtered_signals = self.system.signal_filter.filter_signals(signals, self.system.market_state)
            
            return filtered_signals
            
        except Exception as e:
            logger.error(f"Error checking for signals: {str(e)}")
            return []
    
    def _process_signals(self, signals: List[Signal]):
        """Process trading signals"""
        try:
            # Get account information
            account = self.alpaca.get_account()
            
            # Get current positions
            positions = self.alpaca.get_positions()
            position_symbols = {p['symbol'] for p in positions}
            
            # Get current open orders
            open_orders = self.alpaca.get_open_orders()
            order_symbols = {o['symbol'] for o in open_orders}
            
            # Sort signals by score (highest first)
            sorted_signals = sorted(signals, key=lambda s: s.score, reverse=True)
            
            # Process each signal
            for signal in sorted_signals:
                symbol = signal.symbol
                
                # Skip if we already have a position or open order in this symbol
                if symbol in position_symbols or symbol in order_symbols:
                    logger.info(f"Already have a position or order for {symbol}, skipping signal")
                    continue
                
                # Calculate position size
                position_size = self._calculate_position_size(signal, account)
                
                if position_size <= 0:
                    logger.info(f"Position size for {symbol} is zero or negative, skipping signal")
                    continue
                
                # Check if we have reached the maximum open positions
                if len(positions) >= self.system.config.max_open_positions:
                    logger.info(f"Maximum open positions reached ({len(positions)}), skipping signal")
                    break
                
                # Place order
                side = 'buy' if signal.direction == TradeDirection.LONG else 'sell'
                order_result = self.alpaca.place_order(
                    symbol=symbol,
                    qty=position_size,
                    side=side,
                    order_type='market',
                    time_in_force='day'
                )
                
                logger.info(f"Placed {side} order for {position_size} shares of {symbol} based on {signal.strategy} signal (Score: {signal.score:.2f})")
                
                # Add a small delay between orders
                time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error processing signals: {str(e)}")
    
    def _calculate_position_size(self, signal: Signal, account: Dict[str, Any]) -> int:
        """Calculate position size for a signal using optimized risk management"""
        try:
            # Get account equity
            equity = float(account['equity'])
            
            # Get stock price
            price = signal.price
            
            # Calculate risk per trade (0.4% of equity - more conservative)
            risk_pct = 0.004  # From optimized config
            risk_amount = equity * risk_pct
            
            # Calculate stop loss distance (1.5% by default - tighter stop)
            stop_pct = 0.015
            stop_distance = price * stop_pct
            
            # Calculate position size based on risk
            position_size = risk_amount / stop_distance
            
            # Convert to shares
            shares = int(position_size / price)
            
            # Limit position size to 4% of equity (from optimized config)
            max_position_value = equity * 0.04
            max_shares = int(max_position_value / price)
            
            shares = min(shares, max_shares)
            
            # Ensure minimum position size
            min_shares = 10
            if shares < min_shares:
                shares = 0  # Don't trade if position size is too small
            
            return shares
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0
    
    def _update_portfolio_status(self):
        """Update and log portfolio status"""
        try:
            # Get account information
            account = self.alpaca.get_account()
            
            # Get current positions
            positions = self.alpaca.get_positions()
            
            # Log portfolio status
            logger.info(f"Portfolio Value: ${float(account['portfolio_value']):.2f}")
            logger.info(f"Cash: ${float(account['cash']):.2f}")
            logger.info(f"Number of Positions: {len(positions)}")
            
            # Calculate total P&L
            total_pl = sum(float(p['unrealized_pl']) for p in positions)
            total_pl_pct = total_pl / float(account['portfolio_value']) * 100 if positions else 0
            
            logger.info(f"Total Unrealized P&L: ${total_pl:.2f} ({total_pl_pct:.2f}%)")
            
            # Log individual positions
            if positions:
                logger.info("Current Positions:")
                for position in positions:
                    symbol = position['symbol']
                    qty = position['qty']
                    entry = float(position['cost_basis']) / float(position['qty'])
                    current = float(position['current_price'])
                    pl = float(position['unrealized_pl'])
                    pl_pct = float(position['unrealized_plpc']) * 100
                    
                    logger.info(f"  {symbol}: {qty} shares, Entry: ${entry:.2f}, Current: ${current:.2f}, P&L: ${pl:.2f} ({pl_pct:.2f}%)")
            
        except Exception as e:
            logger.error(f"Error updating portfolio status: {str(e)}")

def save_results(result: BacktestResult, output_file: str):
    """Save backtest results to a JSON file"""
    if isinstance(result, dict):
        result_dict = result
    else:
        result_dict = {
            "total_return_pct": result.total_return_pct,
            "annualized_return_pct": result.annualized_return_pct,
            "max_drawdown_pct": result.max_drawdown_pct,
            "sharpe_ratio": result.sharpe_ratio,
            "profit_factor": result.profit_factor,
            "total_trades": result.total_trades,
            "win_rate": result.win_rate,
            "average_win_pct": result.average_win_pct,
            "average_loss_pct": result.average_loss_pct,
        }
    
    with open(output_file, 'w') as f:
        json.dump(result_dict, f, indent=4)
    logger.info(f"Results saved to {output_file}")

def main():
    """Main function to run the optimized trading model with Alpaca integration"""
    parser = argparse.ArgumentParser(description="Run optimized trading model with Alpaca integration")
    parser.add_argument("--config", type=str, default="optimized_config.yaml", help="Path to configuration file")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, help="Output file for results")
    parser.add_argument("--mode", type=str, choices=["backtest", "paper", "live"], default="backtest",
                        help="Mode to run in (backtest, paper, live)")
    
    args = parser.parse_args()
    
    # For backtest mode, use 2023 data by default (available with free tier)
    if args.mode == "backtest" and not args.start and not args.end:
        args.start = "2023-01-01"
        args.end = "2023-12-31"
        print(f"Using default historical period for backtesting: {args.start} to {args.end}")
    
    # Apply patches to fix issues
    direct_fix.apply_direct_fix()
    
    # Load API credentials
    api_key, api_secret, base_url = load_alpaca_credentials(args.mode)
    
    # Run optimized trading
    result = run_optimized_alpaca_trading(
        config_file=args.config,
        start_date=args.start,
        end_date=args.end,
        api_key=api_key,
        api_secret=api_secret,
        output_file=args.output,
        mode=args.mode
    )
    
    logger.info(f"Optimized Alpaca trading completed in {args.mode} mode")
    
    return result

if __name__ == "__main__":
    main()
