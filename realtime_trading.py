#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Real-time Trading with Expanded Universe
----------------------------------------
This script runs the original trading model (which showed the best performance)
with an expanded universe of stocks from Alpaca in real-time.
"""

import os
import sys
import logging
import datetime as dt
import yaml
import json
import time
import argparse
import signal
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import the multi-strategy system
from multi_strategy_system import (
    MultiStrategySystem, SystemConfig, StockConfig, Signal, 
    MarketRegime, MarketState, CandleData, TradeDirection
)

# Import Alpaca integration
from alpaca_integration import AlpacaIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('realtime_trading.log')
    ]
)

logger = logging.getLogger("RealtimeTrading")

class RealtimeTradingSystem:
    """Real-time trading system using the original model with expanded universe"""
    
    def __init__(self, config_file: str, api_key: str, api_secret: str, paper_trading: bool = True):
        """Initialize the real-time trading system"""
        self.config_file = config_file
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper_trading = paper_trading
        
        # Initialize components
        self.alpaca = None
        self.system = None
        self.config = None
        self.running = False
        self.last_update_time = None
        self.update_interval = dt.timedelta(minutes=5)  # Update every 5 minutes
        
        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Handle termination signals"""
        logger.info("Shutdown signal received, stopping trading system...")
        self.stop()
        sys.exit(0)
    
    def initialize(self):
        """Initialize the trading system with expanded universe"""
        try:
            # Initialize Alpaca integration
            base_url = "https://paper-api.alpaca.markets" if self.paper_trading else "https://api.alpaca.markets"
            self.alpaca = AlpacaIntegration(self.api_key, self.api_secret, base_url)
            logger.info(f"Alpaca integration initialized (Paper Trading: {self.paper_trading})")
            
            # Get tradable assets from Alpaca
            assets = self.alpaca.get_tradable_assets(
                min_price=5.0,          # Minimum price $5
                min_volume=500000,      # Minimum average volume 500K
                max_stocks=100          # Limit to 100 stocks for now
            )
            
            if not assets:
                logger.error("No tradable assets found from Alpaca")
                return False
            
            logger.info(f"Found {len(assets)} tradable assets from Alpaca")
            
            # Load original configuration
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Get a sample stock config to use as template
            sample_stock_config = config['stocks'][0] if config.get('stocks') else {}
            
            # Create new stock configurations
            config['stocks'] = []  # Clear existing stocks
            for asset in assets:
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
                
                config['stocks'].append(stock_config)
            
            # Update other settings
            config['max_open_positions'] = min(50, len(config['stocks']) // 2)  # Allow more open positions
            config['data_source'] = 'ALPACA'  # Use Alpaca as data source
            config['api_key'] = self.api_key
            config['api_secret'] = self.api_secret
            config['enable_auto_trading'] = True
            config['backtesting_mode'] = False
            
            # Save expanded configuration
            expanded_config_file = 'realtime_expanded_config.yaml'
            with open(expanded_config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            logger.info(f"Saved expanded configuration to {expanded_config_file}")
            
            # Create system configuration
            stock_configs = []
            for stock_data in config.get('stocks', []):
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
            market_open_str = config.get('market_hours_start', '09:30')
            market_close_str = config.get('market_hours_end', '16:00')
            
            market_open = dt.datetime.strptime(market_open_str, '%H:%M').time()
            market_close = dt.datetime.strptime(market_close_str, '%H:%M').time()
            
            # Parse strategy weights
            strategy_weights = config.get('strategy_weights', {
                "MeanReversion": 0.25,
                "TrendFollowing": 0.25,
                "VolatilityBreakout": 0.25,
                "GapTrading": 0.25
            })
            
            # Parse rebalance interval
            rebalance_str = config.get('rebalance_interval', '1d')
            rebalance_unit = rebalance_str[-1]
            rebalance_value = int(rebalance_str[:-1])
            
            if rebalance_unit == 'd':
                rebalance_interval = dt.timedelta(days=rebalance_value)
            elif rebalance_unit == 'h':
                rebalance_interval = dt.timedelta(hours=rebalance_value)
            else:
                rebalance_interval = dt.timedelta(days=1)
            
            # Create system configuration
            self.config = SystemConfig(
                stocks=stock_configs,
                initial_capital=config.get('initial_capital', 100000.0),
                max_open_positions=config.get('max_open_positions', 10),
                max_positions_per_symbol=config.get('max_positions_per_symbol', 2),
                max_correlated_positions=config.get('max_correlated_positions', 5),
                max_sector_exposure_pct=config.get('max_sector_exposure_pct', 30.0),
                max_portfolio_risk_daily_pct=config.get('max_portfolio_risk_daily_pct', 2.0),
                strategy_weights=strategy_weights,
                rebalance_interval=rebalance_interval,
                data_lookback_days=config.get('data_lookback_days', 30),
                market_hours_start=market_open,
                market_hours_end=market_close,
                enable_auto_trading=True,
                backtesting_mode=False,
                data_source='ALPACA',
                api_key=self.api_key,
                api_secret=self.api_secret
            )
            
            # Initialize trading system
            self.system = MultiStrategySystem(self.config)
            logger.info("Trading system initialized with expanded universe configuration")
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing trading system: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def start(self):
        """Start the real-time trading system"""
        if self.running:
            logger.warning("Trading system is already running")
            return False
        
        if not self.system:
            logger.error("Trading system not initialized")
            return False
        
        try:
            # Start the trading system
            self.system.start()
            self.running = True
            self.last_update_time = dt.datetime.now()
            
            logger.info("Real-time trading system started")
            
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
            
            logger.info("Real-time trading system stopped")
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
    
    def _update_market_data(self):
        """Update market data from Alpaca"""
        try:
            # Get current date
            today = dt.date.today()
            
            # Get market data for the last 30 days
            start_date = today - dt.timedelta(days=30)
            
            # Fetch market data (SPY and VIX)
            market_data, vix_data = self.alpaca.get_market_data(start_date, today)
            
            # Update system with new data
            self.system._update_market_data(market_data, vix_data)
            
            # Fetch stock data for all symbols
            symbols = [stock.symbol for stock in self.config.stocks]
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
            for stock in self.config.stocks:
                symbol = stock.symbol
                
                # Skip if we don't have data for this symbol
                if symbol not in self.system.candle_data or not self.system.candle_data[symbol]:
                    continue
                
                # Get latest candles
                candles = self.system.candle_data[symbol][-30:]  # Last 30 candles
                
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
            
            # Process each signal
            for signal in signals:
                symbol = signal.symbol
                
                # Skip if we already have a position in this symbol
                if symbol in position_symbols:
                    logger.info(f"Already have a position in {symbol}, skipping signal")
                    continue
                
                # Calculate position size
                position_size = self._calculate_position_size(signal, account)
                
                if position_size <= 0:
                    logger.info(f"Position size for {symbol} is zero or negative, skipping signal")
                    continue
                
                # Place order
                side = 'buy' if signal.direction == TradeDirection.LONG else 'sell'
                order_result = self.alpaca.place_order(
                    symbol=symbol,
                    qty=position_size,
                    side=side,
                    order_type='market',
                    time_in_force='day'
                )
                
                logger.info(f"Placed {side} order for {position_size} shares of {symbol} based on {signal.strategy} signal")
                
                # Add a small delay between orders
                time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error processing signals: {str(e)}")
    
    def _calculate_position_size(self, signal: Signal, account: Dict[str, Any]) -> int:
        """Calculate position size for a signal"""
        try:
            # Get account equity
            equity = account['equity']
            
            # Get stock price
            price = signal.price
            
            # Calculate risk per trade (1% of equity by default)
            risk_pct = 0.01
            risk_amount = equity * risk_pct
            
            # Calculate stop loss distance (2% by default)
            stop_pct = 0.02
            stop_distance = price * stop_pct
            
            # Calculate position size based on risk
            position_size = risk_amount / stop_distance
            
            # Convert to shares
            shares = int(position_size / price)
            
            # Limit position size to 5% of equity
            max_position_value = equity * 0.05
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
            logger.info(f"Portfolio Value: ${account['portfolio_value']:.2f}")
            logger.info(f"Cash: ${account['cash']:.2f}")
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

def main():
    """Main function to run the real-time trading system"""
    parser = argparse.ArgumentParser(description='Run the trading model with an expanded universe of stocks in real-time')
    parser.add_argument('--config', type=str, default='multi_strategy_config.yaml', help='Path to original configuration file')
    parser.add_argument('--paper', action='store_true', help='Use paper trading (default)')
    parser.add_argument('--live', action='store_true', help='Use live trading (use with caution!)')
    
    args = parser.parse_args()
    
    # Determine trading mode
    paper_trading = not args.live  # Default to paper trading unless --live is specified
    
    if args.live:
        logger.warning("LIVE TRADING MODE ENABLED - REAL MONEY WILL BE USED!")
        confirm = input("Are you sure you want to use live trading? (yes/no): ")
        if confirm.lower() != 'yes':
            logger.info("Live trading cancelled. Exiting.")
            return
    
    # Load Alpaca credentials
    api_key, api_secret = load_alpaca_credentials()
    
    if not api_key or not api_secret:
        logger.error("Alpaca API credentials not found. Please set ALPACA_API_KEY and ALPACA_API_SECRET environment variables or create an alpaca_credentials.json file.")
        return
    
    # Initialize and start trading system
    trading_system = RealtimeTradingSystem(args.config, api_key, api_secret, paper_trading)
    
    if trading_system.initialize():
        logger.info("Trading system initialized successfully")
        trading_system.start()
    else:
        logger.error("Failed to initialize trading system")

if __name__ == "__main__":
    main()
