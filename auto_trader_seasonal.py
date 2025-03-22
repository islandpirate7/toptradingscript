#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Automated Trading System with Seasonality
-------------------------------------
This module implements an automated trading system that executes trades based on
seasonal patterns and integrates with the mean reversion strategy.
"""

import os
import json
import logging
import yaml
import pandas as pd
import numpy as np
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import REST

# Import our modules
from seasonality_analyzer import SeasonalityAnalyzer, SeasonType, Direction
from integrate_seasonality import SeasonalityIntegrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("auto_trader_seasonal.log"),
        logging.StreamHandler()
    ]
)

class AutoTraderSeasonal:
    """Automated trading system with seasonality integration"""
    
    def __init__(self, config_file: str, api_credentials_file: str, mode: str = 'paper'):
        """Initialize the automated trading system
        
        Args:
            config_file (str): Path to configuration file
            api_credentials_file (str): Path to API credentials file
            mode (str, optional): Trading mode ('paper' or 'live'). Defaults to 'paper'.
        """
        self.config = self._load_config(config_file)
        self.credentials = self._load_credentials(api_credentials_file)
        self.mode = mode
        self.api = self._initialize_api()
        
        # Initialize seasonality integrator
        seasonality_file = self.config.get('seasonality_file', 'output/seasonal_opportunities.yaml')
        self.seasonality = SeasonalityIntegrator(seasonality_file)
        
        # Trading parameters
        self.max_positions = self.config.get('max_positions', 5)
        self.position_size_pct = self.config.get('position_size_pct', 0.1)
        self.max_risk_per_trade_pct = self.config.get('max_risk_per_trade_pct', 0.01)
        
        # Tracking variables
        self.open_positions = {}
        self.trade_history = []
        
        logging.info(f"Initialized AutoTraderSeasonal in {mode} mode")
        
    def _load_config(self, file_path: str) -> Dict:
        """Load configuration from YAML file
        
        Args:
            file_path (str): Path to configuration file
            
        Returns:
            Dict: Configuration dictionary
        """
        try:
            with open(file_path, 'r') as f:
                config = yaml.safe_load(f)
            logging.info(f"Loaded configuration from {file_path}")
            return config
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            return {}
    
    def _load_credentials(self, file_path: str) -> Dict:
        """Load API credentials from JSON file
        
        Args:
            file_path (str): Path to credentials file
            
        Returns:
            Dict: Credentials dictionary
        """
        try:
            with open(file_path, 'r') as f:
                credentials = json.load(f)
            logging.info(f"Loaded API credentials from {file_path}")
            return credentials
        except Exception as e:
            logging.error(f"Error loading API credentials: {e}")
            return {}
    
    def _initialize_api(self) -> REST:
        """Initialize Alpaca API client
        
        Returns:
            REST: Alpaca API client
        """
        if self.mode == 'paper':
            api_key = self.credentials['paper']['api_key']
            api_secret = self.credentials['paper']['api_secret']
            base_url = self.credentials['paper']['base_url']
        else:  # live
            api_key = self.credentials['live']['api_key']
            api_secret = self.credentials['live']['api_secret']
            base_url = self.credentials['live']['base_url']
            
        api = REST(api_key, api_secret, base_url)
        
        # Test connection
        try:
            account = api.get_account()
            logging.info(f"Connected to Alpaca API. Account status: {account.status}")
            return api
        except Exception as e:
            logging.error(f"Error connecting to Alpaca API: {e}")
            raise
    
    def get_account_info(self) -> Dict:
        """Get account information
        
        Returns:
            Dict: Account information
        """
        try:
            account = self.api.get_account()
            return {
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'equity': float(account.equity),
                'portfolio_value': float(account.portfolio_value),
                'status': account.status,
                'trading_blocked': account.trading_blocked,
                'pattern_day_trader': account.pattern_day_trader
            }
        except Exception as e:
            logging.error(f"Error getting account information: {e}")
            return {}
    
    def get_positions(self) -> Dict[str, Dict]:
        """Get current positions
        
        Returns:
            Dict[str, Dict]: Dictionary mapping symbols to position information
        """
        try:
            positions = self.api.list_positions()
            position_dict = {}
            
            for position in positions:
                position_dict[position.symbol] = {
                    'qty': int(position.qty),
                    'market_value': float(position.market_value),
                    'avg_entry_price': float(position.avg_entry_price),
                    'current_price': float(position.current_price),
                    'unrealized_pl': float(position.unrealized_pl),
                    'unrealized_plpc': float(position.unrealized_plpc),
                    'side': position.side
                }
                
            self.open_positions = position_dict
            return position_dict
            
        except Exception as e:
            logging.error(f"Error getting positions: {e}")
            return {}
    
    def get_seasonal_opportunities(self) -> List[Dict]:
        """Get current seasonal trading opportunities
        
        Returns:
            List[Dict]: List of seasonal opportunities
        """
        try:
            # Load seasonal opportunities
            seasonality_file = self.config.get('seasonality_file', 'output/seasonal_opportunities.yaml')
            
            with open(seasonality_file, 'r') as f:
                data = yaml.safe_load(f)
                
            opportunities = data.get('opportunities', [])
            
            # Filter for current season
            current_season = self.seasonality.get_current_season()
            current_month = current_season['month']
            
            filtered_opps = []
            for opp in opportunities:
                if opp['season'] == current_month:
                    filtered_opps.append(opp)
            
            logging.info(f"Found {len(filtered_opps)} seasonal opportunities for {current_month}")
            return filtered_opps
            
        except Exception as e:
            logging.error(f"Error getting seasonal opportunities: {e}")
            return []
    
    def check_market_hours(self) -> bool:
        """Check if the market is currently open
        
        Returns:
            bool: True if market is open, False otherwise
        """
        try:
            clock = self.api.get_clock()
            return clock.is_open
        except Exception as e:
            logging.error(f"Error checking market hours: {e}")
            return False
    
    def calculate_position_size(self, symbol: str, price: float, risk_pct: float = None) -> int:
        """Calculate position size based on risk parameters
        
        Args:
            symbol (str): Stock symbol
            price (float): Current price
            risk_pct (float, optional): Risk percentage. Defaults to None.
            
        Returns:
            int: Number of shares to trade
        """
        try:
            # Get account information
            account = self.get_account_info()
            portfolio_value = account['portfolio_value']
            
            # Use default risk if not specified
            if risk_pct is None:
                risk_pct = self.max_risk_per_trade_pct
                
            # Calculate position value
            position_value = portfolio_value * self.position_size_pct
            
            # Calculate number of shares
            shares = int(position_value / price)
            
            # Check if we have enough buying power
            buying_power = float(account['buying_power'])
            if price * shares > buying_power:
                shares = int(buying_power / price)
                logging.warning(f"Reduced position size for {symbol} due to buying power constraints")
                
            return shares
            
        except Exception as e:
            logging.error(f"Error calculating position size: {e}")
            return 0
    
    def place_order(self, symbol: str, qty: int, side: str, 
                  order_type: str = 'market', time_in_force: str = 'day',
                  limit_price: float = None, stop_price: float = None) -> Dict:
        """Place an order
        
        Args:
            symbol (str): Stock symbol
            qty (int): Quantity
            side (str): Order side ('buy' or 'sell')
            order_type (str, optional): Order type. Defaults to 'market'.
            time_in_force (str, optional): Time in force. Defaults to 'day'.
            limit_price (float, optional): Limit price. Defaults to None.
            stop_price (float, optional): Stop price. Defaults to None.
            
        Returns:
            Dict: Order information
        """
        try:
            # Check if we're in paper mode or if the market is open
            if self.mode == 'paper' or self.check_market_hours():
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    type=order_type,
                    time_in_force=time_in_force,
                    limit_price=limit_price,
                    stop_price=stop_price
                )
                
                order_info = {
                    'id': order.id,
                    'symbol': symbol,
                    'qty': qty,
                    'side': side,
                    'type': order_type,
                    'time_in_force': time_in_force,
                    'limit_price': limit_price,
                    'stop_price': stop_price,
                    'status': order.status,
                    'created_at': order.created_at
                }
                
                logging.info(f"Placed {side} order for {qty} shares of {symbol}")
                return order_info
            else:
                logging.warning(f"Market is closed. Order for {symbol} not placed.")
                return {}
                
        except Exception as e:
            logging.error(f"Error placing order for {symbol}: {e}")
            return {}
    
    def place_trade_with_stop_loss_take_profit(self, symbol: str, direction: str, 
                                             stop_loss_pct: float = 0.02, 
                                             take_profit_pct: float = 0.03) -> Dict:
        """Place a trade with stop loss and take profit orders
        
        Args:
            symbol (str): Stock symbol
            direction (str): Trade direction ('LONG' or 'SHORT')
            stop_loss_pct (float, optional): Stop loss percentage. Defaults to 0.02.
            take_profit_pct (float, optional): Take profit percentage. Defaults to 0.03.
            
        Returns:
            Dict: Trade information
        """
        try:
            # Get current price
            current_price = float(self.api.get_latest_trade(symbol).price)
            
            # Calculate position size
            qty = self.calculate_position_size(symbol, current_price)
            
            if qty <= 0:
                logging.warning(f"Invalid position size for {symbol}. Trade not placed.")
                return {}
                
            # Determine order side
            side = 'buy' if direction == 'LONG' else 'sell'
            
            # Place primary order
            order_info = self.place_order(symbol, qty, side)
            
            if not order_info:
                return {}
                
            # Wait for order to fill
            filled = False
            retry_count = 0
            
            while not filled and retry_count < 10:
                time.sleep(2)  # Wait 2 seconds
                order = self.api.get_order(order_info['id'])
                
                if order.status == 'filled':
                    filled = True
                    fill_price = float(order.filled_avg_price)
                    logging.info(f"Order for {symbol} filled at {fill_price}")
                else:
                    retry_count += 1
            
            if not filled:
                logging.warning(f"Order for {symbol} not filled after {retry_count} retries")
                return order_info
                
            # Calculate stop loss and take profit prices
            if direction == 'LONG':
                stop_loss_price = fill_price * (1 - stop_loss_pct)
                take_profit_price = fill_price * (1 + take_profit_pct)
                stop_side = 'sell'
            else:  # SHORT
                stop_loss_price = fill_price * (1 + stop_loss_pct)
                take_profit_price = fill_price * (1 - take_profit_pct)
                stop_side = 'buy'
                
            # Place stop loss order
            stop_loss_order = self.place_order(
                symbol=symbol,
                qty=qty,
                side=stop_side,
                order_type='stop',
                stop_price=stop_loss_price
            )
            
            # Place take profit order
            take_profit_order = self.place_order(
                symbol=symbol,
                qty=qty,
                side=stop_side,
                order_type='limit',
                limit_price=take_profit_price
            )
            
            # Record trade
            trade_info = {
                'symbol': symbol,
                'direction': direction,
                'qty': qty,
                'entry_price': fill_price,
                'entry_time': datetime.now().isoformat(),
                'stop_loss_price': stop_loss_price,
                'take_profit_price': take_profit_price,
                'stop_loss_order_id': stop_loss_order.get('id'),
                'take_profit_order_id': take_profit_order.get('id'),
                'status': 'open'
            }
            
            self.trade_history.append(trade_info)
            
            logging.info(f"Placed {direction} trade for {symbol} with stop loss at {stop_loss_price} and take profit at {take_profit_price}")
            
            return trade_info
            
        except Exception as e:
            logging.error(f"Error placing trade for {symbol}: {e}")
            return {}
    
    def execute_seasonal_trades(self) -> List[Dict]:
        """Execute trades based on seasonal opportunities
        
        Returns:
            List[Dict]: List of executed trades
        """
        # Get seasonal opportunities
        opportunities = self.get_seasonal_opportunities()
        
        if not opportunities:
            logging.info("No seasonal opportunities found")
            return []
            
        # Get current positions
        current_positions = self.get_positions()
        
        # Sort opportunities by score (win_rate * avg_return)
        opportunities.sort(key=lambda x: x['win_rate'] * x['avg_return'], reverse=True)
        
        # Determine how many new positions we can take
        available_slots = self.max_positions - len(current_positions)
        
        if available_slots <= 0:
            logging.info("Maximum positions reached. No new trades executed.")
            return []
            
        # Execute trades for top opportunities
        executed_trades = []
        
        for opp in opportunities[:available_slots]:
            symbol = opp['symbol']
            direction = opp['direction']
            
            # Skip if we already have a position in this symbol
            if symbol in current_positions:
                logging.info(f"Already have a position in {symbol}. Skipping.")
                continue
                
            # Get stop loss and take profit percentages from configuration
            symbol_config = self.config.get('symbol_configs', {}).get(symbol, {})
            stop_loss_pct = symbol_config.get('stop_loss_pct', 0.02)
            take_profit_pct = symbol_config.get('take_profit_pct', 0.03)
            
            # Place trade
            trade = self.place_trade_with_stop_loss_take_profit(
                symbol=symbol,
                direction=direction,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct
            )
            
            if trade:
                executed_trades.append(trade)
                
        logging.info(f"Executed {len(executed_trades)} seasonal trades")
        return executed_trades
    
    def update_trade_status(self) -> None:
        """Update the status of open trades"""
        if not self.trade_history:
            return
            
        # Get current positions
        current_positions = self.get_positions()
        
        # Update status of each trade
        for trade in self.trade_history:
            if trade['status'] != 'open':
                continue
                
            symbol = trade['symbol']
            
            # Check if position is still open
            if symbol in current_positions:
                # Position still open, update unrealized P&L
                position = current_positions[symbol]
                trade['current_price'] = position['current_price']
                trade['unrealized_pl'] = position['unrealized_pl']
                trade['unrealized_plpc'] = position['unrealized_plpc']
            else:
                # Position closed, check orders to determine outcome
                try:
                    stop_loss_order = self.api.get_order(trade['stop_loss_order_id'])
                    take_profit_order = self.api.get_order(trade['take_profit_order_id'])
                    
                    if stop_loss_order.status == 'filled':
                        trade['status'] = 'stopped_out'
                        trade['exit_price'] = float(stop_loss_order.filled_avg_price)
                        trade['exit_time'] = stop_loss_order.filled_at
                        trade['pl'] = (trade['exit_price'] - trade['entry_price']) * trade['qty'] * (1 if trade['direction'] == 'LONG' else -1)
                        
                        # Cancel take profit order
                        self.api.cancel_order(trade['take_profit_order_id'])
                        
                    elif take_profit_order.status == 'filled':
                        trade['status'] = 'profit_taken'
                        trade['exit_price'] = float(take_profit_order.filled_avg_price)
                        trade['exit_time'] = take_profit_order.filled_at
                        trade['pl'] = (trade['exit_price'] - trade['entry_price']) * trade['qty'] * (1 if trade['direction'] == 'LONG' else -1)
                        
                        # Cancel stop loss order
                        self.api.cancel_order(trade['stop_loss_order_id'])
                        
                except Exception as e:
                    logging.error(f"Error updating trade status for {symbol}: {e}")
    
    def save_trade_history(self, file_path: str) -> None:
        """Save trade history to a file
        
        Args:
            file_path (str): Path to output file
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(self.trade_history, f, indent=2)
            logging.info(f"Saved trade history to {file_path}")
        except Exception as e:
            logging.error(f"Error saving trade history: {e}")
    
    def run_trading_session(self) -> None:
        """Run a complete trading session"""
        logging.info("Starting trading session")
        
        # Check if market is open
        if not self.check_market_hours() and self.mode != 'paper':
            logging.warning("Market is closed. Trading session aborted.")
            return
            
        # Get account information
        account_info = self.get_account_info()
        logging.info(f"Account equity: ${account_info['equity']}")
        
        # Update status of existing trades
        self.update_trade_status()
        
        # Execute new trades based on seasonal opportunities
        executed_trades = self.execute_seasonal_trades()
        
        # Save trade history
        self.save_trade_history('trade_history.json')
        
        logging.info("Trading session completed")

def main():
    """Main function to run the automated trading system"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Automated trading system with seasonality')
    parser.add_argument('--config', type=str, default='configuration_combined_strategy_seasonal.yaml',
                      help='Path to configuration file')
    parser.add_argument('--credentials', type=str, default='alpaca_credentials.json',
                      help='Path to API credentials file')
    parser.add_argument('--mode', type=str, choices=['paper', 'live'], default='paper',
                      help='Trading mode (paper or live)')
    args = parser.parse_args()
    
    # Initialize trader
    trader = AutoTraderSeasonal(args.config, args.credentials, args.mode)
    
    # Run trading session
    trader.run_trading_session()
    
    logging.info("Automated trading completed successfully")

if __name__ == "__main__":
    main()
