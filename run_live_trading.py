#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Live Trading Script for S&P 500 Trading Strategy
This script runs the S&P 500 trading strategy in live trading mode using Alpaca API
"""

import os
import sys
import json
import yaml
import time
import logging
import argparse
import datetime
import pandas as pd
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"live_trading_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import strategy modules
from final_sp500_strategy import get_sp500_symbols, get_midcap_symbols, generate_signals, prioritize_signals

def load_config(config_file='sp500_config.yaml'):
    """Load configuration from YAML file"""
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Successfully loaded configuration from {config_file}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return None

def load_alpaca_credentials(credentials_file='alpaca_credentials.json'):
    """Load Alpaca API credentials from JSON file"""
    try:
        with open(credentials_file, 'r') as file:
            credentials = json.load(file)
        logger.info(f"Successfully loaded Alpaca credentials from {credentials_file}")
        return credentials
    except Exception as e:
        logger.error(f"Error loading Alpaca credentials: {str(e)}")
        return None

def initialize_alpaca_api(credentials):
    """Initialize Alpaca API client"""
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        
        # Initialize API clients
        trading_client = TradingClient(
            api_key=credentials['API_KEY'],
            secret_key=credentials['API_SECRET'],
            paper=not credentials.get('LIVE', False)  # Default to paper trading unless explicitly set to LIVE
        )
        
        data_client = StockHistoricalDataClient(
            api_key=credentials['API_KEY'],
            secret_key=credentials['API_SECRET']
        )
        
        logger.info(f"Successfully initialized Alpaca API clients (Paper Trading: {not credentials.get('LIVE', False)})")
        
        return {
            'trading_client': trading_client,
            'data_client': data_client,
            'order_request': MarketOrderRequest,
            'order_side': OrderSide,
            'time_in_force': TimeInForce,
            'stock_bars_request': StockBarsRequest,
            'timeframe': TimeFrame
        }
    except Exception as e:
        logger.error(f"Error initializing Alpaca API: {str(e)}")
        return None

def get_account_info(alpaca):
    """Get account information from Alpaca"""
    try:
        account = alpaca['trading_client'].get_account()
        logger.info(f"Account Information:")
        logger.info(f"  Account ID: {account.id}")
        logger.info(f"  Cash: ${float(account.cash):.2f}")
        logger.info(f"  Portfolio Value: ${float(account.portfolio_value):.2f}")
        logger.info(f"  Buying Power: ${float(account.buying_power):.2f}")
        logger.info(f"  Daytrade Count: {account.daytrade_count}")
        logger.info(f"  Status: {account.status}")
        
        return {
            'id': account.id,
            'cash': float(account.cash),
            'portfolio_value': float(account.portfolio_value),
            'buying_power': float(account.buying_power),
            'daytrade_count': account.daytrade_count,
            'status': account.status
        }
    except Exception as e:
        logger.error(f"Error getting account information: {str(e)}")
        return None

def get_positions(alpaca):
    """Get current positions from Alpaca"""
    try:
        positions = alpaca['trading_client'].get_all_positions()
        logger.info(f"Current Positions: {len(positions)}")
        
        position_data = []
        for position in positions:
            position_info = {
                'symbol': position.symbol,
                'qty': int(position.qty),
                'market_value': float(position.market_value),
                'avg_entry_price': float(position.avg_entry_price),
                'current_price': float(position.current_price),
                'unrealized_pl': float(position.unrealized_pl),
                'unrealized_plpc': float(position.unrealized_plpc),
                'side': 'LONG' if int(position.qty) > 0 else 'SHORT'
            }
            logger.info(f"  {position.symbol}: {position_info['qty']} shares, Entry: ${position_info['avg_entry_price']:.2f}, Current: ${position_info['current_price']:.2f}, P/L: ${position_info['unrealized_pl']:.2f} ({position_info['unrealized_plpc']:.2f}%)")
            position_data.append(position_info)
        
        return position_data
    except Exception as e:
        logger.error(f"Error getting positions: {str(e)}")
        return []

def get_market_data(alpaca, symbols, lookback_days=30):
    """Get historical market data for symbols"""
    try:
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=lookback_days)
        
        request_params = alpaca['stock_bars_request'](
            symbol_or_symbols=symbols,
            timeframe=alpaca['timeframe'].Day,
            start=start_date,
            end=end_date
        )
        
        bars = alpaca['data_client'].get_stock_bars(request_params)
        
        # Convert to dictionary of dataframes
        data = {}
        for symbol in symbols:
            if symbol in bars.data:
                symbol_bars = bars.data[symbol]
                df = pd.DataFrame([
                    {
                        'timestamp': bar.timestamp,
                        'open': bar.open,
                        'high': bar.high,
                        'low': bar.low,
                        'close': bar.close,
                        'volume': bar.volume
                    }
                    for bar in symbol_bars
                ])
                if not df.empty:
                    df.set_index('timestamp', inplace=True)
                    data[symbol] = df
        
        logger.info(f"Retrieved market data for {len(data)} symbols")
        return data
    except Exception as e:
        logger.error(f"Error getting market data: {str(e)}")
        return {}

def execute_trade(alpaca, symbol, qty, side, risk_level='medium'):
    """Execute a trade on Alpaca"""
    try:
        # Adjust time in force based on risk level
        if risk_level == 'low':
            time_in_force = alpaca['time_in_force'].DAY
        elif risk_level == 'medium':
            time_in_force = alpaca['time_in_force'].GTC
        else:  # high
            time_in_force = alpaca['time_in_force'].IOC
        
        # Create market order
        market_order_data = alpaca['order_request'](
            symbol=symbol,
            qty=qty,
            side=alpaca['order_side'].BUY if side == 'LONG' else alpaca['order_side'].SELL,
            time_in_force=time_in_force,
            client_order_id=f"sp500_{side}_{symbol}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        )
        
        # Submit order
        order = alpaca['trading_client'].submit_order(market_order_data)
        
        logger.info(f"Submitted {side} order for {qty} shares of {symbol}")
        logger.info(f"  Order ID: {order.id}")
        logger.info(f"  Status: {order.status}")
        
        return {
            'id': order.id,
            'symbol': symbol,
            'qty': qty,
            'side': side,
            'status': order.status
        }
    except Exception as e:
        logger.error(f"Error executing trade for {symbol}: {str(e)}")
        return None

def close_position(alpaca, symbol):
    """Close a position on Alpaca"""
    try:
        # Close position
        response = alpaca['trading_client'].close_position(symbol)
        
        logger.info(f"Closed position for {symbol}")
        logger.info(f"  Order ID: {response.id}")
        logger.info(f"  Status: {response.status}")
        
        return {
            'id': response.id,
            'symbol': symbol,
            'status': response.status
        }
    except Exception as e:
        logger.error(f"Error closing position for {symbol}: {str(e)}")
        return None

def run_live_trading(config, credentials_file='alpaca_credentials.json', max_signals=10, check_interval=5, max_capital=50000, risk_level='medium'):
    """Run live trading with Alpaca API"""
    try:
        # Load Alpaca credentials
        credentials = load_alpaca_credentials(credentials_file)
        if not credentials:
            logger.error("Cannot proceed without Alpaca credentials")
            return False
        
        # Initialize Alpaca API
        alpaca = initialize_alpaca_api(credentials)
        if not alpaca:
            logger.error("Cannot proceed without Alpaca API")
            return False
        
        # Get account information
        account_info = get_account_info(alpaca)
        if not account_info:
            logger.error("Cannot proceed without account information")
            return False
        
        # Check if account is active
        if account_info['status'] != 'ACTIVE':
            logger.error(f"Account is not active. Current status: {account_info['status']}")
            return False
        
        # Set up trading parameters
        available_capital = min(float(account_info['cash']), max_capital)
        logger.info(f"Available capital for trading: ${available_capital:.2f}")
        
        # Main trading loop
        logger.info(f"Starting live trading with check interval of {check_interval} minutes")
        
        while True:
            try:
                current_time = datetime.datetime.now()
                logger.info(f"=== Trading Check at {current_time} ===")
                
                # Check if market is open
                calendar = alpaca['trading_client'].get_calendar(
                    start=current_time.date(),
                    end=current_time.date()
                )
                
                if not calendar or len(calendar) == 0:
                    logger.info("Market is closed today")
                    time.sleep(check_interval * 60)
                    continue
                
                market_open = datetime.datetime.combine(
                    current_time.date(),
                    datetime.time(
                        hour=calendar[0].open.hour,
                        minute=calendar[0].open.minute
                    )
                )
                
                market_close = datetime.datetime.combine(
                    current_time.date(),
                    datetime.time(
                        hour=calendar[0].close.hour,
                        minute=calendar[0].close.minute
                    )
                )
                
                if current_time < market_open:
                    logger.info(f"Market is not open yet. Opens at {market_open}")
                    # Sleep until market open
                    sleep_seconds = (market_open - current_time).total_seconds()
                    if sleep_seconds > 0:
                        logger.info(f"Sleeping for {sleep_seconds / 60:.1f} minutes until market open")
                        time.sleep(min(sleep_seconds, check_interval * 60))
                    continue
                
                if current_time > market_close:
                    logger.info(f"Market is closed. Closed at {market_close}")
                    # Sleep until tomorrow
                    tomorrow = current_time + datetime.timedelta(days=1)
                    tomorrow = datetime.datetime.combine(
                        tomorrow.date(),
                        datetime.time(hour=9, minute=0)
                    )
                    sleep_seconds = (tomorrow - current_time).total_seconds()
                    logger.info(f"Sleeping for {sleep_seconds / 3600:.1f} hours until tomorrow")
                    time.sleep(min(sleep_seconds, check_interval * 60))
                    continue
                
                # Market is open, proceed with trading
                logger.info("Market is open, proceeding with trading")
                
                # Get current positions
                current_positions = get_positions(alpaca)
                current_symbols = [p['symbol'] for p in current_positions]
                
                # Get SP500 symbols
                sp500_symbols = get_sp500_symbols()
                if not sp500_symbols:
                    logger.error("Failed to get S&P 500 symbols")
                    time.sleep(check_interval * 60)
                    continue
                
                # Get mid-cap symbols if enabled
                midcap_config = config.get('strategy', {}).get('midcap_stocks', {})
                midcap_enabled = midcap_config.get('enabled', False)
                midcap_symbols = []
                
                if midcap_enabled:
                    midcap_symbols = get_midcap_symbols()
                    if not midcap_symbols:
                        logger.warning("Failed to get mid-cap symbols, proceeding with S&P 500 only")
                
                # Combine symbols
                all_symbols = list(set(sp500_symbols + midcap_symbols))
                logger.info(f"Total symbols to analyze: {len(all_symbols)} (S&P 500: {len(sp500_symbols)}, Mid-cap: {len(midcap_symbols)})")
                
                # Get market data
                market_data = get_market_data(alpaca, all_symbols, lookback_days=30)
                if not market_data:
                    logger.error("Failed to get market data")
                    time.sleep(check_interval * 60)
                    continue
                
                # Generate signals
                signals = generate_signals(market_data, config)
                if not signals:
                    logger.info("No signals generated")
                    time.sleep(check_interval * 60)
                    continue
                
                # Prioritize signals
                prioritized_signals = prioritize_signals(signals, sp500_symbols, midcap_symbols, config)
                logger.info(f"Generated {len(prioritized_signals)} prioritized signals")
                
                # Filter to LONG-only signals (as per memory)
                long_signals = [s for s in prioritized_signals if s['direction'] == 'LONG']
                logger.info(f"Filtered to {len(long_signals)} LONG signals")
                
                # Take top signals up to max_signals
                top_signals = long_signals[:max_signals]
                logger.info(f"Selected top {len(top_signals)} signals for trading")
                
                # Calculate position sizes
                account_value = float(account_info['portfolio_value'])
                base_position_size = available_capital / max(len(top_signals), 1)
                
                # Execute trades for new signals
                for signal in top_signals:
                    symbol = signal['symbol']
                    
                    # Skip if already in portfolio
                    if symbol in current_symbols:
                        logger.info(f"Already holding {symbol}, skipping")
                        continue
                    
                    # Calculate position size based on signal score
                    score = signal['score']
                    tier_multiplier = 1.0
                    
                    # Apply tiered position sizing
                    tier_thresholds = config.get('strategy', {}).get('signal_thresholds', {})
                    if score >= tier_thresholds.get('tier_1', 0.9):
                        tier_multiplier = config.get('strategy', {}).get('position_sizing', {}).get('tier_multipliers', {}).get('Tier 1 (≥0.9)', 3.0)
                    elif score >= tier_thresholds.get('tier_2', 0.8):
                        tier_multiplier = config.get('strategy', {}).get('position_sizing', {}).get('tier_multipliers', {}).get('Tier 2 (0.8-0.9)', 1.5)
                    else:
                        tier_multiplier = config.get('strategy', {}).get('position_sizing', {}).get('tier_multipliers', {}).get('Below Threshold (<0.8)', 0.0)
                    
                    # Skip if multiplier is zero
                    if tier_multiplier <= 0:
                        logger.info(f"Signal for {symbol} has score {score:.2f} below threshold, skipping")
                        continue
                    
                    # Get current price
                    current_price = float(market_data[symbol].iloc[-1]['close']) if symbol in market_data else 0
                    if current_price <= 0:
                        logger.warning(f"Invalid price for {symbol}, skipping")
                        continue
                    
                    # Calculate position size and shares
                    position_value = base_position_size * tier_multiplier
                    
                    # Check capital balance before trade (as requested)
                    if position_value > available_capital:
                        logger.warning(f"Not enough capital for {symbol}, need ${position_value:.2f}, have ${available_capital:.2f}")
                        position_value = available_capital
                    
                    shares = int(position_value / current_price)
                    
                    # Skip if not enough shares
                    if shares <= 0:
                        logger.warning(f"Not enough capital to buy at least 1 share of {symbol} at ${current_price:.2f}")
                        continue
                    
                    # Execute trade
                    trade_result = execute_trade(alpaca, symbol, shares, 'LONG', risk_level)
                    
                    if trade_result:
                        # Update available capital
                        available_capital -= (shares * current_price)
                        logger.info(f"Remaining available capital: ${available_capital:.2f}")
                        
                        # Break if out of capital
                        if available_capital <= 0:
                            logger.info("Out of available capital, stopping trading for this cycle")
                            break
                
                # Check for positions to close
                for position in current_positions:
                    symbol = position['symbol']
                    
                    # Check if symbol is still in top signals
                    still_valid = any(s['symbol'] == symbol for s in top_signals)
                    
                    # Check stop loss
                    stop_loss_triggered = False
                    if position['unrealized_plpc'] < -2.0:  # 2% stop loss
                        stop_loss_triggered = True
                        logger.info(f"Stop loss triggered for {symbol} with {position['unrealized_plpc']:.2f}% loss")
                    
                    # Close position if not in top signals or stop loss triggered
                    if not still_valid or stop_loss_triggered:
                        logger.info(f"Closing position for {symbol} (still valid: {still_valid}, stop loss: {stop_loss_triggered})")
                        close_result = close_position(alpaca, symbol)
                        
                        if close_result:
                            # Update available capital
                            available_capital += float(position['market_value'])
                            logger.info(f"Updated available capital: ${available_capital:.2f}")
                
                # Sleep until next check
                logger.info(f"Completed trading cycle, sleeping for {check_interval} minutes")
                time.sleep(check_interval * 60)
                
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt detected, stopping trading")
                break
            except Exception as e:
                logger.error(f"Error in trading cycle: {str(e)}")
                time.sleep(check_interval * 60)
        
        return True
    
    except Exception as e:
        logger.error(f"Error in live trading: {str(e)}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run live trading for S&P 500 strategy')
    parser.add_argument('--max_signals', type=int, default=10, help='Maximum number of signals to trade')
    parser.add_argument('--check_interval', type=int, default=5, help='Check interval in minutes')
    parser.add_argument('--max_capital', type=float, default=50000, help='Maximum capital to use')
    parser.add_argument('--risk_level', type=str, default='medium', choices=['low', 'medium', 'high'], help='Risk level')
    parser.add_argument('--live', action='store_true', help='Use live trading instead of paper trading')
    parser.add_argument('--config', type=str, default='sp500_config.yaml', help='Path to configuration file')
    parser.add_argument('--credentials', type=str, default='alpaca_credentials.json', help='Path to Alpaca credentials file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    if not config:
        logger.error("Cannot proceed without configuration")
        return
    
    # Update credentials if using live trading
    if args.live:
        credentials_file = args.credentials
        try:
            with open(credentials_file, 'r') as file:
                credentials = json.load(file)
            
            credentials['LIVE'] = True
            
            with open(credentials_file, 'w') as file:
                json.dump(credentials, file, indent=2)
            
            logger.info("Updated credentials for live trading")
        except Exception as e:
            logger.error(f"Error updating credentials for live trading: {str(e)}")
            return
    
    # Run live trading
    success = run_live_trading(
        config,
        credentials_file=args.credentials,
        max_signals=args.max_signals,
        check_interval=args.check_interval,
        max_capital=args.max_capital,
        risk_level=args.risk_level
    )
    
    if success:
        logger.info("Live trading completed successfully")
    else:
        logger.error("Live trading failed")

if __name__ == "__main__":
    main()
