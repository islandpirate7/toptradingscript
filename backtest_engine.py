#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Backtest engine for trading strategies.
This module provides a backtesting framework for evaluating trading strategies
using historical data from Alpaca.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import time
import talib
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import alpaca.data
from combined_strategy import MarketRegime

logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    Engine for backtesting trading strategies using historical data.
    """
    
    def __init__(self, strategy, symbols, timeframe, initial_capital, 
                 start_date, end_date, api_key, api_secret, base_url=None):
        """
        Initialize the backtest engine.
        
        Args:
            strategy: The trading strategy to backtest
            symbols (list): List of symbols to trade
            timeframe (str): Timeframe for data ('1D', '1H', etc.)
            initial_capital (float): Initial capital for the backtest
            start_date (str): Start date for the backtest (YYYY-MM-DD)
            end_date (str): End date for the backtest (YYYY-MM-DD)
            api_key (str): Alpaca API key
            api_secret (str): Alpaca API secret
            base_url (str): Alpaca API base URL
        """
        self.strategy = strategy
        self.symbols = symbols
        self.timeframe = timeframe
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.portfolio_value = initial_capital
        
        # Parse dates
        self.start_date = start_date if isinstance(start_date, datetime) else datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = end_date if isinstance(end_date, datetime) else datetime.strptime(end_date, '%Y-%m-%d')
        
        # Initialize Alpaca client
        self.data_client = StockHistoricalDataClient(api_key, api_secret)
        
        # Initialize positions and orders
        self.positions = {}
        self.orders = []
        self.trades = []
        
        # Initialize results dataframe
        self.results = pd.DataFrame()
        
        # Map timeframe string to Alpaca TimeFrame
        self.alpaca_timeframe = self._map_timeframe(timeframe)
        
        logger.info(f"Initialized backtest engine for {len(symbols)} symbols from {start_date} to {end_date}")
    
    def _map_timeframe(self, timeframe):
        """
        Map timeframe string to Alpaca TimeFrame object.
        
        Args:
            timeframe (str): Timeframe string ('1D', '1H', etc.)
            
        Returns:
            TimeFrame: Alpaca TimeFrame object
        """
        if timeframe == '1D':
            return TimeFrame.Day
        elif timeframe == '1H':
            return TimeFrame.Hour
        elif timeframe == '15Min':
            return TimeFrame.Minute(15)
        elif timeframe == '5Min':
            return TimeFrame.Minute(5)
        elif timeframe == '1Min':
            return TimeFrame.Minute
        else:
            logger.warning(f"Unsupported timeframe: {timeframe}, defaulting to 1D")
            return TimeFrame.Day
    
    def _fetch_historical_data(self):
        """
        Fetch historical data for all symbols.
        
        Returns:
            dict: Dictionary of dataframes with historical data for each symbol
        """
        logger.info(f"Fetching historical data for {len(self.symbols)} symbols")
        
        data = {}
        
        for symbol in self.symbols:
            try:
                # Create request
                request_params = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=self.alpaca_timeframe,
                    start=self.start_date,
                    end=self.end_date,
                    adjustment='all'  # Apply all adjustments
                )
                
                # Get bars
                bars = self.data_client.get_stock_bars(request_params)
                
                # Convert to dataframe
                if bars and hasattr(bars, 'df'):
                    # If bars returned as a single dataframe
                    df = bars.df
                    
                    # Filter for the specific symbol if needed
                    if 'symbol' in df.columns:
                        df = df[df['symbol'] == symbol]
                    
                    # Reset index to make timestamp a column
                    df = df.reset_index()
                    
                    # Rename columns to match expected format
                    df = df.rename(columns={
                        'timestamp': 'timestamp',
                        'open': 'open',
                        'high': 'high',
                        'low': 'low',
                        'close': 'close',
                        'volume': 'volume',
                        'trade_count': 'trade_count',
                        'vwap': 'vwap'
                    })
                    
                    data[symbol] = df
                    logger.info(f"Fetched {len(df)} bars for {symbol}")
                elif bars and symbol in bars:
                    # If bars returned as a dictionary of dataframes
                    df = bars[symbol].df
                    
                    # Reset index to make timestamp a column
                    df = df.reset_index()
                    
                    # Rename columns to match expected format
                    df = df.rename(columns={
                        'timestamp': 'timestamp',
                        'open': 'open',
                        'high': 'high',
                        'low': 'low',
                        'close': 'close',
                        'volume': 'volume',
                        'trade_count': 'trade_count',
                        'vwap': 'vwap'
                    })
                    
                    data[symbol] = df
                    logger.info(f"Fetched {len(df)} bars for {symbol}")
                else:
                    logger.warning(f"No data found for {symbol}")
            
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
        
        return data
    
    def _process_signals(self, data, date):
        """
        Process trading signals for the given date.
        
        Args:
            data (dict): Dictionary of dataframes with historical data
            date (datetime): Current date
            
        Returns:
            list: List of orders to execute
        """
        orders = []
        market_regime = MarketRegime.UNKNOWN
        
        # Process each symbol to detect market regime
        # We'll use SPY as a proxy for the overall market if available
        if 'SPY' in data:
            df_spy = data['SPY']
            # Convert date to UTC timezone if it's not already timezone-aware
            if date.tzinfo is None:
                # Assume the date is in UTC
                date_utc = pd.Timestamp(date).tz_localize('UTC')
            else:
                date_utc = date
                
            df_spy_current = df_spy[df_spy['timestamp'] <= date_utc].copy()
            
            if len(df_spy_current) >= 20:  # Need enough data for indicators
                # Calculate indicators needed for regime detection
                # Bollinger Bands
                df_spy_current['bb_upper'], df_spy_current['bb_middle'], df_spy_current['bb_lower'] = talib.BBANDS(
                    df_spy_current['close'], 
                    timeperiod=20, 
                    nbdevup=2, 
                    nbdevdn=2
                )
                
                # BB Width
                df_spy_current['bb_width'] = (df_spy_current['bb_upper'] - df_spy_current['bb_lower']) / df_spy_current['bb_middle']
                
                # ADX
                df_spy_current['adx'] = talib.ADX(
                    df_spy_current['high'], 
                    df_spy_current['low'], 
                    df_spy_current['close'], 
                    timeperiod=14
                )
                
                # Detect market regime
                market_regime = self.strategy.detect_market_regime(df_spy_current)
        
        # Process each symbol
        for symbol in self.symbols:
            if symbol not in data:
                continue
            
            # Get data up to current date
            df = data[symbol]
            
            # Convert date to UTC timezone if it's not already timezone-aware
            if date.tzinfo is None:
                # Assume the date is in UTC
                date_utc = pd.Timestamp(date).tz_localize('UTC')
            else:
                date_utc = date
                
            df_current = df[df['timestamp'] <= date_utc].copy()
            
            if len(df_current) < 20:  # Need enough data for indicators
                continue
            
            # Generate signals
            signals = self.strategy.generate_signals(df_current, symbol)
            
            if not signals:
                continue
            
            # Get the latest signal
            latest_signal = signals[-1]
            
            # Check if we have an open position
            has_position = symbol in self.positions and self.positions[symbol]['qty'] != 0
            
            # Map direction to action
            if 'direction' in latest_signal and 'action' not in latest_signal:
                if latest_signal['direction'] == 'LONG':
                    latest_signal['action'] = 'BUY'
                elif latest_signal['direction'] == 'SHORT':
                    latest_signal['action'] = 'SELL'
            
            # Process buy signals
            if latest_signal.get('action') == 'BUY' or latest_signal.get('direction') == 'LONG':
                if not has_position:  # Only buy if we don't already have a position
                    # Get price from signal or use close price
                    price = latest_signal.get('price', df_current['close'].iloc[-1])
                    
                    # Calculate position size
                    position_size = self.strategy.calculate_position_size(
                        signal=latest_signal,
                        capital=self.current_capital,
                        current_positions=len(self.positions)
                    )
                    
                    if position_size > 0:
                        # Create buy order
                        order = {
                            'symbol': symbol,
                            'action': 'BUY',
                            'qty': position_size,
                            'price': price,
                            'timestamp': date,
                            'signal_score': latest_signal.get('score', 0.5),
                            'market_regime': market_regime
                        }
                        
                        orders.append(order)
            
            # Process sell signals
            elif latest_signal.get('action') == 'SELL' or latest_signal.get('direction') == 'SHORT':
                if has_position:  # Only sell if we have a position
                    # Get price from signal or use close price
                    price = latest_signal.get('price', df_current['close'].iloc[-1])
                    
                    # Create sell order
                    order = {
                        'symbol': symbol,
                        'action': 'SELL',
                        'qty': self.positions[symbol]['qty'],
                        'price': price,
                        'timestamp': date,
                        'signal_score': latest_signal.get('score', 0.5),
                        'market_regime': market_regime
                    }
                    
                    orders.append(order)
        
        return orders, market_regime
    
    def _execute_orders(self, orders, date):
        """
        Execute orders and update positions.
        
        Args:
            orders (list): List of orders to execute
            date (datetime): Current date
        """
        for order in orders:
            symbol = order['symbol']
            action = order['action']
            qty = order['qty']
            price = order['price']
            
            # Calculate order value
            order_value = qty * price
            
            # Update positions
            if action == 'BUY':
                # Check if we have enough capital
                if order_value > self.current_capital:
                    logger.warning(f"Not enough capital to buy {qty} shares of {symbol} at {price}")
                    continue
                
                # Update capital
                self.current_capital -= order_value
                
                # Update position
                if symbol not in self.positions:
                    self.positions[symbol] = {
                        'qty': qty,
                        'avg_price': price,
                        'cost_basis': order_value,
                        'entry_date': date
                    }
                else:
                    # Add to existing position
                    current_qty = self.positions[symbol]['qty']
                    current_cost = self.positions[symbol]['cost_basis']
                    
                    new_qty = current_qty + qty
                    new_cost = current_cost + order_value
                    
                    self.positions[symbol] = {
                        'qty': new_qty,
                        'avg_price': new_cost / new_qty,
                        'cost_basis': new_cost,
                        'entry_date': date
                    }
                
                # Record the trade
                trade = {
                    'symbol': symbol,
                    'action': action,
                    'qty': qty,
                    'price': price,
                    'timestamp': date,
                    'value': order_value,
                    'position_value': order_value,
                    'pnl': 0,
                    'market_regime': order.get('market_regime', 'unknown'),
                    'signal_score': order.get('signal_score', 0)
                }
                
                self.trades.append(trade)
                
            elif action == 'SELL':
                # Check if we have the position
                if symbol not in self.positions or self.positions[symbol]['qty'] < qty:
                    logger.warning(f"Not enough shares to sell {qty} of {symbol}")
                    continue
                
                # Calculate P&L
                entry_price = self.positions[symbol]['avg_price']
                entry_value = qty * entry_price
                exit_value = qty * price
                pnl = exit_value - entry_value
                
                # Update capital
                self.current_capital += exit_value
                
                # Update position
                current_qty = self.positions[symbol]['qty']
                current_cost = self.positions[symbol]['cost_basis']
                
                new_qty = current_qty - qty
                new_cost = current_cost * (new_qty / current_qty) if current_qty > 0 else 0
                
                if new_qty > 0:
                    self.positions[symbol] = {
                        'qty': new_qty,
                        'avg_price': self.positions[symbol]['avg_price'],  # Keep the same average price
                        'cost_basis': new_cost,
                        'entry_date': self.positions[symbol]['entry_date']
                    }
                else:
                    # Position closed
                    self.positions.pop(symbol, None)
                
                # Record the trade
                trade = {
                    'symbol': symbol,
                    'action': action,
                    'qty': qty,
                    'price': price,
                    'timestamp': date,
                    'value': exit_value,
                    'position_value': entry_value,
                    'pnl': pnl,
                    'market_regime': order.get('market_regime', 'unknown'),
                    'signal_score': order.get('signal_score', 0)
                }
                
                self.trades.append(trade)
            
            # Record the order
            self.orders.append(order)
    
    def _calculate_portfolio_value(self, data, date):
        """
        Calculate the current portfolio value.
        
        Args:
            data (dict): Dictionary of dataframes with historical data
            date (datetime): Current date
            
        Returns:
            float: Current portfolio value
        """
        portfolio_value = self.current_capital
        
        # Add value of open positions
        for symbol, position in self.positions.items():
            if position['qty'] == 0:
                continue
            
            # Get current price
            if symbol in data:
                df = data[symbol]
                df_current = df[df['timestamp'] <= date]
                
                if len(df_current) > 0:
                    current_price = df_current.iloc[-1]['close']
                    position_value = position['qty'] * current_price
                    portfolio_value += position_value
            
        return portfolio_value
    
    def run(self):
        """
        Run the backtest.
        
        Returns:
            pd.DataFrame: Results of the backtest
        """
        logger.info("Starting backtest")
        
        # Fetch historical data
        data = self._fetch_historical_data()
        
        if not data:
            logger.error("No data available for backtest")
            # Return an empty dataframe with the expected columns
            empty_df = pd.DataFrame(columns=[
                'timestamp', 'portfolio_value', 'cash', 'market_regime',
                'action', 'symbol', 'qty', 'price', 'value', 'position_value', 'pnl'
            ])
            return empty_df
        
        # Get all dates in the data
        all_dates = set()
        for symbol, df in data.items():
            # Convert timezone-aware timestamps to date objects
            dates = pd.to_datetime(df['timestamp']).dt.tz_convert('UTC').dt.date.unique()
            all_dates.update(dates)
        
        if not all_dates:
            logger.error("No dates available in the data")
            # Return an empty dataframe with the expected columns
            empty_df = pd.DataFrame(columns=[
                'timestamp', 'portfolio_value', 'cash', 'market_regime',
                'action', 'symbol', 'qty', 'price', 'value', 'position_value', 'pnl'
            ])
            return empty_df
        
        # Sort dates
        all_dates = sorted(all_dates)
        
        # Initialize results
        results_data = []
        
        # Run backtest for each date
        for date_obj in all_dates:
            # Convert date to datetime with UTC timezone
            date = pd.Timestamp(datetime.combine(date_obj, datetime.min.time())).tz_localize('UTC')
            
            # Process signals
            orders, market_regime = self._process_signals(data, date)
            
            # Execute orders
            self._execute_orders(orders, date)
            
            # Calculate portfolio value
            portfolio_value = self._calculate_portfolio_value(data, date)
            self.portfolio_value = portfolio_value
            
            # Record results
            result = {
                'timestamp': date,
                'portfolio_value': portfolio_value,
                'cash': self.current_capital,
                'market_regime': market_regime
            }
            
            # Add position information
            for symbol, position in self.positions.items():
                result[f'{symbol}_qty'] = position['qty']
                result[f'{symbol}_avg_price'] = position['avg_price']
            
            results_data.append(result)
        
        # Create results dataframe
        self.results = pd.DataFrame(results_data)
        
        # Add trades to results
        trades_df = pd.DataFrame(self.trades)
        if not trades_df.empty:
            self.results = pd.concat([self.results, trades_df], ignore_index=True, sort=False)
        
        logger.info("Backtest completed")
        logger.info(f"Final portfolio value: ${self.portfolio_value:.2f}")
        logger.info(f"Total return: {(self.portfolio_value - self.initial_capital) / self.initial_capital * 100:.2f}%")
        logger.info(f"Total trades: {len(self.trades)}")
        
        return self.results
