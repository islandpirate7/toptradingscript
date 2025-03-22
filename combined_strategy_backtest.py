#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Combined Strategy Backtest Script
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
from typing import List, Dict, Any, Tuple
import yaml
import os
import json
import copy
import argparse

# Import strategy modules
from enhanced_mean_reversion_backtest import EnhancedMeanReversionBacktest
from trend_following_strategy import TrendFollowingStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CombinedStrategyBacktest:
    """Backtest implementation for combined Mean Reversion and Trend Following strategies"""
    
    def __init__(self, config, mean_reversion_strategy, trend_following_strategy, initial_capital=100000.0):
        """Initialize the backtest"""
        self.config = config
        self.mean_reversion = mean_reversion_strategy
        self.trend_following = trend_following_strategy
        
        # Get strategy weights from config
        self.mean_reversion_weight = config.get('strategy_weights', {}).get('MeanReversion', 0.5)
        self.trend_following_weight = config.get('strategy_weights', {}).get('TrendFollowing', 0.5)
        
        # Initialize backtest variables
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.trades = []
        self.signals = []
        self.equity_curve = []
        self.drawdowns = []
        self.peak_equity = initial_capital
        self.max_drawdown = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.latest_candles = {}  # Store latest candles for each symbol
        self.open_positions = []  # Store open positions
        self.data = {}  # Store historical data
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize equity curve with starting point
        self.equity_curve.append({
            'date': datetime.now(),
            'capital': self.current_capital,
            'positions_value': 0,
            'total_value': self.current_capital,
            'open_positions': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0
        })
    
    def _initialize_alpaca(self):
        """Initialize Alpaca API client"""
        try:
            with open('alpaca_credentials.json', 'r') as f:
                credentials = json.load(f)
            
            # Use paper trading credentials by default
            paper_creds = credentials.get('paper', {})
            api = tradeapi.REST(
                key_id=paper_creds.get('api_key'),
                secret_key=paper_creds.get('api_secret'),
                base_url=paper_creds.get('base_url', 'https://paper-api.alpaca.markets')
            )
            logger.info("Alpaca API initialized successfully")
            return api
        except Exception as e:
            logger.error(f"Error initializing Alpaca API: {e}")
            raise
    
    def _get_historical_data(self, symbols, start_date, end_date, timeframe='1Day'):
        """Get historical data from Alpaca"""
        logger.info(f"Fetching historical data for {len(symbols)} symbols from {start_date} to {end_date}")
        
        # Format dates for Alpaca API (YYYY-MM-DD format)
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        data = {}
        for symbol in symbols:
            try:
                # Get bars from Alpaca
                bars = self.alpaca.get_bars(
                    symbol,
                    tradeapi.TimeFrame.Day,
                    start=start_str,
                    end=end_str,
                    adjustment='raw'
                ).df
                
                if len(bars) == 0:
                    logger.warning(f"No data found for {symbol}")
                    continue
                
                # Convert to CandleData format
                candles = []
                for idx, row in bars.iterrows():
                    candle = {
                        'timestamp': idx.to_pydatetime(),
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': int(row['volume'])
                    }
                    candles.append(candle)
                
                data[symbol] = candles
                logger.info(f"Fetched {len(candles)} candles for {symbol}")
            
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
        
        return data
    
    def _combine_signals(self, mean_reversion_signals, trend_following_signals, symbol):
        """Combine signals from both strategies based on weights"""
        combined_signals = []
        
        # Get stock-specific configuration
        stock_config = self.config.get('stocks', {}).get(symbol, {})
        
        # Get strategy weights
        mean_reversion_weight = self.mean_reversion_weight
        trend_following_weight = self.trend_following_weight
        
        # Apply symbol-specific weight multipliers if available
        mr_weight_multiplier = stock_config.get('mean_reversion_params', {}).get('weight_multiplier', 1.0)
        tf_weight_multiplier = stock_config.get('trend_following_params', {}).get('weight_multiplier', 1.0)
        
        # Adjust weights based on multipliers
        mean_reversion_weight *= mr_weight_multiplier
        trend_following_weight *= tf_weight_multiplier
        
        # Process mean reversion signals
        for signal in mean_reversion_signals:
            # Convert strength to numerical value for better filtering
            strength_value = 0.5  # Default medium strength
            if hasattr(signal, 'strength'):
                if signal.strength == 'strong':
                    strength_value = 0.8
                elif signal.strength == 'weak':
                    strength_value = 0.3
            
            # Create a dictionary to store metadata since Signal class doesn't have a metadata attribute
            signal_data = {
                'symbol': signal.symbol,
                'timestamp': signal.timestamp,
                'direction': signal.direction,
                'entry_price': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'strategy': 'MeanReversion',
                'weight': mean_reversion_weight,
                'strength': signal.strength if hasattr(signal, 'strength') else 'medium',
                'strength_value': strength_value,
                'score': strength_value * mean_reversion_weight  # Calculate signal score
            }
            combined_signals.append(signal_data)
        
        # Process trend following signals
        for signal in trend_following_signals:
            # Convert strength to numerical value
            strength_value = 0.5  # Default medium strength
            if hasattr(signal, 'strength'):
                if signal.strength == 'strong':
                    strength_value = 0.8
                elif signal.strength == 'weak':
                    strength_value = 0.3
            
            # Create a dictionary to store metadata
            signal_data = {
                'symbol': signal.symbol,
                'timestamp': signal.timestamp,
                'direction': signal.direction,
                'entry_price': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'strategy': 'TrendFollowing',
                'weight': trend_following_weight,
                'strength': signal.strength if hasattr(signal, 'strength') else 'medium',
                'strength_value': strength_value,
                'score': strength_value * trend_following_weight  # Calculate signal score
            }
            combined_signals.append(signal_data)
        
        # Sort by score (highest first) then by timestamp
        combined_signals.sort(key=lambda x: (-x['score'], x['timestamp']))
        
        # Get signal quality filter settings
        signal_filters = self.config.get('signal_quality_filters', {})
        min_score_threshold = signal_filters.get('min_score_threshold', 0.3)
        
        # Filter signals by minimum score
        filtered_signals = [s for s in combined_signals if s['score'] >= min_score_threshold]
        
        # Limit number of signals per day if needed
        max_signals_per_day = signal_filters.get('max_signals_per_day', 5)
        if len(filtered_signals) > max_signals_per_day:
            filtered_signals = filtered_signals[:max_signals_per_day]
        
        return filtered_signals
    
    def _calculate_position_size(self, signal, available_capital):
        """Calculate position size based on signal strength and available capital"""
        # Get position sizing configuration
        position_sizing_config = self.config.get('position_sizing_config', {})
        base_risk_per_trade = position_sizing_config.get('base_risk_per_trade', 0.01)
        max_position_size = position_sizing_config.get('max_position_size', 0.1)
        min_position_size = position_sizing_config.get('min_position_size', 0.005)
        
        # Get stock-specific configuration if available
        stock_config = None
        for stock in self.config.get('stocks', []):
            if stock.get('symbol') == signal['symbol']:
                stock_config = stock
                break
        
        # Apply stock-specific risk adjustments if available
        if stock_config:
            stock_max_risk = stock_config.get('max_risk_per_trade_pct', 0.5) / 100
            base_risk_per_trade = min(base_risk_per_trade, stock_max_risk)
        
        # Adjust risk based on signal strength
        risk_multiplier = 1.0
        if signal['strength'] == 'strong':
            risk_multiplier = 1.2
        elif signal['strength'] == 'weak':
            risk_multiplier = 0.8
        
        # Adjust risk based on strategy
        strategy_risk_multiplier = 1.0
        if signal['strategy'] == 'MeanReversion':
            # Mean reversion signals tend to have higher win rates but lower R:R
            strategy_risk_multiplier = 1.1
        elif signal['strategy'] == 'TrendFollowing':
            # Trend following signals tend to have lower win rates but higher R:R
            strategy_risk_multiplier = 0.9
        
        # Calculate risk amount
        risk_amount = available_capital * base_risk_per_trade * risk_multiplier * strategy_risk_multiplier
        
        # Calculate position size based on risk and stop loss
        entry_price = signal['entry_price']
        stop_loss = signal['stop_loss']
        risk_per_share = abs(entry_price - stop_loss)
        
        # Avoid division by zero
        if risk_per_share <= 0:
            risk_per_share = entry_price * 0.01  # Default to 1% of price
        
        # Calculate shares based on risk
        shares = risk_amount / risk_per_share
        
        # Calculate position value
        position_value = shares * entry_price
        
        # Apply position size limits
        max_position_value = available_capital * max_position_size
        min_position_value = available_capital * min_position_size
        
        if position_value > max_position_value:
            position_value = max_position_value
            shares = position_value / entry_price
        elif position_value < min_position_value:
            position_value = min_position_value
            shares = position_value / entry_price
        
        # Round down to whole shares
        shares = int(shares)
        
        # Ensure at least 1 share
        shares = max(1, shares)
        
        return shares
    
    def _execute_trade(self, signal, current_date):
        """Execute a trade based on a signal"""
        symbol = signal['symbol']
        direction = signal['direction']
        entry_price = signal['entry_price']
        stop_loss = signal['stop_loss']
        take_profit = signal['take_profit']
        strategy = signal['strategy']
        
        # Calculate available capital
        available_capital = self._calculate_available_capital()
        
        # Calculate position size
        position_size = self._calculate_position_size(signal, available_capital)
        
        # Calculate position value
        position_value = position_size * entry_price
        
        # Check if we have enough capital
        if position_value > available_capital:
            logger.warning(f"Not enough capital to execute trade for {symbol}. Required: {position_value}, Available: {available_capital}")
            return
        
        # Check if we already have too many positions
        if len(self.open_positions) >= self.config.get('max_open_positions', 10):
            logger.warning(f"Maximum number of open positions reached. Cannot execute trade for {symbol}")
            return
        
        # Check if we already have too many positions for this symbol
        symbol_positions = [p for p in self.open_positions if p['symbol'] == symbol]
        if len(symbol_positions) >= self.config.get('max_positions_per_symbol', 2):
            logger.warning(f"Maximum number of positions for {symbol} reached")
            return
        
        # Check for conflicting positions (opposite direction)
        for pos in symbol_positions:
            if pos['direction'] != direction:
                logger.warning(f"Conflicting position for {symbol} exists. Skipping trade")
                return
        
        # Create trade
        trade = {
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'entry_date': current_date,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'position_value': position_value,
            'strategy': strategy,
            'status': 'open',
            'exit_price': None,
            'exit_date': None,
            'profit_loss': 0,
            'profit_loss_pct': 0,
            'exit_reason': None
        }
        
        # Add to open positions
        self.open_positions.append(trade)
        
        # Update capital
        self.current_capital -= position_value
        
        # Log trade
        logger.info(f"Executed {direction} trade for {symbol} at {entry_price} with size {position_size} ({strategy})")
        
        # Add to trade history
        self.trades.append(trade)
        
        return trade
    
    def _calculate_available_capital(self):
        """Calculate available capital"""
        available_capital = self.current_capital
        
        # Subtract value of open positions
        for pos in self.open_positions:
            available_capital -= pos['position_value']
        
        return available_capital
    
    def _check_exits(self, symbol, candle, current_date):
        """Check if any positions should be exited"""
        # Skip if no open positions for this symbol
        symbol_positions = [p for p in self.open_positions if p['symbol'] == symbol and p['status'] == 'open']
        if not symbol_positions:
            return
        
        # Get candle data
        high = candle['high']
        low = candle['low']
        close = candle['close']
        
        for position in symbol_positions:
            # Skip if already closed
            if position['status'] != 'open':
                continue
            
            direction = position['direction']
            entry_price = position['entry_price']
            stop_loss = position['stop_loss']
            take_profit = position['take_profit']
            position_size = position['position_size']
            position_value = position['position_value']
            strategy = position['strategy']
            
            # Initialize exit variables
            exit_triggered = False
            exit_price = None
            exit_reason = None
            
            # Check stop loss hit
            if direction == 'buy' and low <= stop_loss:
                exit_triggered = True
                exit_price = stop_loss
                exit_reason = 'stop_loss'
                logger.info(f"Stop loss hit for {symbol} at {stop_loss}")
            elif direction == 'sell' and high >= stop_loss:
                exit_triggered = True
                exit_price = stop_loss
                exit_reason = 'stop_loss'
                logger.info(f"Stop loss hit for {symbol} at {stop_loss}")
            
            # Check take profit hit
            if not exit_triggered:
                if direction == 'buy' and high >= take_profit:
                    exit_triggered = True
                    exit_price = take_profit
                    exit_reason = 'take_profit'
                    logger.info(f"Take profit hit for {symbol} at {take_profit}")
                elif direction == 'sell' and low <= take_profit:
                    exit_triggered = True
                    exit_price = take_profit
                    exit_reason = 'take_profit'
                    logger.info(f"Take profit hit for {symbol} at {take_profit}")
            
            # Check for strategy-specific exit signals
            if not exit_triggered:
                # For mean reversion, exit when price reverts back to the mean
                if strategy == 'MeanReversion':
                    # Check if price has reverted to the mean
                    if direction == 'buy' and close >= entry_price * 1.01:  # 1% profit is good for mean reversion
                        exit_triggered = True
                        exit_price = close
                        exit_reason = 'reversion_to_mean'
                    elif direction == 'sell' and close <= entry_price * 0.99:  # 1% profit is good for mean reversion
                        exit_triggered = True
                        exit_price = close
                        exit_reason = 'reversion_to_mean'
                
                # For trend following, implement trailing stop
                elif strategy == 'TrendFollowing':
                    # Calculate trailing stop based on ATR
                    atr = self._calculate_atr(symbol, 14)
                    if atr:
                        # Use tighter trailing stop for trend following
                        trail_factor = 1.5
                        if direction == 'buy':
                            trailing_stop = max(stop_loss, close - (atr * trail_factor))
                            if trailing_stop > stop_loss:
                                position['stop_loss'] = trailing_stop
                                logger.info(f"Updated trailing stop for {symbol} to {trailing_stop}")
                        else:  # sell
                            trailing_stop = min(stop_loss, close + (atr * trail_factor))
                            if trailing_stop < stop_loss:
                                position['stop_loss'] = trailing_stop
                                logger.info(f"Updated trailing stop for {symbol} to {trailing_stop}")
            
            # Process exit if triggered
            if exit_triggered:
                # Calculate profit/loss
                if direction == 'buy':
                    profit_loss = (exit_price - entry_price) * position_size
                    profit_loss_pct = (exit_price - entry_price) / entry_price
                else:  # sell
                    profit_loss = (entry_price - exit_price) * position_size
                    profit_loss_pct = (entry_price - exit_price) / entry_price
                
                # Update position
                position['status'] = 'closed'
                position['exit_price'] = exit_price
                position['exit_date'] = current_date
                position['profit_loss'] = profit_loss
                position['profit_loss_pct'] = profit_loss_pct
                position['exit_reason'] = exit_reason
                
                # Update capital
                self.current_capital += position_value + profit_loss
                
                # Update statistics
                self.total_pnl += profit_loss
                if profit_loss > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
                
                # Log exit
                logger.info(f"Exited {direction} trade for {symbol} at {exit_price} with P/L: ${profit_loss:.2f} ({profit_loss_pct:.2%})")
                
                # Remove from open positions
                self.open_positions = [p for p in self.open_positions if p != position]
    
    def _calculate_atr(self, symbol, period=14):
        """Calculate Average True Range for a symbol"""
        if symbol not in self.data or len(self.data[symbol]) < period + 1:
            return None
        
        # Get relevant candles
        candles = self.data[symbol][-period-1:]
        
        # Calculate true ranges
        true_ranges = []
        for i in range(1, len(candles)):
            high = candles[i]['high']
            low = candles[i]['low']
            prev_close = candles[i-1]['close']
            
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            
            true_range = max(tr1, tr2, tr3)
            true_ranges.append(true_range)
        
        # Calculate ATR
        atr = sum(true_ranges) / len(true_ranges)
        return atr
    
    def _update_equity_curve(self, current_date):
        """Update equity curve with current capital and positions value"""
        # Calculate total value of open positions
        positions_value = 0
        for position in self.open_positions:
            if position['status'] == 'open':
                symbol = position['symbol']
                if symbol in self.latest_candles:
                    current_price = self.latest_candles[symbol]['close']
                    position_size = position['position_size']
                    
                    # Calculate current value based on direction
                    if position['direction'] == 'buy':
                        position_value = position_size * current_price
                    else:  # sell
                        # For short positions, we calculate the inverse
                        entry_value = position_size * position['entry_price']
                        current_value = position_size * current_price
                        position_value = entry_value + (entry_value - current_value)
                    
                    positions_value += position_value
        
        # Calculate total portfolio value
        total_value = self.current_capital + positions_value
        
        # Store in equity curve
        self.equity_curve.append({
            'date': current_date,
            'capital': self.current_capital,
            'positions_value': positions_value,
            'total_value': total_value,
            'open_positions': len(self.open_positions),
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'total_pnl': self.total_pnl
        })
        
        # Calculate drawdown
        if len(self.equity_curve) > 1:
            max_value = max([e['total_value'] for e in self.equity_curve])
            current_drawdown = (max_value - total_value) / max_value if max_value > 0 else 0
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        return total_value
    
    def get_performance_metrics(self):
        """Get performance metrics for the backtest"""
        # Calculate returns
        if not self.equity_curve:
            return {
                'total_return_pct': 0,
                'annualized_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_trades': 0
            }
            
        start_value = self.equity_curve[0]['total_value']
        end_value = self.equity_curve[-1]['total_value']
        total_return = end_value - start_value
        total_return_pct = (total_return / start_value) * 100
        
        # Calculate annualized return
        if len(self.equity_curve) > 1:
            start_date = self.equity_curve[0]['date']
            end_date = self.equity_curve[-1]['date']
            days = (end_date - start_date).days
            if days > 0:
                years = days / 365.0
                annualized_return = ((1 + total_return_pct/100) ** (1/years) - 1) * 100
            else:
                annualized_return = 0
        else:
            annualized_return = 0
        
        # Calculate Sharpe ratio (simplified)
        if len(self.equity_curve) > 1:
            daily_returns = []
            for i in range(1, len(self.equity_curve)):
                prev_value = self.equity_curve[i-1]['total_value']
                curr_value = self.equity_curve[i]['total_value']
                daily_return = (curr_value / prev_value) - 1
                daily_returns.append(daily_return)
            
            if daily_returns:
                avg_return = np.mean(daily_returns)
                std_return = np.std(daily_returns)
                if std_return > 0:
                    sharpe_ratio = (avg_return / std_return) * np.sqrt(252)  # Annualized
                else:
                    sharpe_ratio = 0
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        # Calculate win rate
        if self.total_trades > 0:
            win_rate = (self.winning_trades / self.total_trades) * 100
        else:
            win_rate = 0
        
        # Calculate profit factor
        if self.losing_trades > 0:
            total_profit = sum([t['profit_loss'] for t in self.trades if t['profit_loss'] > 0])
            total_loss = abs(sum([t['profit_loss'] for t in self.trades if t['profit_loss'] < 0]))
            profit_factor = total_profit / total_loss if total_loss > 0 else 0
        else:
            profit_factor = 0
        
        # Calculate strategy-specific metrics
        mean_reversion_trades = [t for t in self.trades if t['strategy'] == 'MeanReversion']
        trend_following_trades = [t for t in self.trades if t['strategy'] == 'TrendFollowing']
        
        mr_win_rate = 0
        if mean_reversion_trades:
            mr_winners = len([t for t in mean_reversion_trades if t['profit_loss'] > 0])
            mr_win_rate = (mr_winners / len(mean_reversion_trades)) * 100
        
        tf_win_rate = 0
        if trend_following_trades:
            tf_winners = len([t for t in trend_following_trades if t['profit_loss'] > 0])
            tf_win_rate = (tf_winners / len(trend_following_trades)) * 100
        
        # Create equity curve data for plotting
        equity_data = [{
            'date': point['date'],
            'total_value': point['total_value']
        } for point in self.equity_curve]
        
        return {
            'total_return_pct': total_return_pct,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'mean_reversion_win_rate': mr_win_rate,
            'trend_following_win_rate': tf_win_rate,
            'equity_curve': equity_data,
            'trades': self.trades
        }
    
    def add_data(self, symbol, candles):
        """Add historical data for a symbol"""
        self.data[symbol] = candles
        logger.info(f"Added {len(candles)} candles for {symbol}")
    
    def process_day(self, current_date):
        """Process a single day in the backtest"""
        logger.info(f"Processing day: {current_date}")
        
        # Process each symbol
        for symbol, candles in self.data.items():
            # Find candles for the current date
            day_candles = [c for c in candles if c['timestamp'].date() == current_date.date()]
            
            if not day_candles:
                continue
            
            # Use the last candle of the day
            candle = day_candles[-1]
            
            # Store latest candle
            self.latest_candles[symbol] = candle
            
            # Check for exits
            self._check_exits(symbol, candle, current_date)
            
            # Generate signals
            mean_reversion_signals = self.mean_reversion.generate_signals(symbol, candles, None, None)
            trend_following_signals = self.trend_following.generate_signals(symbol, candles, None, None)
            
            # Combine signals
            combined_signals = self._combine_signals(mean_reversion_signals, trend_following_signals, symbol)
            
            # Execute signals
            for signal in combined_signals:
                # Skip if signal is not for current date
                if signal['timestamp'].date() != current_date.date():
                    continue
                
                # Execute trade
                trade = self._execute_trade(signal, current_date)
                
                if trade:
                    self.signals.append(signal)
        
        # Update equity curve
        self._update_equity_curve(current_date)
        
        return True
    
    def run_backtest(self, start_date, end_date):
        """Run the backtest from start_date to end_date"""
        logger.info(f"Starting combined strategy backtest from {start_date} to {end_date}")
        
        # Get list of symbols from config
        symbols = list(self.config.get('stocks', {}).keys())
        
        if not symbols:
            logger.error("No symbols found in configuration")
            return None
        
        # Get historical data
        data = self._get_historical_data(symbols, start_date, end_date)
        
        if not data:
            logger.error("No historical data found")
            return None
        
        # Process each day
        current_date = start_date
        while current_date <= end_date:
            # Skip weekends
            if current_date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
                current_date += timedelta(days=1)
                continue
            
            logger.info(f"Processing {current_date}")
            
            # Process each symbol
            for symbol in symbols:
                if symbol not in data:
                    continue
                
                # Get candles for this symbol up to current date
                candles = [c for c in data[symbol] if c['timestamp'].date() <= current_date.date()]
                
                if not candles:
                    continue
                
                # Get latest candle
                latest_candle = candles[-1]
                
                # Skip if not current date
                if latest_candle['timestamp'].date() != current_date.date():
                    continue
                
                # Store latest candles for each symbol
                self.latest_candles[symbol] = candles
                
                # Check exits for existing positions
                self._check_exits(symbol, latest_candle, current_date)
                
                # Generate market state
                market_state = {
                    'date': current_date,
                    'regime': "neutral",  # Default value
                    'volatility': 0.0,    # Default value
                    'trend_strength': 0.0, # Default value
                    'is_range_bound': False # Default value
                }
                
                # Try to generate more accurate market state if we have enough data
                if len(candles) >= 50:
                    # Calculate volatility
                    closes = [c['close'] for c in candles[-20:]]
                    returns = [closes[i] / closes[i-1] - 1 for i in range(1, len(closes))]
                    volatility = np.std(returns) * 100  # Convert to percentage
                    
                    # Calculate trend strength using ADX-like approach
                    highs = [c['high'] for c in candles[-14:]]
                    lows = [c['low'] for c in candles[-14:]]
                    
                    # Simple trend strength calculation
                    up_moves = [max(0, highs[i] - highs[i-1]) for i in range(1, len(highs))]
                    down_moves = [max(0, lows[i-1] - lows[i]) for i in range(1, len(lows))]
                    
                    avg_up = sum(up_moves) / len(up_moves) if up_moves else 0
                    avg_down = sum(down_moves) / len(down_moves) if down_moves else 0
                    
                    trend_strength = 0.0
                    if avg_up + avg_down > 0:
                        trend_strength = abs(avg_up - avg_down) / (avg_up + avg_down)
                    
                    # Determine regime
                    regime = "neutral"
                    if trend_strength > 0.3:  # Strong trend
                        if avg_up > avg_down:
                            regime = "bullish"
                        else:
                            regime = "bearish"
                    
                    # Determine if range-bound
                    is_range_bound = trend_strength < 0.2 and volatility < 1.5  # Low trend strength and low volatility
                    
                    # Update market state
                    market_state = {
                        'date': current_date,
                        'regime': regime,
                        'volatility': volatility,
                        'trend_strength': trend_strength,
                        'is_range_bound': is_range_bound
                    }
                
                # Generate signals from both strategies
                mean_reversion_signals = self.mean_reversion.generate_signals(
                    symbol=symbol,
                    candles=candles,
                    market_state=market_state,
                    is_crypto=False  # Assuming we're dealing with stocks
                )
                
                trend_following_signals = self.trend_following.generate_signals(
                    symbol=symbol,
                    candles=candles,
                    market_state=market_state
                )
                
                # Combine signals
                combined_signals = self._combine_signals(
                    mean_reversion_signals or [],
                    trend_following_signals or [],
                    symbol
                )
                
                # Execute trades based on signals
                for signal in combined_signals:
                    # Skip if signal is not for current date
                    if signal['timestamp'].date() != current_date.date():
                        continue
                    
                    # Execute trade
                    trade = self._execute_trade(signal, current_date)
                    
                    if trade:
                        self.signals.append(signal)
            
            # Update equity curve
            self._update_equity_curve(current_date)
            
            # Move to next day
            current_date += timedelta(days=1)
        
        # Close any remaining open positions at the end of the backtest
        for symbol, positions in list(self.positions.items()):
            for position in list(positions):
                if position['status'] == 'open':
                    # Get latest price from the last candle
                    if symbol in data and data[symbol]:
                        latest_price = data[symbol][-1]['close']
                        self._exit_position(position, latest_price, end_date, 'End of Backtest')
        
        # Calculate final statistics
        results = self.get_performance_metrics()
        
        # Plot equity curve
        self._plot_equity_curve(results)
        
        return results
    
    def _plot_equity_curve(self, results):
        """Plot equity curve and drawdowns"""
        if not self.equity_curve:
            logger.warning("No equity curve data to plot")
            return
        
        try:
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
            
            # Plot equity curve
            dates = [date for date, _ in self.equity_curve]
            equity = [value for _, value in self.equity_curve]
            ax1.plot(dates, equity, label='Portfolio Value')
            ax1.set_title('Equity Curve')
            ax1.set_ylabel('Portfolio Value ($)')
            ax1.grid(True)
            ax1.legend()
            
            # Plot drawdowns
            dd_dates = [date for date, _ in self.drawdowns]
            dd_values = [value for _, value in self.drawdowns]
            ax2.fill_between(dd_dates, 0, dd_values, color='red', alpha=0.3)
            ax2.set_title('Drawdowns')
            ax2.set_ylabel('Drawdown (%)')
            ax2.set_xlabel('Date')
            ax2.grid(True)
            
            # Adjust layout and save
            plt.tight_layout()
            
            # Create results directory if it doesn't exist
            os.makedirs('results', exist_ok=True)
            
            # Save figure
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results/equity_curve_{timestamp}.png"
            plt.savefig(filename)
            logger.info(f"Equity curve saved to {filename}")
            
            # Add to results
            results['equity_curve_file'] = filename
            
            plt.close()
        
        except Exception as e:
            logger.error(f"Error plotting equity curve: {e}")
    
    def save_results(self, results):
        """Save backtest results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/combined_backtest_results_{timestamp}.json"
        
        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)
        
        # Convert trade directions to strings for JSON serialization
        serializable_results = copy.deepcopy(results)
        if 'trades' in serializable_results:
            for trade in serializable_results['trades']:
                if isinstance(trade['direction'], Enum):
                    trade['direction'] = trade['direction'].name
                elif not isinstance(trade['direction'], str):
                    trade['direction'] = str(trade['direction'])
        
        # Convert dates to strings for JSON serialization
        for trade in serializable_results['trades']:
            for key in ['entry_time', 'exit_time']:
                if key in trade and trade[key] is not None and not isinstance(trade[key], str):
                    trade[key] = str(trade[key])
        
        # Convert equity curve and drawdown dates to strings
        if 'equity_curve' in serializable_results:
            serializable_results['equity_curve'] = [(str(date), value) for date, value in serializable_results['equity_curve']]
        
        if 'drawdown_curve' in serializable_results:
            serializable_results['drawdown_curve'] = [(str(date), value) for date, value in serializable_results['drawdown_curve']]
        
        try:
            with open(filename, 'w') as f:
                json.dump(serializable_results, f, indent=4, default=str)
            logger.info(f"Results saved to: {filename}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
        
        return filename


def load_config(config_file):
    """Load configuration from YAML file"""
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_file}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise


def main():
    """Main function to run the backtest"""
    parser = argparse.ArgumentParser(description='Run combined strategy backtest')
    parser.add_argument('--config', type=str, default='configuration_mean_reversion_trend_combo.yaml',
                        help='Path to configuration file')
    parser.add_argument('--start_date', type=str, default='2023-01-01',
                        help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2023-12-31',
                        help='End date for backtest (YYYY-MM-DD)')
    parser.add_argument('--log_level', type=str, default='INFO',
                        help='Logging level (DEBUG, INFO, WARNING, ERROR)')
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Parse dates
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    # Load configuration
    config = load_config(args.config)
    
    # Create and run backtest
    mean_reversion_strategy = EnhancedMeanReversionBacktest(config)
    trend_following_strategy = TrendFollowingStrategy(config)
    backtest = CombinedStrategyBacktest(config, mean_reversion_strategy, trend_following_strategy)
    results = backtest.run_backtest(start_date, end_date)
    
    if results:
        # Save results
        results_file = backtest.save_results(results)
        
        # Print summary
        print("\n=== Combined Strategy Backtest Results ===")
        print(f"Period: {start_date.date()} to {end_date.date()}")
        print(f"Initial Capital: ${results['initial_capital']:.2f}")
        print(f"Final Equity: ${results['final_equity']:.2f}")
        print(f"Total Return: {results['total_return_pct']:.2f}%")
        print(f"Annualized Return: {results['annualized_return_pct']:.2f}%")
        print(f"Maximum Drawdown: {results['max_drawdown_pct']:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Win Rate: {results['win_rate']:.2f}%")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
        print(f"Total Trades: {results['total_trades']}")
        
        print("\nStrategy Breakdown:")
        print(f"Mean Reversion: {results['mean_reversion']['total_trades']} trades, "
              f"Win Rate: {results['mean_reversion']['winning_trades'] / max(1, results['mean_reversion']['total_trades']) * 100:.2f}%, "
              f"P&L: ${results['mean_reversion']['total_pnl']:.2f}")
        
        print(f"Trend Following: {results['trend_following']['total_trades']} trades, "
              f"Win Rate: {results['trend_following']['winning_trades'] / max(1, results['trend_following']['total_trades']) * 100:.2f}%, "
              f"P&L: ${results['trend_following']['total_pnl']:.2f}")
        
        if results_file:
            print(f"\nResults saved to: {results_file}")
        
        if 'equity_curve_file' in results:
            print(f"Equity curve saved to: {results['equity_curve_file']}")
    
    else:
        print("Backtest failed to produce results")


if __name__ == "__main__":
    main()
