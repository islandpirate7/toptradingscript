#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Improved Mean Reversion Strategy
--------------------------------
This script implements an improved version of the mean reversion strategy
with the following enhancements:
1. Market regime filtering
2. Dynamic position sizing based on risk
3. Proper equity calculation
4. Multi-factor stock selection
"""

import os
import sys
import yaml
import logging
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import necessary modules from the existing codebase
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from test_optimized_mean_reversion_alpaca import AlpacaBacktest
from combined_strategy import CombinedStrategy
from fix_portfolio_classes import Portfolio, Position

class ImprovedMeanReversionBacktest(AlpacaBacktest):
    """
    Improved version of AlpacaBacktest with:
    1. Market regime filtering
    2. Dynamic position sizing
    3. Multi-factor stock selection
    """
    
    def __init__(self, config_path):
        """Initialize the backtest with the given configuration"""
        super().__init__(config_path)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Get strategy parameters
        self.strategy_params = self.config.get('strategies', {}).get('MeanReversion', {})
        
        # Enhanced features
        self.use_regime_filter = self.strategy_params.get('use_regime_filter', True)
        self.use_seasonality = self.strategy_params.get('use_seasonality', True)
        self.use_multi_factor = True  # Always use multi-factor selection
        
        # Risk parameters
        self.risk_per_trade = self.strategy_params.get('risk_per_trade', 0.01)  # 1% risk per trade
        self.max_position_size = self.strategy_params.get('max_position_size', 0.1)  # Max 10% per position
        
        logger.info(f"Initialized Improved Mean Reversion Backtest with config: {config_path}")
        logger.info(f"Market regime filter: {self.use_regime_filter}")
        logger.info(f"Seasonality filter: {self.use_seasonality}")
        logger.info(f"Multi-factor selection: {self.use_multi_factor}")
    
    def load_historical_data(self, start_date, end_date, symbols=None):
        """Load historical data for the specified symbols and date range"""
        import json
        import alpaca_trade_api as tradeapi
        from alpaca_trade_api.rest import TimeFrame
        from dataclasses import dataclass
        
        @dataclass
        class CandleData:
            symbol: str
            timestamp: dt.datetime
            open: float
            high: float
            low: float
            close: float
            volume: int
        
        # Load Alpaca credentials
        try:
            with open('alpaca_credentials.json', 'r') as f:
                credentials = json.load(f)
                paper_credentials = credentials.get('paper', {})
                api_key = paper_credentials.get('api_key')
                api_secret = paper_credentials.get('api_secret')
                base_url = paper_credentials.get('base_url', 'https://paper-api.alpaca.markets')
        except Exception as e:
            logger.error(f"Error loading Alpaca credentials: {e}")
            return {}
        
        # Initialize Alpaca API
        api = tradeapi.REST(api_key, api_secret, base_url)
        
        # Default symbols if none provided
        if symbols is None:
            symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'AMD']
        
        # Always include SPY for market regime detection
        if 'SPY' not in symbols:
            symbols.append('SPY')
        
        # Convert dates to datetime objects if they are strings
        if isinstance(start_date, str):
            start_date = dt.datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = dt.datetime.strptime(end_date, '%Y-%m-%d')
        
        # Add buffer days for indicator calculation
        start_date_with_buffer = start_date - dt.timedelta(days=50)
        
        # Fetch historical data
        symbol_data = {}
        for symbol in symbols:
            try:
                logger.info(f"Fetching data for {symbol} from {start_date_with_buffer} to {end_date}")
                bars = api.get_bars(
                    symbol,
                    TimeFrame.Day,
                    start=start_date_with_buffer.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    adjustment='raw'
                ).df
                
                # Convert to CandleData objects
                candles = []
                for index, row in bars.iterrows():
                    candle = CandleData(
                        symbol=symbol,
                        timestamp=index.to_pydatetime(),
                        open=row['open'],
                        high=row['high'],
                        low=row['low'],
                        close=row['close'],
                        volume=row['volume']
                    )
                    candles.append(candle)
                
                symbol_data[symbol] = candles
                logger.info(f"Loaded {len(candles)} candles for {symbol}")
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
        
        return symbol_data
    
    def get_unique_dates(self, symbol_data):
        """Get a list of unique dates from the symbol data"""
        all_dates = []
        for symbol, candles in symbol_data.items():
            all_dates.extend([candle.timestamp for candle in candles])
        
        # Remove duplicates and sort
        unique_dates = sorted(list(set(all_dates)))
        return unique_dates
    
    def run_backtest(self, start_date, end_date, symbols=None):
        """Run backtest with improved features"""
        logger.info(f"Running backtest from {start_date} to {end_date}")
        
        # Initialize portfolio
        initial_capital = self.config.get('global', {}).get('initial_capital', 100000)
        self.portfolio = Portfolio(initial_capital)
        
        # Load historical data
        symbol_data = self.load_historical_data(start_date, end_date, symbols)
        logger.info(f"Loaded historical data for {len(symbol_data)} symbols")
        
        # Determine market regime
        market_regime = self.determine_market_regime(symbol_data)
        logger.info(f"Initial market regime: {market_regime}")
        
        # Select stocks using multi-factor approach if enabled
        if self.use_multi_factor:
            selected_symbols = self.select_stocks_multi_factor(symbol_data, start_date, end_date)
            logger.info(f"Selected {len(selected_symbols)} symbols using multi-factor approach")
        else:
            selected_symbols = list(symbol_data.keys())
        
        # Initialize results dictionary
        results = {
            'initial_capital': initial_capital,
            'trades': [],
            'equity_curve': []
        }
        
        # Process each day in the backtest period
        dates = self.get_unique_dates(symbol_data)
        for date_idx, timestamp in enumerate(dates):
            # Skip the first few days to have enough data for indicators
            if date_idx < 20:  # Need at least 20 days for BB calculation
                continue
            
            # Update market regime periodically (every 5 trading days)
            if date_idx % 5 == 0:
                market_regime = self.determine_market_regime(symbol_data, date_idx)
                logger.info(f"Updated market regime: {market_regime}")
            
            # Process each symbol
            for symbol in selected_symbols:
                if symbol not in symbol_data:
                    continue
                
                # Get historical candles up to current date
                historical_candles = symbol_data[symbol][:date_idx+1]
                if len(historical_candles) < 30:  # Need enough data for indicators
                    continue
                
                # Get the current candle
                candle = historical_candles[-1]
                
                # Update current price for open positions
                if symbol in self.portfolio.open_positions:
                    position = self.portfolio.open_positions[symbol]
                    position.current_price = candle.close
                    
                    # Check for stop loss
                    if position.stop_loss is not None:
                        if (position.direction == 'long' and candle.low <= position.stop_loss) or \
                           (position.direction == 'short' and candle.high >= position.stop_loss):
                            # Use the stop loss price for exit
                            exit_price = position.stop_loss
                            self.portfolio.close_position(symbol, exit_price, timestamp, "stop_loss")
                            results['trades'].append({
                                'symbol': symbol,
                                'direction': position.direction,
                                'entry_price': position.entry_price,
                                'entry_time': position.entry_time,
                                'exit_price': exit_price,
                                'exit_time': timestamp,
                                'profit_loss': position.profit_loss,
                                'reason': 'stop_loss'
                            })
                            logger.info(f"Closed {position.direction} position for {symbol} at stop loss: {exit_price:.2f}, P/L: {position.profit_loss:.2f}")
                            continue
                    
                    # Check for take profit
                    if position.take_profit is not None:
                        if (position.direction == 'long' and candle.high >= position.take_profit) or \
                           (position.direction == 'short' and candle.low <= position.take_profit):
                            # Use the take profit price for exit
                            exit_price = position.take_profit
                            self.portfolio.close_position(symbol, exit_price, timestamp, "take_profit")
                            results['trades'].append({
                                'symbol': symbol,
                                'direction': position.direction,
                                'entry_price': position.entry_price,
                                'entry_time': position.entry_time,
                                'exit_price': exit_price,
                                'exit_time': timestamp,
                                'profit_loss': position.profit_loss,
                                'reason': 'take_profit'
                            })
                            logger.info(f"Closed {position.direction} position for {symbol} at take profit: {exit_price:.2f}, P/L: {position.profit_loss:.2f}")
                            continue
                
                # Generate signals with market regime awareness
                signals = self.generate_signals(historical_candles, market_regime)
                
                # Process the most recent signal if any
                if signals:
                    latest_signal = signals[-1]
                    
                    # Only process signals for the current timestamp
                    if latest_signal['timestamp'] == timestamp:
                        # Calculate position size based on risk
                        atr = latest_signal.get('atr', 0)
                        if atr > 0:
                            # Risk-based position sizing
                            risk_amount = self.portfolio.get_equity() * self.risk_per_trade
                            stop_loss_distance = atr * self.strategy_params.get('stop_loss_atr', 1.8)
                            
                            if stop_loss_distance > 0:
                                position_size = int(risk_amount / stop_loss_distance)
                                # Limit position size
                                max_position_size = int(self.portfolio.get_equity() * self.max_position_size / latest_signal['price'])
                                position_size = min(position_size, max_position_size)
                            else:
                                position_size = 100  # Default fallback
                        else:
                            position_size = 100  # Default fallback
                        
                        # Adjust position size based on signal strength
                        strength_multiplier = 1.0
                        if 'strength' in latest_signal:
                            if latest_signal['strength'] == 'strong':
                                strength_multiplier = 1.5
                            elif latest_signal['strength'] == 'weak':
                                strength_multiplier = 0.5
                        
                        position_size = int(position_size * strength_multiplier)
                        
                        # Ensure minimum position size
                        position_size = max(position_size, 10)
                        
                        if latest_signal['signal'] == 'buy':
                            # Only take long positions in bullish or neutral regimes
                            if market_regime in ['bullish', 'neutral']:
                                # Open long position
                                success = self.portfolio.open_position(
                                    symbol=symbol,
                                    entry_price=latest_signal['price'],
                                    entry_time=timestamp,
                                    position_size=position_size,
                                    direction='long',
                                    stop_loss=latest_signal['stop_loss'],
                                    take_profit=latest_signal['take_profit']
                                )
                                
                                if success:
                                    logger.info(f"Opened long position for {symbol}: {position_size} shares at {latest_signal['price']:.2f}")
                            else:
                                logger.info(f"Skipped long signal for {symbol} due to bearish market regime")
                        
                        elif latest_signal['signal'] == 'sell':
                            # Only take short positions in bearish or neutral regimes
                            if market_regime in ['bearish', 'neutral']:
                                # Open short position
                                success = self.portfolio.open_position(
                                    symbol=symbol,
                                    entry_price=latest_signal['price'],
                                    entry_time=timestamp,
                                    position_size=position_size,
                                    direction='short',
                                    stop_loss=latest_signal['stop_loss'],
                                    take_profit=latest_signal['take_profit']
                                )
                                
                                if success:
                                    logger.info(f"Opened short position for {symbol}: {position_size} shares at {latest_signal['price']:.2f}")
                            else:
                                logger.info(f"Skipped short signal for {symbol} due to bullish market regime")
            
            # Update equity curve
            self.portfolio.update_equity_curve(timestamp)
        
        # Close any remaining open positions at the last price
        for symbol, position in list(self.portfolio.open_positions.items()):
            if symbol in symbol_data:
                last_candle = symbol_data[symbol][-1]
                self.portfolio.close_position(symbol, last_candle.close, last_candle.timestamp, "end_of_backtest")
                results['trades'].append({
                    'symbol': symbol,
                    'direction': position.direction,
                    'entry_price': position.entry_price,
                    'entry_time': position.entry_time,
                    'exit_price': last_candle.close,
                    'exit_time': last_candle.timestamp,
                    'profit_loss': position.profit_loss,
                    'reason': 'end_of_backtest'
                })
        
        # Calculate final results
        results['final_capital'] = self.portfolio.get_equity()
        results['return'] = (results['final_capital'] - results['initial_capital']) / results['initial_capital']
        results['win_rate'] = self.portfolio.get_win_rate()
        results['profit_factor'] = self.portfolio.get_profit_factor()
        results['max_drawdown'] = self.portfolio.get_max_drawdown()
        results['total_trades'] = len(self.portfolio.closed_positions)
        
        # Save equity curve
        results['equity_curve'] = [(timestamp.isoformat(), equity) for timestamp, equity in self.portfolio.equity_curve]
        
        logger.info(f"Backtest completed with {results['total_trades']} trades")
        logger.info(f"Final capital: ${results['final_capital']:.2f} (Return: {results['return']:.2%})")
        logger.info(f"Win rate: {results['win_rate']:.2%}")
        logger.info(f"Profit factor: {results['profit_factor']:.2f}")
        logger.info(f"Max drawdown: {results['max_drawdown']:.2%}")
        
        return results
    
    def determine_market_regime(self, symbol_data, current_idx=None):
        """Determine the current market regime (bullish, bearish, neutral)"""
        # Use SPY as a proxy for the overall market
        if 'SPY' not in symbol_data:
            logger.warning("SPY data not available for market regime detection")
            return 'neutral'
        
        spy_data = symbol_data['SPY']
        if current_idx is not None:
            spy_data = spy_data[:current_idx+1]
        
        if len(spy_data) < 50:
            logger.warning("Not enough SPY data for market regime detection")
            return 'neutral'
        
        # Calculate short and long-term moving averages
        closes = [candle.close for candle in spy_data]
        ma20 = sum(closes[-20:]) / 20
        ma50 = sum(closes[-50:]) / 50
        
        # Calculate RSI
        changes = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains = [max(0, change) for change in changes]
        losses = [max(0, -change) for change in changes]
        
        avg_gain = sum(gains[-14:]) / 14
        avg_loss = sum(losses[-14:]) / 14
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        # Determine regime based on moving averages and RSI
        if ma20 > ma50 * 1.02 and rsi > 50:
            return 'bullish'
        elif ma20 < ma50 * 0.98 and rsi < 50:
            return 'bearish'
        else:
            return 'neutral'
    
    def select_stocks_multi_factor(self, symbol_data, start_date, end_date):
        """Select stocks using a multi-factor approach with seasonality"""
        # Initialize configuration for multi-factor selection
        config = {
            'general': {
                'symbols': list(symbol_data.keys()),
                'log_level': 'INFO'
            },
            'stock_selection': {
                'enable_multi_factor': True,
                'technical_weight': 0.6,  # Reduced to make room for seasonality
                'seasonality_weight': 0.4,  # Added significant weight to seasonality
                'technical_factors': {
                    'momentum_weight': 0.25,
                    'trend_weight': 0.25,
                    'volatility_weight': 0.25,
                    'volume_weight': 0.25
                },
                'position_sizing': {
                    'base_position_pct': 0.05,
                    'min_position_pct': 0.02,
                    'max_position_pct': 0.08,
                    'score_scaling_factor': 0.5,
                    'short_size_factor': 0.8
                },
                'regime_adjustments': {
                    'trending': {
                        'long_bias_multiplier': 1.2,
                        'short_bias_multiplier': 0.8
                    },
                    'range_bound': {
                        'long_bias_multiplier': 1.0,
                        'short_bias_multiplier': 1.0
                    }
                },
                'top_n_stocks': 5
            },
            'seasonality': {
                'enabled': True,
                'data_file': 'output/seasonal_opportunities_converted.yaml',
                'min_score_threshold': 0.6,
                'weight_adjustment': True,
                'sector_influence': 0.3,
                'stock_specific_influence': 0.7,
                'top_n_selection': 10,
                'boost_factor': 0.2,
                'penalty_factor': 0.2
            }
        }
        
        # Get the last date in the data
        last_date = end_date
        if isinstance(last_date, str):
            last_date = dt.datetime.strptime(last_date, '%Y-%m-%d')
        
        # Initialize CombinedStrategy with proper configuration
        combined_strategy = CombinedStrategy(config)
        
        # Set the symbol data
        combined_strategy.set_symbol_data(symbol_data)
        
        # Detect market regime using SPY data if available
        market_regime = None
        if 'SPY' in symbol_data and symbol_data['SPY'] is not None:
            spy_data = symbol_data['SPY']
            if len(spy_data) > 20:  # Ensure enough data for regime detection
                market_regime = combined_strategy.detect_market_regime(spy_data)
                logger.info(f"Detected market regime: {market_regime.name}")
        
        # Select top stocks
        try:
            selected_stocks = combined_strategy.select_stocks_multi_factor(
                symbol_data,
                current_date=last_date,
                top_n=5,  # Select top 5 stocks
                direction='ANY',  # Allow both long and short
                market_regime=market_regime
            )
            
            if selected_stocks:
                logger.info(f"Selected {len(selected_stocks)} stocks using multi-factor approach")
                for i, stock in enumerate(selected_stocks):
                    logger.info(f"  {i+1}. {stock['symbol']} - Score: {stock['combined_score']:.4f}, "
                               f"Direction: {stock['technical_direction']}, "
                               f"Seasonal Score: {stock['seasonal_score']:.4f}")
            else:
                logger.warning("No stocks selected using multi-factor approach")
                
            return selected_stocks
        except Exception as e:
            logger.error(f"Error in multi-factor stock selection: {e}")
            # Fallback to all symbols
            return [{'symbol': symbol, 'combined_score': 0.5, 'technical_direction': 'NEUTRAL'} 
                    for symbol in symbol_data.keys()]
    
    def generate_signals(self, candles, market_regime):
        """Generate trading signals with market regime awareness"""
        if len(candles) < 30:
            return []
        
        signals = []
        
        # Extract parameters from config
        bb_period = self.strategy_params.get('bb_period', 20)
        bb_std_dev = self.strategy_params.get('bb_std_dev', 1.9)
        rsi_period = self.strategy_params.get('rsi_period', 14)
        rsi_overbought = self.strategy_params.get('rsi_overbought', 65)
        rsi_oversold = self.strategy_params.get('rsi_oversold', 35)
        require_reversal = self.strategy_params.get('require_reversal', True)
        min_reversal_candles = self.strategy_params.get('min_reversal_candles', 1)
        stop_loss_atr = self.strategy_params.get('stop_loss_atr', 1.8)
        take_profit_atr = self.strategy_params.get('take_profit_atr', 3.0)
        atr_period = self.strategy_params.get('atr_period', 14)
        volume_filter = self.strategy_params.get('volume_filter', True)
        
        # Calculate Bollinger Bands
        closes = [candle.close for candle in candles]
        ma = sum(closes[-bb_period:]) / bb_period
        std_dev = (sum([(close - ma) ** 2 for close in closes[-bb_period:]]) / bb_period) ** 0.5
        
        upper_band = ma + (bb_std_dev * std_dev)
        lower_band = ma - (bb_std_dev * std_dev)
        
        # Calculate RSI
        changes = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains = [max(0, change) for change in changes]
        losses = [max(0, -change) for change in changes]
        
        avg_gain = sum(gains[-rsi_period:]) / rsi_period
        avg_loss = sum(losses[-rsi_period:]) / rsi_period
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        # Calculate ATR for stop loss and take profit
        atr = self.calculate_atr(candles, atr_period)
        
        # Current price
        current_price = candles[-1].close
        
        # Check for buy signal (oversold condition)
        if current_price < lower_band * 1.02 and rsi < rsi_oversold * 1.1:
            # Check for price reversal if required
            reversal = True
            if require_reversal:
                reversal = False
                # Check for min_reversal_candles consecutive higher lows
                if len(candles) > min_reversal_candles + 1:
                    reversal = True
                    for i in range(1, min_reversal_candles + 1):
                        if candles[-i].low <= candles[-i-1].low:
                            reversal = False
                            break
            
            if not reversal:
                return signals
            
            # Check for volume confirmation if required
            if volume_filter:
                volume_increase = candles[-1].volume > sum([c.volume for c in candles[-6:-1]]) / 5 * 1.2
                if not volume_increase:
                    return signals
            
            # Adjust signal based on market regime
            strength = 'moderate'
            if market_regime == 'bullish':
                strength = 'strong'
            elif market_regime == 'bearish':
                strength = 'weak'
            
            # Calculate stop loss and take profit
            stop_loss = current_price - (atr * stop_loss_atr)
            take_profit = current_price + (atr * take_profit_atr)
            
            signals.append({
                'symbol': candles[-1].symbol,
                'signal': 'buy',
                'price': current_price,
                'timestamp': candles[-1].timestamp,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'strength': strength,
                'atr': atr
            })
        
        # Check for sell signal (overbought condition)
        elif current_price > upper_band * 0.98 and rsi > rsi_overbought * 0.9:
            # Check for price reversal if required
            reversal = True
            if require_reversal:
                reversal = False
                # Check for min_reversal_candles consecutive lower highs
                if len(candles) > min_reversal_candles + 1:
                    reversal = True
                    for i in range(1, min_reversal_candles + 1):
                        if candles[-i].high >= candles[-i-1].high:
                            reversal = False
                            break
            
            if not reversal:
                return signals
            
            # Check for volume confirmation if required
            if volume_filter:
                volume_increase = candles[-1].volume > sum([c.volume for c in candles[-6:-1]]) / 5 * 1.2
                if not volume_increase:
                    return signals
            
            # Adjust signal based on market regime
            strength = 'moderate'
            if market_regime == 'bearish':
                strength = 'strong'
            elif market_regime == 'bullish':
                strength = 'weak'
            
            # Calculate stop loss and take profit
            stop_loss = current_price + (atr * stop_loss_atr)
            take_profit = current_price - (atr * take_profit_atr)
            
            signals.append({
                'symbol': candles[-1].symbol,
                'signal': 'sell',
                'price': current_price,
                'timestamp': candles[-1].timestamp,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'strength': strength,
                'atr': atr
            })
        
        return signals
    
    def calculate_atr(self, candles, period):
        """Calculate Average True Range"""
        if len(candles) < period + 1:
            return 0.0
        
        true_ranges = []
        for i in range(1, len(candles)):
            high = candles[i].high
            low = candles[i].low
            prev_close = candles[i-1].close
            
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            
            true_range = max(tr1, tr2, tr3)
            true_ranges.append(true_range)
        
        recent_true_ranges = true_ranges[-period:]
        
        if not recent_true_ranges:
            return 0.0
        
        return sum(recent_true_ranges) / len(recent_true_ranges)

def main():
    """Main function to run the improved backtest"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run improved mean reversion strategy backtest')
    parser.add_argument('--config', type=str, default='configuration_enhanced_mean_reversion.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--start', type=str, default='2023-01-01',
                        help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2023-03-31',
                        help='End date for backtest (YYYY-MM-DD)')
    parser.add_argument('--quarter', type=int, choices=[1, 2, 3, 4],
                        help='Quarter of 2023 to run backtest for (overrides start/end)')
    parser.add_argument('--symbols', type=str, nargs='+',
                        help='List of symbols to include in backtest')
    
    args = parser.parse_args()
    
    # Set start and end dates based on quarter if specified
    if args.quarter:
        if args.quarter == 1:
            args.start = '2023-01-01'
            args.end = '2023-03-31'
        elif args.quarter == 2:
            args.start = '2023-04-01'
            args.end = '2023-06-30'
        elif args.quarter == 3:
            args.start = '2023-07-01'
            args.end = '2023-09-30'
        elif args.quarter == 4:
            args.start = '2023-10-01'
            args.end = '2023-12-31'
    
    # Initialize and run backtest
    backtest = ImprovedMeanReversionBacktest(args.config)
    results = backtest.run_backtest(args.start, args.end, args.symbols)
    
    # Analyze and plot results
    analyze_results(results)

def analyze_results(results):
    """Analyze and plot backtest results"""
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Calculate performance metrics
    trades_df = pd.DataFrame(results['trades'])
    
    if len(trades_df) == 0:
        logger.info("No trades executed during the backtest period")
        return
    
    # Calculate returns
    if 'profit_loss' not in trades_df.columns:
        trades_df['return'] = (trades_df['exit_price'] - trades_df['entry_price']) / trades_df['entry_price']
        trades_df.loc[trades_df['direction'] == 'short', 'return'] = -trades_df.loc[trades_df['direction'] == 'short', 'return']
    else:
        # Calculate returns based on profit_loss
        trades_df['return'] = trades_df['profit_loss'] / (trades_df['entry_price'] * 100)  # Assuming position size of 100
    
    # Overall performance
    total_trades = len(trades_df)
    win_rate = len(trades_df[trades_df['return'] > 0]) / total_trades if total_trades > 0 else 0
    avg_return = trades_df['return'].mean() if total_trades > 0 else 0
    median_return = trades_df['return'].median() if total_trades > 0 else 0
    max_return = trades_df['return'].max() if total_trades > 0 else 0
    min_return = trades_df['return'].min() if total_trades > 0 else 0
    
    # Calculate profit factor
    gross_profit = trades_df.loc[trades_df['return'] > 0, 'return'].sum() if len(trades_df[trades_df['return'] > 0]) > 0 else 0
    gross_loss = abs(trades_df.loc[trades_df['return'] < 0, 'return'].sum()) if len(trades_df[trades_df['return'] < 0]) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    logger.info("\n=== Performance Metrics ===")
    logger.info(f"Total trades: {total_trades}")
    logger.info(f"Win rate: {win_rate:.2%}")
    logger.info(f"Average return: {avg_return:.2%}")
    logger.info(f"Median return: {median_return:.2%}")
    logger.info(f"Maximum return: {max_return:.2%}")
    logger.info(f"Minimum return: {min_return:.2%}")
    logger.info(f"Profit factor: {profit_factor:.2f}")
    
    # Performance by direction
    logger.info("\n=== Performance by Direction ===")
    for direction in trades_df['direction'].unique():
        direction_df = trades_df[trades_df['direction'] == direction]
        direction_trades = len(direction_df)
        direction_win_rate = len(direction_df[direction_df['return'] > 0]) / direction_trades if direction_trades > 0 else 0
        direction_avg_return = direction_df['return'].mean() if direction_trades > 0 else 0
        logger.info(f"{direction}: {direction_trades} trades, Win rate: {direction_win_rate:.2%}, Avg return: {direction_avg_return:.2%}")
    
    # Performance by symbol
    logger.info("\n=== Performance by Symbol ===")
    for symbol in trades_df['symbol'].unique():
        symbol_df = trades_df[trades_df['symbol'] == symbol]
        symbol_trades = len(symbol_df)
        symbol_win_rate = len(symbol_df[symbol_df['return'] > 0]) / symbol_trades if symbol_trades > 0 else 0
        symbol_avg_return = symbol_df['return'].mean() if symbol_trades > 0 else 0
        logger.info(f"{symbol}: {symbol_trades} trades, Win rate: {symbol_win_rate:.2%}, Avg return: {symbol_avg_return:.2%}")
    
    # Plot equity curve
    if 'equity_curve' in results and results['equity_curve']:
        dates = [dt.datetime.fromisoformat(date) for date, _ in results['equity_curve']]
        equity = [value for _, value in results['equity_curve']]
        
        plt.figure(figsize=(12, 6))
        plt.plot(dates, equity)
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity ($)')
        plt.grid(True)
        plt.savefig('output/equity_curve.png')
        plt.close()
    
    # Plot returns distribution
    plt.figure(figsize=(12, 6))
    trades_df['return'].hist(bins=50)
    plt.title('Returns Distribution')
    plt.xlabel('Return (%)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig('output/returns_distribution.png')
    plt.close()
    
    # Save trades to CSV
    trades_df.to_csv('output/trades.csv', index=False)
    
    logger.info("\nAnalysis complete. Plots saved to output directory.")

if __name__ == "__main__":
    main()
