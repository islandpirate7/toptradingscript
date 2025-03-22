#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MeanReversion Strategy Backtest Script
Optimized based on previous testing results
"""

import os
import sys
import json
import yaml
import logging
import argparse
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import alpaca_trade_api as tradeapi
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import time
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Data classes for the backtest
@dataclass
class CandleData:
    timestamp: datetime.datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

@dataclass
class Signal:
    symbol: str
    timestamp: datetime.datetime
    direction: str  # 'long' or 'short'
    strategy_name: str
    score: float
    strength: str  # 'weak', 'moderate', 'strong'
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    is_crypto: bool = False

@dataclass
class Trade:
    symbol: str
    entry_date: datetime.datetime
    entry_price: float
    direction: str  # 'long' or 'short'
    position_size: float
    stop_loss: float
    take_profit: float
    is_crypto: bool = False
    strategy_name: str = ""
    exit_date: Optional[datetime.datetime] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""  # 'stop_loss', 'take_profit', 'time_exit'

@dataclass
class MarketState:
    date: datetime.datetime
    regime: str  # 'bullish', 'bearish', 'neutral'
    volatility: float
    trend_strength: float
    is_range_bound: bool

class MeanReversionStrategy:
    """Mean Reversion Strategy based on Bollinger Bands and RSI"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.name = "MeanReversion"
        self.logger = logging.getLogger(self.name)
    
    def get_param(self, param_name: str, default_value=None):
        """Get parameter from config, first checking symbol-specific params, then global params"""
        mean_reversion_params = self.config.get('mean_reversion_params', {})
        return mean_reversion_params.get(param_name, default_value)
    
    def calculate_indicators(self, candles: List[CandleData]):
        """Calculate Bollinger Bands and RSI for the given candles"""
        if len(candles) < 20:
            return None, None, None, None, None
        
        # Extract close prices
        closes = np.array([c.close for c in candles])
        
        # Calculate Bollinger Bands
        bb_period = self.get_param('bb_period', 20)
        bb_std_dev = self.get_param('bb_std_dev', 1.9)
        
        if len(closes) < bb_period:
            return None, None, None, None, None
        
        sma = np.mean(closes[-bb_period:])
        std_dev = np.std(closes[-bb_period:])
        upper_band = sma + (std_dev * bb_std_dev)
        lower_band = sma - (std_dev * bb_std_dev)
        
        # Calculate RSI
        rsi_period = self.get_param('rsi_period', 14)
        
        if len(closes) < rsi_period + 1:
            return None, None, None, None, None
        
        # Calculate price changes
        delta = np.diff(closes)
        
        # Create arrays for gains and losses
        gains = np.copy(delta)
        losses = np.copy(delta)
        
        # Set gains to 0 where price decreased
        gains[gains < 0] = 0
        
        # Set losses to 0 where price increased, and make losses positive
        losses[losses > 0] = 0
        losses = np.abs(losses)
        
        # Calculate average gains and losses over RSI period
        avg_gain = np.mean(gains[-rsi_period:])
        avg_loss = np.mean(losses[-rsi_period:])
        
        # Calculate RS and RSI
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        # Calculate ATR for stop loss and take profit
        atr_period = self.get_param('atr_period', 14)
        
        if len(candles) < atr_period + 1:
            return upper_band, lower_band, sma, rsi, None
        
        # Calculate true ranges
        tr_values = []
        for i in range(1, len(candles)):
            high = candles[i].high
            low = candles[i].low
            prev_close = candles[i-1].close
            
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            
            tr = max(tr1, tr2, tr3)
            tr_values.append(tr)
        
        # Calculate ATR
        atr = np.mean(tr_values[-atr_period:])
        
        return upper_band, lower_band, sma, rsi, atr

    def generate_signals(self, symbol: str, candles: List[CandleData], market_state: MarketState, is_crypto: bool = False) -> List[Signal]:
        """Generate mean reversion signals based on Bollinger Bands and RSI"""
        if len(candles) < 30:  # Need at least 30 candles for reliable signals
            return []
        
        # Calculate indicators
        upper_band, lower_band, sma, rsi, atr = self.calculate_indicators(candles)
        
        if upper_band is None or lower_band is None or rsi is None:
            return []
        
        # Get parameters
        rsi_overbought = self.get_param('rsi_overbought', 65)
        rsi_oversold = self.get_param('rsi_oversold', 35)
        require_reversal = self.get_param('require_reversal', True)
        min_reversal_candles = self.get_param('min_reversal_candles', 1)
        stop_loss_atr = self.get_param('stop_loss_atr', 1.8)
        take_profit_atr = self.get_param('take_profit_atr', 3.0)
        
        signals = []
        current_candle = candles[-1]
        current_price = current_candle.close
        current_time = current_candle.timestamp
        
        # Check for long signal (price below lower band and RSI oversold)
        if current_price < lower_band and rsi < rsi_oversold:
            # Check for reversal if required
            reversal_confirmed = True
            if require_reversal:
                # Check if price has been declining and is now starting to reverse
                reversal_confirmed = False
                declining_count = 0
                for i in range(2, min(len(candles), min_reversal_candles + 3)):
                    if candles[-i].close < candles[-i-1].close:
                        declining_count += 1
                
                # Confirm reversal if we had enough declining candles and current candle is up
                if declining_count >= min_reversal_candles and current_candle.close > current_candle.open:
                    reversal_confirmed = True
            
            if reversal_confirmed:
                # Calculate stop loss and take profit based on ATR
                if atr is not None:
                    stop_loss = current_price - (atr * stop_loss_atr)
                    take_profit = current_price + (atr * take_profit_atr)
                else:
                    # Fallback if ATR is not available
                    stop_loss = current_price * 0.98  # 2% stop loss
                    take_profit = current_price * 1.03  # 3% take profit
                
                # Determine signal strength based on how far price is from the band and RSI level
                band_distance = (lower_band - current_price) / lower_band
                rsi_distance = (rsi_oversold - rsi) / rsi_oversold
                
                score = (band_distance * 0.6) + (rsi_distance * 0.4)
                
                if score > 0.1:
                    strength = "strong"
                elif score > 0.05:
                    strength = "moderate"
                else:
                    strength = "weak"
                
                signal = Signal(
                    symbol=symbol,
                    timestamp=current_time,
                    direction="long",
                    strategy_name=self.name,
                    score=score,
                    strength=strength,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    is_crypto=is_crypto
                )
                
                signals.append(signal)
                self.logger.info(f"Generated LONG signal for {symbol}: price={current_price:.2f}, lower_band={lower_band:.2f}, RSI={rsi:.2f}")
        
        # Check for short signal (price above upper band and RSI overbought)
        elif current_price > upper_band and rsi > rsi_overbought:
            # Check for reversal if required
            reversal_confirmed = True
            if require_reversal:
                # Check if price has been rising and is now starting to reverse
                reversal_confirmed = False
                rising_count = 0
                for i in range(2, min(len(candles), min_reversal_candles + 3)):
                    if candles[-i].close > candles[-i-1].close:
                        rising_count += 1
                
                # Confirm reversal if we had enough rising candles and current candle is down
                if rising_count >= min_reversal_candles and current_candle.close < current_candle.open:
                    reversal_confirmed = True
            
            if reversal_confirmed:
                # Calculate stop loss and take profit based on ATR
                if atr is not None:
                    stop_loss = current_price + (atr * stop_loss_atr)
                    take_profit = current_price - (atr * take_profit_atr)
                else:
                    # Fallback if ATR is not available
                    stop_loss = current_price * 1.02  # 2% stop loss
                    take_profit = current_price * 0.97  # 3% take profit
                
                # Determine signal strength based on how far price is from the band and RSI level
                band_distance = (current_price - upper_band) / upper_band
                rsi_distance = (rsi - rsi_overbought) / (100 - rsi_overbought)
                
                score = (band_distance * 0.6) + (rsi_distance * 0.4)
                
                if score > 0.1:
                    strength = "strong"
                elif score > 0.05:
                    strength = "moderate"
                else:
                    strength = "weak"
                
                signal = Signal(
                    symbol=symbol,
                    timestamp=current_time,
                    direction="short",
                    strategy_name=self.name,
                    score=score,
                    strength=strength,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    is_crypto=is_crypto
                )
                
                signals.append(signal)
                self.logger.info(f"Generated SHORT signal for {symbol}: price={current_price:.2f}, upper_band={upper_band:.2f}, RSI={rsi:.2f}")
        
        return signals

class MeanReversionBacktest:
    """Backtest implementation for the MeanReversion strategy"""
    
    def __init__(self, config_file: str):
        """Initialize the backtest with the given configuration file"""
        self.config_file = config_file
        self.logger = logging.getLogger("MeanReversionBacktest")
        
        # Load configuration
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize variables
        self.initial_capital = self.config.get('initial_capital', 100000)
        self.cash = self.initial_capital
        self.equity_curve = {}
        self.trade_history = []
        self.current_positions = {}
        
        # Initialize strategy
        self.strategy = MeanReversionStrategy(self.config)
        
        # Initialize stock and crypto configurations
        self.stock_configs = {}
        self.crypto_configs = {}
        
        for stock in self.config.get('stocks', []):
            symbol = stock.get('symbol')
            if symbol:
                self.stock_configs[symbol] = stock
        
        for crypto in self.config.get('cryptos', []):
            symbol = crypto.get('symbol')
            if symbol:
                self.crypto_configs[symbol] = crypto
        
        # Initialize Alpaca API
        self._init_alpaca_api()
    
    def _init_alpaca_api(self):
        """Initialize Alpaca API with credentials from alpaca_credentials.json"""
        try:
            with open('alpaca_credentials.json', 'r') as f:
                credentials = json.load(f)
            
            # Use paper trading credentials by default
            paper_creds = credentials.get('paper', {})
            api_key = paper_creds.get('api_key', '')
            api_secret = paper_creds.get('api_secret', '')
            base_url = paper_creds.get('base_url', 'https://paper-api.alpaca.markets')
            
            self.alpaca = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
            self.logger.info("Initialized Alpaca API with paper trading account")
        except Exception as e:
            self.logger.error(f"Error initializing Alpaca API: {str(e)}")
            sys.exit(1)
    
    def fetch_historical_data(self, symbol: str, start_date: datetime.datetime, end_date: datetime.datetime, is_crypto: bool = False) -> List[CandleData]:
        """Fetch historical data for the given symbol from Alpaca"""
        try:
            # Format symbol for Alpaca API (crypto needs special formatting)
            alpaca_symbol = symbol
            if is_crypto:
                # Convert BTC/USD to BTCUSD for Alpaca
                alpaca_symbol = symbol.replace('/', '')
            
            # Fetch data from Alpaca
            timeframe = '1D'  # Daily candles
            self.logger.info(f"Fetching data for {symbol}")
            
            # Adjust dates for Alpaca API (add one day to end_date to include it)
            adjusted_end_date = end_date + datetime.timedelta(days=1)
            
            # Fetch historical data
            bars = self.alpaca.get_bars(
                alpaca_symbol,
                timeframe,
                start=start_date.strftime('%Y-%m-%d'),
                end=adjusted_end_date.strftime('%Y-%m-%d'),
                adjustment='raw'
            ).df
            
            # Convert to CandleData objects
            candles = []
            for index, row in bars.iterrows():
                candle = CandleData(
                    timestamp=index.to_pydatetime(),
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row['volume']
                )
                candles.append(candle)
            
            self.logger.info(f"Fetched {len(candles)} candles for {symbol}")
            return candles
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return []
    
    def update_market_state(self, date: datetime.datetime, spy_candles: List[CandleData]) -> MarketState:
        """Update market state based on SPY data"""
        if not spy_candles:
            return MarketState(
                date=date,
                regime="neutral",
                volatility=0.0,
                trend_strength=0.0,
                is_range_bound=False
            )
        
        # Get candles up to the current date
        current_candles = [c for c in spy_candles if c.timestamp.replace(tzinfo=None).date() <= date.date()]
        
        if len(current_candles) < 20:
            self.logger.warning("Not enough SPY data for market state update")
            return MarketState(
                date=date,
                regime="neutral",
                volatility=0.0,
                trend_strength=0.0,
                is_range_bound=False
            )
        
        # Calculate 20-day and 50-day moving averages
        closes = [c.close for c in current_candles]
        ma20 = np.mean(closes[-20:])
        ma50 = np.mean(closes[-50:]) if len(closes) >= 50 else ma20
        
        # Calculate volatility (20-day standard deviation)
        volatility = np.std(closes[-20:]) / ma20
        
        # Determine market regime
        if ma20 > ma50 * 1.02:
            regime = "bullish"
        elif ma20 < ma50 * 0.98:
            regime = "bearish"
        else:
            regime = "neutral"
        
        # Calculate trend strength (ADX-like measure)
        trend_strength = 0.0
        if len(closes) >= 14:
            # Calculate directional movement
            up_moves = []
            down_moves = []
            for i in range(1, 14):
                up_move = current_candles[-i].high - current_candles[-i-1].high
                down_move = current_candles[-i-1].low - current_candles[-i].low
                
                if up_move > 0 and up_move > down_move:
                    up_moves.append(up_move)
                else:
                    up_moves.append(0)
                
                if down_move > 0 and down_move > up_move:
                    down_moves.append(down_move)
                else:
                    down_moves.append(0)
            
            # Calculate directional indicators
            avg_up = np.mean(up_moves)
            avg_down = np.mean(down_moves)
            
            if avg_up + avg_down > 0:
                trend_strength = (abs(avg_up - avg_down) / (avg_up + avg_down)) * 100
        
        # Determine if market is range-bound
        is_range_bound = trend_strength < 20
        
        return MarketState(
            date=date,
            regime=regime,
            volatility=volatility,
            trend_strength=trend_strength,
            is_range_bound=is_range_bound
        )
    
    def generate_signals_for_symbol(self, symbol: str, candles: List[CandleData], market_state: MarketState, is_crypto: bool = False) -> List[Signal]:
        """Generate signals for a symbol using the MeanReversion strategy"""
        if not candles or len(candles) < 30:
            return []
        
        # Get the appropriate configuration for the symbol
        config = None
        if is_crypto:
            if symbol in self.crypto_configs:
                config = self.crypto_configs[symbol]
        else:
            if symbol in self.stock_configs:
                config = self.stock_configs[symbol]
        
        if not config:
            self.logger.warning(f"No configuration found for {symbol}")
            return []
        
        # Generate signals
        signals = self.strategy.generate_signals(symbol, candles, market_state, is_crypto)
        self.logger.info(f"Generated {len(signals)} signals for {symbol}")
        
        return signals
    
    def _process_signals(self, date: datetime.datetime, signals: List[Signal], current_prices: Dict[str, float]) -> None:
        """Process signals and create trades based on current equity and position sizing config"""
        if not signals:
            return
        
        # Get position sizing configuration
        position_sizing_config = self.config.get('position_sizing_config', {})
        base_risk_per_trade = position_sizing_config.get('base_risk_per_trade', 0.01)
        max_position_size = position_sizing_config.get('max_position_size', 0.1)
        min_position_size = position_sizing_config.get('min_position_size', 0.005)
        
        # Calculate current equity
        current_equity = self.calculate_current_equity(date, current_prices)
        
        # Process each signal
        for signal in signals:
            symbol = signal.symbol
            
            # Skip if we already have a position in this symbol
            if symbol in self.current_positions:
                self.logger.info(f"Skipping signal for {symbol} as we already have a position")
                continue
            
            # Skip if we don't have the current price
            if symbol not in current_prices:
                self.logger.warning(f"No current price for {symbol}, skipping signal")
                continue
            
            current_price = current_prices[symbol]
            
            # Get symbol-specific config
            symbol_config = None
            if signal.is_crypto and symbol in self.crypto_configs:
                symbol_config = self.crypto_configs[symbol]
            elif not signal.is_crypto and symbol in self.stock_configs:
                symbol_config = self.stock_configs[symbol]
            
            # Calculate position size based on risk
            if signal.stop_loss is not None:
                # Calculate risk amount
                risk_amount = current_equity * base_risk_per_trade
                
                # Calculate risk per share
                risk_per_share = abs(current_price - signal.stop_loss)
                
                # Calculate position size in shares/contracts
                if risk_per_share > 0:
                    position_size = risk_amount / risk_per_share
                else:
                    # Fallback if risk_per_share is zero or very small
                    position_size = (current_equity * min_position_size) / current_price
            else:
                # Fallback if no stop loss is defined
                position_size = (current_equity * min_position_size) / current_price
            
            # Apply symbol-specific limits if available
            if symbol_config:
                max_symbol_position = symbol_config.get('max_position_size', float('inf'))
                min_symbol_position = symbol_config.get('min_position_size', 0)
                
                # Ensure position size is within limits
                position_size = min(position_size, max_symbol_position)
                position_size = max(position_size, min_symbol_position)
            
            # Ensure position size doesn't exceed max percentage of equity
            max_equity_position = (current_equity * max_position_size) / current_price
            position_size = min(position_size, max_equity_position)
            
            # Adjust position size based on signal strength
            if signal.strength == "strong":
                position_size *= 1.2
            elif signal.strength == "weak":
                position_size *= 0.8
            
            # Create trade
            trade = Trade(
                symbol=symbol,
                entry_date=date,
                entry_price=current_price,
                direction=signal.direction,
                position_size=position_size,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                is_crypto=signal.is_crypto,
                strategy_name=signal.strategy_name
            )
            
            # Add to current positions
            self.current_positions[symbol] = trade
            
            # Update cash
            self.cash -= position_size * current_price
            
            self.logger.info(f"Created trade for {symbol}: {signal.direction} {position_size:.2f} shares at {current_price:.2f}")
    
    def _update_positions(self, date: datetime.datetime, current_prices: Dict[str, float]) -> None:
        """Update positions based on current prices and check for exits"""
        if not self.current_positions:
            return
        
        positions_to_remove = []
        
        for symbol, trade in self.current_positions.items():
            # Skip if we don't have the current price
            if symbol not in current_prices:
                self.logger.warning(f"No current price for {symbol}, skipping position update")
                continue
            
            current_price = current_prices[symbol]
            
            # Check for stop loss or take profit
            exit_reason = None
            
            if trade.direction == "long":
                if current_price <= trade.stop_loss:
                    exit_reason = "stop_loss"
                elif current_price >= trade.take_profit:
                    exit_reason = "take_profit"
            else:  # short
                if current_price >= trade.stop_loss:
                    exit_reason = "stop_loss"
                elif current_price <= trade.take_profit:
                    exit_reason = "take_profit"
            
            # Exit the position if needed
            if exit_reason:
                # Update trade
                trade.exit_date = date
                trade.exit_price = current_price
                
                # Calculate P&L
                if trade.direction == "long":
                    trade.pnl = (trade.exit_price - trade.entry_price) * trade.position_size
                    trade.pnl_pct = (trade.exit_price - trade.entry_price) / trade.entry_price
                else:  # short
                    trade.pnl = (trade.entry_price - trade.exit_price) * trade.position_size
                    trade.pnl_pct = (trade.entry_price - trade.exit_price) / trade.entry_price
                
                trade.exit_reason = exit_reason
                
                # Add to trade history
                self.trade_history.append(trade)
                
                # Update cash
                self.cash += (trade.position_size * current_price)
                
                # Mark for removal
                positions_to_remove.append(symbol)
                
                self.logger.info(f"Exited {symbol} position: {exit_reason} at {current_price:.2f}, P&L: {trade.pnl:.2f} ({trade.pnl_pct:.2%})")
        
        # Remove closed positions
        for symbol in positions_to_remove:
            del self.current_positions[symbol]
    
    def calculate_current_equity(self, date: datetime.datetime, current_prices: Dict[str, float]) -> float:
        """Calculate current equity based on cash and open positions"""
        equity = self.cash
        
        for symbol, trade in self.current_positions.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                position_value = trade.position_size * current_price
                equity += position_value
        
        # Update equity curve
        self.equity_curve[date] = equity
        
        return equity
    
    def run_backtest(self, start_date: datetime.datetime, end_date: datetime.datetime) -> Dict:
        """Run the backtest for the specified date range."""
        self.logger.info(f"Starting backtest from {start_date.date()} to {end_date.date()}")
        
        # Debug: Print the symbols from the configuration
        self.logger.info(f"Configuration contains {len(self.config['symbols'])} symbols")
        for symbol_config in self.config['symbols']:
            self.logger.info(f"Symbol in config: {symbol_config['symbol']}")
        
        # Reset backtest state
        self.cash = self.initial_capital
        self.equity_curve = {}
        self.trade_history = []
        self.current_positions = {}
        
        # Initialize dictionaries for stock and crypto symbols
        self.stock_configs = {}
        self.crypto_configs = {}
        
        # Separate stocks and crypto symbols
        for symbol_config in self.config["symbols"]:
            symbol = symbol_config["symbol"]
            if "/" in symbol:  # Crypto symbols contain "/"
                self.crypto_configs[symbol] = symbol_config
            else:
                self.stock_configs[symbol] = symbol_config
        
        self.logger.info(f"Processed {len(self.stock_configs)} stock symbols and {len(self.crypto_configs)} crypto symbols")
        
        # Initialize Alpaca API
        self._init_alpaca_api()
        
        # Fetch SPY data for market state
        self.logger.info(f"Fetching data for SPY")
        spy_candles = self.fetch_historical_data("SPY", start_date - datetime.timedelta(days=100), end_date)
        self.logger.info(f"Fetched {len(spy_candles)} candles for SPY")
        
        # Fetch historical data for all symbols
        stock_data = {}
        for symbol in self.stock_configs.keys():
            self.logger.info(f"Fetching data for {symbol}")
            try:
                candles = self.fetch_historical_data(symbol, start_date - datetime.timedelta(days=30), end_date)
                if candles:
                    stock_data[symbol] = candles
                    self.logger.info(f"Fetched {len(candles)} candles for {symbol}")
                else:
                    self.logger.warning(f"No data available for {symbol} in the specified date range")
            except Exception as e:
                self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
        
        crypto_data = {}
        for symbol in self.crypto_configs.keys():
            self.logger.info(f"Fetching data for {symbol}")
            try:
                candles = self.fetch_historical_data(symbol, start_date - datetime.timedelta(days=30), end_date, is_crypto=True)
                if candles:
                    crypto_data[symbol] = candles
                    self.logger.info(f"Fetched {len(candles)} candles for {symbol}")
                else:
                    self.logger.warning(f"No data available for {symbol} in the specified date range")
            except Exception as e:
                self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
        
        # Generate date range for backtest
        current_date = start_date
        trading_days = []
        
        while current_date <= end_date:
            # Skip weekends for simplicity
            if current_date.weekday() < 5:  # Monday to Friday
                trading_days.append(current_date)
            current_date += datetime.timedelta(days=1)
        
        # Run backtest for each trading day
        for date in trading_days:
            self.logger.info(f"Processing date: {date.date()}")
            
            # Update market state
            market_state = self.update_market_state(date, spy_candles)
            
            # Get current prices for all symbols
            current_prices = {}
            
            # Process stocks
            signals = []
            for symbol, candles in stock_data.items():
                # Get candles up to current date
                current_candles = [c for c in candles if c.timestamp.replace(tzinfo=None).date() < date.date()]
                
                if current_candles:
                    # Get current price
                    current_prices[symbol] = current_candles[-1].close
                    
                    # Generate signals
                    symbol_signals = self.generate_signals_for_symbol(symbol, current_candles, market_state)
                    signals.extend(symbol_signals)
            
            # Process cryptos
            for symbol, candles in crypto_data.items():
                # Get candles up to current date
                current_candles = [c for c in candles if c.timestamp.replace(tzinfo=None).date() < date.date()]
                
                if current_candles:
                    # Get current price
                    current_prices[symbol] = current_candles[-1].close
                    
                    # Generate signals
                    symbol_signals = self.generate_signals_for_symbol(symbol, current_candles, market_state, is_crypto=True)
                    signals.extend(symbol_signals)
            
            # Update existing positions
            self._update_positions(date, current_prices)
            
            # Process new signals
            self._process_signals(date, signals, current_prices)
            
            # Calculate equity for the day
            self.calculate_current_equity(date, current_prices)
        
        # Close any remaining positions at the end of the backtest
        final_trades = []
        for symbol, trade in self.current_positions.items():
            if symbol in current_prices:
                trade.exit_date = end_date
                trade.exit_price = current_prices[symbol]
                
                # Calculate P&L
                if trade.direction == "long":
                    trade.pnl = (trade.exit_price - trade.entry_price) * trade.position_size
                    trade.pnl_pct = (trade.exit_price - trade.entry_price) / trade.entry_price
                else:  # short
                    trade.pnl = (trade.entry_price - trade.exit_price) * trade.position_size
                    trade.pnl_pct = (trade.entry_price - trade.exit_price) / trade.entry_price
                
                trade.exit_reason = "backtest_end"
                final_trades.append(trade)
        
        # Add final trades to history
        self.trade_history.extend(final_trades)
        
        # Calculate performance metrics
        performance = self.calculate_performance()
        
        return performance
    
    def calculate_performance(self) -> Dict:
        """Calculate performance metrics for the backtest"""
        if not self.equity_curve:
            return {
                "initial_capital": self.initial_capital,
                "final_equity": self.initial_capital,
                "total_return": 0.0,
                "annualized_return": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_trades": 0
            }
        
        # Sort equity curve by date
        sorted_dates = sorted(self.equity_curve.keys())
        equity_values = [self.equity_curve[date] for date in sorted_dates]
        
        # Calculate returns
        initial_capital = self.initial_capital
        final_equity = equity_values[-1]
        total_return = (final_equity - initial_capital) / initial_capital
        
        # Calculate annualized return
        days = (sorted_dates[-1] - sorted_dates[0]).days
        if days > 0:
            annualized_return = ((1 + total_return) ** (365 / days)) - 1
        else:
            annualized_return = 0.0
        
        # Calculate maximum drawdown
        max_drawdown = 0.0
        peak = equity_values[0]
        
        for equity in equity_values:
            if equity > peak:
                peak = equity
            
            drawdown = (peak - equity) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate daily returns for Sharpe ratio
        daily_returns = []
        for i in range(1, len(equity_values)):
            daily_return = (equity_values[i] - equity_values[i-1]) / equity_values[i-1]
            daily_returns.append(daily_return)
        
        # Calculate Sharpe ratio
        if daily_returns and len(daily_returns) > 1:
            avg_return = np.mean(daily_returns)
            std_return = np.std(daily_returns)
            
            if std_return > 0:
                sharpe_ratio = (avg_return / std_return) * (252 ** 0.5)  # Annualized
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0
        
        # Calculate win rate and profit factor
        winning_trades = [t for t in self.trade_history if t.pnl > 0]
        losing_trades = [t for t in self.trade_history if t.pnl <= 0]
        
        total_trades = len(self.trade_history)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
        
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            "initial_capital": initial_capital,
            "final_equity": final_equity,
            "total_return": total_return * 100,  # Convert to percentage
            "annualized_return": annualized_return * 100,  # Convert to percentage
            "max_drawdown": max_drawdown * 100,  # Convert to percentage
            "sharpe_ratio": sharpe_ratio,
            "win_rate": win_rate * 100,  # Convert to percentage
            "profit_factor": profit_factor,
            "total_trades": total_trades
        }
    
    def plot_equity_curve(self, save_path: str = None) -> None:
        """Plot the equity curve"""
        if not self.equity_curve:
            self.logger.warning("No equity curve data to plot")
            return
        
        # Sort equity curve by date
        sorted_dates = sorted(self.equity_curve.keys())
        equity_values = [self.equity_curve[date] for date in sorted_dates]
        
        # Create figure
        plt.figure(figsize=(12, 6))
        plt.plot(sorted_dates, equity_values, label='Equity Curve')
        
        # Add horizontal line for initial capital
        plt.axhline(y=self.initial_capital, color='r', linestyle='--', label='Initial Capital')
        
        # Add labels and title
        plt.xlabel('Date')
        plt.ylabel('Equity ($)')
        plt.title('MeanReversion Strategy Equity Curve')
        plt.legend()
        plt.grid(True)
        
        # Format x-axis dates
        plt.gcf().autofmt_xdate()
        
        # Save or show
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Equity curve saved to {save_path}")
        else:
            plt.show()
    
    def save_results(self, performance: Dict, save_dir: str = "backtest_results") -> str:
        """Save backtest results to JSON file"""
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate timestamp for filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{save_dir}/mean_reversion_results_{timestamp}.json"
        
        # Convert trade history to serializable format
        trade_data = []
        for trade in self.trade_history:
            trade_dict = {
                "symbol": trade.symbol,
                "entry_date": trade.entry_date.isoformat(),
                "entry_price": trade.entry_price,
                "direction": trade.direction,
                "position_size": trade.position_size,
                "stop_loss": trade.stop_loss,
                "take_profit": trade.take_profit,
                "is_crypto": trade.is_crypto,
                "strategy_name": trade.strategy_name
            }
            
            if trade.exit_date:
                trade_dict["exit_date"] = trade.exit_date.isoformat()
                trade_dict["exit_price"] = trade.exit_price
                trade_dict["pnl"] = trade.pnl
                trade_dict["pnl_pct"] = trade.pnl_pct
                trade_dict["exit_reason"] = trade.exit_reason
            
            trade_data.append(trade_dict)
        
        # Convert equity curve to serializable format
        equity_data = {date.isoformat(): value for date, value in self.equity_curve.items()}
        
        # Create results dictionary
        results = {
            "performance": performance,
            "trades": trade_data,
            "equity_curve": equity_data,
            "config_file": self.config_file,
            "backtest_date": datetime.datetime.now().isoformat()
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)
        
        self.logger.info(f"Results saved to {filename}")
        return filename

def main():
    """Main function to run the backtest"""
    parser = argparse.ArgumentParser(description='Run MeanReversion strategy backtest')
    parser.add_argument('--quarter', type=int, choices=[1, 2, 3, 4], help='Quarter to backtest (1-4)')
    parser.add_argument('--year', type=int, default=2023, help='Year to backtest (default: 2023)')
    parser.add_argument('--config', type=str, default='configuration_mean_reversion_2024.yaml', 
                        help='Configuration file to use (default: configuration_mean_reversion_2024.yaml)')
    args = parser.parse_args()
    
    # Determine date range based on quarter and year
    if args.quarter:
        year = args.year
        if args.quarter == 1:
            start_date = datetime.datetime(year, 1, 1)
            end_date = datetime.datetime(year, 3, 31)
        elif args.quarter == 2:
            start_date = datetime.datetime(year, 4, 1)
            end_date = datetime.datetime(year, 6, 30)
        elif args.quarter == 3:
            start_date = datetime.datetime(year, 7, 1)
            end_date = datetime.datetime(year, 9, 30)
        elif args.quarter == 4:
            start_date = datetime.datetime(year, 10, 1)
            end_date = datetime.datetime(year, 12, 31)
    else:
        # Default to full year if no quarter specified
        start_date = datetime.datetime(args.year, 1, 1)
        end_date = datetime.datetime(args.year, 12, 31)
    
    # Run backtest
    backtest = MeanReversionBacktest(config_file=args.config)
    results = backtest.run_backtest(start_date, end_date)
    
    # Print results
    print("\n=== MeanReversion Strategy Backtest Results ===")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Initial Capital: ${results['initial_capital']:.2f}")
    print(f"Final Equity: ${results['final_equity']:.2f}")
    print(f"Total Return: {results['total_return']:.2f}%")
    print(f"Annualized Return: {results['annualized_return']:.2f}%")
    print(f"Maximum Drawdown: {results['max_drawdown']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Win Rate: {results['win_rate']:.2f}%")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print(f"Total Trades: {results['total_trades']}")
    print(f"\nResults saved to: {backtest.save_results(results)}")
    print(f"Equity curve saved to: {backtest.plot_equity_curve()}")

if __name__ == "__main__":
    main()
