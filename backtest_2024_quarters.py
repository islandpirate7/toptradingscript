#!/usr/bin/env python
# -*- coding: utf-8 -*-

import alpaca_trade_api as tradeapi
import datetime
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import yaml
import argparse
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import time
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('MultiStrategyBacktest')

# Data classes for storing market data and signals
@dataclass
class CandleData:
    timestamp: datetime.datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

@dataclass
class Signal:
    symbol: str
    timestamp: datetime.datetime
    direction: str  # 'long' or 'short'
    strength: str   # 'strong', 'moderate', 'weak'
    strategy_name: str
    entry_price: float
    stop_loss: float
    take_profit: float
    score: float = 0.0
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
    regime: str = "neutral"  # 'bullish', 'bearish', 'neutral', 'volatile'
    trend_strength: str = "neutral"  # 'strong', 'weak', 'neutral'
    volatility: str = "normal"  # 'high', 'normal', 'low'

@dataclass
class StockConfig:
    symbol: str
    max_position_size: float
    min_position_size: float
    max_risk_per_trade_pct: float
    min_volume: float

@dataclass
class CryptoConfig:
    symbol: str
    max_position_size: float
    min_position_size: float
    max_risk_per_trade_pct: float
    min_volume: float

class MeanReversionStrategy:
    def __init__(self, config: Dict):
        self.name = "MeanReversionStrategy"
        self.config = config
        
    def _calculate_atr(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(highs) < period + 1:
            return 0.01  # Default value if not enough data
        
        tr_values = []
        for i in range(1, len(highs)):
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i-1])
            tr3 = abs(lows[i] - closes[i-1])
            tr_values.append(max(tr1, tr2, tr3))
        
        return sum(tr_values[-period:]) / period
    
    def generate_signals(self, symbol: str, candles: List[CandleData], 
                         market_state: MarketState, is_crypto: bool = False) -> List[Signal]:
        """Generate mean reversion signals based on Bollinger Bands and RSI"""
        if len(candles) < 30:  # Need at least 30 candles for indicators
            return []
        
        # Get configuration for the symbol
        symbol_config = {}
        if is_crypto:
            crypto_configs = self.config.get('crypto', [])
            for config in crypto_configs:
                if config.get('symbol') == symbol:
                    symbol_config = config
                    break
        else:
            stock_configs = self.config.get('stocks', [])
            for config in stock_configs:
                if config.get('symbol') == symbol:
                    symbol_config = config
                    break
        
        # Get mean reversion parameters with defaults
        mean_reversion_params = symbol_config.get('mean_reversion_params', {})
        bb_period = mean_reversion_params.get('bb_period', 20)
        bb_std_dev = mean_reversion_params.get('bb_std_dev', 1.9)
        rsi_period = mean_reversion_params.get('rsi_period', 14)
        rsi_overbought = mean_reversion_params.get('rsi_overbought', 65)
        rsi_oversold = mean_reversion_params.get('rsi_oversold', 35)
        min_reversal_candles = mean_reversion_params.get('min_reversal_candles', 1)
        
        # Extract prices
        closes = [candle.close for candle in candles]
        highs = [candle.high for candle in candles]
        lows = [candle.low for candle in candles]
        
        # Calculate Bollinger Bands
        bb_middle = np.mean(closes[-bb_period:])
        bb_std = np.std(closes[-bb_period:])
        bb_upper = bb_middle + bb_std_dev * bb_std
        bb_lower = bb_middle - bb_std_dev * bb_std
        
        # Calculate RSI
        deltas = np.diff(closes)
        seed = deltas[:rsi_period+1]
        up = seed[seed >= 0].sum()/rsi_period
        down = -seed[seed < 0].sum()/rsi_period
        rs = up/down if down != 0 else float('inf')
        rsi = 100 - (100/(1+rs))
        
        # Calculate ATR for stop loss and take profit
        atr = self._calculate_atr(highs, lows, closes, period=14)
        
        # Current price
        current_price = closes[-1]
        current_candle = candles[-1]
        
        signals = []
        
        # Check for oversold condition (long signal)
        if current_price < bb_lower and rsi < rsi_oversold:
            # Check for price reversal (price moving back up)
            if min(closes[-min_reversal_candles-1:-1]) < closes[-1]:
                # Calculate signal strength based on distance from BB and RSI extremity
                bb_distance = (bb_lower - current_price) / bb_lower
                rsi_distance = (rsi_oversold - rsi) / rsi_oversold
                signal_score = 0.3 + 0.3 * bb_distance + 0.4 * rsi_distance
                
                # Determine signal strength
                if signal_score > 0.7:
                    strength = "strong"
                elif signal_score > 0.4:
                    strength = "moderate"
                else:
                    strength = "weak"
                
                # Calculate stop loss and take profit
                stop_loss = current_price * (1 - 1.8 * atr / current_price)
                take_profit = current_price * (1 + 3.0 * atr / current_price)
                
                # Create signal
                signal = Signal(
                    symbol=symbol,
                    timestamp=current_candle.timestamp,
                    direction="long",
                    strength=strength,
                    strategy_name=self.name,
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    score=signal_score,
                    is_crypto=is_crypto
                )
                signals.append(signal)
        
        # Check for overbought condition (short signal)
        if current_price > bb_upper and rsi > rsi_overbought:
            # Check for price reversal (price moving back down)
            if max(closes[-min_reversal_candles-1:-1]) > closes[-1]:
                # Calculate signal strength based on distance from BB and RSI extremity
                bb_distance = (current_price - bb_upper) / bb_upper
                rsi_distance = (rsi - rsi_overbought) / (100 - rsi_overbought)
                signal_score = 0.3 + 0.3 * bb_distance + 0.4 * rsi_distance
                
                # Determine signal strength
                if signal_score > 0.7:
                    strength = "strong"
                elif signal_score > 0.4:
                    strength = "moderate"
                else:
                    strength = "weak"
                
                # Calculate stop loss and take profit
                stop_loss = current_price * (1 + 1.8 * atr / current_price)
                take_profit = current_price * (1 - 3.0 * atr / current_price)
                
                # Create signal
                signal = Signal(
                    symbol=symbol,
                    timestamp=current_candle.timestamp,
                    direction="short",
                    strength=strength,
                    strategy_name=self.name,
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    score=signal_score,
                    is_crypto=is_crypto
                )
                signals.append(signal)
        
        return signals
    
class MultiStrategyBacktest:
    def __init__(self, config_file: str, start_date: datetime.datetime, end_date: datetime.datetime,
                 alpaca_credentials_file: str = 'alpaca_credentials.json', results_dir: str = 'backtest_results'):
        """Initialize the MultiStrategy backtest"""
        self.config_file = config_file
        self.start_date = start_date
        self.end_date = end_date
        self.alpaca_credentials_file = alpaca_credentials_file
        self.results_dir = results_dir
        
        # Load configuration
        self.config = self._load_config(config_file)
        logger.info(f"Loaded configuration from {config_file}")
        
        # System version
        self.version = "2.5"
        logger.info(f"MultiStrategy Trading System v{self.version}")
        
        # Initialize Alpaca API
        self.api = self._init_alpaca_api()
        logger.info("Initialized Alpaca API with paper trading account")
        
        # Initialize strategies
        self.strategies = [MeanReversionStrategy(self.config)]
        logger.info(f"Initialized {len(self.strategies)} trading strategies")
        
        # Initialize stock and crypto configurations
        self.stock_configs = self._init_stock_configs()
        logger.info(f"Initialized {len(self.stock_configs)} stock configurations")
        
        self.crypto_configs = self._init_crypto_configs()
        logger.info(f"Initialized {len(self.crypto_configs)} crypto configurations")
        
        # Initialize market state
        self.market_state = MarketState()
        
        # Initialize backtest state
        self.equity_curve = {self.start_date.date().isoformat(): self.config['initial_capital']}
        self.daily_returns = []
        self.trade_history = []
        self.current_positions = {}
        self.signals_generated = []
        self.cash = self.config['initial_capital']
        
        # Create results directory if it doesn't exist
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
    
    def _load_config(self, config_file: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise
    
    def _init_alpaca_api(self) -> tradeapi.REST:
        """Initialize Alpaca API client"""
        try:
            # Load credentials from JSON file
            with open(self.alpaca_credentials_file, 'r') as f:
                credentials = json.load(f)
            
            # Use paper trading credentials
            api = tradeapi.REST(
                key_id=credentials['paper']['api_key'],
                secret_key=credentials['paper']['api_secret'],
                base_url=credentials['paper']['base_url']
            )
            
            return api
        except Exception as e:
            logger.error(f"Error initializing Alpaca API: {e}")
            raise
    
    def _init_stock_configs(self) -> Dict[str, StockConfig]:
        """Initialize stock configurations"""
        stock_configs = {}
        
        # Get stock configurations from config
        stocks_list = self.config.get('stocks', [])
        
        for stock_config in stocks_list:
            symbol = stock_config.get('symbol')
            if not symbol:
                continue
                
            # Create StockConfig object
            config = StockConfig(
                symbol=symbol,
                max_position_size=stock_config.get('max_position_size', 100),
                min_position_size=stock_config.get('min_position_size', 1),
                max_risk_per_trade_pct=stock_config.get('max_risk_per_trade_pct', 0.5) / 100,  # Convert to decimal
                min_volume=stock_config.get('min_volume', 1000)
            )
            
            stock_configs[symbol] = config
            
        logger.info(f"Initialized {len(stock_configs)} stock configurations")
        return stock_configs
    
    def _init_crypto_configs(self) -> Dict[str, CryptoConfig]:
        """Initialize crypto configurations"""
        crypto_configs = {}
        
        # Get crypto configurations from config
        crypto_list = self.config.get('crypto', [])
        
        for crypto_config in crypto_list:
            symbol = crypto_config.get('symbol')
            if not symbol:
                continue
                
            # Create CryptoConfig object
            config = CryptoConfig(
                symbol=symbol,
                max_position_size=crypto_config.get('max_position_size', 1.0),
                min_position_size=crypto_config.get('min_position_size', 0.01),
                max_risk_per_trade_pct=crypto_config.get('max_risk_per_trade_pct', 0.5) / 100,  # Convert to decimal
                min_volume=crypto_config.get('min_volume', 1000)
            )
            
            # Add USD suffix for Alpaca API
            alpaca_symbol = f"{symbol}USD"
            crypto_configs[alpaca_symbol] = config
            
        logger.info(f"Initialized {len(crypto_configs)} crypto configurations")
        return crypto_configs
    
    def fetch_historical_data(self, symbols: List[str], is_crypto: bool = False) -> Dict[str, List[CandleData]]:
        """Fetch historical data for the given symbols"""
        if not symbols:
            return {}
        
        logger.info(f"Fetching historical data for {len(symbols)} {'crypto' if is_crypto else 'stock'} symbols")
        
        # Calculate start date with lookback period
        lookback_days = 60  # Need enough data for indicators
        start_date = self.start_date - datetime.timedelta(days=lookback_days)
        
        # Initialize result dictionary
        result = {}
        
        # Fetch data for each symbol
        for symbol in symbols:
            try:
                logger.info(f"Fetching data for {symbol}")
                
                # Format symbol for crypto
                api_symbol = symbol
                if is_crypto:
                    # Format for Alpaca API: BTC -> BTCUSD
                    if not symbol.endswith('USD'):
                        api_symbol = f"{symbol}USD"
                
                # Fetch data from Alpaca
                bars = self.api.get_bars(
                    api_symbol,
                    tradeapi.TimeFrame.Day,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=self.end_date.strftime('%Y-%m-%d'),
                    adjustment='raw'
                ).df
                
                if bars.empty:
                    logger.warning(f"No data found for {symbol}")
                    continue
                
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
                
                logger.info(f"Fetched {len(candles)} candles for {symbol}")
                result[symbol] = candles
                
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
        
        return result
    
    def update_market_state(self, date: datetime.datetime) -> None:
        """Update market state based on current market conditions"""
        # Use SPY as a proxy for market conditions
        spy_candles = self.stock_data.get('SPY', [])
        
        if not spy_candles:
            logger.warning("SPY data not available for market state update")
            return
        
        # Compare dates properly
        current_candles = [c for c in spy_candles if c.timestamp.replace(tzinfo=None).date() == date.date()]
        
        if len(current_candles) < 20:
            logger.warning("Not enough SPY data for market state update")
            return
        
        # Calculate moving averages
        closes = [c.close for c in current_candles]
        ma20 = sum(closes[-20:]) / 20
        ma50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else ma20
        
        # Calculate volatility (standard deviation of returns)
        returns = [closes[i] / closes[i-1] - 1 for i in range(1, len(closes))]
        volatility = np.std(returns[-20:]) * np.sqrt(252)  # Annualized
        
        # Determine market regime
        if ma20 > ma50 and closes[-1] > ma20:
            self.market_state.regime = "bullish"
        elif ma20 < ma50 and closes[-1] < ma20:
            self.market_state.regime = "bearish"
        else:
            self.market_state.regime = "neutral"
        
        # Determine volatility state
        if volatility > 0.25:
            self.market_state.volatility = "high"
        elif volatility < 0.10:
            self.market_state.volatility = "low"
        else:
            self.market_state.volatility = "normal"
        
        logger.info(f"Market state updated: {self.market_state.regime}, volatility: {self.market_state.volatility}")
    
    def generate_signals_for_symbol(self, symbol: str, candles: List[CandleData], 
                                   is_crypto: bool = False) -> List[Signal]:
        """Generate trading signals for a symbol using all strategies"""
        if len(candles) < 30:  # Need at least 30 candles for indicators
            logger.warning(f"Not enough candles for {symbol}, skipping signal generation")
            return []
        
        # Get configuration for the symbol
        if is_crypto:
            config = self.crypto_configs.get(symbol)
        else:
            config = self.stock_configs.get(symbol)
        
        if not config:
            logger.warning(f"No configuration found for {symbol}, skipping signal generation")
            return []
        
        # Generate signals from all strategies
        all_signals = []
        for strategy in self.strategies:
            signals = strategy.generate_signals(symbol, candles, self.market_state, is_crypto)
            all_signals.extend(signals)
        
        logger.info(f"Generated {len(all_signals)} signals for {symbol}")
        return all_signals
    
    def _process_signals(self, date: datetime.datetime, signals: List[Signal], current_prices: Dict[str, float]) -> None:
        """Process trading signals and create trades"""
        if not signals:
            return
            
        logger.info(f"Processing {len(signals)} signals for {date.date()}")
        
        # Sort signals by score (highest first)
        signals.sort(key=lambda x: x.score, reverse=True)
        
        # Get current equity
        current_equity = self.cash
        for symbol, trade in self.current_positions.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                position_value = trade.position_size * current_price
                current_equity += position_value
        
        # Get position sizing config
        position_sizing = self.config.get('position_sizing_config', {})
        base_risk_per_trade = position_sizing.get('base_risk_per_trade', 0.01)  # Default to 1% risk per trade
        max_position_size_pct = position_sizing.get('max_position_size', 0.1)   # Default to 10% max position
        min_position_size_pct = position_sizing.get('min_position_size', 0.005) # Default to 0.5% min position
        
        # Process each signal
        for signal in signals:
            # Skip if we already have a position for this symbol
            if signal.symbol in self.current_positions:
                continue
                
            # Skip if we've reached the maximum number of positions
            if len(self.current_positions) >= self.config.get('max_open_positions', 10):
                logger.info(f"Maximum positions reached ({len(self.current_positions)}), skipping signal for {signal.symbol}")
                break
                
            # Get current price for the symbol
            if signal.symbol not in current_prices:
                logger.warning(f"No current price available for {signal.symbol}, skipping signal")
                continue
                
            current_price = current_prices[signal.symbol]
            
            # Get configuration for the symbol
            config = None
            if signal.is_crypto:
                if signal.symbol in self.crypto_configs:
                    config = self.crypto_configs[signal.symbol]
            else:
                if signal.symbol in self.stock_configs:
                    config = self.stock_configs[signal.symbol]
                    
            if not config:
                logger.warning(f"No configuration found for {signal.symbol}, skipping signal")
                continue
            
            # Calculate stop loss and take profit levels
            if signal.direction == "long":
                stop_loss = signal.stop_loss if signal.stop_loss else current_price * (1 - config.max_risk_per_trade_pct)
                take_profit = signal.take_profit if signal.take_profit else current_price * (1 + config.max_risk_per_trade_pct * 2)  # 2:1 reward-risk ratio
            else:  # short
                stop_loss = signal.stop_loss if signal.stop_loss else current_price * (1 + config.max_risk_per_trade_pct)
                take_profit = signal.take_profit if signal.take_profit else current_price * (1 - config.max_risk_per_trade_pct * 2)  # 2:1 reward-risk ratio
            
            # Calculate position size based on risk
            risk_per_share = abs(current_price - stop_loss)
            risk_per_trade = current_equity * base_risk_per_trade  # Risk 1% of equity per trade by default
            
            # Adjust position size based on signal strength
            if signal.strength == "strong":
                risk_per_trade *= 1.5
            elif signal.strength == "weak":
                risk_per_trade *= 0.5
                
            # Calculate position size
            position_size = risk_per_trade / risk_per_share if risk_per_share > 0 else 0
            
            # Apply min/max position size constraints
            max_position_size = current_equity * max_position_size_pct
            min_position_size = current_equity * min_position_size_pct
            
            position_value = position_size * current_price
            if position_value > max_position_size:
                position_size = max_position_size / current_price
            elif position_value < min_position_size:
                position_size = min_position_size / current_price
                
            # Round position size to appropriate precision
            if signal.is_crypto:
                position_size = round(position_size, 4)  # 4 decimal places for crypto
            else:
                position_size = int(position_size)  # Whole shares for stocks
                
            # Ensure we have enough cash for the trade
            if position_size * current_price > self.cash:
                logger.warning(f"Not enough cash for {signal.symbol} position, adjusting size")
                position_size = math.floor(self.cash / current_price) if not signal.is_crypto else self.cash / current_price
                
            if position_size <= 0:
                logger.warning(f"Position size for {signal.symbol} is zero or negative, skipping trade")
                continue
                
            # Create trade
            trade = Trade(
                symbol=signal.symbol,
                direction=signal.direction,
                entry_date=date,
                entry_price=current_price,
                position_size=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                is_crypto=signal.is_crypto,
                strategy_name=signal.strategy_name
            )
            
            # Add to current positions
            self.current_positions[signal.symbol] = trade
            
            # Update cash
            self.cash -= position_size * current_price
            
            logger.info(f"Created {signal.direction} trade for {signal.symbol} at {current_price:.2f}, size: {position_size}, stop: {stop_loss:.2f}, target: {take_profit:.2f}")
    
    def _update_positions(self, date: datetime.datetime, current_prices: Dict[str, float]) -> None:
        """Update positions based on current prices"""
        if not self.current_positions:
            return
        
        logger.info(f"Updating {len(self.current_positions)} positions for {date.date()}")
        
        # Track positions to remove
        positions_to_remove = []
        
        # Update each position
        for symbol, trade in self.current_positions.items():
            # Skip if we don't have current price
            if symbol not in current_prices:
                logger.warning(f"No current price for {symbol}, skipping position update")
                continue
            
            current_price = current_prices[symbol]
            
            # Check for stop loss
            if trade.direction == "long" and current_price <= trade.stop_loss:
                # Long position hit stop loss
                trade.exit_date = date
                trade.exit_price = trade.stop_loss
                trade.exit_reason = "stop_loss"
                trade.pnl = (trade.exit_price - trade.entry_price) * trade.position_size
                trade.pnl_pct = (trade.exit_price / trade.entry_price - 1) * 100
                
                # Update cash
                self.cash += trade.position_size * trade.exit_price
                
                # Add to trade history
                self.trade_history.append(trade)
                
                # Mark for removal
                positions_to_remove.append(symbol)
                
                logger.info(f"Long position for {symbol} hit stop loss: entry: {trade.entry_price}, "
                           f"exit: {trade.exit_price}, PnL: {trade.pnl:.2f} ({trade.pnl_pct:.2f}%)")
            
            elif trade.direction == "short" and current_price >= trade.stop_loss:
                # Short position hit stop loss
                trade.exit_date = date
                trade.exit_price = trade.stop_loss
                trade.exit_reason = "stop_loss"
                trade.pnl = (trade.entry_price - trade.exit_price) * trade.position_size
                trade.pnl_pct = (trade.entry_price / trade.exit_price - 1) * 100
                
                # Update cash
                self.cash += trade.position_size * (2 * trade.entry_price - trade.exit_price)
                
                # Add to trade history
                self.trade_history.append(trade)
                
                # Mark for removal
                positions_to_remove.append(symbol)
                
                logger.info(f"Short position for {symbol} hit stop loss: entry: {trade.entry_price}, "
                           f"exit: {trade.exit_price}, PnL: {trade.pnl:.2f} ({trade.pnl_pct:.2f}%)")
            
            # Check for take profit
            elif trade.direction == "long" and current_price >= trade.take_profit:
                # Long position hit take profit
                trade.exit_date = date
                trade.exit_price = trade.take_profit
                trade.exit_reason = "take_profit"
                trade.pnl = (trade.exit_price - trade.entry_price) * trade.position_size
                trade.pnl_pct = (trade.exit_price / trade.entry_price - 1) * 100
                
                # Update cash
                self.cash += trade.position_size * trade.exit_price
                
                # Add to trade history
                self.trade_history.append(trade)
                
                # Mark for removal
                positions_to_remove.append(symbol)
                
                logger.info(f"Long position for {symbol} hit take profit: entry: {trade.entry_price}, "
                           f"exit: {trade.exit_price}, PnL: {trade.pnl:.2f} ({trade.pnl_pct:.2f}%)")
            
            elif trade.direction == "short" and current_price <= trade.take_profit:
                # Short position hit take profit
                trade.exit_date = date
                trade.exit_price = trade.take_profit
                trade.exit_reason = "take_profit"
                trade.pnl = (trade.entry_price - trade.exit_price) * trade.position_size
                trade.pnl_pct = (trade.entry_price / trade.exit_price - 1) * 100
                
                # Update cash
                self.cash += trade.position_size * (2 * trade.entry_price - trade.exit_price)
                
                # Add to trade history
                self.trade_history.append(trade)
                
                # Mark for removal
                positions_to_remove.append(symbol)
                
                logger.info(f"Short position for {symbol} hit take profit: entry: {trade.entry_price}, "
                           f"exit: {trade.exit_price}, PnL: {trade.pnl:.2f} ({trade.pnl_pct:.2f}%)")
            
            # Check for time-based exit (5 days)
            elif (date - trade.entry_date.replace(tzinfo=None)).days >= 5:
                # Time-based exit
                trade.exit_date = date
                trade.exit_price = current_price
                trade.exit_reason = "time_exit"
                
                if trade.direction == "long":
                    trade.pnl = (trade.exit_price - trade.entry_price) * trade.position_size
                    trade.pnl_pct = (trade.exit_price / trade.entry_price - 1) * 100
                    # Update cash
                    self.cash += trade.position_size * trade.exit_price
                else:  # short
                    trade.pnl = (trade.entry_price - trade.exit_price) * trade.position_size
                    trade.pnl_pct = (trade.entry_price / trade.exit_price - 1) * 100
                    # Update cash
                    self.cash += trade.position_size * (2 * trade.entry_price - trade.exit_price)
                
                # Add to trade history
                self.trade_history.append(trade)
                
                # Mark for removal
                positions_to_remove.append(symbol)
                
                logger.info(f"Position for {symbol} exited due to time: entry: {trade.entry_price}, "
                           f"exit: {trade.exit_price}, PnL: {trade.pnl:.2f} ({trade.pnl_pct:.2f}%)")
        
        # Remove closed positions
        for symbol in positions_to_remove:
            del self.current_positions[symbol]
    
    def run_backtest(self) -> Dict:
        """Run the backtest and return results"""
        logger.info(f"Starting backtest from {self.start_date.date()} to {self.end_date.date()}")
        
        # Ensure start and end dates are timezone-naive
        if self.start_date.tzinfo is not None:
            self.start_date = self.start_date.replace(tzinfo=None)
        if self.end_date.tzinfo is not None:
            self.end_date = self.end_date.replace(tzinfo=None)
            
        # Generate date range for backtest
        current_date = self.start_date
        date_range = []
        while current_date <= self.end_date:
            # Skip weekends
            if current_date.weekday() < 5:  # Monday to Friday
                date_range.append(current_date)
            current_date += datetime.timedelta(days=1)
        
        # Initialize equity curve with initial capital
        self.equity_curve = {self.start_date.date().isoformat(): self.config['initial_capital']}
        self.cash = self.config['initial_capital']
        current_equity = self.config['initial_capital']
        
        # Track daily returns
        daily_returns = []
        
        # Progress tracking
        total_days = len(date_range)
        progress = 0
        
        # Run backtest for each day
        for i, date in enumerate(date_range):
            try:
                # Update progress
                progress = (i + 1) / total_days * 100
                if (i + 1) % 5 == 0 or i == 0 or i == total_days - 1:
                    logger.info(f"Backtest progress: {progress:.1f}% ({i+1}/{total_days} days)")
                
                # Update market state
                self.update_market_state(date)
                
                # Get current prices for all symbols
                current_prices = {}
                
                # Stock prices
                for symbol, candles in self.stock_data.items():
                    # Find the candle for the current date
                    matching_candles = [c for c in candles if c.timestamp.replace(tzinfo=None).date() == date.date()]
                    if matching_candles:
                        current_prices[symbol] = matching_candles[0].close
                
                # Crypto prices
                for symbol, candles in self.crypto_data.items():
                    # Find the candle for the current date
                    matching_candles = [c for c in candles if c.timestamp.replace(tzinfo=None).date() == date.date()]
                    if matching_candles:
                        current_prices[symbol] = matching_candles[0].close
                
                # Update existing positions
                self._update_positions(date, current_prices)
                
                # Generate signals for stocks
                stock_signals = []
                for symbol, candles in self.stock_data.items():
                    if symbol == 'SPY':  # Skip SPY as it's used for market state only
                        continue
                    
                    # Filter candles up to current date
                    current_candles = [c for c in candles if c.timestamp.replace(tzinfo=None) <= date]
                    
                    # Generate signals
                    signals = self.generate_signals_for_symbol(symbol, current_candles, is_crypto=False)
                    stock_signals.extend(signals)
                
                # Generate signals for crypto
                crypto_signals = []
                for symbol, candles in self.crypto_data.items():
                    # Filter candles up to current date
                    current_candles = [c for c in candles if c.timestamp.replace(tzinfo=None) <= date]
                    
                    # Generate signals
                    signals = self.generate_signals_for_symbol(symbol, current_candles, is_crypto=True)
                    crypto_signals.extend(signals)
                
                # Combine signals
                all_signals = stock_signals + crypto_signals
                
                # Process signals
                self._process_signals(date, all_signals, current_prices)
                
                # Calculate current equity
                position_value = 0
                for symbol, trade in self.current_positions.items():
                    if symbol in current_prices:
                        current_price = current_prices[symbol]
                        if trade.direction == "long":
                            position_value += trade.position_size * current_price
                        else:  # short
                            position_value += trade.position_size * (2 * trade.entry_price - current_price)
                
                current_equity = self.cash + position_value
                
                # Update equity curve
                self.equity_curve[date.date().isoformat()] = current_equity
                
                # Calculate daily return
                prev_date = date - datetime.timedelta(days=1)
                while prev_date.weekday() >= 5 and prev_date >= self.start_date:  # Skip weekends
                    prev_date -= datetime.timedelta(days=1)
                
                prev_date_str = prev_date.date().isoformat()
                if prev_date_str in self.equity_curve:
                    prev_equity = self.equity_curve[prev_date_str]
                    daily_return = current_equity / prev_equity - 1
                    daily_returns.append(daily_return)
            
            except Exception as e:
                logger.error(f"Error on backtest day {date.date()}: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Calculate performance metrics
        total_return = (current_equity / self.config['initial_capital'] - 1) * 100
        
        # Annualized return
        days = (self.end_date - self.start_date).days
        annualized_return = ((1 + total_return / 100) ** (365 / days) - 1) * 100
        
        # Max drawdown
        max_drawdown = 0
        peak = self.config['initial_capital']
        for equity in self.equity_curve.values():
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        # Sharpe ratio
        risk_free_rate = 0.02  # 2% annual risk-free rate
        daily_risk_free = (1 + risk_free_rate) ** (1/252) - 1
        excess_returns = [r - daily_risk_free for r in daily_returns]
        sharpe_ratio = (np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)) if len(excess_returns) > 0 and np.std(excess_returns) > 0 else 0
        
        # Win rate
        winning_trades = [t for t in self.trade_history if t.pnl > 0]
        win_rate = len(winning_trades) / len(self.trade_history) * 100 if self.trade_history else 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in self.trade_history if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trade_history if t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
        
        # Create results dictionary
        results = {
            "initial_capital": self.config['initial_capital'],
            "final_equity": current_equity,
            "total_return": total_return,
            "annualized_return": annualized_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": len(self.trade_history),
            "equity_curve": self.equity_curve
        }
        
        return results

    def _calculate_performance_metrics(self) -> Dict:
        """Calculate performance metrics from backtest results"""
        logger.info("Calculating performance metrics")
        
        # Basic metrics
        initial_capital = self.config['initial_capital']
        final_equity = self.equity_curve[-1]
        total_return = (final_equity / initial_capital - 1) * 100
        
        # Calculate annualized return
        days = (self.end_date - self.start_date).days
        annualized_return = ((1 + total_return / 100) ** (365 / days) - 1) * 100
        
        # Calculate max drawdown
        max_equity = initial_capital
        max_drawdown = 0
        
        for equity in self.equity_curve:
            max_equity = max(max_equity, equity)
            drawdown = (max_equity - equity) / max_equity * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate Sharpe ratio
        if len(self.daily_returns) > 0:
            avg_daily_return = np.mean(self.daily_returns)
            std_daily_return = np.std(self.daily_returns)
            risk_free_rate = 0.02 / 252  # Assuming 2% annual risk-free rate
            
            if std_daily_return > 0:
                sharpe_ratio = np.sqrt(252) * (avg_daily_return - risk_free_rate) / std_daily_return
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        # Calculate win rate and profit factor
        if len(self.trade_history) > 0:
            winning_trades = [t for t in self.trade_history if t.pnl > 0]
            losing_trades = [t for t in self.trade_history if t.pnl <= 0]
            
            win_rate = len(winning_trades) / len(self.trade_history) * 100
            
            total_profit = sum(t.pnl for t in winning_trades)
            total_loss = abs(sum(t.pnl for t in losing_trades))
            
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        else:
            win_rate = 0
            profit_factor = 0
        
        # Compile results
        results = {
            'initial_capital': initial_capital,
            'final_equity': final_equity,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(self.trade_history),
            'equity_curve': self.equity_curve,
            'trade_history': [vars(t) for t in self.trade_history]
        }
        
        # Log summary
        logger.info(f"Initial Capital: ${initial_capital:.2f}")
        logger.info(f"Final Equity: ${final_equity:.2f}")
        logger.info(f"Total Return: {total_return:.2f}%")
        logger.info(f"Annualized Return: {annualized_return:.2f}%")
        logger.info(f"Max Drawdown: {max_drawdown:.2f}%")
        logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        logger.info(f"Win Rate: {win_rate:.2f}%")
        logger.info(f"Profit Factor: {profit_factor:.2f}")
        logger.info(f"Total Trades: {len(self.trade_history)}")
        
        return results
    
    def _save_results(self, results: Dict) -> None:
        """Save backtest results to files"""
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        period = f"{self.start_date.strftime('%Y%m%d')}-{self.end_date.strftime('%Y%m%d')}"
        
        # Save metrics to JSON
        metrics_file = os.path.join(self.results_dir, f"metrics_v{self.version}_{period}_{timestamp}.json")
        with open(metrics_file, 'w') as f:
            # Create a copy of results without equity curve and trade history for more readable metrics file
            metrics = {k: v for k, v in results.items() if k not in ['equity_curve', 'trade_history']}
            json.dump(metrics, f, indent=4)
        
        logger.info(f"Saved metrics to {metrics_file}")
        
        # Save trades to CSV
        trades_file = os.path.join(self.results_dir, f"trades_v{self.version}_{period}_{timestamp}.csv")
        
        if self.trade_history:
            # Convert trade history to DataFrame
            trades_df = pd.DataFrame([vars(t) for t in self.trade_history])
            
            # Convert datetime objects to strings
            if 'entry_date' in trades_df.columns:
                trades_df['entry_date'] = trades_df['entry_date'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
            if 'exit_date' in trades_df.columns:
                trades_df['exit_date'] = trades_df['exit_date'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if x else None)
            
            # Save to CSV
            trades_df.to_csv(trades_file, index=False)
            
            logger.info(f"Saved {len(self.trade_history)} trades to {trades_file}")
        else:
            logger.warning("No trades to save")
        
        # Plot equity curve
        equity_curve_file = os.path.join(self.results_dir, f"equity_curve_v{self.version}_{period}_{timestamp}.png")
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.equity_curve.values())
        plt.title(f"Equity Curve - {self.start_date.date()} to {self.end_date.date()}")
        plt.xlabel("Trading Days")
        plt.ylabel("Equity ($)")
        plt.grid(True)
        plt.savefig(equity_curve_file)
        
        logger.info(f"Saved equity curve to {equity_curve_file}")

def run_quarter_backtest(config_file: str, year: int, quarter: int) -> Dict:
    """Run backtest for a specific quarter"""
    logger.info(f"Starting backtest for Q{quarter} {year}")
    
    # Define quarter date ranges
    quarter_ranges = {
        1: ((1, 1), (3, 31)),
        2: ((4, 1), (6, 30)),
        3: ((7, 1), (9, 30)),
        4: ((10, 1), (12, 31))
    }
    
    # Get date range for the quarter
    start_month, start_day = quarter_ranges[quarter][0]
    end_month, end_day = quarter_ranges[quarter][1]
    
    # Create datetime objects
    # Use the actual requested year
    actual_year = year  # Using the actual requested year
    start_date = datetime.datetime(actual_year, start_month, start_day)
    end_date = datetime.datetime(actual_year, end_month, end_day)
    
    logger.info(f"Running backtest for Q{quarter} {year} ({start_date.date()} to {end_date.date()})")
    
    # Create a temporary config file with the quarter-specific settings
    temp_config_file = f"temp_config_{year}_Q{quarter}.yaml"
    
    # Load the original config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config with quarter-specific settings
    config['backtest_start_date'] = start_date.strftime('%Y-%m-%d')
    config['backtest_end_date'] = end_date.strftime('%Y-%m-%d')
    
    # Save the temporary config
    with open(temp_config_file, 'w') as f:
        yaml.dump(config, f)
    
    try:
        # Run backtest
        backtest = MultiStrategyBacktest(
            config_file=temp_config_file,
            start_date=start_date,
            end_date=end_date
        )
        
        # Set version for file naming
        backtest.version = f"{year}_Q{quarter}"
        
        # Fetch historical data for stocks
        stock_symbols = list(backtest.stock_configs.keys())
        # Add SPY for market state
        if 'SPY' not in stock_symbols:
            stock_symbols.append('SPY')
        backtest.stock_data = backtest.fetch_historical_data(stock_symbols, is_crypto=False)
        
        # Fetch historical data for crypto
        crypto_symbols = list(backtest.crypto_configs.keys())
        backtest.crypto_data = backtest.fetch_historical_data(crypto_symbols, is_crypto=True)
        
        # Run the backtest
        results = backtest.run_backtest()
        
        # Save results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(backtest.results_dir, f"results_{year}_Q{quarter}_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        
        # Save trades to CSV
        trades_file = os.path.join(backtest.results_dir, f"trades_{year}_Q{quarter}_{timestamp}.csv")
        if backtest.trade_history:
            trades_df = pd.DataFrame([
                {
                    'symbol': t.symbol,
                    'direction': t.direction,
                    'entry_date': t.entry_date,
                    'entry_price': t.entry_price,
                    'exit_date': t.exit_date,
                    'exit_price': t.exit_price,
                    'position_size': t.position_size,
                    'pnl': t.pnl,
                    'pnl_pct': t.pnl_pct,
                    'exit_reason': t.exit_reason,
                    'is_crypto': t.is_crypto
                }
                for t in backtest.trade_history
            ])
            trades_df.to_csv(trades_file, index=False)
        
        return results
    
    except Exception as e:
        logger.error(f"Error running backtest for Q{quarter} {year}: {e}")
        return None
    
    finally:
        # Clean up temporary config file
        if os.path.exists(temp_config_file):
            os.remove(temp_config_file)

def main():
    """Main function to run quarterly backtests"""
    parser = argparse.ArgumentParser(description='Run backtests for multiple quarters')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--year', type=int, default=2024, help='Year to backtest')
    parser.add_argument('--quarters', type=str, default='1,2,3,4', help='Comma-separated list of quarters to backtest')
    
    args = parser.parse_args()
    
    # Parse quarters
    quarters = [int(q) for q in args.quarters.split(',')]
    
    # Run backtests for each quarter
    results = {}
    for quarter in quarters:
        quarter_results = run_quarter_backtest(args.config, args.year, quarter)
        if quarter_results:
            results[f"Q{quarter}"] = quarter_results
    
    # Compare quarterly performance
    if results:
        logger.info("\nQuarterly Performance Comparison:")
        logger.info("=" * 80)
        logger.info(f"{'Quarter':<10}{'Return':<12}{'Ann. Return':<15}{'Max DD':<10}{'Sharpe':<10}{'Win Rate':<10}{'Profit Factor':<15}{'Trades'}")
        logger.info("-" * 80)
        
        for quarter, result in results.items():
            logger.info(f"{quarter:<10}{result['total_return']:<12.2f}{result['annualized_return']:<15.2f}"
                       f"{result['max_drawdown']:<10.2f}{result['sharpe_ratio']:<10.2f}"
                       f"{result['win_rate']:<10.2f}{result['profit_factor']:<15.2f}{result['total_trades']}")
        
        logger.info("=" * 80)
        
        # Save combined results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_file = os.path.join("backtest_results", f"combined_results_{args.year}_{timestamp}.json")
        with open(combined_file, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        
        logger.info(f"Saved combined results to {combined_file}")

if __name__ == "__main__":
    main()
