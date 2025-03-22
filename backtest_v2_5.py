#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MultiStrategy Trading System v2.5 Backtest
------------------------------------------
Version: 2.5 - Enhanced MeanReversion Strategy with Crypto Support
Last Updated: 2025-03-15

This script runs a comprehensive backtest of the MultiStrategy v2.5 trading system
using the Alpaca paper trading account. It tests both stocks and crypto assets
with the optimized MeanReversion strategy configuration.
"""

import os
import sys
import json
import yaml
import logging
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import REST, TimeFrame

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("backtest_v2_5.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MultiStrategyBacktest")

# Create results directory if it doesn't exist
class CandleData:
    def __init__(self, timestamp, open_price, high, low, close, volume):
        self.timestamp = timestamp
        self.open = open_price
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

class Signal:
    def __init__(self, symbol, timestamp, direction, strength, strategy_name, 
                 entry_price, stop_loss, take_profit, score=0.0):
        self.symbol = symbol
        self.timestamp = timestamp
        self.direction = direction  # 'long' or 'short'
        self.strength = strength    # 'weak', 'moderate', 'strong'
        self.strategy_name = strategy_name
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.score = score

class Trade:
    def __init__(self, symbol, entry_time, entry_price, direction, position_size,
                 stop_loss, take_profit, strategy_name):
        self.symbol = symbol
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.exit_time = None
        self.exit_price = None
        self.direction = direction
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.strategy_name = strategy_name
        self.pnl = 0.0
        self.pnl_pct = 0.0
        self.status = "open"
        self.exit_reason = None

class MarketState:
    def __init__(self):
        self.regime = "neutral"  # neutral, bullish, bearish, volatile
        self.volatility = "normal"  # low, normal, high
        self.trend_strength = "neutral"  # weak, neutral, strong
        self.market_breadth = 0.0  # -1.0 to 1.0
        self.sector_performance = {}  # sector name -> performance score
        self.vix = 0.0
        self.risk_on = True

class StockConfig:
    def __init__(self, symbol, config_data):
        self.symbol = symbol
        self.max_position_size = config_data.get("max_position_size", 100)
        self.min_position_size = config_data.get("min_position_size", 10)
        self.max_risk_per_trade_pct = config_data.get("max_risk_per_trade_pct", 0.5)
        self.sector = config_data.get("sector", "Unknown")
        self.industry = config_data.get("industry", "Unknown")
        self.mean_reversion_params = config_data.get("mean_reversion_params", {})
        self.trend_following_params = config_data.get("trend_following_params", {})
        self.volatility_breakout_params = config_data.get("volatility_breakout_params", {})
        self.gap_trading_params = config_data.get("gap_trading_params", {})

# Strategy base class
class Strategy:
    def __init__(self, config):
        self.config = config
        self.name = self.__class__.__name__
        
    def generate_signals(self, symbol: str, candles: List[CandleData], 
                         stock_config: StockConfig, market_state: MarketState) -> List[Signal]:
        """Generate trading signals for a symbol"""
        raise NotImplementedError("Subclasses must implement generate_signals")
    
    def calculate_regime_weight(self, market_state: MarketState) -> float:
        """Calculate strategy weight based on market regime"""
        raise NotImplementedError("Subclasses must implement calculate_regime_weight")

# Utility functions
def calculate_bollinger_bands(prices, period=20, std_dev=2.0):
    """Calculate Bollinger Bands for a price series"""
    if len(prices) < period:
        return None, None, None
    
    rolling_mean = np.mean(prices[-period:])
    rolling_std = np.std(prices[-period:])
    
    upper_band = rolling_mean + (rolling_std * std_dev)
    lower_band = rolling_mean - (rolling_std * std_dev)
    
    return upper_band, rolling_mean, lower_band

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    if len(prices) < period + 1:
        return 50  # Default to neutral if not enough data
    
    # Calculate price changes
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    
    if down == 0:
        return 100
    
    rs = up / down
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_atr(candles, period=14):
    """Calculate Average True Range"""
    if len(candles) < period:
        return None
    
    true_ranges = []
    for i in range(1, len(candles)):
        high_low = candles[i].high - candles[i].low
        high_close_prev = abs(candles[i].high - candles[i-1].close)
        low_close_prev = abs(candles[i].low - candles[i-1].close)
        true_range = max(high_low, high_close_prev, low_close_prev)
        true_ranges.append(true_range)
    
    if len(true_ranges) < period:
        return sum(true_ranges) / len(true_ranges)
    
    return sum(true_ranges[-period:]) / period

# MeanReversion Strategy Implementation
class MeanReversionStrategy(Strategy):
    def __init__(self, config):
        super().__init__(config)
        self.strategy_config = config.get("strategy_configs", {}).get("MeanReversion", {})
        self.bb_period = self.strategy_config.get("bb_period", 20)
        self.bb_std_dev = self.strategy_config.get("bb_std_dev", 1.9)
        self.rsi_period = self.strategy_config.get("rsi_period", 14)
        self.rsi_overbought = self.strategy_config.get("rsi_overbought", 65)
        self.rsi_oversold = self.strategy_config.get("rsi_oversold", 35)
        self.require_reversal = self.strategy_config.get("require_reversal", True)
        self.min_reversal_candles = self.strategy_config.get("min_reversal_candles", 1)
        self.stop_loss_atr = self.strategy_config.get("stop_loss_atr", 1.8)
        self.take_profit_atr = self.strategy_config.get("take_profit_atr", 3.0)
        self.max_position_size = self.strategy_config.get("max_position_size", 0.1)
    
    def generate_signals(self, symbol: str, candles: List[CandleData], 
                        stock_config: StockConfig, market_state: MarketState) -> List[Signal]:
        """Generate mean reversion signals based on Bollinger Bands and RSI"""
        if len(candles) < max(self.bb_period, self.rsi_period) + 10:
            logger.warning(f"Not enough candles for {symbol} to generate mean reversion signals")
            return []
        
        # Get symbol-specific parameters if available
        symbol_params = stock_config.mean_reversion_params
        bb_period = symbol_params.get("bb_period", self.bb_period)
        bb_std_dev = symbol_params.get("bb_std_dev", self.bb_std_dev)
        rsi_period = symbol_params.get("rsi_period", self.rsi_period)
        rsi_overbought = symbol_params.get("rsi_overbought", self.rsi_overbought)
        rsi_oversold = symbol_params.get("rsi_oversold", self.rsi_oversold)
        
        # Extract closing prices
        close_prices = [candle.close for candle in candles]
        
        # Calculate indicators
        upper_band, middle_band, lower_band = calculate_bollinger_bands(
            close_prices, period=bb_period, std_dev=bb_std_dev
        )
        
        if upper_band is None:
            return []
        
        rsi = calculate_rsi(close_prices, period=rsi_period)
        atr = calculate_atr(candles, period=14)
        
        if atr is None:
            return []
        
        # Check for signal conditions
        signals = []
        current_price = candles[-1].close
        timestamp = candles[-1].timestamp
        
        # Price distance from bands as percentage
        upper_band_distance = (upper_band - current_price) / current_price * 100
        lower_band_distance = (current_price - lower_band) / current_price * 100
        
        # Check for reversal pattern if required
        reversal_condition_met = True
        if self.require_reversal:
            # For long signals, check if price was below lower band and is now moving up
            if current_price < upper_band and rsi > 30 and rsi < rsi_overbought:
                # Check last few candles for reversal pattern
                reversal_count = 0
                for i in range(2, min(self.min_reversal_candles + 2, len(candles))):
                    if candles[-i].close < candles[-i+1].close:
                        reversal_count += 1
                reversal_condition_met = reversal_count >= self.min_reversal_candles
            
            # For short signals, check if price was above upper band and is now moving down
            elif current_price > lower_band and rsi < 70 and rsi > rsi_oversold:
                # Check last few candles for reversal pattern
                reversal_count = 0
                for i in range(2, min(self.min_reversal_candles + 2, len(candles))):
                    if candles[-i].close > candles[-i+1].close:
                        reversal_count += 1
                reversal_condition_met = reversal_count >= self.min_reversal_candles
        
        # Generate long signal
        # More lenient conditions: price near lower band OR RSI oversold
        if ((current_price <= lower_band * 1.01) or (rsi <= rsi_oversold + 5)) and reversal_condition_met:
            # Calculate stop loss and take profit levels
            stop_loss = current_price * (1 - self.stop_loss_atr * atr / current_price)
            take_profit = current_price * (1 + self.take_profit_atr * atr / current_price)
            
            # Determine signal strength based on RSI and distance from band
            if rsi < rsi_oversold and current_price <= lower_band:
                strength = "strong"
                signal_score = 0.9
            elif rsi < rsi_oversold + 5 or current_price <= lower_band * 1.01:
                strength = "moderate"
                signal_score = 0.7
            else:
                strength = "weak"
                signal_score = 0.5
            
            # Apply market regime adjustment
            regime_weight = self.calculate_regime_weight(market_state)
            signal_score *= regime_weight
            
            # Create signal
            signal = Signal(
                symbol=symbol,
                timestamp=timestamp,
                direction="long",
                strength=strength,
                strategy_name=self.name,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                score=signal_score
            )
            signals.append(signal)
        
        # Generate short signal
        # More lenient conditions: price near upper band OR RSI overbought
        if ((current_price >= upper_band * 0.99) or (rsi >= rsi_overbought - 5)) and reversal_condition_met:
            # Calculate stop loss and take profit levels
            stop_loss = current_price * (1 + self.stop_loss_atr * atr / current_price)
            take_profit = current_price * (1 - self.take_profit_atr * atr / current_price)
            
            # Determine signal strength based on RSI and distance from band
            if rsi > rsi_overbought and current_price >= upper_band:
                strength = "strong"
                signal_score = 0.9
            elif rsi > rsi_overbought - 5 or current_price >= upper_band * 0.99:
                strength = "moderate"
                signal_score = 0.7
            else:
                strength = "weak"
                signal_score = 0.5
            
            # Apply market regime adjustment
            regime_weight = self.calculate_regime_weight(market_state)
            signal_score *= regime_weight
            
            # Create signal
            signal = Signal(
                symbol=symbol,
                timestamp=timestamp,
                direction="short",
                strength=strength,
                strategy_name=self.name,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                score=signal_score
            )
            signals.append(signal)
        
        return signals
    
    def calculate_regime_weight(self, market_state: MarketState) -> float:
        """Calculate strategy weight based on market regime"""
        # Mean reversion works best in neutral or volatile markets
        if market_state.regime == "neutral":
            return 1.0
        elif market_state.regime == "volatile":
            return 0.8
        else:  # bullish or bearish
            return 0.6

# Backtest Implementation
class MultiStrategyBacktest:
    def __init__(self, config_file: str, alpaca_credentials_file: str, start_date: str, end_date: str):
        """Initialize the backtest with configuration and date range"""
        self.config_file = config_file
        self.alpaca_credentials_file = alpaca_credentials_file
        self.start_date = start_date
        self.end_date = end_date
        self.config = self._load_config()
        self.alpaca_api = self._initialize_alpaca_api()
        self.strategies = self._initialize_strategies()
        self.stock_configs = self._initialize_stock_configs()
        self.crypto_configs = self._initialize_crypto_configs()
        self.market_state = MarketState()
        self.initial_capital = self.config.get("initial_capital", 100000.0)
        self.equity_curve = [self.initial_capital]
        self.trade_history = []
        self.current_positions = {}
        self.daily_returns = []
        self.daily_equity = []
        self.signals_generated = []
        self.timestamp = []
        
        # Create results directory
        self.results_dir = "backtest_results"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        with open(self.config_file, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Loaded configuration from {self.config_file}")
        logger.info(f"MultiStrategy Trading System v{config.get('Version', '2.5')}")
        return config
    
    def _initialize_alpaca_api(self) -> REST:
        """Initialize Alpaca API client"""
        with open(self.alpaca_credentials_file, 'r') as file:
            credentials = json.load(file)
        
        # Use paper trading credentials
        paper_credentials = credentials.get("paper", {})
        api_key = paper_credentials.get("api_key")
        api_secret = paper_credentials.get("api_secret")
        base_url = paper_credentials.get("base_url", "https://paper-api.alpaca.markets")
        
        if not api_key or not api_secret:
            raise ValueError("Missing Alpaca API credentials")
        
        logger.info(f"Initializing Alpaca API with paper trading account")
        return REST(api_key, api_secret, base_url)
    
    def _initialize_strategies(self) -> Dict[str, Strategy]:
        """Initialize trading strategies"""
        strategies = {
            "MeanReversion": MeanReversionStrategy(self.config)
        }
        logger.info(f"Initialized {len(strategies)} trading strategies")
        return strategies
    
    def _initialize_stock_configs(self) -> Dict[str, StockConfig]:
        """Initialize stock configurations"""
        stock_configs = {}
        for stock_config in self.config.get("stocks", []):
            symbol = stock_config.get("symbol")
            if symbol:
                stock_configs[symbol] = StockConfig(symbol, stock_config)
        logger.info(f"Initialized {len(stock_configs)} stock configurations")
        return stock_configs
    
    def _initialize_crypto_configs(self) -> Dict[str, StockConfig]:
        """Initialize crypto configurations"""
        crypto_configs = {}
        crypto_section = self.config.get("crypto", {})
        if crypto_section.get("enabled", False):
            for crypto_config in crypto_section.get("assets", []):
                symbol = crypto_config.get("symbol")
                if symbol:
                    crypto_configs[symbol] = StockConfig(symbol, crypto_config)
            logger.info(f"Initialized {len(crypto_configs)} crypto configurations")
        else:
            logger.info("Crypto trading is disabled in configuration")
        return crypto_configs
    
    def fetch_historical_data(self, symbol: str, is_crypto: bool = False) -> List[CandleData]:
        """Fetch historical data for a symbol from Alpaca API"""
        try:
            # Calculate start date with lookback period
            start_date = datetime.datetime.strptime(self.start_date, "%Y-%m-%d")
            lookback_start = start_date - datetime.timedelta(days=60)  # 60-day lookback for indicators
            
            # Format dates for API
            start_str = lookback_start.strftime("%Y-%m-%d")
            end_str = self.end_date
            
            logger.info(f"Fetching historical data for {symbol} from {start_str} to {end_str}")
            
            # Fetch data from Alpaca
            if is_crypto:
                # For crypto, we need to use the crypto API endpoint
                crypto_symbol = symbol.replace('USD', '/USD')  # Convert BTCUSD to BTC/USD format
                bars = self.alpaca_api.get_crypto_bars(
                    crypto_symbol,
                    TimeFrame.Day,
                    start_str,
                    end_str
                ).df
                
                if bars.empty:
                    logger.warning(f"No data returned for crypto {symbol} ({crypto_symbol})")
                    return []
            else:
                # For stocks, use the regular bars endpoint
                bars = self.alpaca_api.get_bars(
                    symbol,
                    TimeFrame.Day,
                    start_str,
                    end_str
                ).df
                
                if bars.empty:
                    logger.warning(f"No data returned for stock {symbol}")
                    return []
            
            # Convert to CandleData objects
            candles = []
            for index, row in bars.iterrows():
                candle = CandleData(
                    timestamp=index.to_pydatetime(),
                    open_price=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row['volume']
                )
                candles.append(candle)
            
            logger.info(f"Fetched {len(candles)} candles for {symbol}")
            return candles
            
        except Exception as e:
            if is_crypto:
                logger.error(f"Error fetching data for crypto {symbol}: {str(e)}")
                # Try alternative format if the first one fails
                try:
                    # Some exchanges might use a different format
                    crypto_symbol = symbol  # Try using the original format
                    bars = self.alpaca_api.get_crypto_bars(
                        crypto_symbol,
                        TimeFrame.Day,
                        start_str,
                        end_str
                    ).df
                    
                    if bars.empty:
                        logger.warning(f"No data returned for crypto {symbol} (alternative format)")
                        return []
                    
                    # Convert to CandleData objects
                    candles = []
                    for index, row in bars.iterrows():
                        candle = CandleData(
                            timestamp=index.to_pydatetime(),
                            open_price=row['open'],
                            high=row['high'],
                            low=row['low'],
                            close=row['close'],
                            volume=row['volume']
                        )
                        candles.append(candle)
                    
                    logger.info(f"Fetched {len(candles)} candles for {symbol} (alternative format)")
                    return candles
                    
                except Exception as e2:
                    logger.error(f"Error fetching data for crypto {symbol} (alternative format): {str(e2)}")
                    return []
            else:
                logger.error(f"Error fetching data for stock {symbol}: {str(e)}")
                return []
    
    def generate_signals_for_symbol(self, symbol: str, candles: List[CandleData], 
                                   is_crypto: bool = False) -> List[Signal]:
        """Generate signals for a symbol using all strategies"""
        if not candles:
            logger.warning(f"No candles available for {symbol}")
            return []
        
        # Get the appropriate config
        if is_crypto:
            if symbol not in self.crypto_configs:
                logger.warning(f"No configuration found for crypto {symbol}")
                return []
            config = self.crypto_configs[symbol]
        else:
            if symbol not in self.stock_configs:
                logger.warning(f"No configuration found for stock {symbol}")
                return []
            config = self.stock_configs[symbol]
        
        # Generate signals from all strategies
        all_signals = []
        for strategy_name, strategy in self.strategies.items():
            signals = strategy.generate_signals(symbol, candles, config, self.market_state)
            all_signals.extend(signals)
            logger.info(f"Generated {len(signals)} {strategy_name} signals for {symbol}")
        
        return all_signals
    
    def update_market_state(self, date: datetime.datetime) -> None:
        """Update market state based on current market conditions"""
        # In a real implementation, we would analyze market data to determine the regime
        # For this backtest, we'll use a simplified approach
        
        # Check if we have SPY data to determine market regime
        spy_candles = self.fetch_historical_data("SPY")
        if not spy_candles:
            logger.warning("Could not fetch SPY data to determine market regime")
            return
        
        # Calculate market trend using simple moving averages
        spy_closes = [candle.close for candle in spy_candles]
        if len(spy_closes) < 50:
            logger.warning("Not enough SPY data to determine market regime")
            return
        
        # Calculate short and long-term moving averages
        short_ma = np.mean(spy_closes[-20:])
        long_ma = np.mean(spy_closes[-50:])
        
        # Calculate volatility
        recent_volatility = np.std(spy_closes[-20:]) / np.mean(spy_closes[-20:]) * 100
        
        # Determine market regime
        if short_ma > long_ma * 1.05:
            self.market_state.regime = "bullish"
            self.market_state.trend_strength = "strong"
        elif short_ma > long_ma:
            self.market_state.regime = "bullish"
            self.market_state.trend_strength = "weak"
        elif short_ma < long_ma * 0.95:
            self.market_state.regime = "bearish"
            self.market_state.trend_strength = "strong"
        elif short_ma < long_ma:
            self.market_state.regime = "bearish"
            self.market_state.trend_strength = "weak"
        else:
            self.market_state.regime = "neutral"
            self.market_state.trend_strength = "neutral"
        
        # Determine volatility regime
        if recent_volatility > 2.0:
            self.market_state.volatility = "high"
        elif recent_volatility < 1.0:
            self.market_state.volatility = "low"
        else:
            self.market_state.volatility = "normal"
        
        # If high volatility and strong trend, mark as volatile regime
        if self.market_state.volatility == "high" and self.market_state.trend_strength == "strong":
            self.market_state.regime = "volatile"
        
        logger.info(f"Market state updated: {self.market_state.regime} regime, {self.market_state.volatility} volatility")
    
    def run_backtest(self) -> Dict:
        """Run the backtest and return results"""
        logger.info(f"Starting backtest from {self.start_date} to {self.end_date}")
        
        # Process each day in the backtest period
        current_date = datetime.datetime.strptime(self.start_date, "%Y-%m-%d")
        end_date = datetime.datetime.strptime(self.end_date, "%Y-%m-%d")
        
        # Track daily equity for performance metrics
        daily_equity = [self.initial_capital]
        dates = [current_date]
        
        # For progress reporting
        progress_report_time = time.time()
        total_days = (end_date - current_date).days
        processed_days = 0
        
        while current_date <= end_date:
            # Skip weekends
            if current_date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
                current_date += datetime.timedelta(days=1)
                continue
            
            logger.info(f"Processing date: {current_date.strftime('%Y-%m-%d')}")
            
            # Update market state for the current day
            self.update_market_state(current_date)
            
            # Process all symbols
            all_signals = []
            
            # Process stocks
            for symbol in self.stock_configs.keys():
                candles = self.fetch_historical_data(symbol)
                signals = self.generate_signals_for_symbol(symbol, candles)
                all_signals.extend(signals)
            
            # Process crypto if enabled
            if self.crypto_configs:
                for symbol in self.crypto_configs.keys():
                    candles = self.fetch_historical_data(symbol, is_crypto=True)
                    signals = self.generate_signals_for_symbol(symbol, candles, is_crypto=True)
                    all_signals.extend(signals)
            
            # Process signals and update positions
            self._process_signals(current_date, all_signals)
            
            # Update positions and calculate daily P&L
            self._update_positions(current_date)
            
            # Record daily equity
            daily_equity.append(self.equity_curve[-1])
            dates.append(current_date)
            
            # Progress reporting
            current_time = time.time()
            if current_time - progress_report_time >= 10:
                progress_pct = processed_days / total_days * 100 if total_days > 0 else 100
                current_equity = self.equity_curve[-1]
                total_return_pct = (current_equity - self.initial_capital) / self.initial_capital * 100
                
                print(f"\n{'='*50}")
                print(f"Progress Report - {current_date.strftime('%Y-%m-%d')} ({progress_pct:.1f}% complete)")
                print(f"Current Equity: ${current_equity:.2f}")
                print(f"Return to Date: {total_return_pct:.2f}%")
                print(f"Open Positions: {len(self.current_positions)}")
                print(f"Closed Trades: {len(self.trade_history)}")
                
                if self.trade_history:
                    winning_trades = [t for t in self.trade_history if t.pnl > 0]
                    win_rate = len(winning_trades) / len(self.trade_history) * 100
                    print(f"Win Rate: {win_rate:.1f}%")
                
                print(f"{'='*50}")
                
                progress_report_time = current_time
            
            # Move to the next day
            current_date += datetime.timedelta(days=1)
            processed_days += 1
        
        # Calculate performance metrics
        results = self._calculate_performance_metrics(daily_equity, dates)
        
        # Save results
        self._save_results(results)
        
        logger.info(f"Backtest completed. Final equity: ${self.equity_curve[-1]:.2f}")
        return results
    
    def _process_signals(self, date: datetime.datetime, signals: List[Signal]) -> None:
        """Process trading signals and execute trades"""
        if not signals:
            return
        
        # Log all signals
        for signal in signals:
            logger.info(f"Signal: {signal.symbol} {signal.direction} {signal.strength} score={signal.score:.2f} "
                       f"entry={signal.entry_price:.2f} stop={signal.stop_loss:.2f} target={signal.take_profit:.2f}")
        
        # Calculate available capital
        used_capital = sum(trade.position_size for trade in self.current_positions.values())
        available_capital = self.equity_curve[-1] - used_capital
        logger.info(f"Available capital for new positions: ${available_capital:.2f}")
        
        # Process signals
        for signal in signals:
            # Skip if we already have a position in this symbol
            if signal.symbol in self.current_positions:
                logger.info(f"Skipping {signal.symbol} signal - already have a position")
                continue
            
            # Get the latest price data for the symbol
            is_crypto = signal.symbol in self.crypto_configs
            candles = self.fetch_historical_data(signal.symbol, is_crypto=is_crypto)
            
            if not candles:
                logger.warning(f"No price data available for {signal.symbol} on {date.strftime('%Y-%m-%d')}")
                continue
            
            # Find the candle for the current date
            current_candle = None
            for candle in candles:
                if candle.timestamp.date() == date.date():
                    current_candle = candle
                    break
            
            # If no candle for current date, use the latest candle
            if not current_candle:
                logger.warning(f"No candle found for {signal.symbol} on {date.strftime('%Y-%m-%d')}, using latest available")
                current_candle = candles[-1]
            
            # Update the signal's entry price to the current price
            signal.entry_price = current_candle.close
            
            # Recalculate stop loss and take profit based on updated entry price
            if signal.direction == "long":
                stop_pct = (signal.entry_price - signal.stop_loss) / signal.entry_price
                take_profit_pct = (signal.take_profit - signal.entry_price) / signal.entry_price
                signal.stop_loss = signal.entry_price * (1 - stop_pct)
                signal.take_profit = signal.entry_price * (1 + take_profit_pct)
            else:  # short
                stop_pct = (signal.stop_loss - signal.entry_price) / signal.entry_price
                take_profit_pct = (signal.entry_price - signal.take_profit) / signal.entry_price
                signal.stop_loss = signal.entry_price * (1 + stop_pct)
                signal.take_profit = signal.entry_price * (1 - take_profit_pct)
            
            # Calculate position size
            if is_crypto:
                config = self.crypto_configs[signal.symbol]
            else:
                config = self.stock_configs[signal.symbol]
            
            # Get position size limits from config
            max_position_size = getattr(config, "max_position_size", 1000)
            min_position_size = getattr(config, "min_position_size", 100)
            max_risk_pct = getattr(config, "max_risk_per_trade_pct", 1.0) / 100.0
            
            # Calculate risk-based position size
            risk_amount = self.equity_curve[-1] * max_risk_pct
            
            if signal.direction == "long":
                risk_per_share = signal.entry_price - signal.stop_loss
            else:  # short
                risk_per_share = signal.stop_loss - signal.entry_price
            
            # Avoid division by zero
            if risk_per_share <= 0:
                logger.warning(f"Invalid risk per share for {signal.symbol}: {risk_per_share}")
                continue
            
            # Calculate position size based on risk
            position_size = risk_amount / (risk_per_share / signal.entry_price)
            
            # Limit position size
            if position_size > max_position_size:
                position_size = max_position_size
                logger.info(f"Reduced position size to maximum ${position_size:.2f}")
            elif position_size < min_position_size:
                position_size = min_position_size
                logger.info(f"Reduced position size to minimum ${position_size:.2f}")
            
            # Ensure we have enough capital
            if position_size > available_capital:
                if available_capital >= min_position_size:
                    position_size = available_capital
                    logger.info(f"Reduced position size to available capital ${position_size:.2f}")
                else:
                    logger.info(f"Not enough capital to open position in {signal.symbol}")
                    continue
            
            # Create trade
            trade = Trade(
                symbol=signal.symbol,
                entry_time=date,
                entry_price=signal.entry_price,
                direction=signal.direction,
                position_size=position_size,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                strategy_name=signal.strategy_name
            )
            
            # Add to current positions
            self.current_positions[signal.symbol] = trade
            
            # Update available capital
            available_capital -= position_size
            
            # Log the trade
            logger.info(f"Opened {signal.direction} position in {signal.symbol} at {signal.entry_price:.2f}, "
                       f"size: ${position_size:.2f}, stop: {signal.stop_loss:.2f}, target: {signal.take_profit:.2f}")
            
            # Store signal
            self.signals_generated.append(signal)
    
    def _update_positions(self, current_date: datetime.datetime) -> None:
        """Update open positions and check for exits"""
        if not self.current_positions:
            return
        
        # Track closed positions to remove from current_positions
        closed_positions = []
        
        # Calculate daily P&L
        daily_pnl = 0.0
        
        for symbol, trade in self.current_positions.items():
            # Fetch latest price data
            is_crypto = symbol in self.crypto_configs
            candles = self.fetch_historical_data(symbol, is_crypto=is_crypto)
            
            if not candles:
                logger.warning(f"No price data available for {symbol} on {current_date.strftime('%Y-%m-%d')}")
                continue
            
            # Find the candle for the current date
            current_candle = None
            for candle in candles:
                if candle.timestamp.date() == current_date.date():
                    current_candle = candle
                    break
            
            # If no candle for current date, use the latest candle
            if not current_candle:
                logger.warning(f"No candle found for {symbol} on {current_date.strftime('%Y-%m-%d')}, using latest available")
                current_candle = candles[-1]
            
            current_price = current_candle.close
            
            # Check for stop loss or take profit
            exit_reason = None
            exit_price = None
            
            if trade.direction == "long":
                # Check if price hit stop loss
                if current_candle.low <= trade.stop_loss:
                    exit_reason = "stop_loss"
                    exit_price = trade.stop_loss
                # Check if price hit take profit
                elif current_candle.high >= trade.take_profit:
                    exit_reason = "take_profit"
                    exit_price = trade.take_profit
                # Check for time-based exit (max holding period of 10 trading days)
                elif (current_date - trade.entry_time).days >= 10:
                    exit_reason = "time_exit"
                    exit_price = current_price
                
                # Calculate unrealized P&L
                unrealized_pnl = (current_price - trade.entry_price) * trade.position_size / trade.entry_price
            else:  # short
                # Check if price hit stop loss
                if current_candle.high >= trade.stop_loss:
                    exit_reason = "stop_loss"
                    exit_price = trade.stop_loss
                # Check if price hit take profit
                elif current_candle.low <= trade.take_profit:
                    exit_reason = "take_profit"
                    exit_price = trade.take_profit
                # Check for time-based exit (max holding period of 10 trading days)
                elif (current_date - trade.entry_time).days >= 10:
                    exit_reason = "time_exit"
                    exit_price = current_price
                
                # Calculate unrealized P&L
                unrealized_pnl = (trade.entry_price - current_price) * trade.position_size / trade.entry_price
            
            # If exit triggered, close the position
            if exit_reason:
                trade.exit_time = current_date
                trade.exit_price = exit_price
                trade.status = "closed"
                trade.exit_reason = exit_reason
                
                # Calculate P&L
                if trade.direction == "long":
                    trade.pnl = (exit_price - trade.entry_price) * trade.position_size / trade.entry_price
                else:  # short
                    trade.pnl = (trade.entry_price - exit_price) * trade.position_size / trade.entry_price
                
                trade.pnl_pct = trade.pnl / trade.position_size * 100
                
                # Add to trade history
                self.trade_history.append(trade)
                
                # Mark for removal from current positions
                closed_positions.append(symbol)
                
                # Add to daily P&L
                daily_pnl += trade.pnl
                
                # Update equity
                self.equity_curve.append(self.equity_curve[-1] + trade.pnl)
                
                # Log the exit
                logger.info(f"Closed {trade.direction} position in {symbol} at {exit_price:.2f}, "
                           f"P&L: ${trade.pnl:.2f} ({trade.pnl_pct:.2f}%), reason: {exit_reason}")
            else:
                # Add unrealized P&L to daily total
                daily_pnl += unrealized_pnl
        
        # Remove closed positions from current_positions
        for symbol in closed_positions:
            if symbol in self.current_positions:
                del self.current_positions[symbol]
        
        # If no positions were closed but we have open positions, update equity with unrealized P&L
        if not closed_positions and self.current_positions:
            self.equity_curve.append(self.equity_curve[-1] + daily_pnl)
        elif not closed_positions and not self.current_positions:
            # No change in equity if no positions
            self.equity_curve.append(self.equity_curve[-1])
        
        # Calculate daily return
        if len(self.equity_curve) > 1:
            daily_return = (self.equity_curve[-1] - self.equity_curve[-2]) / self.equity_curve[-2]
            self.daily_returns.append(daily_return)
    
    def _calculate_performance_metrics(self, daily_equity: List[float], dates: List[datetime.datetime]) -> Dict:
        """Calculate performance metrics from backtest results"""
        if len(daily_equity) < 2:
            return {
                "total_return_pct": 0.0,
                "annualized_return_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "sharpe_ratio": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_trades": 0
            }
        
        # Calculate total return
        total_return = (daily_equity[-1] - daily_equity[0]) / daily_equity[0]
        
        # Calculate annualized return
        days = (dates[-1] - dates[0]).days
        if days > 0:
            annualized_return = (1 + total_return) ** (365 / days) - 1
        else:
            annualized_return = 0.0
        
        # Calculate max drawdown
        max_drawdown = 0.0
        peak = daily_equity[0]
        
        for equity in daily_equity:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate Sharpe ratio
        if self.daily_returns:
            avg_daily_return = np.mean(self.daily_returns)
            std_daily_return = np.std(self.daily_returns)
            
            if std_daily_return > 0:
                sharpe_ratio = avg_daily_return / std_daily_return * np.sqrt(252)  # Annualized
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0
        
        # Calculate win rate and profit factor
        winning_trades = [t for t in self.trade_history if t.pnl > 0]
        losing_trades = [t for t in self.trade_history if t.pnl <= 0]
        
        win_rate = len(winning_trades) / len(self.trade_history) if self.trade_history else 0.0
        
        total_profit = sum(t.pnl for t in winning_trades)
        total_loss = abs(sum(t.pnl for t in losing_trades))
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 0.0
        
        # Calculate average trade metrics
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0.0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0.0
        
        # Calculate strategy-specific metrics
        strategy_performance = {}
        for strategy_name in self.strategies.keys():
            strategy_trades = [t for t in self.trade_history if t.strategy_name == strategy_name]
            if strategy_trades:
                strategy_winning_trades = [t for t in strategy_trades if t.pnl > 0]
                strategy_win_rate = len(strategy_winning_trades) / len(strategy_trades)
                strategy_profit = sum(t.pnl for t in strategy_trades)
                
                strategy_performance[strategy_name] = {
                    "win_rate": strategy_win_rate,
                    "total_profit": strategy_profit,
                    "num_trades": len(strategy_trades)
                }
        
        # Return all metrics
        return {
            "initial_capital": daily_equity[0],
            "final_equity": daily_equity[-1],
            "total_return_pct": total_return * 100,
            "annualized_return_pct": annualized_return * 100,
            "max_drawdown_pct": max_drawdown * 100,
            "sharpe_ratio": sharpe_ratio,
            "win_rate": win_rate * 100,
            "profit_factor": profit_factor,
            "total_trades": len(self.trade_history),
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "strategy_performance": strategy_performance,
            "equity_curve": daily_equity,
            "dates": [d.strftime("%Y-%m-%d") for d in dates]
        }
    
    def _save_results(self, results: Dict) -> None:
        """Save backtest results to files"""
        # Create timestamp for filenames
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save performance metrics to JSON
        metrics_file = f"backtest_results/metrics_v2_5_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump({k: v for k, v in results.items() if k not in ['equity_curve', 'dates']}, f, indent=4)
        
        # Save trade history to CSV
        trades_file = f"backtest_results/trades_v2_5_{timestamp}.csv"
        trades_df = pd.DataFrame([
            {
                "symbol": t.symbol,
                "strategy": t.strategy_name,
                "direction": t.direction,
                "entry_time": t.entry_time.strftime("%Y-%m-%d"),
                "exit_time": t.exit_time.strftime("%Y-%m-%d") if t.exit_time else "Open",
                "entry_price": t.entry_price,
                "exit_price": t.exit_price if t.exit_price else 0.0,
                "position_size": t.position_size,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "exit_reason": t.exit_reason if t.exit_reason else "Open"
            }
            for t in self.trade_history
        ])
        trades_df.to_csv(trades_file, index=False)
        
        # Plot equity curve
        plt.figure(figsize=(12, 6))
        plt.plot(results['dates'], results['equity_curve'])
        plt.title(f"MultiStrategy v2.5 Equity Curve ({self.start_date} to {self.end_date})")
        plt.xlabel("Date")
        plt.ylabel("Equity ($)")
        plt.grid(True)
        plt.xticks(rotation=45)
        
        # Save plot
        plot_file = f"backtest_results/equity_curve_v2_5_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        
        logger.info(f"Results saved to {metrics_file}, {trades_file}, and {plot_file}")


# Main execution
if __name__ == "__main__":
    import time
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run MultiStrategy Trading System v2.5 Backtest')
    parser.add_argument('--config', type=str, default='multi_strategy_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--credentials', type=str, default='alpaca_credentials.json',
                        help='Path to Alpaca credentials file')
    parser.add_argument('--start_date', type=str, default='2023-01-01',
                        help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2023-03-31',
                        help='End date for backtest (YYYY-MM-DD)')
    parser.add_argument('--quarter', type=int, choices=[1, 2, 3, 4], 
                        help='Quarter of 2023 to backtest (1-4)')
    
    args = parser.parse_args()
    
    # If quarter is specified, set the start and end dates accordingly
    if args.quarter:
        if args.quarter == 1:
            args.start_date = '2023-01-01'
            args.end_date = '2023-03-31'
        elif args.quarter == 2:
            args.start_date = '2023-04-01'
            args.end_date = '2023-06-30'
        elif args.quarter == 3:
            args.start_date = '2023-07-01'
            args.end_date = '2023-09-30'
        elif args.quarter == 4:
            args.start_date = '2023-10-01'
            args.end_date = '2023-12-31'
    
    # Ensure config file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file '{args.config}' not found.")
        sys.exit(1)
    
    # Ensure credentials file exists
    if not os.path.exists(args.credentials):
        print(f"Error: Alpaca credentials file '{args.credentials}' not found.")
        sys.exit(1)
    
    print(f"Starting backtest from {args.start_date} to {args.end_date}...")
    print(f"Using configuration file: {args.config}")
    print(f"Using Alpaca credentials file: {args.credentials}")
    
    # Initialize and run backtest
    start_time = time.time()
    backtest = MultiStrategyBacktest(args.config, args.credentials, args.start_date, args.end_date)
    results = backtest.run_backtest()
    end_time = time.time()
    
    # Print summary
    print("\n" + "="*50)
    print("BACKTEST RESULTS SUMMARY")
    print("="*50)
    print(f"Initial Capital: ${results['initial_capital']:.2f}")
    print(f"Final Equity: ${results['final_equity']:.2f}")
    print(f"Total Return: {results['total_return_pct']:.2f}%")
    print(f"Annualized Return: {results['annualized_return_pct']:.2f}%")
    print(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Win Rate: {results['win_rate']:.2f}%")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Execution Time: {(end_time - start_time):.2f} seconds")
    print("="*50)
    
    print(f"\nResults saved to: {backtest.results_dir}")
