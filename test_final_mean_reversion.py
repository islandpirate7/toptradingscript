#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test the final optimized MeanReversion strategy configuration with both synthetic and real data.
"""

import os
import sys
import json
import datetime as dt
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional
import alpaca_trade_api as tradeapi
from dataclasses import dataclass
import yaml
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MeanReversionTest')

# Define necessary classes and enums
class TradeDirection:
    LONG = "LONG"
    SHORT = "SHORT"

class SignalStrength:
    STRONG_BUY = "STRONG_BUY"
    MODERATE_BUY = "MODERATE_BUY"
    WEAK_BUY = "WEAK_BUY"
    WEAK_SELL = "WEAK_SELL"
    MODERATE_SELL = "MODERATE_SELL"
    STRONG_SELL = "STRONG_SELL"

@dataclass
class CandleData:
    """Class to represent candle data."""
    timestamp: dt.datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    symbol: str = None

@dataclass
class Signal:
    """Class to represent a trading signal."""
    symbol: str
    direction: str
    strategy: str
    entry_price: float
    stop_loss: float
    take_profit: float
    timestamp: dt.datetime
    strength: str
    expiration: dt.datetime

def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_file}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}

def calculate_bollinger_bands(candles: List[CandleData], period: int = 20, std_dev: float = 2.0) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Calculate Bollinger Bands for the given candles"""
    if len(candles) < period:
        return None, None, None
    
    # Calculate SMA
    closes = [candle.close for candle in candles[-period:]]
    sma = sum(closes) / period
    
    # Calculate standard deviation
    variance = sum((price - sma) ** 2 for price in closes) / period
    std = variance ** 0.5
    
    # Calculate upper and lower bands
    upper_band = sma + (std_dev * std)
    lower_band = sma - (std_dev * std)
    
    return sma, upper_band, lower_band

def calculate_rsi(candles: List[CandleData], period: int = 14) -> Optional[float]:
    """Calculate RSI for the given candles"""
    if len(candles) < period + 1:
        return None
    
    # Calculate price changes
    deltas = [candles[i].close - candles[i-1].close for i in range(1, len(candles))]
    
    # Calculate gains and losses
    gains = [delta if delta > 0 else 0 for delta in deltas]
    losses = [-delta if delta < 0 else 0 for delta in deltas]
    
    # Calculate average gains and losses
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    
    if avg_loss == 0:
        return 100
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_atr(candles: List[CandleData], period: int = 14) -> Optional[float]:
    """Calculate ATR for the given candles"""
    if len(candles) < period + 1:
        return None
    
    # Calculate true ranges
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
    
    # Calculate ATR
    atr = sum(true_ranges[-period:]) / period
    
    return atr

def create_realistic_candles(symbol: str, num_candles: int = 100, start_price: float = 100.0) -> List[CandleData]:
    """Create realistic synthetic candle data with extreme price movements to test mean reversion"""
    candles = []
    current_price = start_price
    timestamp = dt.datetime.now() - dt.timedelta(days=num_candles)
    
    # Create base candles with normal price movements
    for i in range(num_candles):
        # Add some randomness to price movements
        daily_volatility = current_price * 0.015  # 1.5% daily volatility
        price_change = np.random.normal(0, daily_volatility)
        
        # Create more extreme movements occasionally to trigger signals
        if i > 20 and i % 15 == 0:  # Every 15 candles after the first 20
            # Create an oversold condition (price drops significantly)
            price_change = -current_price * random.uniform(0.05, 0.08)  # 5-8% drop
            logger.debug(f"Creating oversold condition at candle {i}, price change: {price_change:.2f}")
        elif i > 20 and i % 15 == 5:  # 5 candles after each oversold condition
            # Create a reversal (price increases)
            price_change = current_price * random.uniform(0.02, 0.03)  # 2-3% increase
            logger.debug(f"Creating reversal after oversold at candle {i}, price change: {price_change:.2f}")
        elif i > 20 and i % 15 == 8:  # Every 15 candles, offset by 8
            # Create an overbought condition (price rises significantly)
            price_change = current_price * random.uniform(0.05, 0.08)  # 5-8% rise
            logger.debug(f"Creating overbought condition at candle {i}, price change: {price_change:.2f}")
        elif i > 20 and i % 15 == 13:  # 5 candles after each overbought condition
            # Create a reversal (price decreases)
            price_change = -current_price * random.uniform(0.02, 0.03)  # 2-3% decrease
            logger.debug(f"Creating reversal after overbought at candle {i}, price change: {price_change:.2f}")
        
        # Update current price
        current_price += price_change
        
        # Ensure price doesn't go negative
        current_price = max(current_price, 5.0)
        
        # Create intraday price movements
        daily_open = current_price * (1 + np.random.normal(0, 0.005))
        daily_close = current_price
        
        # Determine high and low
        if daily_open > daily_close:
            daily_high = daily_open * (1 + abs(np.random.normal(0, 0.003)))
            daily_low = daily_close * (1 - abs(np.random.normal(0, 0.003)))
        else:
            daily_high = daily_close * (1 + abs(np.random.normal(0, 0.003)))
            daily_low = daily_open * (1 - abs(np.random.normal(0, 0.003)))
        
        # Ensure high >= open, close and low <= open, close
        daily_high = max(daily_high, daily_open, daily_close)
        daily_low = min(daily_low, daily_open, daily_close)
        
        # Create volume with some randomness
        daily_volume = int(np.random.normal(1000000, 300000))
        
        # Create candle
        candle = CandleData(
            timestamp=timestamp + dt.timedelta(days=i),
            open=daily_open,
            high=daily_high,
            low=daily_low,
            close=daily_close,
            volume=daily_volume,
            symbol=symbol
        )
        
        candles.append(candle)
    
    return candles

def fetch_historical_data(symbol: str, start_date: dt.datetime, end_date: dt.datetime) -> List[CandleData]:
    """Fetch historical data from Alpaca API"""
    try:
        # Load Alpaca credentials
        with open('alpaca_credentials.json', 'r') as f:
            credentials = json.load(f)
        
        # Use paper trading credentials
        paper_creds = credentials.get('paper', {})
        api_key = paper_creds.get('api_key')
        api_secret = paper_creds.get('api_secret')
        base_url = paper_creds.get('base_url', 'https://paper-api.alpaca.markets/v2')
        
        if not api_key or not api_secret:
            logger.error("Missing Alpaca API credentials")
            return []
        
        # Initialize Alpaca API
        api = tradeapi.REST(api_key, api_secret, base_url)
        
        # Format dates
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Fetch historical data
        logger.info(f"Fetching historical data for {symbol} from {start_str} to {end_str}")
        bars = api.get_bars(symbol, '1D', start=start_str, end=end_str).df
        
        # Convert to CandleData objects
        candles = []
        for index, row in bars.iterrows():
            candle = CandleData(
                timestamp=index.to_pydatetime(),
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=int(row['volume'])
            )
            candle.symbol = symbol
            candles.append(candle)
        
        logger.info(f"Fetched {len(candles)} candles for {symbol}")
        return candles
    
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        return []

def generate_mean_reversion_signals(candles: List[CandleData], config: Dict[str, Any], symbol: str) -> List[Signal]:
    """Generate signals using the MeanReversion strategy with the provided configuration"""
    signals = []
    
    # Extract parameters from config
    bb_period = config.get('bb_period', 20)
    bb_std_dev = config.get('bb_std_dev', 2.0)
    rsi_period = config.get('rsi_period', 14)
    rsi_overbought = config.get('rsi_overbought', 70)
    rsi_oversold = config.get('rsi_oversold', 30)
    min_reversal_candles = config.get('min_reversal_candles', 2)
    require_reversal = config.get('require_reversal', True)
    stop_loss_atr = config.get('stop_loss_atr', 2.0)
    take_profit_atr = config.get('take_profit_atr', 3.0)
    
    # Need enough candles for calculations
    if len(candles) < max(bb_period, rsi_period) + 1:
        logger.warning(f"Not enough candles for calculations. Need at least {max(bb_period, rsi_period) + 1}, got {len(candles)}")
        return signals
    
    # Generate signals for each candle (except the first few needed for indicators)
    for i in range(max(bb_period, rsi_period) + 1, len(candles)):
        # Use candles up to current point for calculations
        current_candles = candles[:i+1]
        current_candle = current_candles[-1]
        
        # Calculate indicators
        sma, upper_band, lower_band = calculate_bollinger_bands(current_candles, bb_period, bb_std_dev)
        rsi = calculate_rsi(current_candles, rsi_period)
        atr = calculate_atr(current_candles, 14)
        
        if sma is None or rsi is None or atr is None:
            continue
        
        current_price = current_candle.close
        
        # Log detailed calculations for debugging
        logger.debug(f"Candle {i}: Price={current_price:.2f}, SMA={sma:.2f}, BB_Lower={lower_band:.2f}, BB_Upper={upper_band:.2f}, RSI={rsi:.2f}")
        
        # Check for oversold condition (buy signal)
        if current_price < lower_band * 1.05 and rsi < rsi_oversold * 1.2:
            logger.debug(f"Potential buy signal: Price < Lower Band and RSI < Oversold")
            
            # Check for price reversal if required
            reversal = True
            if require_reversal:
                reversal = current_candle.close > current_candles[-2].close
                logger.debug(f"Reversal check: {reversal} (Current close: {current_candle.close:.2f}, Previous close: {current_candles[-2].close:.2f})")
            
            if reversal:
                # Calculate stop loss and take profit using ATR
                stop_loss = current_price - (atr * stop_loss_atr)
                take_profit = current_price + (atr * take_profit_atr)
                
                # Determine signal strength
                rsi_strength = (rsi_oversold * 1.2 - rsi) / (rsi_oversold * 1.2)
                price_strength = (lower_band * 1.05 - current_price) / (lower_band * 1.05)
                
                if rsi_strength > 0.2 and price_strength > 0.05:
                    strength = SignalStrength.STRONG_BUY
                elif rsi_strength > 0.1 or price_strength > 0.02:
                    strength = SignalStrength.MODERATE_BUY
                else:
                    strength = SignalStrength.WEAK_BUY
                
                logger.debug(f"Generated BUY signal with strength {strength}")
                
                signal = Signal(
                    symbol=symbol,
                    direction=TradeDirection.LONG,
                    strategy="MeanReversion",
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    timestamp=current_candle.timestamp,
                    strength=strength,
                    expiration=current_candle.timestamp + dt.timedelta(days=3)
                )
                signals.append(signal)
        
        # Check for overbought condition (sell signal)
        elif current_price > upper_band / 1.05 and rsi > rsi_overbought / 1.2:
            logger.debug(f"Potential sell signal: Price > Upper Band and RSI > Overbought")
            
            # Check for price reversal if required
            reversal = True
            if require_reversal:
                reversal = current_candle.close < current_candles[-2].close
                logger.debug(f"Reversal check: {reversal} (Current close: {current_candle.close:.2f}, Previous close: {current_candles[-2].close:.2f})")
            
            if reversal:
                # Calculate stop loss and take profit using ATR
                stop_loss = current_price + (atr * stop_loss_atr)
                take_profit = current_price - (atr * take_profit_atr)
                
                # Determine signal strength
                rsi_strength = (rsi - (rsi_overbought / 1.2)) / (100 - (rsi_overbought / 1.2))
                price_strength = (current_price - (upper_band / 1.05)) / (upper_band / 1.05)
                
                if rsi_strength > 0.2 and price_strength > 0.05:
                    strength = SignalStrength.STRONG_SELL
                elif rsi_strength > 0.1 or price_strength > 0.02:
                    strength = SignalStrength.MODERATE_SELL
                else:
                    strength = SignalStrength.WEAK_SELL
                
                logger.debug(f"Generated SELL signal with strength {strength}")
                
                signal = Signal(
                    symbol=symbol,
                    direction=TradeDirection.SHORT,
                    strategy="MeanReversion",
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    timestamp=current_candle.timestamp,
                    strength=strength,
                    expiration=current_candle.timestamp + dt.timedelta(days=3)
                )
                signals.append(signal)
    
    return signals

def plot_signals(candles: List[CandleData], signals: List[Signal], symbol: str, config_name: str):
    """Plot candles with signals for visualization"""
    if not candles or not signals:
        return
    
    # Extract data for plotting
    dates = [c.timestamp for c in candles]
    closes = [c.close for c in candles]
    
    # Calculate Bollinger Bands for plotting
    bb_period = 20
    bb_std_dev = 1.9  # From final config
    
    upper_bands = []
    lower_bands = []
    smas = []
    
    for i in range(bb_period, len(candles)):
        sma, upper, lower = calculate_bollinger_bands(candles[:i+1], bb_period, bb_std_dev)
        smas.append(sma)
        upper_bands.append(upper)
        lower_bands.append(lower)
    
    # Pad with None for the first bb_period candles
    smas = [None] * bb_period + smas
    upper_bands = [None] * bb_period + upper_bands
    lower_bands = [None] * bb_period + lower_bands
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Plot price and Bollinger Bands
    plt.subplot(2, 1, 1)
    plt.plot(dates, closes, label='Close Price')
    plt.plot(dates, smas, label='SMA', linestyle='--')
    plt.plot(dates, upper_bands, label='Upper Band', linestyle=':')
    plt.plot(dates, lower_bands, label='Lower Band', linestyle=':')
    
    # Plot buy signals
    buy_signals = [s for s in signals if s.direction == TradeDirection.LONG]
    if buy_signals:
        buy_dates = [s.timestamp for s in buy_signals]
        buy_prices = [s.entry_price for s in buy_signals]
        plt.scatter(buy_dates, buy_prices, marker='^', color='green', s=100, label='Buy Signal')
    
    # Plot sell signals
    sell_signals = [s for s in signals if s.direction == TradeDirection.SHORT]
    if sell_signals:
        sell_dates = [s.timestamp for s in sell_signals]
        sell_prices = [s.entry_price for s in sell_signals]
        plt.scatter(sell_dates, sell_prices, marker='v', color='red', s=100, label='Sell Signal')
    
    plt.title(f'{symbol} Price with Bollinger Bands and Signals - {config_name}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    # Plot RSI
    plt.subplot(2, 1, 2)
    rsi_values = []
    rsi_period = 14
    
    for i in range(rsi_period, len(candles)):
        rsi = calculate_rsi(candles[:i+1], rsi_period)
        rsi_values.append(rsi)
    
    # Pad with None for the first rsi_period candles
    rsi_values = [None] * rsi_period + rsi_values
    
    plt.plot(dates, rsi_values, label='RSI')
    plt.axhline(y=70, color='r', linestyle='--', label='Overbought (70)')
    plt.axhline(y=30, color='g', linestyle='--', label='Oversold (30)')
    plt.axhline(y=50, color='k', linestyle='-', alpha=0.2)
    
    # Plot RSI at signal points
    if buy_signals:
        buy_rsi = []
        for s in buy_signals:
            # Find the RSI value at the signal timestamp
            idx = dates.index(s.timestamp)
            buy_rsi.append(rsi_values[idx])
        plt.scatter(buy_dates, buy_rsi, marker='^', color='green', s=100)
    
    if sell_signals:
        sell_rsi = []
        for s in sell_signals:
            # Find the RSI value at the signal timestamp
            idx = dates.index(s.timestamp)
            sell_rsi.append(rsi_values[idx])
        plt.scatter(sell_dates, sell_rsi, marker='v', color='red', s=100)
    
    plt.title(f'{symbol} RSI with Signals')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Create directory for plots if it doesn't exist
    os.makedirs('signal_plots', exist_ok=True)
    
    # Save the plot
    plt.savefig(f'signal_plots/{symbol}_{config_name}_signals.png')
    plt.close()

def main():
    """Main function to test the MeanReversion strategy"""
    # Load the final optimized configuration
    config = load_config('configuration_mean_reversion_final.yaml')
    
    if not config:
        logger.error("Failed to load configuration")
        return
    
    # Extract MeanReversion strategy configuration
    mr_config = config.get('strategies', {}).get('MeanReversion', {})
    
    if not mr_config:
        logger.error("MeanReversion strategy configuration not found")
        return
    
    # Define symbols to test
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    
    # Test with synthetic data
    logger.info("\nTesting with synthetic data:")
    synthetic_signals = []
    
    for symbol in symbols:
        # Create synthetic candles
        candles = create_realistic_candles(symbol, num_candles=100, start_price=100.0)
        
        # Generate signals
        signals = generate_mean_reversion_signals(candles, mr_config, symbol)
        
        logger.info(f"Generated {len(signals)} signals for {symbol} with synthetic data")
        
        # Count signals by direction
        long_signals = [s for s in signals if s.direction == TradeDirection.LONG]
        short_signals = [s for s in signals if s.direction == TradeDirection.SHORT]
        
        logger.info(f"  Long signals: {len(long_signals)}, Short signals: {len(short_signals)}")
        
        # Count signals by strength
        strong_signals = [s for s in signals if s.strength in [SignalStrength.STRONG_BUY, SignalStrength.STRONG_SELL]]
        moderate_signals = [s for s in signals if s.strength in [SignalStrength.MODERATE_BUY, SignalStrength.MODERATE_SELL]]
        weak_signals = [s for s in signals if s.strength in [SignalStrength.WEAK_BUY, SignalStrength.WEAK_SELL]]
        
        logger.info(f"  Signal strength: Strong: {len(strong_signals)}, Moderate: {len(moderate_signals)}, Weak: {len(weak_signals)}")
        
        # Plot signals
        plot_signals(candles, signals, symbol, "Synthetic")
        
        synthetic_signals.extend(signals)
    
    # Test with real historical data from 2023 Q4
    logger.info("\nTesting with real historical data:")
    real_signals = []
    
    # Define date range for 2023 Q4
    start_date = dt.datetime(2023, 10, 1)
    end_date = dt.datetime(2023, 12, 31)
    
    for symbol in symbols:
        # Fetch historical data
        candles = fetch_historical_data(symbol, start_date, end_date)
        
        if not candles:
            logger.warning(f"No historical data available for {symbol}")
            continue
        
        # Generate signals
        signals = generate_mean_reversion_signals(candles, mr_config, symbol)
        
        logger.info(f"Generated {len(signals)} signals for {symbol} with real data")
        
        # Count signals by direction
        long_signals = [s for s in signals if s.direction == TradeDirection.LONG]
        short_signals = [s for s in signals if s.direction == TradeDirection.SHORT]
        
        logger.info(f"  Long signals: {len(long_signals)}, Short signals: {len(short_signals)}")
        
        # Count signals by strength
        strong_signals = [s for s in signals if s.strength in [SignalStrength.STRONG_BUY, SignalStrength.STRONG_SELL]]
        moderate_signals = [s for s in signals if s.strength in [SignalStrength.MODERATE_BUY, SignalStrength.MODERATE_SELL]]
        weak_signals = [s for s in signals if s.strength in [SignalStrength.WEAK_BUY, SignalStrength.WEAK_SELL]]
        
        logger.info(f"  Signal strength: Strong: {len(strong_signals)}, Moderate: {len(moderate_signals)}, Weak: {len(weak_signals)}")
        
        # Plot signals
        plot_signals(candles, signals, symbol, "Real")
        
        real_signals.extend(signals)
    
    # Summary of all signals
    logger.info("\nSummary of synthetic data test:")
    logger.info(f"Generated {len(synthetic_signals)} signals across {len(symbols)} symbols with synthetic data")
    
    # Count signals by direction
    long_signals = [s for s in synthetic_signals if s.direction == TradeDirection.LONG]
    short_signals = [s for s in synthetic_signals if s.direction == TradeDirection.SHORT]
    
    logger.info(f"Long signals: {len(long_signals)}, Short signals: {len(short_signals)}")
    
    # Count signals by strength
    strong_signals = [s for s in synthetic_signals if s.strength in [SignalStrength.STRONG_BUY, SignalStrength.STRONG_SELL]]
    moderate_signals = [s for s in synthetic_signals if s.strength in [SignalStrength.MODERATE_BUY, SignalStrength.MODERATE_SELL]]
    weak_signals = [s for s in synthetic_signals if s.strength in [SignalStrength.WEAK_BUY, SignalStrength.WEAK_SELL]]
    
    logger.info(f"Signal strength distribution: Strong: {len(strong_signals)}, Moderate: {len(moderate_signals)}, Weak: {len(weak_signals)}")
    
    logger.info("\nSummary of real data test:")
    logger.info(f"Generated {len(real_signals)} signals across {len(symbols)} symbols with real data")
    
    # Count signals by direction
    long_signals = [s for s in real_signals if s.direction == TradeDirection.LONG]
    short_signals = [s for s in real_signals if s.direction == TradeDirection.SHORT]
    
    logger.info(f"Long signals: {len(long_signals)}, Short signals: {len(short_signals)}")
    
    # Count signals by strength
    strong_signals = [s for s in real_signals if s.strength in [SignalStrength.STRONG_BUY, SignalStrength.STRONG_SELL]]
    moderate_signals = [s for s in real_signals if s.strength in [SignalStrength.MODERATE_BUY, SignalStrength.MODERATE_SELL]]
    weak_signals = [s for s in real_signals if s.strength in [SignalStrength.WEAK_BUY, SignalStrength.WEAK_SELL]]
    
    logger.info(f"Signal strength distribution: Strong: {len(strong_signals)}, Moderate: {len(moderate_signals)}, Weak: {len(weak_signals)}")

if __name__ == "__main__":
    main()
