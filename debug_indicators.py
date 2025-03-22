import os
import sys
import yaml
import logging
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

from mean_reversion_enhanced import EnhancedMeanReversionStrategy, CandleData, MarketState
from enhanced_mean_reversion_backtest import EnhancedMeanReversionBacktest

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Debug indicator values for specific symbols"""
    # Load configuration
    config_path = 'configuration_mean_reversion_enhanced_optimized.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create backtest instance
    backtest = EnhancedMeanReversionBacktest(config_path)
    
    # Create strategy instance
    strategy = EnhancedMeanReversionStrategy(config)
    
    # Define date range - use a shorter period for debugging
    # Using 2023 data since the user's Alpaca account has a free tier subscription
    # which doesn't permit querying recent market data
    start_date = datetime.datetime.strptime('2023-01-01', '%Y-%m-%d')
    end_date = datetime.datetime.strptime('2023-03-31', '%Y-%m-%d')  # Just use Q1 for faster debugging
    
    # Select symbols to debug
    debug_symbols = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'BTC/USD']
    
    # Fetch data for each symbol
    for symbol in debug_symbols:
        logger.info(f"Analyzing {symbol}...")
        
        # Fetch historical data
        is_crypto = '/' in symbol
        candles = backtest.fetch_historical_data(symbol, start_date, end_date, is_crypto=is_crypto)
        
        if not candles:
            logger.warning(f"No data fetched for {symbol}, skipping")
            continue
        
        logger.info(f"Fetched {len(candles)} candles for {symbol} from {candles[0].timestamp.strftime('%Y-%m-%d')} to {candles[-1].timestamp.strftime('%Y-%m-%d')}")
        
        # Calculate indicators
        middle_band, upper_band, lower_band = strategy.calculate_bollinger_bands(candles)
        rsi = strategy.calculate_rsi(candles)
        atr = strategy.calculate_atr(candles)
        
        # Create dataframe for analysis
        data = []
        for i in range(len(candles)):
            if i < max(strategy.bb_period, strategy.rsi_period):
                continue
                
            candle = candles[i]
            
            # Skip if indicators are not available
            if (upper_band[i] is None or lower_band[i] is None or 
                rsi[i] is None or atr[i] is None):
                continue
            
            # Calculate band distances
            lower_band_distance = (candle.close - lower_band[i]) / candle.close if candle.close > 0 else 0
            upper_band_distance = (upper_band[i] - candle.close) / candle.close if candle.close > 0 else 0
            
            # Check if RSI conditions are met
            rsi_oversold = rsi[i] < strategy.rsi_oversold or min(rsi[max(0, i-3):i+1]) < strategy.rsi_oversold
            rsi_overbought = rsi[i] > strategy.rsi_overbought or max(rsi[max(0, i-3):i+1]) > strategy.rsi_overbought
            
            # Check if price is near bands
            near_lower_band = lower_band_distance <= 0.005 or candle.close < lower_band[i]
            near_upper_band = upper_band_distance <= 0.005 or candle.close > upper_band[i]
            
            # Check if signal conditions are met
            long_signal = near_lower_band and rsi_oversold
            short_signal = near_upper_band and rsi_overbought
            
            data.append({
                'date': candle.timestamp.strftime('%Y-%m-%d'),
                'close': candle.close,
                'lower_band': lower_band[i],
                'middle_band': middle_band[i],
                'upper_band': upper_band[i],
                'rsi': rsi[i],
                'atr': atr[i],
                'lower_band_distance': lower_band_distance,
                'upper_band_distance': upper_band_distance,
                'near_lower_band': near_lower_band,
                'near_upper_band': near_upper_band,
                'rsi_oversold': rsi_oversold,
                'rsi_overbought': rsi_overbought,
                'long_signal': long_signal,
                'short_signal': short_signal
            })
        
        # Convert to dataframe
        df = pd.DataFrame(data)
        
        # Print summary
        logger.info(f"Summary for {symbol}:")
        logger.info(f"Total days analyzed: {len(df)}")
        logger.info(f"Days near lower band: {df['near_lower_band'].sum()} ({df['near_lower_band'].mean():.2%})")
        logger.info(f"Days near upper band: {df['near_upper_band'].sum()} ({df['near_upper_band'].mean():.2%})")
        logger.info(f"Days with RSI oversold: {df['rsi_oversold'].sum()} ({df['rsi_oversold'].mean():.2%})")
        logger.info(f"Days with RSI overbought: {df['rsi_overbought'].sum()} ({df['rsi_overbought'].mean():.2%})")
        logger.info(f"Days with long signal: {df['long_signal'].sum()} ({df['long_signal'].mean():.2%})")
        logger.info(f"Days with short signal: {df['short_signal'].sum()} ({df['short_signal'].mean():.2%})")
        
        # Print days with signals
        if df['long_signal'].sum() > 0 or df['short_signal'].sum() > 0:
            logger.info("Days with signals:")
            signal_days = df[(df['long_signal'] == True) | (df['short_signal'] == True)]
            for _, row in signal_days.iterrows():
                signal_type = "LONG" if row['long_signal'] else "SHORT"
                logger.info(f"{row['date']}: {signal_type} signal - Close: {row['close']:.2f}, "
                           f"Lower BB: {row['lower_band']:.2f}, Upper BB: {row['upper_band']:.2f}, "
                           f"RSI: {row['rsi']:.2f}")
        else:
            logger.info("No signals generated for this symbol in the time period")
        
        # Save to CSV for further analysis
        output_dir = 'debug_output'
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(f"{output_dir}/{symbol.replace('/', '_')}_indicators.csv", index=False)
        
        # Plot indicators
        plt.figure(figsize=(12, 10))
        
        # Price and Bollinger Bands
        plt.subplot(3, 1, 1)
        plt.plot(df['date'], df['close'], label='Close')
        plt.plot(df['date'], df['upper_band'], 'r--', label='Upper BB')
        plt.plot(df['date'], df['middle_band'], 'g--', label='Middle BB')
        plt.plot(df['date'], df['lower_band'], 'r--', label='Lower BB')
        plt.title(f'{symbol} Price and Bollinger Bands')
        plt.xticks(rotation=45)
        plt.legend()
        
        # RSI
        plt.subplot(3, 1, 2)
        plt.plot(df['date'], df['rsi'], label='RSI')
        plt.axhline(y=strategy.rsi_overbought, color='r', linestyle='--', label=f'Overbought ({strategy.rsi_overbought})')
        plt.axhline(y=strategy.rsi_oversold, color='g', linestyle='--', label=f'Oversold ({strategy.rsi_oversold})')
        plt.title(f'{symbol} RSI')
        plt.xticks(rotation=45)
        plt.legend()
        
        # Signals
        plt.subplot(3, 1, 3)
        long_signals = df[df['long_signal'] == True]
        short_signals = df[df['short_signal'] == True]
        
        plt.plot(df['date'], df['close'], label='Close')
        if not long_signals.empty:
            plt.scatter(long_signals['date'], long_signals['close'], color='g', marker='^', s=100, label='Long Signal')
        if not short_signals.empty:
            plt.scatter(short_signals['date'], short_signals['close'], color='r', marker='v', s=100, label='Short Signal')
        
        plt.title(f'{symbol} Signals')
        plt.xticks(rotation=45)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{symbol.replace('/', '_')}_indicators.png")
        plt.close()
        
        logger.info(f"Saved analysis for {symbol} to {output_dir}/{symbol.replace('/', '_')}_indicators.csv and .png")
        logger.info("-" * 80)

if __name__ == "__main__":
    main()
