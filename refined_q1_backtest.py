#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Refined Q1 2023 Backtest for S&P 500 with 100 stocks
Based on initial backtest results showing SHORT positions outperforming LONG positions
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Top 100 S&P 500 symbols by market cap
TOP_100_SYMBOLS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "GOOG", "TSLA", "UNH",
    "LLY", "JPM", "V", "XOM", "AVGO", "PG", "MA", "HD", "COST", "MRK",
    "CVX", "ABBV", "PEP", "KO", "ADBE", "WMT", "CRM", "BAC", "TMO", "MCD",
    "CSCO", "PFE", "NFLX", "CMCSA", "ABT", "ORCL", "TMUS", "AMD", "DIS", "ACN",
    "DHR", "VZ", "NKE", "TXN", "NEE", "WFC", "PM", "INTC", "INTU", "COP",
    "AMGN", "IBM", "RTX", "HON", "QCOM", "UPS", "CAT", "LOW", "SPGI", "BA",
    "GE", "ELV", "DE", "AMAT", "ISRG", "AXP", "BKNG", "MDLZ", "GILD", "ADI",
    "SBUX", "TJX", "MMC", "SYK", "VRTX", "PLD", "MS", "BLK", "SCHW", "C",
    "ZTS", "CB", "AMT", "ADP", "GS", "ETN", "LRCX", "NOW", "MO", "REGN",
    "EOG", "SO", "BMY", "EQIX", "BSX", "CME", "CI", "PANW", "TGT", "SLB"
]

def load_alpaca_credentials(mode='paper'):
    """Load Alpaca API credentials from JSON file"""
    try:
        with open('alpaca_credentials.json', 'r') as f:
            credentials = json.load(f)
        return credentials[mode]
    except Exception as e:
        logger.error(f"Error loading Alpaca credentials: {str(e)}")
        raise

def run_refined_backtest():
    """Run refined backtest for Q1 2023 with 100 stocks"""
    # Initialize Alpaca API
    credentials = load_alpaca_credentials(mode='paper')
    api = tradeapi.REST(
        key_id=credentials['api_key'],
        secret_key=credentials['api_secret'],
        base_url=credentials['base_url']
    )
    
    # Define backtest parameters
    start_date = '2023-01-01'
    end_date = '2023-03-31'
    holding_period = 3  # Shorter holding period
    base_position_size = 1000  # $1,000 per position
    
    # Get historical data for each symbol
    all_returns = []
    
    for symbol in tqdm(TOP_100_SYMBOLS, desc="Processing symbols"):
        try:
            # Get daily bars
            bars = api.get_bars(
                symbol,
                '1D',
                start=pd.Timestamp(start_date, tz='UTC').isoformat(),
                end=pd.Timestamp(end_date, tz='UTC').isoformat(),
                adjustment='raw'
            ).df
            
            if len(bars) < 10:  # Skip if not enough data
                logger.warning(f"Not enough data for {symbol}")
                continue
            
            # Calculate RSI (14-day)
            delta = bars['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            bars['rsi'] = 100 - (100 / (1 + rs))
            
            # Calculate MACD
            ema12 = bars['close'].ewm(span=12, adjust=False).mean()
            ema26 = bars['close'].ewm(span=26, adjust=False).mean()
            bars['macd'] = ema12 - ema26
            bars['macd_signal'] = bars['macd'].ewm(span=9, adjust=False).mean()
            
            # Calculate Bollinger Bands
            bars['sma20'] = bars['close'].rolling(window=20).mean()
            bars['std20'] = bars['close'].rolling(window=20).std()
            bars['upper_band'] = bars['sma20'] + (bars['std20'] * 2)
            bars['lower_band'] = bars['sma20'] - (bars['std20'] * 2)
            
            # Calculate ATR for volatility-based position sizing
            high_low = bars['high'] - bars['low']
            high_close = np.abs(bars['high'] - bars['close'].shift())
            low_close = np.abs(bars['low'] - bars['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            bars['atr'] = true_range.rolling(14).mean()
            
            # Drop rows with NaN values
            bars = bars.dropna()
            
            # Generate trades for each trading day
            for i in range(len(bars) - holding_period):
                # Get current day data
                current_day = bars.iloc[i]
                
                # Calculate combined technical score (0-1)
                rsi = current_day['rsi']
                macd = current_day['macd']
                macd_signal = current_day['macd_signal']
                close = current_day['close']
                lower_band = current_day['lower_band']
                upper_band = current_day['upper_band']
                
                # RSI component (0-1)
                if rsi < 30:
                    rsi_score = 1.0  # Strongly oversold - bullish
                elif rsi > 70:
                    rsi_score = 0.0  # Strongly overbought - bearish
                else:
                    rsi_score = 1 - ((rsi - 30) / 40)  # Linear scale between 30-70
                
                # MACD component (0-1)
                if macd > macd_signal:
                    macd_score = 1.0  # Bullish
                else:
                    macd_score = 0.0  # Bearish
                
                # Bollinger Bands component (0-1)
                if close < lower_band:
                    bb_score = 1.0  # Below lower band - bullish
                elif close > upper_band:
                    bb_score = 0.0  # Above upper band - bearish
                else:
                    bb_score = 1 - ((close - lower_band) / (upper_band - lower_band))
                
                # Combined score (weighted average)
                # REFINED: Adjust weights to favor SHORT signals
                combined_score = (rsi_score * 0.3) + (macd_score * 0.3) + (bb_score * 0.4)
                
                # REFINED: Invert the score to favor SHORT signals
                inverted_score = 1 - combined_score
                
                # Determine trade direction based on combined score
                # REFINED: Lower the threshold for SHORT positions
                if inverted_score >= 0.55:  # Favoring SHORT positions
                    direction = 'SHORT'
                elif combined_score >= 0.65:  # Higher threshold for LONG positions
                    direction = 'LONG'
                else:
                    direction = 'NEUTRAL'
                
                # Skip neutral positions
                if direction == 'NEUTRAL':
                    continue
                
                # Calculate position size with optimizations
                position_size = base_position_size
                
                # REFINED: Increase position size for SHORT positions
                if direction == 'SHORT':
                    position_size *= 1.3  # 30% increase for SHORT positions
                
                # REFINED: Increase position size for optimal score ranges
                if direction == 'SHORT' and 0.6 <= inverted_score <= 0.8:
                    position_size *= 1.2  # 20% additional increase for optimal SHORT scores
                elif direction == 'LONG' and 0.7 <= combined_score <= 0.9:
                    position_size *= 1.1  # 10% additional increase for optimal LONG scores
                
                # Calculate return
                start_price = current_day['close']
                end_price = bars.iloc[i + holding_period]['close']
                
                if direction == 'LONG':
                    pct_return = (end_price / start_price - 1) * 100
                else:  # SHORT
                    pct_return = (start_price / end_price - 1) * 100
                
                # Calculate dollar return
                dollar_return = (pct_return / 100) * position_size
                
                # Add to results
                all_returns.append({
                    'symbol': symbol,
                    'date': current_day.name.strftime('%Y-%m-%d'),
                    'direction': direction,
                    'combined_score': combined_score if direction == 'LONG' else inverted_score,
                    'rsi': rsi,
                    'macd': macd,
                    'position_size': position_size,
                    'start_price': start_price,
                    'end_price': end_price,
                    'pct_return': pct_return,
                    'dollar_return': dollar_return
                })
                
        except Exception as e:
            logger.error(f"Error processing {symbol}: {str(e)}")
    
    # Create results dataframe
    results_df = pd.DataFrame(all_returns)
    
    # Calculate overall statistics
    if len(results_df) > 0:
        # Save results
        results_df.to_csv('refined_q1_2023_backtest_results.csv', index=False)
        
        # Calculate statistics
        total_trades = len(results_df)
        win_rate = len(results_df[results_df['pct_return'] > 0]) / total_trades * 100
        avg_pct_return = results_df['pct_return'].mean()
        total_pct_return = results_df['pct_return'].sum()
        avg_dollar_return = results_df['dollar_return'].mean()
        total_dollar_return = results_df['dollar_return'].sum()
        
        # Calculate statistics by direction
        long_results = results_df[results_df['direction'] == 'LONG']
        short_results = results_df[results_df['direction'] == 'SHORT']
        
        long_win_rate = len(long_results[long_results['pct_return'] > 0]) / len(long_results) * 100 if len(long_results) > 0 else 0
        short_win_rate = len(short_results[short_results['pct_return'] > 0]) / len(short_results) * 100 if len(short_results) > 0 else 0
        
        long_total_return = long_results['pct_return'].sum()
        short_total_return = short_results['pct_return'].sum()
        
        long_dollar_return = long_results['dollar_return'].sum()
        short_dollar_return = short_results['dollar_return'].sum()
        
        # Print statistics
        logger.info(f"Refined Backtest completed for Q1 2023 with 100 stocks")
        logger.info(f"Total trades: {total_trades}")
        logger.info(f"Win rate: {win_rate:.2f}%")
        logger.info(f"Average return per trade: {avg_pct_return:.2f}%")
        logger.info(f"Total return: {total_pct_return:.2f}%")
        logger.info(f"Average dollar return per trade: ${avg_dollar_return:.2f}")
        logger.info(f"Total dollar return: ${total_dollar_return:.2f}")
        logger.info(f"")
        logger.info(f"LONG trades: {len(long_results)}")
        logger.info(f"LONG win rate: {long_win_rate:.2f}%")
        logger.info(f"LONG total return: {long_total_return:.2f}%")
        logger.info(f"LONG dollar return: ${long_dollar_return:.2f}")
        logger.info(f"")
        logger.info(f"SHORT trades: {len(short_results)}")
        logger.info(f"SHORT win rate: {short_win_rate:.2f}%")
        logger.info(f"SHORT total return: {short_total_return:.2f}%")
        logger.info(f"SHORT dollar return: ${short_dollar_return:.2f}")
        
        # Calculate annualized return
        days_in_period = 90  # Q1 is approximately 90 days
        annualized_return = total_pct_return * (365 / days_in_period)
        annualized_dollar_return = total_dollar_return * (365 / days_in_period)
        
        logger.info(f"")
        logger.info(f"Annualized return: {annualized_return:.2f}%")
        logger.info(f"Annualized dollar return: ${annualized_dollar_return:.2f}")
        
        # Calculate return on $100,000 capital
        capital = 100000
        return_on_capital = (total_dollar_return / capital) * 100
        annualized_return_on_capital = (annualized_dollar_return / capital) * 100
        
        logger.info(f"")
        logger.info(f"Return on $100,000 capital: {return_on_capital:.2f}%")
        logger.info(f"Dollar return on $100,000 capital: ${total_dollar_return:.2f}")
        logger.info(f"Annualized return on capital: {annualized_return_on_capital:.2f}%")
        logger.info(f"Annualized dollar return on capital: ${annualized_dollar_return:.2f}")
        
        return results_df
    else:
        logger.warning("No valid trades found in the backtest period")
        return pd.DataFrame()

if __name__ == "__main__":
    run_refined_backtest()
