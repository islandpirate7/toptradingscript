#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Backtest script for S&P 500 stock selection with 100 stocks
"""

import os
import json
import logging
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pytz
import requests
from bs4 import BeautifulSoup
import alpaca_trade_api as tradeapi
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('backtest_100_stocks.log')
    ]
)
logger = logging.getLogger(__name__)

def load_alpaca_credentials(mode='paper'):
    """Load Alpaca API credentials from JSON file"""
    try:
        with open('alpaca_credentials.json', 'r') as f:
            credentials = json.load(f)
        return credentials[mode]
    except Exception as e:
        logger.error(f"Error loading Alpaca credentials: {str(e)}")
        raise

def get_sp500_symbols():
    """Get the current list of S&P 500 symbols from Wikipedia"""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table', {'class': 'wikitable'})
        symbols = []
        for row in table.find_all('tr')[1:]:
            symbol = row.find_all('td')[0].text.strip()
            symbols.append(symbol.replace('.', '-'))
        logger.info(f"Successfully fetched {len(symbols)} S&P 500 symbols")
        return symbols
    except Exception as e:
        logger.error(f"Error fetching S&P 500 symbols: {str(e)}")
        # Fall back to a static list of top 100 symbols
        logger.info("Using static list of top 100 S&P 500 symbols")
        return [
            "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "GOOG", "BRK.B", "TSLA", "UNH",
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

def load_historical_data(api, symbols, start_date, end_date):
    """Load historical data for symbols"""
    logger.info(f"Loading historical data for {len(symbols)} symbols from {start_date} to {end_date}")
    
    # Convert dates to datetime objects
    start_dt = pd.to_datetime(start_date).tz_localize('UTC')
    end_dt = pd.to_datetime(end_date).tz_localize('UTC')
    
    # Load data for each symbol
    data = {}
    for symbol in tqdm(symbols, desc="Loading data"):
        try:
            # Get daily bars
            bars = api.get_bars(
                symbol,
                '1D',
                start=start_dt.isoformat(),
                end=end_dt.isoformat(),
                adjustment='raw'
            ).df
            
            if len(bars) > 0:
                data[symbol] = bars
            else:
                logger.warning(f"No data available for {symbol}")
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {str(e)}")
    
    logger.info(f"Loaded data for {len(data)} symbols")
    return data

def calculate_technical_indicators(data):
    """Calculate technical indicators for each symbol"""
    for symbol, df in data.items():
        try:
            # Calculate RSI (14-day)
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Calculate MACD
            ema12 = df['close'].ewm(span=12, adjust=False).mean()
            ema26 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = ema12 - ema26
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # Calculate Bollinger Bands
            df['sma20'] = df['close'].rolling(window=20).mean()
            df['std20'] = df['close'].rolling(window=20).std()
            df['upper_band'] = df['sma20'] + (df['std20'] * 2)
            df['lower_band'] = df['sma20'] - (df['std20'] * 2)
            df['bb_width'] = (df['upper_band'] - df['lower_band']) / df['sma20']
            
            # Calculate Stochastic Oscillator
            low_14 = df['low'].rolling(window=14).min()
            high_14 = df['high'].rolling(window=14).max()
            df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
            df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
            
            # Calculate ADX
            plus_dm = df['high'].diff()
            minus_dm = df['low'].diff(-1)
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift(1))
            tr3 = abs(df['low'] - df['close'].shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean()
            plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
            minus_di = abs(100 * (minus_dm.rolling(window=14).mean() / atr))
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            df['adx'] = dx.rolling(window=14).mean()
            
            # Calculate ATR
            df['atr'] = atr
            
            # Calculate momentum
            df['momentum'] = df['close'] / df['close'].shift(10) - 1
            
            # Calculate rate of change
            df['roc'] = df['close'].pct_change(10) * 100
            
        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol}: {str(e)}")
    
    return data

def score_stocks(data, date, config):
    """Score stocks based on technical indicators"""
    scores = []
    
    # Get technical analysis config
    ta_config = config.get('technical_analysis', {})
    ta_weight = ta_config.get('weight', 0.925)
    indicators = ta_config.get('indicators', {})
    
    # Get seasonality config
    seasonality_config = config.get('seasonality', {})
    seasonality_weight = seasonality_config.get('weight', 0.075)
    
    for symbol, df in data.items():
        try:
            # Get data for the given date
            date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
            if date_str not in df.index.strftime('%Y-%m-%d').values:
                continue
            
            idx = df.index[df.index.strftime('%Y-%m-%d') == date_str][0]
            row = df.loc[idx]
            
            # Calculate technical score
            tech_score = 0
            direction = 'NEUTRAL'
            
            # RSI
            if indicators.get('rsi', {}).get('enabled', True):
                rsi_weight = indicators.get('rsi', {}).get('weight', 0.2)
                rsi = row['rsi']
                overbought = indicators.get('rsi', {}).get('overbought', 70)
                oversold = indicators.get('rsi', {}).get('oversold', 35)  # Adjusted to favor LONG
                
                if rsi < oversold:
                    # Oversold - bullish
                    rsi_score = 1.0
                    rsi_direction = 'LONG'
                elif rsi > overbought:
                    # Overbought - bearish
                    rsi_score = 0.0
                    rsi_direction = 'SHORT'
                else:
                    # Neutral
                    rsi_score = 1 - ((rsi - oversold) / (overbought - oversold))
                    rsi_direction = 'NEUTRAL'
                
                tech_score += rsi_score * rsi_weight
                if rsi_direction != 'NEUTRAL':
                    direction = rsi_direction
            
            # MACD
            if indicators.get('macd', {}).get('enabled', True):
                macd_weight = indicators.get('macd', {}).get('weight', 0.25)  # Increased weight
                macd = row['macd']
                macd_signal = row['macd_signal']
                
                if macd > macd_signal:
                    # Bullish
                    macd_score = 1.0
                    macd_direction = 'LONG'
                else:
                    # Bearish
                    macd_score = 0.0
                    macd_direction = 'SHORT'
                
                tech_score += macd_score * macd_weight
                if direction == 'NEUTRAL':
                    direction = macd_direction
            
            # Bollinger Bands
            if indicators.get('bollinger_bands', {}).get('enabled', True):
                bb_weight = indicators.get('bollinger_bands', {}).get('weight', 0.2)
                close = row['close']
                upper_band = row['upper_band']
                lower_band = row['lower_band']
                
                if close < lower_band:
                    # Below lower band - bullish
                    bb_score = 1.0
                    bb_direction = 'LONG'
                elif close > upper_band:
                    # Above upper band - bearish
                    bb_score = 0.0
                    bb_direction = 'SHORT'
                else:
                    # Within bands - neutral
                    bb_score = 1 - ((close - lower_band) / (upper_band - lower_band))
                    bb_direction = 'NEUTRAL'
                
                tech_score += bb_score * bb_weight
                if direction == 'NEUTRAL' and bb_direction != 'NEUTRAL':
                    direction = bb_direction
            
            # Stochastic
            if indicators.get('stochastic', {}).get('enabled', True):
                stoch_weight = indicators.get('stochastic', {}).get('weight', 0.2)
                stoch_k = row['stoch_k']
                stoch_d = row['stoch_d']
                
                if stoch_k < 20 and stoch_k > stoch_d:
                    # Oversold and rising - bullish
                    stoch_score = 1.0
                    stoch_direction = 'LONG'
                elif stoch_k > 80 and stoch_k < stoch_d:
                    # Overbought and falling - bearish
                    stoch_score = 0.0
                    stoch_direction = 'SHORT'
                else:
                    # Neutral
                    stoch_score = 0.5
                    stoch_direction = 'NEUTRAL'
                
                tech_score += stoch_score * stoch_weight
                if direction == 'NEUTRAL' and stoch_direction != 'NEUTRAL':
                    direction = stoch_direction
            
            # ADX
            if indicators.get('adx', {}).get('enabled', True):
                adx_weight = indicators.get('adx', {}).get('weight', 0.2)
                adx = row['adx']
                
                if adx > 25:
                    # Strong trend
                    adx_score = 1.0
                else:
                    # Weak trend
                    adx_score = adx / 25
                
                tech_score += adx_score * adx_weight
            
            # Simple seasonality score (placeholder)
            current_month = pd.to_datetime(date).month
            seasonality_score = 0.5  # Neutral default
            
            # Combine scores
            combined_score = (tech_score * ta_weight) + (seasonality_score * seasonality_weight)
            
            scores.append({
                'symbol': symbol,
                'date': date,
                'combined_score': combined_score,
                'technical_score': tech_score,
                'seasonality_score': seasonality_score,
                'direction': direction,
                'close': row['close'],
                'market_regime': config.get('market_regime', 'mixed')
            })
        
        except Exception as e:
            logger.error(f"Error scoring {symbol}: {str(e)}")
    
    return scores

def select_top_stocks(scores, top_n=100):
    """Select top N stocks based on combined score"""
    # Sort by combined score
    sorted_scores = sorted(scores, key=lambda x: x['combined_score'], reverse=True)
    
    # ENHANCEMENT: Prioritize stocks with scores in the 0.6-0.7 range
    optimal_range = [s for s in sorted_scores if 0.6 <= s['combined_score'] <= 0.7]
    other_stocks = [s for s in sorted_scores if s['combined_score'] < 0.6 or s['combined_score'] > 0.7]
    
    # Combine and take top N
    prioritized_stocks = optimal_range + other_stocks
    return prioritized_stocks[:top_n]

def calculate_returns(selections, data, holding_period=5):
    """Calculate returns for selected stocks over the holding period"""
    results = []
    
    for selection in selections:
        symbol = selection['symbol']
        date = selection['date']
        direction = selection['direction']
        
        try:
            # Get symbol data
            df = data[symbol]
            
            # Find index of selection date
            date_dt = pd.to_datetime(date)
            if date_dt.tzinfo is not None:
                date_dt = date_dt.tz_localize(None)
                
            # Find the closest date in the dataframe
            df_dates = df.index.tz_localize(None) if df.index.tzinfo is not None else df.index
            idx = (df_dates - date_dt).abs().argmin()
            
            # Calculate end index
            end_idx = min(idx + holding_period, len(df) - 1)
            
            # Get prices
            start_price = df.iloc[idx]['close']
            end_price = df.iloc[end_idx]['close']
            
            # Calculate return based on direction
            if direction == 'LONG':
                ret = (end_price / start_price - 1) * 100
            elif direction == 'SHORT':
                ret = (start_price / end_price - 1) * 100
            else:
                ret = 0  # Neutral position
            
            # Add to results
            results.append({
                'symbol': symbol,
                'date': date,
                'direction': direction,
                'score': selection['combined_score'],
                'start_price': start_price,
                'end_price': end_price,
                'return': ret,
                'holding_period': end_idx - idx,
                'market_regime': selection['market_regime']
            })
        
        except Exception as e:
            logger.error(f"Error calculating return for {symbol}: {str(e)}")
    
    return results

def run_backtest(config_file, start_date, end_date, top_n=100, holding_period=5):
    """Run backtest for the given period"""
    # Load configuration
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Initialize Alpaca API
    credentials = load_alpaca_credentials(mode='paper')
    api = tradeapi.REST(
        key_id=credentials['api_key'],
        secret_key=credentials['api_secret'],
        base_url=credentials['base_url']
    )
    
    # Get S&P 500 symbols
    symbols = get_sp500_symbols()
    
    # Load historical data
    data = load_historical_data(api, symbols, start_date, end_date)
    
    # Calculate technical indicators
    data = calculate_technical_indicators(data)
    
    # Generate date range
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    dates = pd.date_range(start=start_dt, end=end_dt, freq='B')
    
    all_selections = []
    all_results = []
    
    # Run backtest for each date
    for date in tqdm(dates, desc="Running backtest"):
        date_str = date.strftime('%Y-%m-%d')
        
        # Score stocks
        scores = score_stocks(data, date_str, config)
        
        # Select top stocks
        selections = select_top_stocks(scores, top_n)
        
        # Add to all selections
        all_selections.extend(selections)
        
        # Calculate returns
        results = calculate_returns(selections, data, holding_period)
        
        # Add to all results
        all_results.extend(results)
    
    # Create results dataframe
    results_df = pd.DataFrame(all_results)
    
    # Calculate overall statistics
    total_trades = len(results_df)
    trades_with_returns = len(results_df[~results_df['return'].isna()])
    win_rate = len(results_df[results_df['return'] > 0]) / trades_with_returns * 100
    avg_return = results_df['return'].mean()
    total_return = results_df['return'].sum()
    
    logger.info(f"Backtest completed for period {start_date} to {end_date}")
    logger.info(f"Total trades: {total_trades}")
    logger.info(f"Win rate: {win_rate:.2f}%")
    logger.info(f"Average return: {avg_return:.2f}%")
    logger.info(f"Total return: {total_return:.2f}%")
    
    return results_df

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Backtest S&P 500 stock selection with 100 stocks')
    parser.add_argument('--config', type=str, default='strategy_optimization/optimized_config.json',
                        help='Path to configuration file')
    parser.add_argument('--start_date', type=str, required=True,
                        help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, required=True,
                        help='End date for backtest (YYYY-MM-DD)')
    parser.add_argument('--top_n', type=int, default=100,
                        help='Number of top stocks to select')
    parser.add_argument('--holding_period', type=int, default=5,
                        help='Holding period in days')
    parser.add_argument('--output', type=str,
                        help='Output file for results (CSV)')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs('backtest_results', exist_ok=True)
    
    # Run backtest
    results_df = run_backtest(
        args.config,
        args.start_date,
        args.end_date,
        args.top_n,
        args.holding_period
    )
    
    # Save results
    if args.output:
        output_file = args.output
    else:
        start_str = args.start_date.replace('-', '')
        end_str = args.end_date.replace('-', '')
        output_file = f'backtest_results_{start_str}_{end_str}_100stocks.csv'
    
    results_df.to_csv(output_file, index=False)
    logger.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
