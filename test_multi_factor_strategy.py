#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the enhanced multi-factor stock selection strategy.
This script loads the enhanced configuration, runs a backtest, and analyzes the results.
"""

# Set matplotlib to use non-interactive backend before importing it
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid threading issues

import os
import sys
import yaml
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import alpaca_trade_api as tradeapi
import json
import time
import argparse

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import strategy modules
from combined_strategy import CombinedStrategy
from data_loader import AlpacaDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def load_alpaca_credentials():
    """Load Alpaca API credentials from file"""
    try:
        with open('alpaca_credentials.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading Alpaca credentials: {e}")
        return None

def load_data_parallel(data_loader, symbols, start_date, end_date, timeframe, max_workers=10):
    """
    Load historical data for multiple symbols in parallel
    
    Args:
        data_loader: Data loader instance
        symbols: List of symbols to load data for
        start_date: Start date for data
        end_date: End date for data
        timeframe: Timeframe for data
        max_workers: Maximum number of parallel workers
        
    Returns:
        dict: Dictionary of symbol -> dataframe
    """
    symbol_data = {}
    
    def load_symbol_data(symbol):
        try:
            df = data_loader.load_historical_data(
                symbol, 
                start=start_date, 
                end=end_date, 
                timeframe=timeframe
            )
            
            if df is not None and not df.empty:
                # Convert timestamp to timezone-naive for consistent comparisons
                if 'timestamp' in df.columns and hasattr(df['timestamp'], 'dt') and df['timestamp'].dt.tz is not None:
                    df['timestamp'] = df['timestamp'].dt.tz_localize(None)
                
                # Set timestamp as index for easier date-based operations
                if 'timestamp' in df.columns:
                    df = df.set_index('timestamp')
                
                logger.info(f"Loaded {len(df)} bars for {symbol}")
                return symbol, df
            else:
                logger.warning(f"No data loaded for {symbol}")
                return symbol, None
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            return symbol, None
    
    # Use ThreadPoolExecutor to load data in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks
        future_to_symbol = {executor.submit(load_symbol_data, symbol): symbol for symbol in symbols}
        
        # Process results as they complete
        for future in as_completed(future_to_symbol):
            symbol, df = future.result()
            if df is not None:
                symbol_data[symbol] = df
    
    logger.info(f"Successfully loaded data for {len(symbol_data)} out of {len(symbols)} symbols")
    return symbol_data

def calculate_returns(selections_df, symbol_data, holding_period=5):
    """
    Calculate returns for selected stocks
    
    Args:
        selections_df: DataFrame with stock selections
        symbol_data: Dictionary of symbol -> dataframe with price data
        holding_period: Number of days to hold each position
        
    Returns:
        DataFrame: Updated selections DataFrame with return metrics
    """
    # Add columns for return calculation
    selections_df['entry_price'] = None
    selections_df['exit_price'] = None
    selections_df['return'] = None
    selections_df['exit_date'] = None
    selections_df['holding_days'] = None
    
    # Calculate returns for each selection
    for idx, row in selections_df.iterrows():
        symbol = row['symbol']
        date = row['date']
        direction = row['direction']
        
        # Convert date to datetime if it's not already
        if isinstance(date, str):
            date_dt = datetime.strptime(date, '%Y-%m-%d')
        else:
            date_dt = pd.Timestamp(date)
        
        # Calculate exit date (holding_period trading days later)
        end_date = date_dt + timedelta(days=holding_period)
        
        # Get symbol data
        if symbol not in symbol_data:
            continue
            
        symbol_df = symbol_data[symbol]
        
        # Find entry date (first trading day on or after selection date)
        try:
            # First try using boolean indexing with Series.idxmax()
            entry_mask = symbol_df.index >= date_dt
            if isinstance(entry_mask, pd.Series) and entry_mask.any():
                entry_idx = entry_mask.idxmax()
            else:
                # Fallback to using numpy where
                entry_indices = np.where(symbol_df.index >= date_dt)[0]
                if len(entry_indices) > 0:
                    entry_idx = symbol_df.index[entry_indices[0]]
                else:
                    continue
        except Exception as e:
            logger.error(f"Error finding entry date for {symbol}: {e}")
            # Try a different approach
            try:
                entry_dates = [idx for idx in symbol_df.index if idx >= date_dt]
                if entry_dates:
                    entry_idx = entry_dates[0]
                else:
                    continue
            except Exception as e2:
                logger.error(f"Second attempt failed for {symbol}: {e2}")
                continue
        
        entry_price = symbol_df.loc[entry_idx, 'close']
        
        # Find exit date (holding_period trading days later or last available day)
        try:
            # First try using boolean indexing with Series.idxmax()
            exit_mask = symbol_df.index > end_date
            if isinstance(exit_mask, pd.Series) and exit_mask.any():
                exit_idx = exit_mask.idxmax()
                if exit_idx > 0:
                    exit_idx_loc = symbol_df.index.get_loc(exit_idx)
                    if exit_idx_loc > 0:
                        exit_idx = symbol_df.index[exit_idx_loc - 1]
            else:
                # Fallback to using numpy where
                exit_indices = np.where(symbol_df.index > end_date)[0]
                if len(exit_indices) > 0:
                    exit_idx_loc = exit_indices[0]
                    if exit_idx_loc > 0:
                        exit_idx = symbol_df.index[exit_idx_loc - 1]
                    else:
                        exit_idx = symbol_df.index[exit_idx_loc]
                else:
                    # If no data after end_date, use the last available date
                    exit_idx = symbol_df.index[-1]
        except Exception as e:
            logger.error(f"Error finding exit date for {symbol}: {e}")
            # Try a different approach
            try:
                exit_dates = [idx for idx in symbol_df.index if idx > end_date]
                if exit_dates:
                    exit_idx = exit_dates[0]
                    # Get the previous day if possible
                    exit_idx_loc = list(symbol_df.index).index(exit_idx)
                    if exit_idx_loc > 0:
                        exit_idx = symbol_df.index[exit_idx_loc - 1]
                else:
                    # If no data after end_date, use the last available date
                    exit_idx = symbol_df.index[-1]
            except Exception as e2:
                logger.error(f"Second attempt failed for {symbol}: {e2}")
                continue
            
        exit_price = symbol_df.loc[exit_idx, 'close']
        exit_date = exit_idx
        
        # Calculate return based on direction
        if direction == 'LONG':
            ret = (exit_price - entry_price) / entry_price
        else:  # SHORT
            ret = (entry_price - exit_price) / entry_price
            
        # Calculate holding days
        holding_days = (exit_date - date_dt).days
        
        # Update DataFrame
        selections_df.at[idx, 'entry_price'] = entry_price
        selections_df.at[idx, 'exit_price'] = exit_price
        selections_df.at[idx, 'return'] = ret
        selections_df.at[idx, 'exit_date'] = exit_date
        selections_df.at[idx, 'holding_days'] = holding_days
    
    return selections_df

def analyze_performance(results_df):
    """
    Analyze performance of stock selections
    
    Args:
        results_df: DataFrame with stock selections and returns
        
    Returns:
        dict: Performance metrics
    """
    if results_df.empty:
        logger.warning("No results to analyze")
        return {}
    
    # Basic performance metrics
    total_trades = len(results_df)
    winning_trades = len(results_df[results_df['return'] > 0])
    losing_trades = len(results_df[results_df['return'] <= 0])
    
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # Calculate average returns
    avg_return = results_df['return'].mean()
    avg_win = results_df[results_df['return'] > 0]['return'].mean() if winning_trades > 0 else 0
    avg_loss = results_df[results_df['return'] <= 0]['return'].mean() if losing_trades > 0 else 0
    
    # Calculate profit factor
    gross_profit = results_df[results_df['return'] > 0]['return'].sum()
    gross_loss = abs(results_df[results_df['return'] <= 0]['return'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
    
    # Calculate drawdown
    cumulative_returns = (1 + results_df.sort_values('date')['return']).cumprod() - 1
    if len(cumulative_returns) > 0:
        max_drawdown = (cumulative_returns + 1).div((cumulative_returns + 1).cummax()).min() - 1
    else:
        max_drawdown = 0
    
    # Calculate Sharpe ratio (assuming risk-free rate of 0)
    daily_returns = results_df.groupby('date')['return'].mean()
    sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 and daily_returns.std() > 0 else 0
    
    # Calculate Sortino ratio (downside deviation)
    downside_returns = daily_returns[daily_returns < 0]
    sortino_ratio = daily_returns.mean() / downside_returns.std() * np.sqrt(252) if len(downside_returns) > 1 and downside_returns.std() > 0 else 0
    
    # Calculate Calmar ratio
    annualized_return = (1 + daily_returns.mean()) ** 252 - 1 if len(daily_returns) > 0 else 0
    calmar_ratio = abs(annualized_return / max_drawdown) if max_drawdown != 0 else float('inf')
    
    # Analyze performance by direction
    long_trades = results_df[results_df['direction'] == 'LONG']
    short_trades = results_df[results_df['direction'] == 'SHORT']
    
    long_win_rate = len(long_trades[long_trades['return'] > 0]) / len(long_trades) if len(long_trades) > 0 else 0
    short_win_rate = len(short_trades[short_trades['return'] > 0]) / len(short_trades) if len(short_trades) > 0 else 0
    
    long_avg_return = long_trades['return'].mean() if len(long_trades) > 0 else 0
    short_avg_return = short_trades['return'].mean() if len(short_trades) > 0 else 0
    
    # Analyze performance by market regime
    regime_performance = {}
    if 'market_regime' in results_df.columns:
        for regime in results_df['market_regime'].unique():
            regime_trades = results_df[results_df['market_regime'] == regime]
            regime_win_rate = len(regime_trades[regime_trades['return'] > 0]) / len(regime_trades) if len(regime_trades) > 0 else 0
            regime_avg_return = regime_trades['return'].mean() if len(regime_trades) > 0 else 0
            regime_performance[regime] = {
                'trades': len(regime_trades),
                'win_rate': regime_win_rate,
                'avg_return': regime_avg_return
            }
    
    # Analyze performance by sector
    sector_performance = {}
    if 'sector' in results_df.columns:
        for sector in results_df['sector'].unique():
            if pd.isna(sector):
                continue
            sector_trades = results_df[results_df['sector'] == sector]
            sector_win_rate = len(sector_trades[sector_trades['return'] > 0]) / len(sector_trades) if len(sector_trades) > 0 else 0
            sector_avg_return = sector_trades['return'].mean() if len(sector_trades) > 0 else 0
            sector_performance[sector] = {
                'trades': len(sector_trades),
                'win_rate': sector_win_rate,
                'avg_return': sector_avg_return
            }
    
    # Analyze performance by technical score
    results_df['technical_score_bin'] = pd.qcut(results_df['technical_score'], 4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
    technical_score_performance = {}
    for score_bin in results_df['technical_score_bin'].unique():
        if pd.isna(score_bin):
            continue
        bin_trades = results_df[results_df['technical_score_bin'] == score_bin]
        bin_win_rate = len(bin_trades[bin_trades['return'] > 0]) / len(bin_trades) if len(bin_trades) > 0 else 0
        bin_avg_return = bin_trades['return'].mean() if len(bin_trades) > 0 else 0
        technical_score_performance[score_bin] = {
            'trades': len(bin_trades),
            'win_rate': bin_win_rate,
            'avg_return': bin_avg_return
        }
    
    # Compile all metrics
    metrics = {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'avg_return': avg_return,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'direction_performance': {
            'long': {
                'trades': len(long_trades),
                'win_rate': long_win_rate,
                'avg_return': long_avg_return
            },
            'short': {
                'trades': len(short_trades),
                'win_rate': short_win_rate,
                'avg_return': short_avg_return
            }
        },
        'regime_performance': regime_performance,
        'sector_performance': sector_performance,
        'technical_score_performance': technical_score_performance
    }
    
    # Log key metrics
    logger.info(f"Performance Summary:")
    logger.info(f"Total Trades: {total_trades}, Win Rate: {win_rate:.2%}")
    logger.info(f"Avg Return: {avg_return:.2%}, Profit Factor: {profit_factor:.2f}")
    logger.info(f"Max Drawdown: {max_drawdown:.2%}")
    logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}, Sortino Ratio: {sortino_ratio:.2f}, Calmar Ratio: {calmar_ratio:.2f}")
    
    logger.info(f"Direction Performance:")
    logger.info(f"Long: {len(long_trades)} trades, Win Rate: {long_win_rate:.2%}, Avg Return: {long_avg_return:.2%}")
    logger.info(f"Short: {len(short_trades)} trades, Win Rate: {short_win_rate:.2%}, Avg Return: {short_avg_return:.2%}")
    
    if regime_performance:
        logger.info(f"Regime Performance:")
        for regime, perf in regime_performance.items():
            logger.info(f"{regime}: {perf['trades']} trades, Win Rate: {perf['win_rate']:.2%}, Avg Return: {perf['avg_return']:.2%}")
    
    return metrics

def plot_performance(results_df, output_prefix=''):
    """
    Create performance visualizations
    
    Args:
        results_df: DataFrame with stock selections and returns
        output_prefix: Prefix for output files
    """
    if results_df.empty or 'return' not in results_df.columns:
        logger.warning("Cannot create performance plots: no return data available")
        return
    
    # Filter out rows with NaN returns
    valid_results = results_df.dropna(subset=['return'])
    
    if valid_results.empty:
        logger.warning("Cannot create performance plots: no valid return data available")
        return
    
    # Set the style
    sns.set(style="whitegrid")
    
    # 1. Distribution of returns
    plt.figure(figsize=(12, 6))
    sns.histplot(valid_results['return'], kde=True)
    plt.title('Distribution of Returns')
    plt.xlabel('Return (%)')
    plt.ylabel('Frequency')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.savefig(f'{output_prefix}return_distribution.png')
    plt.close()
    
    # 2. Returns by direction
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='direction', y='return', data=valid_results)
    plt.title('Returns by Direction')
    plt.xlabel('Direction')
    plt.ylabel('Return (%)')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.savefig(f'{output_prefix}returns_by_direction.png')
    plt.close()
    
    # 3. Returns by market regime (if available)
    if 'market_regime' in valid_results.columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='market_regime', y='return', data=valid_results)
        plt.title('Returns by Market Regime')
        plt.xlabel('Market Regime')
        plt.ylabel('Return (%)')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.savefig(f'{output_prefix}returns_by_regime.png')
        plt.close()
    
    # 4. Top performing stocks
    top_stocks = valid_results.groupby('symbol')['return'].mean().sort_values(ascending=False).head(10)
    plt.figure(figsize=(12, 6))
    top_stocks.plot(kind='bar')
    plt.title('Top 10 Performing Stocks (Average Return)')
    plt.xlabel('Symbol')
    plt.ylabel('Average Return (%)')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.savefig(f'{output_prefix}top_performing_stocks.png')
    plt.close()
    
    # 5. Cumulative returns over time
    if 'date' in valid_results.columns:
        # Convert date to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(valid_results['date']):
            valid_results['date'] = pd.to_datetime(valid_results['date'])
        
        # Calculate daily average returns
        daily_returns = valid_results.groupby('date')['return'].mean().reset_index()
        daily_returns.set_index('date', inplace=True)
        
        # Calculate cumulative returns (assuming equal weighting)
        daily_returns['cum_return'] = (1 + daily_returns['return']).cumprod() - 1
        daily_returns['cum_return'] = daily_returns['cum_return'] * 100  # Convert to percentage
        
        plt.figure(figsize=(12, 6))
        daily_returns['cum_return'].plot()
        plt.title('Cumulative Returns Over Time')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return (%)')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.savefig(f'{output_prefix}cumulative_returns.png')
        plt.close()
    
    logger.info("Performance plots created successfully")

def run_backtest(config_file, start_date, end_date, holding_period=5):
    """
    Run a backtest using the specified configuration.
    
    Args:
        config_file (str): Path to the configuration file
        start_date (str): Start date for the backtest (YYYY-MM-DD)
        end_date (str): End date for the backtest (YYYY-MM-DD)
        holding_period (int): Number of days to hold each position
        
    Returns:
        dict: Backtest results
    """
    # Load configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Configure logging
    log_level = getattr(logging, config['general'].get('log_level', 'INFO'))
    logger.setLevel(log_level)
    
    # Get symbols from config
    symbols = config['general']['symbols']
    logger.info(f"Running backtest with {len(symbols)} symbols")
    
    # Get timeframe from config
    timeframe = config['general'].get('timeframe', '1D')
    
    # Load Alpaca credentials
    credentials = load_alpaca_credentials()
    
    # Use paper trading credentials
    api_key = credentials['paper']['api_key']
    api_secret = credentials['paper']['api_secret']
    base_url = credentials['paper']['base_url']
    
    # Initialize API
    api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
    
    # Initialize data loader
    data_loader = AlpacaDataLoader(api)
    
    # Adjust start date to include enough historical data for indicators
    lookback_days = 100  # Enough for most indicators
    adjusted_start_date = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    
    # Load historical data
    logger.info(f"Loading historical data from {adjusted_start_date} to {end_date}")
    symbol_data = load_data_parallel(data_loader, symbols, adjusted_start_date, end_date, timeframe)
    
    # Check if we have data
    if not symbol_data:
        logger.error("No data loaded, exiting")
        return {}
    
    # Log data summary
    data_summary = {symbol: len(df) for symbol, df in symbol_data.items() if df is not None}
    logger.info(f"Loaded data for {len(data_summary)} symbols")
    logger.info(f"Average bars per symbol: {sum(data_summary.values()) / len(data_summary) if data_summary else 0:.1f}")
    
    # Initialize strategy
    strategy = CombinedStrategy(config)
    
    # Set market data
    strategy.set_symbol_data(symbol_data)
    
    # Get trading days
    trading_days = []
    for symbol, df in symbol_data.items():
        if df is not None and not df.empty:
            # Get dates after the start_date (excluding lookback period)
            start_timestamp = pd.Timestamp(start_date)
            logger.info(f"Looking for trading days from {start_timestamp} onwards")
            
            # Check the index type
            logger.info(f"Index type for {symbol}: {type(df.index)}")
            logger.info(f"Sample index values for {symbol}: {df.index[:3]}")
            
            # Get all dates on or after start_date
            dates = [date for date in df.index if pd.Timestamp(date) >= start_timestamp]
            
            logger.info(f"Found {len(dates)} trading days for {symbol}")
            trading_days.extend(dates)
            break
    
    # Sort and deduplicate trading days
    trading_days = sorted(list(set(trading_days)))
    logger.info(f"Total trading days after deduplication: {len(trading_days)}")
    if trading_days:
        logger.info(f"First trading day: {trading_days[0]}, Last trading day: {trading_days[-1]}")
    
    # Initialize results DataFrame
    selections_df = pd.DataFrame()
    
    # Track market regimes
    market_regimes = {}
    
    # Process each trading day
    for day in trading_days:
        current_date = pd.Timestamp(day).to_pydatetime()
        
        # Skip if current_date is after end_date
        if current_date > datetime.strptime(end_date, '%Y-%m-%d'):
            continue
        
        logger.info(f"Processing {current_date.strftime('%Y-%m-%d')}")
        
        # Detect market regime using SPY data
        market_regime = None
        if 'SPY' in symbol_data and symbol_data['SPY'] is not None:
            spy_data = symbol_data['SPY'].loc[:day]
            if len(spy_data) > 20:  # Ensure enough data for regime detection
                market_regime = strategy.detect_market_regime(spy_data)
                market_regimes[day] = market_regime.name
                logger.info(f"Detected market regime: {market_regime.name}")
        
        # Select stocks using multi-factor approach
        top_n = config['stock_selection'].get('top_n_stocks', 10)
        try:
            # Get stock selections with scores
            logger.info(f"Calling select_stocks_multi_factor with {len(symbol_data)} symbols, top_n={top_n}, direction='ANY', market_regime={market_regime}")
            
            # Debug: Check if we can get any technical scores
            technical_scores = {}
            for symbol, df in symbol_data.items():
                if df is None or df.empty:
                    continue
                    
                try:
                    # Get data up to current_date
                    recent_data = df[df.index <= current_date]
                    
                    if len(recent_data) < 20:  # Need at least 20 bars for technical analysis
                        continue
                        
                    # Calculate technical score
                    score = strategy.calculate_technical_score(recent_data)
                    technical_scores[symbol] = score
                except Exception as e:
                    logger.error(f"Error calculating technical score for {symbol}: {e}")
            
            logger.info(f"DEBUG: Calculated technical scores for {len(technical_scores)} symbols")
            if technical_scores:
                logger.info(f"DEBUG: Sample technical scores: {list(technical_scores.items())[:3]}")
            
            selections = strategy.select_stocks_multi_factor(
                symbol_data, 
                current_date=current_date,
                top_n=top_n,
                direction='ANY',  # Allow both long and short
                market_regime=market_regime
            )
            
            if not selections:
                logger.warning(f"No stocks selected for {current_date.strftime('%Y-%m-%d')}")
                continue
            
            # Log selection details
            logger.info(f"Got {len(selections)} selections from multi-factor strategy")
            for i, sel in enumerate(selections[:3]):
                logger.info(f"Selection {i}: {sel}")
            
            # Convert to DataFrame
            day_selections = pd.DataFrame(selections)
            
            # Add date and market regime
            day_selections['date'] = current_date
            if market_regime:
                day_selections['market_regime'] = market_regime.name
            
            # Add sector information if not already present
            if 'sector' not in day_selections.columns:
                day_selections['sector'] = day_selections['symbol'].apply(lambda x: strategy.get_sector_for_symbol(x))
            
            # Append to results
            selections_df = pd.concat([selections_df, day_selections], ignore_index=True)
            
            # Log selections
            logger.info(f"Selected {len(selections)} stocks for {current_date.strftime('%Y-%m-%d')}")
            for i, selection in enumerate(selections[:5]):  # Log top 5
                logger.info(f"  {i+1}. {selection['symbol']} - Score: {selection['combined_score']:.4f}, "
                           f"Direction: {selection['direction']}, "
                           f"Position Size: {selection['position_size']:.1%}")
            
            if len(selections) > 5:
                logger.info(f"  ... and {len(selections) - 5} more")
            
        except Exception as e:
            logger.error(f"Error selecting stocks for {current_date.strftime('%Y-%m-%d')}: {e}")
            continue
    
    # Calculate returns for selected stocks
    results_df = calculate_returns(selections_df, symbol_data, holding_period)
    
    # Analyze performance
    performance = analyze_performance(results_df)
    
    # Plot performance
    try:
        output_prefix = f"{os.path.basename(config_file).split('.')[0]}_{start_date}_{end_date}"
        plot_performance(results_df, output_prefix)
    except Exception as e:
        logger.error(f"Error plotting performance: {e}")
    
    # Calculate performance metrics
    results = {
        'total_selections': len(selections_df),
        'unique_days': len(selections_df['date'].unique()) if 'date' in selections_df.columns else 0,
        'unique_symbols': len(selections_df['symbol'].unique()) if 'symbol' in selections_df.columns else 0,
        'selections_df': selections_df,
        'results_df': results_df,
        'performance': performance,
        'market_regimes': market_regimes,
        'config': config
    }
    
    return results

def main():
    """
    Main function to run the backtest
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run multi-factor strategy backtest')
    parser.add_argument('--config', type=str, default='configuration_enhanced_multi_factor_500.yaml',
                        help='Path to configuration file')
    parser.add_argument('--start_date', type=str, default='2023-01-01',
                        help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2023-03-31',
                        help='End date for backtest (YYYY-MM-DD)')
    parser.add_argument('--holding_period', type=int, default=5,
                        help='Holding period in days')
    args = parser.parse_args()
    
    # Get configuration file path
    config_file = args.config
    
    # Check if configuration file exists
    if not os.path.exists(config_file):
        logger.error(f"Configuration file {config_file} not found")
        return
    
    # Run backtest
    logger.info(f"\n=== Running backtest for period: {args.start_date} to {args.end_date} ===")
    run_backtest(config_file, args.start_date, args.end_date, args.holding_period)

def compile_overall_performance(all_results):
    """
    Compile overall performance metrics across all test periods
    
    Args:
        all_results (dict): Dictionary of period -> results
    """
    if not all_results:
        logger.warning("No results to compile")
        return
    
    # Combine all results DataFrames
    all_selections = []
    all_trades = []
    
    for period, results in all_results.items():
        if 'selections_df' in results and not results['selections_df'].empty:
            period_selections = results['selections_df'].copy()
            period_selections['period'] = period
            all_selections.append(period_selections)
        
        if 'results_df' in results and not results['results_df'].empty:
            period_results = results['results_df'].copy()
            period_results['period'] = period
            all_trades.append(period_results)
    
    if not all_selections or not all_trades:
        logger.warning("No valid results to compile")
        return
    
    # Combine DataFrames
    combined_selections = pd.concat(all_selections, ignore_index=True)
    combined_trades = pd.concat(all_trades, ignore_index=True)
    
    # Analyze overall performance
    overall_performance = analyze_performance(combined_trades)
    
    # Log overall performance
    logger.info("\n=== Overall Performance Across All Periods ===")
    logger.info(f"Total Trades: {overall_performance['total_trades']}")
    logger.info(f"Win Rate: {overall_performance['win_rate']:.2%}")
    logger.info(f"Profit Factor: {overall_performance['profit_factor']:.2f}")
    logger.info(f"Average Return: {overall_performance['avg_return']:.2%}")
    logger.info(f"Sharpe Ratio: {overall_performance['sharpe_ratio']:.2f}")
    
    # Plot overall performance
    try:
        plot_performance(combined_trades, "overall_performance")
    except Exception as e:
        logger.error(f"Error plotting overall performance: {e}")
    
    # Save combined results to CSV for further analysis
    try:
        combined_trades.to_csv("overall_backtest_results.csv", index=False)
        logger.info("Saved overall results to overall_backtest_results.csv")
    except Exception as e:
        logger.error(f"Error saving overall results: {e}")

if __name__ == "__main__":
    main()
