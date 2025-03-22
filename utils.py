#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for the trading system.
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

def setup_logging(log_level=logging.INFO):
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (default: INFO)
    """
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_config(config_file):
    """
    Load configuration from a YAML file.
    
    Args:
        config_file (str): Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        return None

def calculate_performance_metrics(results):
    """
    Calculate performance metrics from backtest results.
    
    Args:
        results (pd.DataFrame): Backtest results
        
    Returns:
        dict: Dictionary of performance metrics
    """
    # Check if results is empty or None
    if results is None or results.empty:
        logging.warning("No results available for performance calculation")
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'trade_count': 0
        }
    
    # Check if required columns exist
    required_columns = ['timestamp', 'portfolio_value']
    if not all(col in results.columns for col in required_columns):
        logging.warning(f"Results missing required columns: {required_columns}")
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'trade_count': 0
        }
    
    # Filter to get unique portfolio values per day
    portfolio_values = results.drop_duplicates('timestamp')[['timestamp', 'portfolio_value']]
    
    if len(portfolio_values) < 2:
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'trade_count': 0
        }
    
    # Calculate returns
    portfolio_values['prev_value'] = portfolio_values['portfolio_value'].shift(1)
    portfolio_values['daily_return'] = (portfolio_values['portfolio_value'] - portfolio_values['prev_value']) / portfolio_values['prev_value']
    
    # Calculate cumulative returns
    initial_value = portfolio_values.iloc[0]['portfolio_value']
    final_value = portfolio_values.iloc[-1]['portfolio_value']
    
    total_return = (final_value - initial_value) / initial_value * 100
    
    # Calculate annualized return
    days = (portfolio_values.iloc[-1]['timestamp'] - portfolio_values.iloc[0]['timestamp']).days
    if days > 0:
        annualized_return = ((final_value / initial_value) ** (365 / days) - 1) * 100
    else:
        annualized_return = 0.0
    
    # Calculate Sharpe ratio (assuming risk-free rate of 0)
    daily_returns = portfolio_values['daily_return'].dropna()
    if len(daily_returns) > 0 and daily_returns.std() > 0:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)  # Annualized
    else:
        sharpe_ratio = 0.0
    
    # Calculate maximum drawdown
    portfolio_values['cumulative_max'] = portfolio_values['portfolio_value'].cummax()
    portfolio_values['drawdown'] = (portfolio_values['portfolio_value'] - portfolio_values['cumulative_max']) / portfolio_values['cumulative_max'] * 100
    max_drawdown = portfolio_values['drawdown'].min()
    
    # Calculate win rate and profit factor from trades
    trades = results[results['action'].isin(['BUY', 'SELL'])]
    
    if len(trades) > 0:
        winning_trades = trades[trades['pnl'] > 0]
        losing_trades = trades[trades['pnl'] < 0]
        
        win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0.0
        
        gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0.0
        gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0.0
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    else:
        win_rate = 0.0
        profit_factor = 0.0
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'trade_count': len(trades) // 2  # Divide by 2 because each round trip is 2 trades (buy + sell)
    }

def analyze_trades_by_symbol(results):
    """
    Analyze trades grouped by symbol.
    
    Args:
        results (pd.DataFrame): Backtest results
        
    Returns:
        pd.DataFrame: Trade analysis by symbol
    """
    trades = results[results['action'].isin(['BUY', 'SELL'])]
    
    if len(trades) == 0:
        return pd.DataFrame()
    
    # Group by symbol
    symbol_groups = trades.groupby('symbol')
    
    symbol_metrics = []
    
    for symbol, group in symbol_groups:
        # Calculate metrics
        trades_count = len(group)
        winning_trades = group[group['pnl'] > 0]
        losing_trades = group[group['pnl'] < 0]
        
        win_rate = len(winning_trades) / trades_count if trades_count > 0 else 0.0
        
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0.0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0.0
        
        gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0.0
        gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0.0
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        symbol_metrics.append({
            'symbol': symbol,
            'trades_count': trades_count,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'net_profit': gross_profit - gross_loss,
            'profit_factor': profit_factor
        })
    
    return pd.DataFrame(symbol_metrics)

def get_monthly_returns(results):
    """
    Calculate monthly returns from backtest results.
    
    Args:
        results (pd.DataFrame): Backtest results
        
    Returns:
        pd.DataFrame: Monthly returns
    """
    # Filter to get unique portfolio values per day
    portfolio_values = results.drop_duplicates('timestamp')[['timestamp', 'portfolio_value']]
    
    if len(portfolio_values) < 2:
        return pd.DataFrame()
    
    # Convert timestamp to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(portfolio_values['timestamp']):
        portfolio_values['timestamp'] = pd.to_datetime(portfolio_values['timestamp'])
    
    # Add month column
    portfolio_values['year_month'] = portfolio_values['timestamp'].dt.strftime('%Y-%m')
    
    # Get month start and end values
    monthly_start = portfolio_values.groupby('year_month')['portfolio_value'].first()
    monthly_end = portfolio_values.groupby('year_month')['portfolio_value'].last()
    
    # Calculate monthly returns
    monthly_returns = (monthly_end - monthly_start) / monthly_start * 100
    
    # Convert to DataFrame
    monthly_returns_df = pd.DataFrame({
        'year_month': monthly_returns.index,
        'return_pct': monthly_returns.values
    })
    
    return monthly_returns_df
