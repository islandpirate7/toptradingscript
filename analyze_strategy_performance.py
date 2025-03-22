#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to analyze and compare performance of different trading strategies.
This script loads backtest results from CSV files and generates comparative visualizations.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import glob
import json
import yaml
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def load_backtest_results(file_path):
    """
    Load backtest results from a CSV file
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame: Loaded results
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} records from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

def calculate_performance_metrics(results_df):
    """
    Calculate performance metrics for a results DataFrame
    
    Args:
        results_df: DataFrame with backtest results
        
    Returns:
        dict: Performance metrics
    """
    if results_df.empty or 'return' not in results_df.columns:
        logger.warning("Cannot calculate metrics: no return data available")
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
    
    return {
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
        'annualized_return': annualized_return
    }

def compare_strategies(results_files, strategy_names=None):
    """
    Compare performance of different strategies
    
    Args:
        results_files: List of paths to results CSV files
        strategy_names: List of strategy names (optional)
        
    Returns:
        dict: Comparative performance metrics
    """
    if not results_files:
        logger.warning("No results files provided")
        return {}
    
    # Use filenames as strategy names if not provided
    if not strategy_names:
        strategy_names = [os.path.basename(f).split('.')[0] for f in results_files]
    
    # Load results
    results = {}
    for i, file_path in enumerate(results_files):
        name = strategy_names[i] if i < len(strategy_names) else f"Strategy {i+1}"
        df = load_backtest_results(file_path)
        if not df.empty:
            results[name] = df
    
    if not results:
        logger.warning("No valid results loaded")
        return {}
    
    # Calculate metrics for each strategy
    metrics = {}
    for name, df in results.items():
        metrics[name] = calculate_performance_metrics(df)
    
    # Calculate equity curves
    equity_curves = {}
    for name, df in results.items():
        if 'date' in df.columns and 'return' in df.columns:
            # Calculate daily returns
            daily_returns = df.groupby('date')['return'].mean().reset_index()
            daily_returns.set_index('date', inplace=True)
            daily_returns.index = pd.to_datetime(daily_returns.index)
            
            # Calculate equity curve
            equity_curve = (1 + daily_returns['return']).cumprod()
            equity_curves[name] = equity_curve
    
    return {
        'metrics': metrics,
        'equity_curves': equity_curves,
        'results': results
    }

def plot_comparative_metrics(comparison_results, output_prefix='strategy_comparison'):
    """
    Create comparative performance visualizations
    
    Args:
        comparison_results: Dict with comparative metrics
        output_prefix: Prefix for output files
    """
    if not comparison_results or 'metrics' not in comparison_results:
        logger.warning("No metrics to plot")
        return
    
    metrics = comparison_results['metrics']
    
    # Create a DataFrame for easy plotting
    metrics_df = pd.DataFrame(metrics).T
    
    # 1. Win Rate Comparison
    plt.figure(figsize=(12, 6))
    ax = metrics_df['win_rate'].sort_values().plot(kind='barh')
    plt.title('Win Rate Comparison')
    plt.xlabel('Win Rate')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, v in enumerate(metrics_df['win_rate'].sort_values()):
        ax.text(v + 0.01, i, f'{v:.2%}', va='center')
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_win_rate.png")
    plt.close()
    
    # 2. Average Return Comparison
    plt.figure(figsize=(12, 6))
    ax = metrics_df['avg_return'].sort_values().plot(kind='barh')
    plt.title('Average Return Comparison')
    plt.xlabel('Average Return')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, v in enumerate(metrics_df['avg_return'].sort_values()):
        ax.text(v + 0.001, i, f'{v:.2%}', va='center')
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_avg_return.png")
    plt.close()
    
    # 3. Profit Factor Comparison
    plt.figure(figsize=(12, 6))
    ax = metrics_df['profit_factor'].sort_values().plot(kind='barh')
    plt.title('Profit Factor Comparison')
    plt.xlabel('Profit Factor')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, v in enumerate(metrics_df['profit_factor'].sort_values()):
        ax.text(v + 0.1, i, f'{v:.2f}', va='center')
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_profit_factor.png")
    plt.close()
    
    # 4. Sharpe Ratio Comparison
    plt.figure(figsize=(12, 6))
    ax = metrics_df['sharpe_ratio'].sort_values().plot(kind='barh')
    plt.title('Sharpe Ratio Comparison')
    plt.xlabel('Sharpe Ratio')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, v in enumerate(metrics_df['sharpe_ratio'].sort_values()):
        ax.text(v + 0.1, i, f'{v:.2f}', va='center')
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_sharpe_ratio.png")
    plt.close()
    
    # 5. Equity Curves Comparison
    if 'equity_curves' in comparison_results and comparison_results['equity_curves']:
        plt.figure(figsize=(12, 6))
        
        for name, curve in comparison_results['equity_curves'].items():
            plt.plot(curve, label=name)
        
        plt.title('Equity Curves Comparison')
        plt.xlabel('Date')
        plt.ylabel('Equity (Starting at 1.0)')
        plt.grid(linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_equity_curves.png")
        plt.close()
    
    # 6. Summary Table
    plt.figure(figsize=(14, len(metrics) * 0.8))
    
    # Select key metrics for the table
    table_metrics = metrics_df[['total_trades', 'win_rate', 'avg_return', 'profit_factor', 'sharpe_ratio', 'max_drawdown']].copy()
    
    # Format metrics for display
    table_metrics['win_rate'] = table_metrics['win_rate'].apply(lambda x: f"{x:.2%}")
    table_metrics['avg_return'] = table_metrics['avg_return'].apply(lambda x: f"{x:.2%}")
    table_metrics['profit_factor'] = table_metrics['profit_factor'].apply(lambda x: f"{x:.2f}")
    table_metrics['sharpe_ratio'] = table_metrics['sharpe_ratio'].apply(lambda x: f"{x:.2f}")
    table_metrics['max_drawdown'] = table_metrics['max_drawdown'].apply(lambda x: f"{x:.2%}")
    
    # Rename columns for display
    table_metrics.columns = ['Total Trades', 'Win Rate', 'Avg Return', 'Profit Factor', 'Sharpe Ratio', 'Max Drawdown']
    
    # Create table
    plt.axis('off')
    table = plt.table(
        cellText=table_metrics.values,
        rowLabels=table_metrics.index,
        colLabels=table_metrics.columns,
        loc='center',
        cellLoc='center',
        colWidths=[0.12] * len(table_metrics.columns)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    plt.title('Strategy Performance Summary', y=0.9)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_summary_table.png", bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved comparative plots with prefix: {output_prefix}")

def analyze_market_regime_performance(results_df, output_prefix='regime_analysis'):
    """
    Analyze performance across different market regimes
    
    Args:
        results_df: DataFrame with backtest results
        output_prefix: Prefix for output files
    """
    if results_df.empty or 'market_regime' not in results_df.columns or 'return' not in results_df.columns:
        logger.warning("Cannot analyze market regimes: missing required columns")
        return
    
    # Group by market regime
    regime_performance = {}
    for regime in results_df['market_regime'].unique():
        if pd.isna(regime):
            continue
            
        regime_df = results_df[results_df['market_regime'] == regime]
        regime_performance[regime] = calculate_performance_metrics(regime_df)
    
    # Create a DataFrame for easy plotting
    regime_df = pd.DataFrame(regime_performance).T
    
    # 1. Win Rate by Regime
    plt.figure(figsize=(10, 6))
    ax = regime_df['win_rate'].sort_values().plot(kind='bar')
    plt.title('Win Rate by Market Regime')
    plt.xlabel('Market Regime')
    plt.ylabel('Win Rate')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, v in enumerate(regime_df['win_rate']):
        ax.text(i, v + 0.01, f'{v:.2%}', ha='center')
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_win_rate_by_regime.png")
    plt.close()
    
    # 2. Average Return by Regime
    plt.figure(figsize=(10, 6))
    ax = regime_df['avg_return'].sort_values().plot(kind='bar')
    plt.title('Average Return by Market Regime')
    plt.xlabel('Market Regime')
    plt.ylabel('Average Return')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, v in enumerate(regime_df['avg_return']):
        ax.text(i, v + 0.001, f'{v:.2%}', ha='center')
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_avg_return_by_regime.png")
    plt.close()
    
    # 3. Trade Distribution by Regime
    plt.figure(figsize=(10, 6))
    ax = regime_df['total_trades'].plot(kind='pie', autopct='%1.1f%%')
    plt.title('Trade Distribution by Market Regime')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_trade_distribution_by_regime.png")
    plt.close()
    
    # 4. Summary Table
    plt.figure(figsize=(12, len(regime_performance) * 0.8))
    
    # Select key metrics for the table
    table_metrics = regime_df[['total_trades', 'win_rate', 'avg_return', 'profit_factor']].copy()
    
    # Format metrics for display
    table_metrics['win_rate'] = table_metrics['win_rate'].apply(lambda x: f"{x:.2%}")
    table_metrics['avg_return'] = table_metrics['avg_return'].apply(lambda x: f"{x:.2%}")
    table_metrics['profit_factor'] = table_metrics['profit_factor'].apply(lambda x: f"{x:.2f}")
    
    # Rename columns for display
    table_metrics.columns = ['Total Trades', 'Win Rate', 'Avg Return', 'Profit Factor']
    
    # Create table
    plt.axis('off')
    table = plt.table(
        cellText=table_metrics.values,
        rowLabels=table_metrics.index,
        colLabels=table_metrics.columns,
        loc='center',
        cellLoc='center',
        colWidths=[0.15] * len(table_metrics.columns)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    plt.title('Performance by Market Regime', y=0.9)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_regime_summary_table.png", bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved market regime analysis plots with prefix: {output_prefix}")

def analyze_sector_performance(results_df, output_prefix='sector_analysis'):
    """
    Analyze performance across different sectors
    
    Args:
        results_df: DataFrame with backtest results
        output_prefix: Prefix for output files
    """
    if results_df.empty or 'sector' not in results_df.columns or 'return' not in results_df.columns:
        logger.warning("Cannot analyze sectors: missing required columns")
        return
    
    # Group by sector
    sector_performance = {}
    for sector in results_df['sector'].unique():
        if pd.isna(sector):
            continue
            
        sector_df = results_df[results_df['sector'] == sector]
        if len(sector_df) < 5:  # Skip sectors with too few trades
            continue
            
        sector_performance[sector] = calculate_performance_metrics(sector_df)
    
    if not sector_performance:
        logger.warning("No valid sectors to analyze")
        return
    
    # Create a DataFrame for easy plotting
    sector_df = pd.DataFrame(sector_performance).T
    
    # 1. Win Rate by Sector
    plt.figure(figsize=(12, 8))
    ax = sector_df['win_rate'].sort_values().plot(kind='barh')
    plt.title('Win Rate by Sector')
    plt.xlabel('Win Rate')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, v in enumerate(sector_df['win_rate'].sort_values()):
        ax.text(v + 0.01, i, f'{v:.2%}', va='center')
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_win_rate_by_sector.png")
    plt.close()
    
    # 2. Average Return by Sector
    plt.figure(figsize=(12, 8))
    ax = sector_df['avg_return'].sort_values().plot(kind='barh')
    plt.title('Average Return by Sector')
    plt.xlabel('Average Return')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, v in enumerate(sector_df['avg_return'].sort_values()):
        ax.text(v + 0.001, i, f'{v:.2%}', va='center')
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_avg_return_by_sector.png")
    plt.close()
    
    # 3. Trade Count by Sector
    plt.figure(figsize=(12, 8))
    ax = sector_df['total_trades'].sort_values().plot(kind='barh')
    plt.title('Trade Count by Sector')
    plt.xlabel('Number of Trades')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, v in enumerate(sector_df['total_trades'].sort_values()):
        ax.text(v + 1, i, str(int(v)), va='center')
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_trade_count_by_sector.png")
    plt.close()
    
    # 4. Summary Table
    plt.figure(figsize=(14, len(sector_performance) * 0.6))
    
    # Select key metrics for the table
    table_metrics = sector_df[['total_trades', 'win_rate', 'avg_return', 'profit_factor']].copy()
    
    # Format metrics for display
    table_metrics['win_rate'] = table_metrics['win_rate'].apply(lambda x: f"{x:.2%}")
    table_metrics['avg_return'] = table_metrics['avg_return'].apply(lambda x: f"{x:.2%}")
    table_metrics['profit_factor'] = table_metrics['profit_factor'].apply(lambda x: f"{x:.2f}")
    
    # Rename columns for display
    table_metrics.columns = ['Total Trades', 'Win Rate', 'Avg Return', 'Profit Factor']
    
    # Create table
    plt.axis('off')
    table = plt.table(
        cellText=table_metrics.values,
        rowLabels=table_metrics.index,
        colLabels=table_metrics.columns,
        loc='center',
        cellLoc='center',
        colWidths=[0.15] * len(table_metrics.columns)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    plt.title('Performance by Sector', y=0.95)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_sector_summary_table.png", bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved sector analysis plots with prefix: {output_prefix}")

def main():
    """Main function to analyze strategy performance"""
    # Find all backtest result files
    result_files = glob.glob("overall_backtest_results*.csv")
    
    if not result_files:
        logger.warning("No backtest result files found")
        return
    
    logger.info(f"Found {len(result_files)} backtest result files")
    
    # Load the most recent overall results file
    if "overall_backtest_results.csv" in result_files:
        results_df = load_backtest_results("overall_backtest_results.csv")
        
        # Analyze market regime performance
        analyze_market_regime_performance(results_df)
        
        # Analyze sector performance
        analyze_sector_performance(results_df)
    
    # Compare different strategy versions if multiple result files exist
    if len(result_files) > 1:
        # Extract strategy names from filenames
        strategy_names = [os.path.basename(f).split('.')[0] for f in result_files]
        
        # Compare strategies
        comparison_results = compare_strategies(result_files, strategy_names)
        
        # Plot comparative metrics
        plot_comparative_metrics(comparison_results)
    
    logger.info("Analysis complete")

if __name__ == "__main__":
    main()
