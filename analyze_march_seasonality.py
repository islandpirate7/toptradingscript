#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analyze March Seasonality
------------------------
This script analyzes which stocks have strong seasonality scores for March
and would be selected by the strategy based on seasonality.
"""

import os
import yaml
import logging
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from tabulate import tabulate

from seasonality_enhanced import SeasonalityEnhanced
from combined_strategy import CombinedStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_file):
    """Load configuration from YAML file
    
    Args:
        config_file (str): Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def analyze_seasonality_for_month(month, year=2024, config_file='configuration_combined_strategy_new_stocks.yaml'):
    """Analyze seasonality scores for a specific month
    
    Args:
        month (int): Month to analyze (1-12)
        year (int): Year to analyze
        config_file (str): Path to configuration file
    
    Returns:
        pd.DataFrame: DataFrame with seasonality scores
    """
    # Load configuration
    config = load_config(config_file)
    
    # Ensure seasonality is enabled
    if 'seasonality' not in config:
        config['seasonality'] = {
            'enabled': True,
            'data_file': 'output/seasonal_opportunities_converted.yaml'
        }
    else:
        config['seasonality']['enabled'] = True
    
    # Initialize seasonality analyzer
    seasonality_file = config['seasonality']['data_file']
    logger.info(f"Initializing seasonality analyzer with data file: {seasonality_file}")
    seasonality_analyzer = SeasonalityEnhanced(seasonality_file)
    
    # Get all available symbols from the seasonality data
    symbols = list(seasonality_analyzer.seasonality_data.keys())
    logger.info(f"Found {len(symbols)} symbols in seasonality data")
    
    # Set the date to analyze (first day of the month)
    current_date = dt.datetime(year, month, 1)
    month_name = current_date.strftime('%B')
    logger.info(f"Analyzing seasonality for {month_name} {year}")
    
    # Collect seasonality scores for all symbols
    seasonality_scores = []
    for symbol in symbols:
        score = seasonality_analyzer.get_seasonal_score(symbol, current_date)
        
        # Get detailed data if available for this month
        month_str = str(month)
        detailed_data = {}
        if month_str in seasonality_analyzer.seasonality_data[symbol]:
            data = seasonality_analyzer.seasonality_data[symbol][month_str]
            direction = data.get('direction', 'NEUTRAL')
            detailed_data = {
                'win_rate': data.get('win_rate', 0),
                'avg_return': data.get('avg_return', 0),
                'correlation': data.get('correlation', 0),
                'trade_count': data.get('trade_count', 0)
            }
        else:
            direction = 'NEUTRAL'
        
        seasonality_scores.append({
            'symbol': symbol,
            'score': score,
            'direction': direction,
            'win_rate': detailed_data.get('win_rate', 0),
            'avg_return': detailed_data.get('avg_return', 0),
            'correlation': detailed_data.get('correlation', 0),
            'trade_count': detailed_data.get('trade_count', 0)
        })
    
    # Convert to DataFrame and sort by score
    df = pd.DataFrame(seasonality_scores)
    df = df.sort_values(by='score', ascending=False)
    
    return df

def plot_seasonality_scores(df, month, year=2024, top_n=10):
    """Plot seasonality scores
    
    Args:
        df (pd.DataFrame): DataFrame with seasonality scores
        month (int): Month being analyzed
        year (int): Year being analyzed
        top_n (int): Number of top stocks to highlight
    """
    month_name = dt.datetime(year, month, 1).strftime('%B')
    
    # Filter for top N stocks
    top_df = df.head(top_n)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    # Plot 1: Seasonality Scores
    colors = ['green' if d == 'LONG' else 'red' for d in top_df['direction']]
    ax1.barh(top_df['symbol'], top_df['score'], color=colors)
    ax1.set_title(f'Top {top_n} Stocks by Seasonality Score for {month_name}')
    ax1.set_xlabel('Seasonality Score')
    ax1.set_ylabel('Symbol')
    ax1.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add direction labels
    for i, (score, direction) in enumerate(zip(top_df['score'], top_df['direction'])):
        ax1.text(max(0.01, score + 0.05), i, direction, va='center')
    
    # Plot 2: Win Rate vs. Average Return
    scatter = ax2.scatter(
        top_df['win_rate'] * 100, 
        top_df['avg_return'] * 100,
        s=top_df['correlation'] * 100, 
        c=colors,
        alpha=0.7
    )
    
    # Add symbol labels
    for i, symbol in enumerate(top_df['symbol']):
        ax2.annotate(
            symbol, 
            (top_df['win_rate'].iloc[i] * 100, top_df['avg_return'].iloc[i] * 100),
            xytext=(5, 5),
            textcoords='offset points'
        )
    
    ax2.set_title(f'Win Rate vs. Average Return for {month_name}')
    ax2.set_xlabel('Win Rate (%)')
    ax2.set_ylabel('Average Return (%)')
    ax2.grid(linestyle='--', alpha=0.7)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='LONG'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='SHORT')
    ]
    ax2.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    
    # Save the figure
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.savefig(f'{output_dir}/seasonality_analysis_{month_name.lower()}_{year}.png')
    logger.info(f"Saved seasonality analysis plot to {output_dir}/seasonality_analysis_{month_name.lower()}_{year}.png")
    
    plt.close()

def print_seasonality_table(df, top_n=10):
    """Print seasonality scores in a table format
    
    Args:
        df (pd.DataFrame): DataFrame with seasonality scores
        top_n (int): Number of top stocks to display
    """
    # Format the data for tabulate
    table_data = []
    for _, row in df.head(top_n).iterrows():
        table_data.append([
            row['symbol'],
            f"{row['score']:.2f}",
            row['direction'],
            f"{row['win_rate']*100:.1f}%",
            f"{row['avg_return']*100:.2f}%",
            f"{row['correlation']:.2f}",
            int(row['trade_count']) if not np.isnan(row['trade_count']) else 0
        ])
    
    # Print the table
    headers = ['Symbol', 'Score', 'Direction', 'Win Rate', 'Avg Return', 'Correlation', 'Trade Count']
    print("\nTop Stocks by Seasonality Score:")
    print(tabulate(table_data, headers=headers, tablefmt='grid'))

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze seasonality for a specific month')
    parser.add_argument('--month', type=int, default=3,
                        help='Month to analyze (1-12)')
    parser.add_argument('--year', type=int, default=2024,
                        help='Year to analyze')
    parser.add_argument('--config', type=str, default='configuration_combined_strategy_new_stocks.yaml',
                        help='Path to configuration file')
    parser.add_argument('--top_n', type=int, default=10,
                        help='Number of top stocks to display')
    
    args = parser.parse_args()
    
    # Analyze seasonality
    df = analyze_seasonality_for_month(args.month, args.year, args.config)
    
    # Print results
    print_seasonality_table(df, args.top_n)
    
    # Plot results
    plot_seasonality_scores(df, args.month, args.year, args.top_n)
    
    return df

if __name__ == "__main__":
    main()
