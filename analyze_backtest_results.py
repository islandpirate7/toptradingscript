#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analyze Backtest Results
This script analyzes the backtest results and generates visualizations
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_latest_backtest_file():
    """Find the latest backtest results file"""
    files = [f for f in os.listdir() if f.startswith('backtest_signals_') and f.endswith('.csv')]
    if not files:
        logger.error("No backtest results files found")
        return None
    
    # Sort by modification time
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return files[0]

def analyze_backtest_results(file_path=None):
    """Analyze backtest results"""
    if file_path is None:
        file_path = find_latest_backtest_file()
        if file_path is None:
            return
    
    logger.info(f"Analyzing backtest results from {file_path}")
    
    # Load results
    results = pd.read_csv(file_path)
    
    # Basic statistics
    total_signals = len(results)
    long_signals = len(results[results['direction'] == 'LONG'])
    short_signals = len(results[results['direction'] == 'SHORT'])
    
    avg_score = results['score'].mean()
    avg_position_size = results['position_size'].mean()
    total_position_size = results['position_size'].sum()
    
    logger.info(f"Total signals: {total_signals}")
    logger.info(f"LONG signals: {long_signals} ({long_signals/total_signals*100:.2f}%)")
    logger.info(f"SHORT signals: {short_signals} ({short_signals/total_signals*100:.2f}%)")
    logger.info(f"Average score: {avg_score:.4f}")
    logger.info(f"Average position size: ${avg_position_size:.2f}")
    logger.info(f"Total position size: ${total_position_size:.2f}")
    
    # Create visualizations directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    # Plot distribution of scores
    plt.figure(figsize=(10, 6))
    sns.histplot(results['score'], bins=20, kde=True)
    plt.title('Distribution of Signal Scores')
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.savefig('visualizations/score_distribution.png')
    
    # Plot direction distribution
    plt.figure(figsize=(8, 8))
    direction_counts = results['direction'].value_counts()
    plt.pie(direction_counts, labels=direction_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Trade Direction Distribution')
    plt.savefig('visualizations/direction_distribution.png')
    
    # Plot position sizes by direction
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='direction', y='position_size', data=results)
    plt.title('Position Sizes by Direction')
    plt.xlabel('Direction')
    plt.ylabel('Position Size ($)')
    plt.savefig('visualizations/position_size_by_direction.png')
    
    # Plot score vs position size
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='score', y='position_size', hue='direction', data=results)
    plt.title('Score vs Position Size')
    plt.xlabel('Score')
    plt.ylabel('Position Size ($)')
    plt.savefig('visualizations/score_vs_position_size.png')
    
    # Analyze score ranges
    score_ranges = [
        (0.5, 0.6),
        (0.6, 0.7),
        (0.7, 0.8),
        (0.8, 0.9),
        (0.9, 1.0)
    ]
    
    range_counts = []
    for low, high in score_ranges:
        count = len(results[(results['score'] >= low) & (results['score'] < high)])
        range_counts.append({
            'range': f"{low:.1f}-{high:.1f}",
            'count': count,
            'percentage': count / total_signals * 100
        })
    
    range_df = pd.DataFrame(range_counts)
    logger.info("\nScore Range Distribution:")
    logger.info(range_df.to_string(index=False))
    
    # Plot score range distribution
    plt.figure(figsize=(10, 6))
    sns.barplot(x='range', y='count', data=range_df)
    plt.title('Score Range Distribution')
    plt.xlabel('Score Range')
    plt.ylabel('Count')
    plt.savefig('visualizations/score_range_distribution.png')
    
    # Generate HTML report
    html_report = f"""
    <html>
    <head>
        <title>Backtest Results Analysis</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #2c3e50; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .stats {{ display: flex; flex-wrap: wrap; margin-bottom: 20px; }}
            .stat-box {{ background-color: #f8f9fa; border-radius: 5px; padding: 15px; margin: 10px; flex: 1; min-width: 200px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            .stat-value {{ font-size: 24px; font-weight: bold; color: #3498db; }}
            .stat-label {{ font-size: 14px; color: #7f8c8d; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            .image-container {{ display: flex; flex-wrap: wrap; justify-content: center; }}
            .image-box {{ margin: 10px; }}
            img {{ max-width: 100%; height: auto; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Backtest Results Analysis</h1>
            <p>Analysis of {file_path} generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Summary Statistics</h2>
            <div class="stats">
                <div class="stat-box">
                    <div class="stat-value">{total_signals}</div>
                    <div class="stat-label">Total Signals</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{long_signals} ({long_signals/total_signals*100:.1f}%)</div>
                    <div class="stat-label">LONG Signals</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{short_signals} ({short_signals/total_signals*100:.1f}%)</div>
                    <div class="stat-label">SHORT Signals</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">${avg_position_size:.2f}</div>
                    <div class="stat-label">Avg Position Size</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">${total_position_size:.2f}</div>
                    <div class="stat-label">Total Position Size</div>
                </div>
            </div>
            
            <h2>Score Range Distribution</h2>
            <table>
                <tr>
                    <th>Score Range</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
    """
    
    for _, row in range_df.iterrows():
        html_report += f"""
                <tr>
                    <td>{row['range']}</td>
                    <td>{row['count']}</td>
                    <td>{row['percentage']:.1f}%</td>
                </tr>
        """
    
    html_report += """
            </table>
            
            <h2>Visualizations</h2>
            <div class="image-container">
                <div class="image-box">
                    <img src="score_distribution.png" alt="Score Distribution">
                    <p>Distribution of Signal Scores</p>
                </div>
                <div class="image-box">
                    <img src="direction_distribution.png" alt="Direction Distribution">
                    <p>Trade Direction Distribution</p>
                </div>
                <div class="image-box">
                    <img src="position_size_by_direction.png" alt="Position Size by Direction">
                    <p>Position Sizes by Direction</p>
                </div>
                <div class="image-box">
                    <img src="score_vs_position_size.png" alt="Score vs Position Size">
                    <p>Score vs Position Size</p>
                </div>
                <div class="image-box">
                    <img src="score_range_distribution.png" alt="Score Range Distribution">
                    <p>Score Range Distribution</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open('visualizations/backtest_analysis.html', 'w') as f:
        f.write(html_report)
    
    logger.info(f"Analysis complete. Report saved to visualizations/backtest_analysis.html")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze Backtest Results')
    parser.add_argument('--file', type=str, help='Path to backtest results file')
    
    args = parser.parse_args()
    
    analyze_backtest_results(args.file)
