#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run Seasonal Backtest
-------------------------------------
This script runs backtests for the combined strategy with seasonality integration
to evaluate performance across different time periods.
"""

import os
import logging
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# Import our backtest module
from backtest_combined_strategy import Backtester, BacktestResults, run_quarterly_backtests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_seasonal_backtest(config_file: str = 'configuration_combined_strategy_seasonal.yaml',
                         output_dir: str = 'output/seasonal_backtest'):
    """Run backtest for the combined strategy with seasonality
    
    Args:
        config_file (str, optional): Path to configuration file. 
                                    Defaults to 'configuration_combined_strategy_seasonal.yaml'.
        output_dir (str, optional): Output directory. Defaults to 'output/seasonal_backtest'.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize backtester with the config file directly
    # The Backtester class will handle timeframe conversion internally
    backtester = Backtester(config_file)
    
    # Run backtest
    results = backtester.run_backtest()
    
    # Generate report
    report_path = os.path.join(output_dir, 'seasonal_backtest_report.html')
    results.generate_report(report_path)
    
    # Save results
    results_path = os.path.join(output_dir, 'seasonal_backtest_results.json')
    results.save_results(results_path)
    
    # Plot equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(results.equity_curve.index, results.equity_curve.values)
    plt.title('Equity Curve - Combined Strategy with Seasonality')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'seasonal_equity_curve.png'))
    
    logging.info(f"Backtest completed. Results saved to {output_dir}")
    
    return results

def compare_strategies(seasonal_config: str = 'configuration_combined_strategy_seasonal.yaml',
                      base_config: str = 'configuration_mean_reversion_final.yaml',
                      output_dir: str = 'output/strategy_comparison'):
    """Compare performance of seasonal and base strategies
    
    Args:
        seasonal_config (str, optional): Path to seasonal configuration file. 
                                        Defaults to 'configuration_combined_strategy_seasonal.yaml'.
        base_config (str, optional): Path to base configuration file.
                                    Defaults to 'configuration_mean_reversion_final.yaml'.
        output_dir (str, optional): Output directory. Defaults to 'output/strategy_comparison'.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize backtester for seasonal strategy directly with the config file
    # The Backtester class will handle timeframe conversion internally
    seasonal_backtester = Backtester(seasonal_config)
    
    # Initialize backtester for base strategy directly with the config file
    # The Backtester class will handle timeframe conversion internally
    base_backtester = Backtester(base_config)
    
    # Run backtests
    seasonal_results = seasonal_backtester.run_backtest()
    base_results = base_backtester.run_backtest()
    
    # Compare metrics
    metrics_comparison = pd.DataFrame({
        'Metric': list(seasonal_results.metrics.keys()),
        'Seasonal Strategy': list(seasonal_results.metrics.values()),
        'Base Strategy': list(base_results.metrics.values())
    })
    
    # Save metrics comparison
    metrics_path = os.path.join(output_dir, 'metrics_comparison.csv')
    metrics_comparison.to_csv(metrics_path, index=False)
    
    # Plot equity curves
    plt.figure(figsize=(12, 6))
    plt.plot(seasonal_results.equity_curve.index, seasonal_results.equity_curve.values, label='Seasonal Strategy')
    plt.plot(base_results.equity_curve.index, base_results.equity_curve.values, label='Base Strategy')
    plt.title('Equity Curve Comparison')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'equity_curve_comparison.png'))
    
    # Calculate monthly returns
    seasonal_monthly = seasonal_results.equity_curve.resample('M').last().pct_change().dropna()
    base_monthly = base_results.equity_curve.resample('M').last().pct_change().dropna()
    
    # Plot monthly returns comparison
    plt.figure(figsize=(12, 6))
    plt.bar(seasonal_monthly.index.strftime('%Y-%m'), seasonal_monthly.values, width=0.4, alpha=0.7, label='Seasonal Strategy')
    plt.bar(base_monthly.index.strftime('%Y-%m'), base_monthly.values, width=0.4, alpha=0.7, label='Base Strategy')
    plt.title('Monthly Returns Comparison')
    plt.xlabel('Month')
    plt.ylabel('Return (%)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(output_dir, 'monthly_returns_comparison.png'))
    
    logging.info(f"Strategy comparison completed. Results saved to {output_dir}")
    
    return seasonal_results, base_results

def run_quarterly_seasonal_backtests(config_file: str = 'configuration_combined_strategy_seasonal.yaml',
                                    output_dir: str = 'output/quarterly_backtests',
                                    year: int = 2023):
    """Run quarterly backtests for the combined strategy with seasonality
    
    Args:
        config_file (str, optional): Path to configuration file. 
                                    Defaults to 'configuration_combined_strategy_seasonal.yaml'.
        output_dir (str, optional): Output directory. Defaults to 'output/quarterly_backtests'.
        year (int, optional): Year to run quarterly backtests for. Defaults to 2023.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Define quarters
    quarters = [
        (f"{year}-01-01", f"{year}-03-31", "Q1"),
        (f"{year}-04-01", f"{year}-06-30", "Q2"),
        (f"{year}-07-01", f"{year}-09-30", "Q3"),
        (f"{year}-10-01", f"{year}-12-31", "Q4")
    ]
    
    # Run backtests for each quarter
    results = {}
    for start_date, end_date, quarter in quarters:
        # Update config with quarter dates
        config['general']['backtest_start_date'] = start_date
        config['general']['backtest_end_date'] = end_date
        
        # Write updated config to a temporary file
        temp_config_file = os.path.join(output_dir, f'temp_config_{quarter}.yaml')
        with open(temp_config_file, 'w') as f:
            yaml.dump(config, f)
        
        # Run backtest
        logging.info(f"Running backtest for {quarter} {year}")
        backtester = Backtester(temp_config_file)
        quarter_results = backtester.run_backtest()
        
        # Save results
        quarter_output_dir = os.path.join(output_dir, f"{year}_{quarter}")
        os.makedirs(quarter_output_dir, exist_ok=True)
        
        # Generate report
        report_path = os.path.join(quarter_output_dir, f'backtest_report_{quarter}.html')
        quarter_results.generate_report(report_path)
        
        # Save results
        results_path = os.path.join(quarter_output_dir, f'backtest_results_{quarter}.json')
        quarter_results.save_results(results_path)
        
        # Plot equity curve
        plt.figure(figsize=(12, 6))
        plt.plot(quarter_results.equity_curve.index, quarter_results.equity_curve.values)
        plt.title(f'Equity Curve - {quarter} {year}')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        plt.savefig(os.path.join(quarter_output_dir, f'equity_curve_{quarter}.png'))
        
        # Store results
        results[quarter] = quarter_results
    
    # Compare quarterly performance
    metrics = ['total_return', 'annualized_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
    comparison = pd.DataFrame(index=metrics, columns=[q for _, _, q in quarters])
    
    for _, _, quarter in quarters:
        for metric in metrics:
            comparison.loc[metric, quarter] = results[quarter].metrics.get(metric, 0)
    
    # Save comparison
    comparison_path = os.path.join(output_dir, f'quarterly_comparison_{year}.csv')
    comparison.to_csv(comparison_path)
    
    # Plot comparison
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        plt.bar(comparison.columns, comparison.loc[metric])
        plt.title(f'{metric.replace("_", " ").title()} by Quarter')
        plt.xlabel('Quarter')
        plt.ylabel(metric.replace("_", " ").title())
        plt.grid(True, axis='y')
        plt.savefig(os.path.join(output_dir, f'{metric}_by_quarter.png'))
    
    logging.info(f"Quarterly backtests completed. Results saved to {output_dir}")
    
    return results

def main():
    """Main function to run the script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run seasonal backtest')
    parser.add_argument('--config', type=str, default='configuration_combined_strategy_seasonal.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output', type=str, default='output/seasonal_backtest',
                        help='Output directory')
    parser.add_argument('--compare', action='store_true',
                        help='Compare with base strategy')
    parser.add_argument('--base', type=str, default='configuration_mean_reversion_final.yaml',
                        help='Path to base configuration file for comparison')
    parser.add_argument('--quarterly', action='store_true',
                        help='Run quarterly backtests')
    parser.add_argument('--year', type=int, default=2023,
                        help='Year for quarterly backtests')
    
    args = parser.parse_args()
    
    if args.quarterly:
        run_quarterly_seasonal_backtests(args.config, args.output, args.year)
    elif args.compare:
        compare_strategies(args.config, args.base, args.output)
    else:
        run_seasonal_backtest(args.config, args.output)

if __name__ == "__main__":
    main()
