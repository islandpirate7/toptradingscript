import os
import pandas as pd
import alpaca_trade_api as tradeapi
from final_sp500_strategy import run_backtest
from datetime import datetime
import yaml
import logging
import argparse
import sys
import traceback
import json
import multiprocessing
import numpy as np

# Initialize logger
logger = logging.getLogger(__name__)

def display_performance_metrics(summary):
    """Display performance metrics from a backtest summary"""
    if not summary:
        print("No performance metrics available")
        return
    
    print("\n===== PERFORMANCE METRICS =====")
    print(f"Win Rate: {summary.get('win_rate', 0):.2f}%")
    print(f"Profit Factor: {summary.get('profit_factor', 0):.2f}")
    print(f"Average Win: ${summary.get('avg_win', 0):.2f}")
    print(f"Average Loss: ${summary.get('avg_loss', 0):.2f}")
    print(f"Average Holding Period: {summary.get('avg_holding_period', 0):.1f} days")
    
    # Check if tier_metrics is available and has long/short win rates
    if 'tier_metrics' in summary:
        tier_metrics = summary['tier_metrics']
        if isinstance(tier_metrics, dict):
            if 'long_win_rate' in tier_metrics:
                print(f"LONG Win Rate: {tier_metrics['long_win_rate']:.2f}%")
            if 'short_win_rate' in tier_metrics:
                print(f"SHORT Win Rate: {tier_metrics['short_win_rate']:.2f}%")
        
        # Display tier-specific metrics if available
        if 'tier_metrics' in tier_metrics and isinstance(tier_metrics['tier_metrics'], dict):
            print("\n===== TIER PERFORMANCE METRICS =====")
            for tier_name, tier_data in tier_metrics['tier_metrics'].items():
                if isinstance(tier_data, dict):
                    print(f"\n{tier_name}:")
                    print(f"  Win Rate: {tier_data.get('win_rate', 0):.2f}%")
                    print(f"  Average P/L: ${tier_data.get('avg_pl', 0):.2f}")
                    print(f"  Trade Count: {tier_data.get('trade_count', 0)}")
                    
                    if tier_data.get('long_count', 0) > 0:
                        print(f"  LONG Win Rate: {tier_data.get('long_win_rate', 0):.2f}% ({tier_data.get('long_count', 0)} trades)")
                    if tier_data.get('short_count', 0) > 0:
                        print(f"  SHORT Win Rate: {tier_data.get('short_win_rate', 0):.2f}% ({tier_data.get('short_count', 0)} trades)")
                else:
                    print(f"{tier_name}: {tier_data}")

def run_quarter_backtest(quarter, start_date, end_date, max_signals, initial_capital):
    """Run backtest for a specific quarter in a separate process"""
    print(f"\n{'=' * 50}")
    print(f"Running backtest for {quarter}: {start_date} to {end_date}")
    print(f"{'=' * 50}")
    
    # Import the run_backtest function from final_sp500_strategy
    from final_sp500_strategy import run_backtest
    
    # Run backtest for this quarter
    summary, signals = run_backtest(
        start_date, 
        end_date, 
        mode='backtest', 
        max_signals=max_signals, 
        initial_capital=initial_capital
    )
    
    # Create a unique filename for this quarter's results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"./backtest_results/quarter_results_{quarter}_{timestamp}.json"
    
    # Save results to a temporary file
    os.makedirs("./backtest_results", exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump({
            'summary': summary,
            'signals': signals if signals else []
        }, f, default=str)
    
    return results_file

def run_multiple_backtests(quarter, start_date, end_date, max_signals=100, initial_capital=300, num_runs=5, random_seed=42):
    """
    Run multiple backtests and average the results to get a more stable assessment
    
    Args:
        quarter (str): Quarter identifier (e.g., 'Q1_2023')
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        max_signals (int): Maximum number of signals to use
        initial_capital (float): Initial capital for the backtest
        num_runs (int): Number of backtest runs to perform
        random_seed (int): Base random seed for reproducibility
        
    Returns:
        dict: Averaged backtest results
    """
    print(f"\n{'=' * 50}")
    print(f"Running {num_runs} backtests for {quarter}: {start_date} to {end_date}")
    print(f"{'=' * 50}")
    
    # Import the run_backtest function from final_sp500_strategy
    from final_sp500_strategy import run_backtest
    
    # Lists to store results
    all_summaries = []
    all_signals = []
    
    # Create a directory for multiple backtest results
    results_dir = "./backtest_results/multiple_runs"
    os.makedirs(results_dir, exist_ok=True)
    
    # Run multiple backtests
    for run_idx in range(num_runs):
        # Set a unique random seed for each run based on the base seed
        current_seed = random_seed + run_idx
        
        print(f"\nRun {run_idx + 1}/{num_runs} (Seed: {current_seed})")
        
        # Set the random seed for numpy
        np.random.seed(current_seed)
        
        # Run backtest for this quarter with the current seed
        summary, signals = run_backtest(
            start_date, 
            end_date, 
            mode='backtest', 
            max_signals=max_signals, 
            initial_capital=initial_capital,
            random_seed=current_seed  # Pass the seed to run_backtest
        )
        
        if summary:
            all_summaries.append(summary)
            
            # Display brief summary for this run
            print(f"  Win Rate: {summary.get('win_rate', 0):.2f}%")
            print(f"  Profit Factor: {summary.get('profit_factor', 0):.2f}")
            print(f"  Total Return: {summary.get('total_return', 0):.2f}%")
            
        if signals:
            all_signals.append(signals)
    
    # Calculate averaged metrics
    if all_summaries:
        # Initialize the averaged summary with the structure of the first summary
        avg_summary = {k: 0 for k in all_summaries[0].keys() if isinstance(all_summaries[0][k], (int, float))}
        
        # Add non-numeric fields
        for k in all_summaries[0].keys():
            if not isinstance(all_summaries[0][k], (int, float)):
                avg_summary[k] = all_summaries[0][k]
        
        # Calculate averages for numeric fields
        for metric in avg_summary.keys():
            if isinstance(avg_summary[metric], (int, float)):
                values = [s.get(metric, 0) for s in all_summaries]
                avg_summary[metric] = sum(values) / len(values)
        
        # Calculate standard deviations for key metrics
        std_devs = {}
        for metric in ['win_rate', 'profit_factor', 'total_return']:
            if metric in avg_summary:
                values = [s.get(metric, 0) for s in all_summaries]
                std_devs[f"{metric}_std"] = np.std(values)
        
        # Add standard deviations to the summary
        avg_summary.update(std_devs)
        
        # Handle tier_metrics separately if they exist
        if 'tier_metrics' in all_summaries[0]:
            # Initialize with the structure of the first summary's tier_metrics
            tier_metrics = {}
            
            # Get all tier keys from all summaries
            all_tier_keys = set()
            for summary in all_summaries:
                if 'tier_metrics' in summary:
                    all_tier_keys.update(summary['tier_metrics'].keys())
            
            # Average each tier metric
            for tier_key in all_tier_keys:
                tier_values = []
                for summary in all_summaries:
                    if 'tier_metrics' in summary and tier_key in summary['tier_metrics']:
                        tier_values.append(summary['tier_metrics'][tier_key])
                
                if tier_values:
                    # If tier_values are dictionaries, average their numeric values
                    if isinstance(tier_values[0], dict):
                        tier_metrics[tier_key] = {}
                        for k in tier_values[0].keys():
                            if isinstance(tier_values[0][k], (int, float)):
                                values = [tv.get(k, 0) for tv in tier_values if k in tv]
                                tier_metrics[tier_key][k] = sum(values) / len(values)
                            else:
                                tier_metrics[tier_key][k] = tier_values[0][k]
                    else:
                        # If tier_values are not dictionaries, just average them
                        tier_metrics[tier_key] = sum(tier_values) / len(tier_values)
            
            avg_summary['tier_metrics'] = tier_metrics
        
        # Create a unique filename for the averaged results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        avg_results_file = f"{results_dir}/avg_results_{quarter}_{timestamp}.json"
        
        # Save averaged results
        with open(avg_results_file, 'w') as f:
            json.dump({
                'avg_summary': avg_summary,
                'num_runs': num_runs,
                'random_seed_base': random_seed,
                'std_devs': std_devs
            }, f, default=str)
        
        print(f"\n{'=' * 50}")
        print(f"AVERAGED RESULTS ({num_runs} runs)")
        print(f"{'=' * 50}")
        print(f"Win Rate: {avg_summary.get('win_rate', 0):.2f}% (±{std_devs.get('win_rate_std', 0):.2f})")
        print(f"Profit Factor: {avg_summary.get('profit_factor', 0):.2f} (±{std_devs.get('profit_factor_std', 0):.2f})")
        print(f"Total Return: {avg_summary.get('total_return', 0):.2f}% (±{std_devs.get('total_return_std', 0):.2f})")
        
        if 'tier_metrics' in avg_summary and 'long_win_rate' in avg_summary['tier_metrics']:
            print(f"LONG Win Rate: {avg_summary['tier_metrics']['long_win_rate']:.2f}%")
        
        return avg_summary
    
    return None

def run_comprehensive_backtest(quarter, max_signals=100, initial_capital=300, multiple_runs=False, num_runs=5):
    """Run a comprehensive backtest for a specific quarter with detailed signal analysis"""
    try:
        # Load configuration
        config_path = 'sp500_config.yaml'
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        logger.info(f"Successfully loaded configuration from {config_path}")
        
        # Define date ranges for each quarter
        quarters = {
            'Q1_2023': ('2023-01-01', '2023-03-31'),
            'Q2_2023': ('2023-04-01', '2023-06-30'),
            'Q3_2023': ('2023-07-01', '2023-09-30'),
            'Q4_2023': ('2023-10-01', '2023-12-31'),
            'Q1_2024': ('2024-01-01', '2024-03-31')
        }
        
        all_metrics = []
        
        # If 'all' is specified, run for all quarters
        if quarter.lower() == 'all':
            summary_data = []
            
            for q, (start_date, end_date) in quarters.items():
                if multiple_runs:
                    # Run multiple backtests and average results
                    summary = run_multiple_backtests(q, start_date, end_date, max_signals, initial_capital, num_runs)
                    signals = None  # Signals are not returned from multiple runs
                else:
                    # Run a single backtest
                    print(f"\n{'=' * 50}")
                    print(f"Running backtest for {q}: {start_date} to {end_date}")
                    print(f"{'=' * 50}")
                    
                    # Run backtest for this quarter
                    results_file = run_quarter_backtest(q, start_date, end_date, max_signals, initial_capital)
                    
                    with open(results_file, 'r') as f:
                        data = json.load(f)
                        summary = data['summary']
                        signals = data['signals']
                
                if summary:
                    all_metrics.append(summary)
                    
                    # Display summary for this quarter
                    print(f"\nTotal signals: {summary['total_signals']}")
                    print(f"LONG signals: {summary['long_signals']} ({summary['long_signals']/summary['total_signals']*100 if summary['total_signals'] > 0 else 0:.1f}%)")
                    print(f"SHORT signals: {summary['short_signals']} ({summary['short_signals']/summary['total_signals']*100 if summary['total_signals'] > 0 else 0:.1f}%)")
                    print(f"\nAverage LONG score: {summary['avg_long_score']:.3f}")
                    print(f"Average SHORT score: {summary['avg_short_score']:.3f}")
                    
                    # Add performance metrics if available
                    if summary and 'win_rate' in summary:
                        print("\n===== PERFORMANCE METRICS =====")
                        print(f"Win Rate: {summary['win_rate']:.2f}%")
                        print(f"Profit Factor: {summary['profit_factor']:.2f}")
                        print(f"Average Win: ${summary['avg_win']:.2f}")
                        print(f"Average Loss: ${summary['avg_loss']:.2f}")
                        print(f"Average Holding Period: {summary['avg_holding_period']:.1f} days")
                        print(f"LONG Win Rate: {summary['tier_metrics']['long_win_rate']:.2f}%")
                        print(f"SHORT Win Rate: {summary['tier_metrics']['short_win_rate']:.2f}%")
                        
                        # Add performance metrics to summary data
                        summary_data.append({
                            'quarter': q,
                            'win_rate': summary['win_rate'],
                            'profit_factor': summary['profit_factor'],
                            'avg_win': summary['avg_win'],
                            'avg_loss': summary['avg_loss'],
                            'avg_holding_period': summary['avg_holding_period'],
                            'long_win_rate': summary['tier_metrics']['long_win_rate'],
                            'short_win_rate': summary['tier_metrics']['short_win_rate']
                        })
            
            # Create a summary DataFrame
            if summary_data:
                # Add quarter name to each summary
                for i, q in enumerate(quarters.keys()):
                    if i < len(summary_data):
                        summary_data[i]['quarter'] = q
                
                summary_df = pd.DataFrame(summary_data)
                
                # Reorder columns to put quarter first
                cols = ['quarter'] + [col for col in summary_df.columns if col != 'quarter']
                summary_df = summary_df[cols]
                
                # Display overall summary
                print("\n" + "=" * 50)
                print("OVERALL SUMMARY")
                print("=" * 50)
                
                # Display the summary DataFrame
                print(summary_df)
                
                # Save the summary to a CSV file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                summary_path = f"./results/analysis/all_quarters_summary_{timestamp}.csv"
                summary_df.to_csv(summary_path, index=False)
                print(f"\nSummary saved to {summary_path}")
                
                # Calculate overall statistics
                total_signals = summary_df['total_signals'].sum()
                total_long = summary_df['long_signals'].sum()
                total_short = summary_df['short_signals'].sum()
                
                print("\nOverall statistics:")
                print(f"Total signals across all quarters: {total_signals}")
                print(f"Total LONG signals: {total_long} ({total_long/total_signals*100 if total_signals > 0 else 0:.1f}%)")
                print(f"Total SHORT signals: {total_short} ({total_short/total_signals*100 if total_signals > 0 else 0:.1f}%)")
                print(f"\nWeighted average LONG score: {(summary_df['avg_long_score'] * summary_df['long_signals']).sum() / total_long if total_long > 0 else 0:.3f}")
                print(f"Weighted average SHORT score: {(summary_df['avg_short_score'] * summary_df['short_signals']).sum() / total_short if total_short > 0 else 0:.3f}")
                
                # Calculate weighted performance metrics if available
                if 'win_rate' in summary_df.columns:
                    weighted_win_rate = (summary_df['win_rate'] * summary_df['total_signals']).sum() / total_signals if total_signals > 0 else 0
                    weighted_profit_factor = (summary_df['profit_factor'] * summary_df['total_signals']).sum() / total_signals if total_signals > 0 else 0
                    weighted_avg_win = (summary_df['avg_win'] * summary_df['total_signals']).sum() / total_signals if total_signals > 0 else 0
                    weighted_avg_loss = (summary_df['avg_loss'] * summary_df['total_signals']).sum() / total_signals if total_signals > 0 else 0
                    weighted_avg_holding = (summary_df['avg_holding_period'] * summary_df['total_signals']).sum() / total_signals if total_signals > 0 else 0
                    weighted_long_win_rate = (summary_df['long_win_rate'] * summary_df['long_signals']).sum() / total_long if total_long > 0 else 0
                    weighted_short_win_rate = (summary_df['short_win_rate'] * summary_df['short_signals']).sum() / total_short if total_short > 0 else 0
                    
                    print("\n===== AVERAGE PERFORMANCE METRICS =====")
                    print(f"Overall Win Rate: {weighted_win_rate:.2f}%")
                    print(f"Overall Profit Factor: {weighted_profit_factor:.2f}")
                    print(f"Overall Average Win: ${weighted_avg_win:.2f}")
                    print(f"Overall Average Loss: ${weighted_avg_loss:.2f}")
                    print(f"Overall Average Holding Period: {weighted_avg_holding:.1f} days")
                    print(f"Overall LONG Win Rate: {weighted_long_win_rate:.2f}%")
                    print(f"Overall SHORT Win Rate: {weighted_short_win_rate:.2f}%")
                    
                    # Display tier performance metrics if available in any of the backtest results
                    tier_metrics_available = any('tier_metrics' in metrics for metrics in all_metrics)
                    if tier_metrics_available:
                        print("\n===== SIGNAL SCORE TIER PERFORMANCE =====")
                        
                        # Collect tier metrics across all backtests
                        all_tier_metrics = {}
                        for metrics in all_metrics:
                            if 'tier_metrics' in metrics:
                                for tier, tier_data in metrics['tier_metrics'].items():
                                    if tier not in all_tier_metrics:
                                        all_tier_metrics[tier] = {
                                            'win_rate': [],
                                            'avg_pl': [],
                                            'trade_count': [],
                                            'long_win_rate': [],
                                            'short_win_rate': [],
                                            'long_count': [],
                                            'short_count': []
                                        }
                                    
                                    # Add metrics with weights based on trade count
                                    trade_count = tier_data.get('trade_count', 0)
                                    if trade_count > 0:
                                        all_tier_metrics[tier]['win_rate'].append((tier_data.get('win_rate', 0), trade_count))
                                        all_tier_metrics[tier]['avg_pl'].append((tier_data.get('avg_pl', 0), trade_count))
                                        all_tier_metrics[tier]['trade_count'].append(trade_count)
                                        
                                        long_count = tier_data.get('long_count', 0)
                                        if long_count > 0:
                                            all_tier_metrics[tier]['long_win_rate'].append((tier_data.get('long_win_rate', 0), long_count))
                                            all_tier_metrics[tier]['long_count'].append(long_count)
                                        
                                        short_count = tier_data.get('short_count', 0)
                                        if short_count > 0:
                                            all_tier_metrics[tier]['short_win_rate'].append((tier_data.get('short_win_rate', 0), short_count))
                                            all_tier_metrics[tier]['short_count'].append(short_count)
                        
                        # Calculate weighted averages for each tier
                        combined_tier_metrics = {}
                        for tier, metrics in all_tier_metrics.items():
                            total_trade_count = sum(metrics['trade_count'])
                            if total_trade_count > 0:
                                # Calculate weighted win rate
                                weighted_win_rate = sum(rate * count for rate, count in metrics['win_rate']) / total_trade_count
                                
                                # Calculate weighted average P/L
                                weighted_avg_pl = sum(pl * count for pl, count in metrics['avg_pl']) / total_trade_count
                                
                                # Calculate LONG metrics
                                total_long_count = sum(metrics['long_count']) if metrics['long_count'] else 0
                                long_win_rate = 0
                                if total_long_count > 0:
                                    long_win_rate = sum(rate * count for rate, count in metrics['long_win_rate']) / total_long_count
                                
                                # Calculate SHORT metrics
                                total_short_count = sum(metrics['short_count']) if metrics['short_count'] else 0
                                short_win_rate = 0
                                if total_short_count > 0:
                                    short_win_rate = sum(rate * count for rate, count in metrics['short_win_rate']) / total_short_count
                                
                                combined_tier_metrics[tier] = {
                                    'win_rate': weighted_win_rate,
                                    'avg_pl': weighted_avg_pl,
                                    'trade_count': total_trade_count,
                                    'long_win_rate': long_win_rate,
                                    'short_win_rate': short_win_rate,
                                    'long_count': total_long_count,
                                    'short_count': total_short_count
                                }
                        
                        # Save combined tier metrics to CSV
                        tier_metrics_data = []
                        for tier, tier_data in combined_tier_metrics.items():
                            tier_metrics_data.append({
                                'tier': tier,
                                'win_rate': tier_data['win_rate'],
                                'avg_pl': tier_data['avg_pl'],
                                'trade_count': tier_data['trade_count'],
                                'long_win_rate': tier_data['long_win_rate'],
                                'short_win_rate': tier_data['short_win_rate'],
                                'long_count': tier_data['long_count'],
                                'short_count': tier_data['short_count']
                            })
                        
                        tier_metrics_df = pd.DataFrame(tier_metrics_data)
                        tier_metrics_path = f"./results/analysis/tier_metrics_{timestamp}.csv"
                        tier_metrics_df.to_csv(tier_metrics_path, index=False)
                        print(f"\nTier metrics saved to: {tier_metrics_path}")
                        
                        # Display combined tier metrics
                        print("\n===== COMBINED TIER PERFORMANCE METRICS =====")
                        for tier, tier_data in combined_tier_metrics.items():
                            print(f"\n{tier}:")
                            print(f"  Win Rate: {tier_data['win_rate']:.2f}%")
                            print(f"  Average P/L: ${tier_data['avg_pl']:.2f}")
                            print(f"  Trade Count: {tier_data['trade_count']}")
                            if tier_data['long_count'] > 0:
                                print(f"  LONG Win Rate: {tier_data['long_win_rate']:.2f}% ({tier_data['long_count']} trades)")
                            if tier_data['short_count'] > 0:
                                print(f"  SHORT Win Rate: {tier_data['short_win_rate']:.2f}% ({tier_data['short_count']} trades)")
        else:
            # Run backtest for the specified quarter
            start_date, end_date = quarters[quarter]
            if multiple_runs:
                # Run multiple backtests and average results
                summary = run_multiple_backtests(quarter, start_date, end_date, max_signals, initial_capital, num_runs)
                signals = None  # Signals are not returned from multiple runs
            else:
                # Run a single backtest
                results_file = run_quarter_backtest(quarter, start_date, end_date, max_signals, initial_capital)
                
                with open(results_file, 'r') as f:
                    data = json.load(f)
                    summary = data['summary']
                    signals = data['signals']
            
            if summary:
                print(f"\nTotal signals: {summary['total_signals']}")
                print(f"LONG signals: {summary['long_signals']} ({summary['long_signals']/summary['total_signals']*100 if summary['total_signals'] > 0 else 0:.1f}%)")
                print(f"SHORT signals: {summary['short_signals']} ({summary['short_signals']/summary['total_signals']*100 if summary['total_signals'] > 0 else 0:.1f}%)")
                print(f"\nAverage LONG score: {summary['avg_long_score']:.3f}")
                print(f"Average SHORT score: {summary['avg_short_score']:.3f}")
                
                # Add performance metrics if available
                if summary and 'win_rate' in summary:
                    print("\n===== PERFORMANCE METRICS =====")
                    print(f"Win Rate: {summary['win_rate']:.2f}%")
                    print(f"Profit Factor: {summary['profit_factor']:.2f}")
                    print(f"Average Win: ${summary['avg_win']:.2f}")
                    print(f"Average Loss: ${summary['avg_loss']:.2f}")
                    print(f"Average Holding Period: {summary['avg_holding_period']:.1f} days")
                    print(f"LONG Win Rate: {summary['tier_metrics']['long_win_rate']:.2f}%")
                    print(f"SHORT Win Rate: {summary['tier_metrics']['short_win_rate']:.2f}%")
                    
                    # Add signal score tier performance analysis if available
                    # We'll check the most recent backtest's performance metrics file
                    performance_dir = config['paths']['performance']
                    if os.path.exists(performance_dir):
                        perf_files = [os.path.join(performance_dir, f) for f in os.listdir(performance_dir) if f.endswith('.csv')]
                        if perf_files:
                            # Sort by modification time, newest first
                            perf_files.sort(key=os.path.getmtime, reverse=True)
                            latest_perf_file = perf_files[0]
                            
                            try:
                                # Load the performance metrics
                                perf_df = pd.read_csv(latest_perf_file)
                                
                                # Check if we have tier metrics
                                tier_columns = [col for col in perf_df.columns if 'Tier' in col and 'win_rate' in col]
                                if tier_columns:
                                    print("\n===== SIGNAL SCORE TIER PERFORMANCE =====")
                                    
                                    # Map the column names back to readable tier names
                                    tier_mapping = {
                                        'Tier_1_ge09_win_rate': 'Tier 1 (≥0.9)',
                                        'Tier_2_08to09_win_rate': 'Tier 2 (0.8-0.9)',
                                        'Tier_3_07to08_win_rate': 'Tier 3 (0.7-0.8)',
                                        'Tier_4_lt07_win_rate': 'Tier 4 (<0.7)'
                                    }
                                    
                                    for tier_col in tier_columns:
                                        # Extract the base name without _win_rate
                                        base_name = tier_col.replace('_win_rate', '')
                                        
                                        # Get the readable tier name
                                        tier_name = tier_mapping.get(base_name, base_name)
                                        
                                        # Get the metrics
                                        win_rate = perf_df[tier_col].values[0]
                                        avg_pl = perf_df[f"{base_name}_avg_pl"].values[0] if f"{base_name}_avg_pl" in perf_df.columns else 0
                                        avg_pl_pct = perf_df[f"{base_name}_avg_pl_pct"].values[0] if f"{base_name}_avg_pl_pct" in perf_df.columns else 0
                                        trades = perf_df[f"{base_name}_trades"].values[0] if f"{base_name}_trades" in perf_df.columns else 0
                                        
                                        # Display the tier performance
                                        print(f"{tier_name}: Win Rate {win_rate:.2f}%, Avg P/L ${avg_pl:.2f} ({avg_pl_pct:.2f}%), Trades: {int(trades)}")
                                        
                                        # Get direction-specific metrics if available
                                        long_win_rate = perf_df[f"{base_name}_long_win_rate"].values[0] if f"{base_name}_long_win_rate" in perf_df.columns else 0
                                        short_win_rate = perf_df[f"{base_name}_short_win_rate"].values[0] if f"{base_name}_short_win_rate" in perf_df.columns else 0
                                        long_trades = perf_df[f"{base_name}_long_trades"].values[0] if f"{base_name}_long_trades" in perf_df.columns else 0
                                        short_trades = perf_df[f"{base_name}_short_trades"].values[0] if f"{base_name}_short_trades" in perf_df.columns else 0
                                        
                                        # Display direction-specific metrics
                                        if long_trades > 0:
                                            print(f"  LONG: Win Rate {long_win_rate:.2f}% ({int(long_trades)} trades)")
                                        if short_trades > 0:
                                            print(f"  SHORT: Win Rate {short_win_rate:.2f}% ({int(short_trades)} trades)")
                            except Exception as e:
                                logger.warning(f"Error loading tier performance metrics: {e}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

def run_all_quarters_backtest(max_signals=100, initial_capital=300, multiple_runs=False, num_runs=5):
    """Run comprehensive backtests for all quarters"""
    try:
        # Load configuration
        config_path = 'sp500_config.yaml'
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        logger.info(f"Successfully loaded configuration from {config_path}")
        
        # Define quarters to backtest
        quarters = {
            'Q1_2023': ('2023-01-01', '2023-03-31'),
            'Q2_2023': ('2023-04-01', '2023-06-30'),
            'Q3_2023': ('2023-07-01', '2023-09-30'),
            'Q4_2023': ('2023-10-01', '2023-12-31'),
            'Q1_2024': ('2024-01-01', '2024-03-31')
        }
        
        all_metrics = {}
        summary_data = []
        
        for q, (start_date, end_date) in quarters.items():
            if multiple_runs:
                # Run multiple backtests and average results
                summary = run_multiple_backtests(q, start_date, end_date, max_signals, initial_capital, num_runs)
                signals = None  # Signals are not returned from multiple runs
            else:
                # Run a single backtest
                print(f"\n{'=' * 50}")
                print(f"Running backtest for {q}: {start_date} to {end_date}")
                print(f"{'=' * 50}")
                
                # Run backtest for this quarter
                results_file = run_quarter_backtest(q, start_date, end_date, max_signals, initial_capital)
                
                with open(results_file, 'r') as f:
                    data = json.load(f)
                    summary = data['summary']
                    signals = data['signals']
            
            # Store metrics for this quarter
            if summary:
                all_metrics[q] = summary
            
            if signals:
                # Convert signals to DataFrame
                signals_df = pd.DataFrame(signals)
                
                # Count signals by direction
                long_signals = signals_df[signals_df["direction"] == "LONG"]
                short_signals = signals_df[signals_df["direction"] == "SHORT"]
                
                print(f"\nTotal signals: {len(signals_df)}")
                print(f"LONG signals: {len(long_signals)} ({len(long_signals)/len(signals_df)*100:.1f}%)")
                print(f"SHORT signals: {len(short_signals)} ({len(short_signals)/len(signals_df)*100:.1f}%)")
                
                # Average scores
                avg_long_score = long_signals['score'].mean() if not long_signals.empty else 0
                avg_short_score = short_signals['score'].mean() if not short_signals.empty else 0
                print(f"\nAverage LONG score: {avg_long_score:.3f}")
                print(f"Average SHORT score: {avg_short_score:.3f}")
                
                # Add to summary data
                quarter_summary = {
                    'quarter': q,
                    'total_signals': len(signals_df),
                    'long_signals': len(long_signals),
                    'short_signals': len(short_signals),
                    'long_pct': len(long_signals)/len(signals_df)*100 if len(signals_df) > 0 else 0,
                    'short_pct': len(short_signals)/len(signals_df)*100 if len(signals_df) > 0 else 0,
                    'avg_long_score': avg_long_score,
                    'avg_short_score': avg_short_score
                }
                
                # Add performance metrics if available
                if summary and 'win_rate' in summary:
                    print("\n===== PERFORMANCE METRICS =====")
                    print(f"Win Rate: {summary['win_rate']:.2f}%")
                    print(f"Profit Factor: {summary['profit_factor']:.2f}")
                    print(f"Average Win: ${summary['avg_win']:.2f}")
                    print(f"Average Loss: ${summary['avg_loss']:.2f}")
                    print(f"Average Holding Period: {summary['avg_holding_period']:.1f} days")
                    print(f"Initial Capital: ${summary['initial_capital']:.2f}")
                    print(f"Final Capital: ${summary['final_capital']:.2f}")
                    print(f"Total Return: {summary['total_return']:.2f}%")
                    
                    # Add performance metrics to summary data
                    quarter_summary.update({
                        'win_rate': summary['win_rate'],
                        'profit_factor': summary['profit_factor'],
                        'avg_win': summary['avg_win'],
                        'avg_loss': summary['avg_loss'],
                        'avg_holding_period': summary['avg_holding_period'],
                        'initial_capital': summary['initial_capital'],
                        'final_capital': summary['final_capital'],
                        'total_return': summary['total_return'],
                        'total_trades': summary['total_trades'],
                        'winning_trades': summary['winning_trades'],
                        'losing_trades': summary['losing_trades']
                    })
                    
                    # Add LONG and SHORT win rates if available
                    if 'tier_metrics' in summary:
                        tier_metrics = summary['tier_metrics']
                        tier1_metrics = tier_metrics.get('Tier 1 (≥0.9)', {})
                        quarter_summary['long_win_rate'] = tier1_metrics.get('long_win_rate', 0)
                        quarter_summary['short_win_rate'] = tier1_metrics.get('short_win_rate', 0)
                
                summary_data.append(quarter_summary)
        
        # Create summary DataFrame
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            print("\n" + "="*50)
            print("OVERALL SUMMARY")
            print("="*50)
            print(summary_df)
            
            # Save summary to CSV
            os.makedirs("./results/analysis", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_path = f"./results/analysis/all_quarters_summary_{timestamp}.csv"
            summary_df.to_csv(summary_path, index=False)
            print(f"\nSummary saved to {summary_path}")
            
            # Calculate overall statistics
            total_signals = summary_df['total_signals'].sum()
            total_long = summary_df['long_signals'].sum()
            total_short = summary_df['short_signals'].sum()
            
            print(f"\nOverall statistics:")
            print(f"Total signals across all quarters: {total_signals}")
            print(f"Total LONG signals: {total_long} ({total_long/total_signals*100:.1f}%)")
            print(f"Total SHORT signals: {total_short} ({total_short/total_signals*100:.1f}%)")
            
            # Average scores weighted by signal count
            weighted_long_score = (summary_df['avg_long_score'] * summary_df['long_signals']).sum() / total_long if total_long > 0 else 0
            weighted_short_score = (summary_df['avg_short_score'] * summary_df['short_signals']).sum() / total_short if total_short > 0 else 0
            
            print(f"Weighted average LONG score: {weighted_long_score:.3f}")
            print(f"Weighted average SHORT score: {weighted_short_score:.3f}")
            
            # Display average performance metrics if available
            if 'win_rate' in summary_df.columns:
                print("\n===== AVERAGE PERFORMANCE METRICS =====")
                # Calculate weighted averages based on total signals per quarter
                weighted_win_rate = (summary_df['win_rate'] * summary_df['total_signals']).sum() / total_signals
                weighted_profit_factor = (summary_df['profit_factor'] * summary_df['total_signals']).sum() / total_signals
                weighted_avg_win = (summary_df['avg_win'] * summary_df['total_signals']).sum() / total_signals
                weighted_avg_loss = (summary_df['avg_loss'] * summary_df['total_signals']).sum() / total_signals
                weighted_avg_holding = (summary_df['avg_holding_period'] * summary_df['total_signals']).sum() / total_signals
                weighted_long_win_rate = (summary_df['long_win_rate'] * summary_df['long_signals']).sum() / total_long if total_long > 0 else 0
                weighted_short_win_rate = (summary_df['short_win_rate'] * summary_df['short_signals']).sum() / total_short if total_short > 0 else 0
                
                print(f"Overall Win Rate: {weighted_win_rate:.2f}%")
                print(f"Overall Profit Factor: {weighted_profit_factor:.2f}")
                print(f"Overall Average Win: ${weighted_avg_win:.2f}")
                print(f"Overall Average Loss: ${weighted_avg_loss:.2f}")
                print(f"Overall Average Holding Period: {weighted_avg_holding:.1f} days")
                print(f"Overall LONG Win Rate: {weighted_long_win_rate:.2f}%")
                print(f"Overall SHORT Win Rate: {weighted_short_win_rate:.2f}%")
                
        return all_metrics
    except Exception as e:
        print(f"Error running all quarters backtest: {str(e)}")
        traceback.print_exc()
        return []

def main():
    """Main function to run the comprehensive backtest"""
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Run comprehensive backtest for specified quarters')
        parser.add_argument('quarters', nargs='+', help='Quarters to run backtest for (e.g., Q1_2023 Q2_2023)')
        parser.add_argument('--max_signals', type=int, default=100, help='Maximum number of signals to use')
        parser.add_argument('--initial_capital', type=float, default=300, help='Initial capital for the backtest')
        parser.add_argument('--multiple_runs', action='store_true', help='Run multiple backtests and average results')
        parser.add_argument('--num_runs', type=int, default=5, help='Number of backtest runs to perform when using --multiple_runs')
        parser.add_argument('--random_seed', type=int, default=42, help='Base random seed for reproducibility')
        args = parser.parse_args()
        
        # Define quarters mapping
        quarters_map = {
            'Q1_2023': ('2023-01-01', '2023-03-31'),
            'Q2_2023': ('2023-04-01', '2023-06-30'),
            'Q3_2023': ('2023-07-01', '2023-09-30'),
            'Q4_2023': ('2023-10-01', '2023-12-31'),
            'Q1_2024': ('2024-01-01', '2024-03-31')
        }
        
        # Check if 'all' is specified
        if 'all' in args.quarters:
            print("Running backtest for all quarters")
            run_all_quarters_backtest(
                max_signals=args.max_signals, 
                initial_capital=args.initial_capital,
                multiple_runs=args.multiple_runs,
                num_runs=args.num_runs
            )
            return
        
        # Run backtest for each quarter
        for quarter in args.quarters:
            run_comprehensive_backtest(
                quarter, 
                max_signals=args.max_signals, 
                initial_capital=args.initial_capital,
                multiple_runs=args.multiple_runs,
                num_runs=args.num_runs
            )
    except Exception as e:
        logger.error(f"Error running comprehensive backtest: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
