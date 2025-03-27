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
import time

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', f"comprehensive_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"), mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add a console handler to ensure output is visible in web interface
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

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
            # Check for long_win_rate directly in the tier_metrics dictionary
            for tier_name, tier_data in tier_metrics.items():
                if isinstance(tier_data, dict) and 'long_win_rate' in tier_data:
                    print(f"LONG Win Rate: {tier_data['long_win_rate']:.2f}%")
                    break
        
        # Display tier-specific metrics if available
        print("\n===== TIER PERFORMANCE METRICS =====")
        for tier_name, tier_data in tier_metrics.items():
            if isinstance(tier_data, dict):
                print(f"\n{tier_name}:")
                print(f"  Win Rate: {tier_data.get('win_rate', 0):.2f}%")
                print(f"  Average P/L: ${tier_data.get('avg_pl', 0):.2f}")
                print(f"  Trade Count: {tier_data.get('trade_count', 0)}")
                
                if tier_data.get('long_count', 0) > 0:
                    print(f"  LONG Win Rate: {tier_data.get('long_win_rate', 0):.2f}% ({tier_data.get('long_count', 0)} trades)")
            else:
                print(f"{tier_name}: {tier_data}")

def run_quarter_backtest(quarter, start_date, end_date, max_signals=100, initial_capital=300, random_seed=42, continuous_capital=False, previous_capital=None, weekly_selection=False, tier1_threshold=0.8, tier2_threshold=0.7, tier3_threshold=0.6):
    """
    Run a backtest for a specific quarter
    
    Args:
        quarter (str): Quarter identifier (e.g., 'Q1_2023')
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        max_signals (int): Maximum number of signals to use
        initial_capital (float): Initial capital for the backtest
        random_seed (int): Random seed for reproducibility
        continuous_capital (bool): Whether to use continuous capital across quarters
        previous_capital (float): Previous capital from a prior run (if continuous_capital is True)
        weekly_selection (bool): Whether to enable weekly stock selection refresh
        tier1_threshold (float): Threshold for tier 1 signals
        tier2_threshold (float): Threshold for tier 2 signals
        tier3_threshold (float): Threshold for tier 3 signals
        
    Returns:
        tuple: (summary, signals, results_file) where summary is a dict of backtest metrics, 
               signals is a dict of signal data, and results_file is the path to the saved results
    """
    try:
        # Set initial capital based on previous run if continuous capital is enabled
        if continuous_capital and previous_capital is not None:
            run_initial_capital = previous_capital
        else:
            run_initial_capital = initial_capital
            
        logger.info(f"Running backtest for {quarter}: {start_date} to {end_date}")
        logger.info(f"Initial capital: ${run_initial_capital}")
        logger.info(f"Max signals: {max_signals}")
        logger.info(f"Weekly selection: {weekly_selection}")
        logger.info(f"Tier thresholds: Tier1={tier1_threshold}, Tier2={tier2_threshold}, Tier3={tier3_threshold}")
        
        # Create results directory if it doesn't exist
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backtest_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Define results file path
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(results_dir, f"backtest_{quarter}_{start_date}_to_{end_date}_{timestamp}.json")
        
        # Run backtest using final_sp500_strategy.py
        from final_sp500_strategy import run_backtest
        
        # Run the backtest
        summary, signals = run_backtest(
            start_date=start_date,
            end_date=end_date,
            initial_capital=run_initial_capital,
            max_signals=max_signals,
            random_seed=random_seed,
            weekly_selection=weekly_selection,
            tier1_threshold=tier1_threshold,
            tier2_threshold=tier2_threshold,
            tier3_threshold=tier3_threshold
        )
        
        # Check if summary is None and create a minimal summary to avoid null values
        if summary is None:
            logger.error(f"Backtest for {quarter} returned None summary. This indicates an error in the backtest execution.")
            summary = {
                'quarter': quarter,
                'start_date': start_date,
                'end_date': end_date,
                'error': 'Backtest execution failed to return valid results',
                'win_rate': 0,
                'profit_factor': 0,
                'total_return': 0,
                'initial_capital': run_initial_capital,
                'final_capital': run_initial_capital,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'calmar_ratio': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'avg_holding_period': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0
            }
        else:
            # Ensure all required fields are present
            if 'max_drawdown' not in summary:
                summary['max_drawdown'] = 0
            if 'sharpe_ratio' not in summary:
                summary['sharpe_ratio'] = 0
            if 'sortino_ratio' not in summary:
                summary['sortino_ratio'] = 0
            if 'calmar_ratio' not in summary:
                summary['calmar_ratio'] = 0
                
            logger.info(f"Backtest for {quarter} completed successfully with win_rate: {summary.get('win_rate', 'N/A')}%, profit_factor: {summary.get('profit_factor', 'N/A')}")
        
        # Save results to file
        with open(results_file, 'w') as f:
            json.dump({
                'summary': summary,
                'signals': signals if signals else [],
                'quarter': quarter,
                'start_date': start_date,
                'end_date': end_date,
                'max_signals': max_signals,
                'initial_capital': run_initial_capital,
                'weekly_selection': weekly_selection,
                'tier1_threshold': tier1_threshold,
                'tier2_threshold': tier2_threshold,
                'tier3_threshold': tier3_threshold,
                'trading_parameters': {
                    'position_sizing': {
                        'base_position_pct': 5,
                        'tier1_factor': 3.0,
                        'tier2_factor': 1.5,
                        'midcap_factor': 0.8
                    },
                    'stop_loss_pct': 5,
                    'take_profit_pct': 10,
                    'max_drawdown_pct': 15,
                    'large_cap_percentage': 70,
                    'avg_holding_period': {
                        'win': 12,
                        'loss': 5
                    },
                    'win_rate_adjustments': {
                        'base_long_win_rate': 0.62,
                        'market_regime_adjustments': {
                            'STRONG_BULLISH': 0.15,
                            'BULLISH': 0.10,
                            'NEUTRAL': 0.00,
                            'BEARISH': -0.10,
                            'STRONG_BEARISH': -0.20
                        }
                    }
                }
            }, f, default=str, indent=4)
        
        logger.info(f"Backtest results saved to {results_file}")
        
        return summary, signals, results_file
    except Exception as e:
        logger.error(f"Error running backtest for {quarter}: {str(e)}", exc_info=True)
        
        # Create a minimal summary with error information
        error_summary = {
            'quarter': quarter,
            'start_date': start_date,
            'end_date': end_date,
            'error': str(e),
            'win_rate': 0,
            'profit_factor': 0,
            'total_return': 0,
            'initial_capital': initial_capital,
            'final_capital': initial_capital,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'calmar_ratio': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'avg_holding_period': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0
        }
        
        # Create error results file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(results_dir, f"backtest_{quarter}_{start_date}_to_{end_date}_error_{timestamp}.json")
        
        # Save error results
        with open(results_file, 'w') as f:
            json.dump({
                'summary': error_summary,
                'signals': [],
                'error': str(e),
                'traceback': traceback.format_exc()
            }, f, default=str, indent=4)
            
        logger.info(f"Error results saved to {results_file}")
        
        return error_summary, [], results_file

def run_multiple_backtests(quarter, start_date, end_date, max_signals=100, initial_capital=300, num_runs=5, random_seed=42, continuous_capital=False, previous_capital=None, weekly_selection=False, tier1_threshold=0.8, tier2_threshold=0.7, tier3_threshold=0.6):
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
        continuous_capital (bool): Whether to use continuous capital across quarters
        previous_capital (float): Previous ending capital to use as initial capital (if continuous_capital is True)
        weekly_selection (bool): Whether to enable weekly stock selection refresh
        tier1_threshold (float): Threshold for tier 1 signals
        tier2_threshold (float): Threshold for tier 2 signals
        tier3_threshold (float): Threshold for tier 3 signals
        
    Returns:
        dict: Averaged backtest results and final capital
    """
    print(f"\n{'=' * 50}")
    print(f"Running {num_runs} backtests for {quarter}: {start_date} to {end_date}")
    if continuous_capital and previous_capital is not None:
        print(f"Using continuous capital: Starting with ${previous_capital:.2f} from previous quarter")
    print(f"{'=' * 50}")
    
    # Import the run_backtest function from final_sp500_strategy
    from final_sp500_strategy import run_backtest
    
    # Lists to store results
    all_summaries = []
    all_signals = []
    final_capitals = []
    
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
        
        # Determine the initial capital for this run
        run_initial_capital = previous_capital if continuous_capital and previous_capital is not None else initial_capital
        
        # Run backtest for this quarter with the current seed
        summary, signals, _ = run_quarter_backtest(
            quarter, 
            start_date, 
            end_date, 
            max_signals, 
            initial_capital,
            random_seed=current_seed,  # Pass the seed to run_backtest
            continuous_capital=continuous_capital,
            previous_capital=previous_capital,
            weekly_selection=weekly_selection,
            tier1_threshold=tier1_threshold,
            tier2_threshold=tier2_threshold,
            tier3_threshold=tier3_threshold
        )
        
        if summary:
            all_summaries.append(summary)
            
            # Store the final capital for this run
            final_capital = run_initial_capital * (1 + summary.get('total_return', 0) / 100)
            final_capitals.append(final_capital)
            
            # Display brief summary for this run
            print(f"  Win Rate: {summary.get('win_rate', 0):.2f}%")
            print(f"  Profit Factor: {summary.get('profit_factor', 0):.2f}")
            print(f"  Total Return: {summary.get('total_return', 0):.2f}%")
            print(f"  Final Capital: ${final_capital:.2f}")
            
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
        
        # Ensure the backtest_results directory exists at the root level (where the web interface expects it)
        backtest_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_results")
        os.makedirs(backtest_dir, exist_ok=True)
        
        # Create a standardized filename format that the web interface can recognize
        avg_results_file = os.path.join(backtest_dir, f"backtest_{quarter}_{start_date}_to_{end_date}_avg{num_runs}runs_{timestamp}.json")
        
        # Save averaged results
        with open(avg_results_file, 'w') as f:
            json.dump({
                'summary': avg_summary,  # Use 'summary' key to match the format expected by the web interface
                'num_runs': num_runs,
                'random_seed_base': random_seed,
                'std_devs': std_devs,
                'final_capital': sum(final_capitals) / len(final_capitals) if final_capitals else 0
            }, f, default=str)
        
        print(f"\n{'=' * 50}")
        print(f"AVERAGED RESULTS ({num_runs} runs)")
        print(f"{'=' * 50}")
        print(f"Win Rate: {avg_summary.get('win_rate', 0):.2f}% (±{std_devs.get('win_rate_std', 0):.2f})")
        print(f"Profit Factor: {avg_summary.get('profit_factor', 0):.2f} (±{std_devs.get('profit_factor_std', 0):.2f})")
        print(f"Total Return: {avg_summary.get('total_return', 0):.2f}% (±{std_devs.get('total_return_std', 0):.2f})")
        
        # Check if long_win_rate exists directly in the summary
        if 'long_win_rate' in avg_summary:
            print(f"LONG Win Rate: {avg_summary['long_win_rate']:.2f}%")
        # Otherwise, try to find it in tier_metrics
        elif 'tier_metrics' in avg_summary and isinstance(avg_summary['tier_metrics'], dict):
            for tier_name, tier_data in avg_summary['tier_metrics'].items():
                if isinstance(tier_data, dict) and 'long_win_rate' in tier_data:
                    print(f"LONG Win Rate: {tier_data['long_win_rate']:.2f}%")
                    break
        
        return avg_summary
    
    return None

def run_comprehensive_backtest(quarters, max_signals=100, initial_capital=300, multiple_runs=False, num_runs=5, random_seed=42, continuous_capital=False, weekly_selection=False, tier1_threshold=0.8, tier2_threshold=0.7, tier3_threshold=0.6):
    """
    Run a comprehensive backtest across multiple quarters
    
    Args:
        quarters (list): List of quarter identifiers (e.g., ['Q1_2023', 'Q2_2023'])
        max_signals (int): Maximum number of signals to use
        initial_capital (float): Initial capital for the backtest
        multiple_runs (bool): Whether to run multiple backtests and average results
        num_runs (int): Number of backtest runs to perform
        random_seed (int): Base random seed for reproducibility
        continuous_capital (bool): Whether to use continuous capital across quarters
        weekly_selection (bool): Whether to enable weekly stock selection refresh
        tier1_threshold (float): Threshold for tier 1 signals
        tier2_threshold (float): Threshold for tier 2 signals
        tier3_threshold (float): Threshold for tier 3 signals
        
    Returns:
        dict: Combined backtest results
    """
    try:
        # Load configuration
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sp500_config.yaml')
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        logger.info(f"Successfully loaded configuration from {config_path}")
        
        # Define date ranges for each quarter
        quarters_map = {
            'Q1_2023': ('2023-01-01', '2023-03-31'),
            'Q2_2023': ('2023-04-01', '2023-06-30'),
            'Q3_2023': ('2023-07-01', '2023-09-30'),
            'Q4_2023': ('2023-10-01', '2023-12-31'),
            'Q1_2024': ('2024-01-01', '2024-03-31')
        }
        
        all_metrics = []
        
        # If 'all' is specified, run for all quarters
        if 'all' in quarters:
            summary_data = []
            
            previous_capital = None
            for quarter, (start_date, end_date) in quarters_map.items():
                if multiple_runs:
                    # Run multiple backtests and average results
                    summary = run_multiple_backtests(
                        quarter, 
                        start_date, 
                        end_date, 
                        max_signals, 
                        initial_capital, 
                        num_runs, 
                        random_seed, 
                        continuous_capital=continuous_capital, 
                        previous_capital=previous_capital,
                        weekly_selection=weekly_selection,
                        tier1_threshold=tier1_threshold,
                        tier2_threshold=tier2_threshold,
                        tier3_threshold=tier3_threshold
                    )
                    signals = None  # Signals are not returned from multiple runs
                    if summary and continuous_capital:
                        previous_capital = summary.get('final_capital', initial_capital)
                else:
                    # Run a single backtest
                    print(f"\n{'=' * 50}")
                    print(f"Running backtest for {quarter}: {start_date} to {end_date}")
                    print(f"{'=' * 50}")
                    
                    # Set initial capital for this run
                    run_initial_capital = previous_capital if continuous_capital and previous_capital else initial_capital
                    
                    # Run backtest for this quarter
                    summary, signals, results_file = run_quarter_backtest(
                        quarter, 
                        start_date, 
                        end_date, 
                        max_signals, 
                        initial_capital,
                        random_seed,
                        continuous_capital=continuous_capital,
                        previous_capital=previous_capital,
                        weekly_selection=weekly_selection,
                        tier1_threshold=tier1_threshold,
                        tier2_threshold=tier2_threshold,
                        tier3_threshold=tier3_threshold
                    )
                    if summary and continuous_capital:
                        previous_capital = summary.get('final_capital', initial_capital)
            
                if summary:
                    # Add quarter info to the summary
                    summary['quarter'] = quarter
                    summary['start_date'] = start_date
                    summary['end_date'] = end_date
                    
                    # Store summary for combined results
                    summary_data.append(summary)
            
            # Create a summary DataFrame
            if summary_data:
                # Add quarter name to each summary
                for i, quarter in enumerate(quarters_map.keys()):
                    if i < len(summary_data):
                        summary_data[i]['quarter'] = quarter
                
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
                
                print("\nOverall statistics:")
                print(f"Total signals across all quarters: {total_signals}")
                print(f"Total LONG signals: {total_long} ({total_long/total_signals*100 if total_signals > 0 else 0:.1f}%)")
                
                # Calculate weighted performance metrics if available
                if 'win_rate' in summary_df.columns:
                    weighted_win_rate = (summary_df['win_rate'] * summary_df['total_signals']).sum() / total_signals if total_signals > 0 else 0
                    weighted_profit_factor = (summary_df['profit_factor'] * summary_df['total_signals']).sum() / total_signals if total_signals > 0 else 0
                    weighted_avg_win = (summary_df['avg_win'] * summary_df['total_signals']).sum() / total_signals if total_signals > 0 else 0
                    weighted_avg_loss = (summary_df['avg_loss'] * summary_df['total_signals']).sum() / total_signals if total_signals > 0 else 0
                    weighted_avg_holding = (summary_df['avg_holding_period'] * summary_df['total_signals']).sum() / total_signals if total_signals > 0 else 0
                    weighted_long_win_rate = (summary_df['long_win_rate'] * summary_df['long_signals']).sum() / total_long if total_long > 0 else 0
                    
                    print("\n===== AVERAGE PERFORMANCE METRICS =====")
                    print(f"Overall Win Rate: {weighted_win_rate:.2f}%")
                    print(f"Overall Profit Factor: {weighted_profit_factor:.2f}")
                    print(f"Overall Average Win: ${weighted_avg_win:.2f}")
                    print(f"Overall Average Loss: ${weighted_avg_loss:.2f}")
                    print(f"Overall Average Holding Period: {weighted_avg_holding:.1f} days")
                    print(f"Overall LONG Win Rate: {weighted_long_win_rate:.2f}%")
                    
                    # Display tier performance metrics if available
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
                                            'long_count': []
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
                                
                                combined_tier_metrics[tier] = {
                                    'win_rate': weighted_win_rate,
                                    'avg_pl': weighted_avg_pl,
                                    'trade_count': total_trade_count,
                                    'long_win_rate': long_win_rate,
                                    'long_count': total_long_count
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
                                'long_count': tier_data['long_count']
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
        else:
            # Run backtest for the specified quarter
            start_date, end_date = quarters_map[quarters[0]]
            if multiple_runs:
                # Run multiple backtests and average results
                summary = run_multiple_backtests(quarters[0], start_date, end_date, max_signals, initial_capital, num_runs, random_seed, continuous_capital=continuous_capital, previous_capital=None, weekly_selection=weekly_selection, tier1_threshold=tier1_threshold, tier2_threshold=tier2_threshold, tier3_threshold=tier3_threshold)
                signals = None  # Signals are not returned from multiple runs
            else:
                # Run a single backtest
                summary, signals, results_file = run_quarter_backtest(quarters[0], start_date, end_date, max_signals, initial_capital, random_seed, weekly_selection=weekly_selection, tier1_threshold=tier1_threshold, tier2_threshold=tier2_threshold, tier3_threshold=tier3_threshold)
                
                if results_file:
                    with open(results_file, 'r') as f:
                        data = json.load(f)
                        
                    if 'summary' in data:
                        summary = data['summary']
                    if 'signals' in data:
                        signals = data['signals']
            
            if summary:
                print(f"\nTotal signals: {summary['total_signals']}")
                print(f"LONG signals: {summary['long_signals']} ({summary['long_signals']/summary['total_signals']*100 if summary['total_signals'] > 0 else 0:.1f}%)")
                
                # Add performance metrics if available
                if summary and 'win_rate' in summary:
                    print("\n===== PERFORMANCE METRICS =====")
                    print(f"Win Rate: {summary['win_rate']:.2f}%")
                    print(f"Profit Factor: {summary['profit_factor']:.2f}")
                    print(f"Average Win: ${summary['avg_win']:.2f}")
                    print(f"Average Loss: ${summary['avg_loss']:.2f}")
                    print(f"Average Holding Period: {summary['avg_holding_period']:.1f} days")
                    
                    # Access long_win_rate directly from the summary if available
                    if 'long_win_rate' in summary:
                        print(f"LONG Win Rate: {summary['long_win_rate']:.2f}%")
                    # Otherwise, try to find it in tier_metrics
                    elif 'tier_metrics' in summary and isinstance(summary['tier_metrics'], dict):
                        # Check if any tier has long_win_rate
                        for tier_name, tier_data in summary['tier_metrics'].items():
                            if isinstance(tier_data, dict) and 'long_win_rate' in tier_data:
                                print(f"LONG Win Rate: {tier_data['long_win_rate']:.2f}%")
                                break
                    
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
                                        long_trades = perf_df[f"{base_name}_long_trades"].values[0] if f"{base_name}_long_trades" in perf_df.columns else 0
                                        
                                        # Display direction-specific metrics
                                        if long_trades > 0:
                                            print(f"  LONG: Win Rate {long_win_rate:.2f}% ({int(long_trades)} trades)")
                            except Exception as e:
                                logger.warning(f"Error loading tier performance metrics: {e}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

def run_all_quarters_backtest(max_signals=100, initial_capital=300, multiple_runs=False, num_runs=5, continuous_capital=False, weekly_selection=False, tier1_threshold=0.8, tier2_threshold=0.7, tier3_threshold=0.6):
    """Run comprehensive backtests for all quarters"""
    try:
        # Load configuration
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sp500_config.yaml')
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
        quarter_results = []
        
        previous_capital = None
        for quarter, (start_date, end_date) in quarters.items():
            if multiple_runs:
                # Run multiple backtests and average results
                summary = run_multiple_backtests(
                    quarter, 
                    start_date, 
                    end_date, 
                    max_signals, 
                    initial_capital, 
                    num_runs, 
                    continuous_capital=continuous_capital, 
                    previous_capital=previous_capital,
                    weekly_selection=weekly_selection,
                    tier1_threshold=tier1_threshold,
                    tier2_threshold=tier2_threshold,
                    tier3_threshold=tier3_threshold
                )
                signals = None  # Signals are not returned from multiple runs
                if summary and continuous_capital:
                    previous_capital = summary.get('final_capital', initial_capital)
            else:
                # Run a single backtest
                print(f"\n{'=' * 50}")
                print(f"Running backtest for {quarter}: {start_date} to {end_date}")
                print(f"{'=' * 50}")
                
                # Set initial capital for this run
                run_initial_capital = previous_capital if continuous_capital and previous_capital else initial_capital
                
                # Run backtest for this quarter
                summary, signals, results_file = run_quarter_backtest(
                    quarter, 
                    start_date, 
                    end_date, 
                    max_signals=max_signals, 
                    initial_capital=run_initial_capital,
                    random_seed=42,
                    continuous_capital=continuous_capital,
                    previous_capital=previous_capital,
                    weekly_selection=weekly_selection,
                    tier1_threshold=tier1_threshold,
                    tier2_threshold=tier2_threshold,
                    tier3_threshold=tier3_threshold
                )
                
                with open(results_file, 'r') as f:
                    data = json.load(f)
                    summary = data['summary']
                    signals = data['signals']
            
            # Store metrics for this quarter
            if summary:
                all_metrics[quarter] = summary
            
            if signals:
                # Convert signals to DataFrame
                signals_df = pd.DataFrame(signals)
                
                # Count signals by direction
                long_signals = signals_df[signals_df["direction"] == "LONG"]
                
                print(f"\nTotal signals: {len(signals_df)}")
                print(f"LONG signals: {len(long_signals)} ({len(long_signals)/len(signals_df)*100:.1f}%)")
                
                # Average scores
                avg_long_score = long_signals['score'].mean() if not long_signals.empty else 0
                print(f"\nAverage LONG score: {avg_long_score:.3f}")
                
                # Add to summary data
                quarter_summary = {
                    'quarter': quarter,
                    'total_signals': len(signals_df),
                    'long_signals': len(long_signals),
                    'long_pct': len(long_signals)/len(signals_df)*100 if len(signals_df) > 0 else 0,
                    'avg_long_score': avg_long_score
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
                    
                    # Add LONG win rate if available
                    if 'tier_metrics' in summary:
                        tier_metrics = summary['tier_metrics']
                        tier1_metrics = tier_metrics.get('Tier 1 (≥0.9)', {})
                        quarter_summary['long_win_rate'] = tier1_metrics.get('long_win_rate', 0)
                
                quarter_results.append(quarter_summary)
        
        # Create summary DataFrame
        if quarter_results:
            summary_df = pd.DataFrame(quarter_results)
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
            
            print(f"\nOverall statistics:")
            print(f"Total signals across all quarters: {total_signals}")
            print(f"Total LONG signals: {total_long} ({total_long/total_signals*100:.1f}%)")
            
            # Average scores weighted by signal count
            weighted_long_score = (summary_df['avg_long_score'] * summary_df['long_signals']).sum() / total_long if total_long > 0 else 0
            
            print(f"Weighted average LONG score: {weighted_long_score:.3f}")
            
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
                
                print(f"Overall Win Rate: {weighted_win_rate:.2f}%")
                print(f"Overall Profit Factor: {weighted_profit_factor:.2f}")
                print(f"Overall Average Win: ${weighted_avg_win:.2f}")
                print(f"Overall Average Loss: ${weighted_avg_loss:.2f}")
                print(f"Overall Average Holding Period: {weighted_avg_holding:.1f} days")
                print(f"Overall LONG Win Rate: {weighted_long_win_rate:.2f}%")
                
        return all_metrics
    except Exception as e:
        print(f"Error running all quarters backtest: {str(e)}")
        traceback.print_exc()
        return []

def run_backtest_for_web(start_date, end_date, max_signals=100, initial_capital=300, continuous_capital=False, weekly_selection=False):
    """
    Run a backtest for a specific date range using real data - designed for web interface integration
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        max_signals (int): Maximum number of signals to generate
        initial_capital (float): Initial capital for the backtest
        continuous_capital (bool): Whether to use continuous capital
        weekly_selection (bool): Whether to use weekly selection
        
    Returns:
        dict: Backtest results in the format expected by the web interface
    """
    logger.info(f"[DEBUG] Starting run_backtest_for_web in run_comprehensive_backtest.py")
    logger.info(f"[DEBUG] Running backtest from {start_date} to {end_date}")
    logger.info(f"[DEBUG] Parameters: max_signals={max_signals}, initial_capital={initial_capital}, continuous_capital={continuous_capital}, weekly_selection={weekly_selection}")
    
    # Extract quarter information from dates
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    
    year = start_dt.year
    quarter = (start_dt.month - 1) // 3 + 1
    quarter_name = f"Q{quarter}_{year}"
    
    try:
        # Import the run_backtest function from final_sp500_strategy
        logger.info(f"[DEBUG] Importing run_backtest from final_sp500_strategy")
        from final_sp500_strategy import run_backtest
        
        # Run the backtest
        logger.info(f"[DEBUG] Calling run_backtest from final_sp500_strategy")
        logger.info(f"[DEBUG] This should fetch real data from Alpaca API")
        start_time = time.time()
        
        summary, signals = run_backtest(
            start_date, 
            end_date, 
            mode='backtest', 
            max_signals=max_signals, 
            initial_capital=initial_capital,
            weekly_selection=weekly_selection,
            continuous_capital=continuous_capital)
        
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"[DEBUG] run_backtest execution time: {execution_time:.2f} seconds")
        
        # Check if summary is None
        if summary is None:
            logger.error("[DEBUG] Backtest returned None summary. This indicates an error in the backtest execution.")
            raise Exception("Backtest execution failed to return valid results")
        
        # Log some details about the results to verify they're real
        logger.info(f"[DEBUG] Backtest returned summary with {len(signals) if signals else 0} signals")
        if signals and len(signals) > 0:
            logger.info(f"[DEBUG] First few signals: {signals[:3]}")
        
        # Create result object in the format expected by the web interface
        result = {
            'summary': summary,
            'trades': signals if signals else [],
            'parameters': {
                'max_signals': max_signals,
                'initial_capital': initial_capital,
                'continuous_capital': continuous_capital,
                'weekly_selection': weekly_selection
            }
        }
        
        logger.info(f"[DEBUG] Backtest completed with {summary.get('total_trades', 0)} trades")
        logger.info(f"[DEBUG] Win rate: {summary.get('win_rate', 0)}%, Profit factor: {summary.get('profit_factor', 0)}, Total return: {summary.get('total_return', 0)}%")
        
        return result
    
    except Exception as e:
        logger.error(f"[DEBUG] Error running backtest: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Raise the exception to be handled by the caller
        raise Exception(f"Error running backtest: {str(e)}")

def main():
    """Main function to run the comprehensive backtest"""
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Run comprehensive backtest for specified quarters')
        parser.add_argument('quarters', nargs='*', help='Quarters to run backtest for (e.g., Q1_2023 Q2_2023)')
        parser.add_argument('--start_date', type=str, help='Custom start date (YYYY-MM-DD) for backtest')
        parser.add_argument('--end_date', type=str, help='Custom end date (YYYY-MM-DD) for backtest')
        parser.add_argument('--max_signals', type=int, default=100, help='Maximum number of signals to use')
        parser.add_argument('--initial_capital', type=float, default=300, help='Initial capital for the backtest')
        parser.add_argument('--multiple_runs', action='store_true', help='Run multiple backtests and average results')
        parser.add_argument('--num_runs', type=int, default=5, help='Number of backtest runs to perform when using --multiple_runs')
        parser.add_argument('--random_seed', type=int, default=42, help='Base random seed for reproducibility')
        parser.add_argument('--continuous_capital', action='store_true', help='Use continuous capital across quarters')
        parser.add_argument('--weekly_selection', action='store_true', help='Enable weekly stock selection refresh')
        parser.add_argument('--tier1_threshold', type=float, default=0.8, help='Threshold for tier 1 signals')
        parser.add_argument('--tier2_threshold', type=float, default=0.7, help='Threshold for tier 2 signals')
        parser.add_argument('--tier3_threshold', type=float, default=0.6, help='Threshold for tier 3 signals')
        args = parser.parse_args()
        
        # Define quarters mapping
        quarters_map = {
            'Q1_2023': ('2023-01-01', '2023-03-31'),
            'Q2_2023': ('2023-04-01', '2023-06-30'),
            'Q3_2023': ('2023-07-01', '2023-09-30'),
            'Q4_2023': ('2023-10-01', '2023-12-31'),
            'Q1_2024': ('2024-01-01', '2024-03-31')
        }
        
        # Check if custom date range is provided
        if args.start_date and args.end_date:
            print(f"Running backtest with custom date range: {args.start_date} to {args.end_date}")
            # Create a unique identifier for this custom date range
            custom_quarter = f"Custom_{args.start_date}_to_{args.end_date}"
            run_quarter_backtest(
                custom_quarter,
                args.start_date,
                args.end_date,
                max_signals=args.max_signals,
                initial_capital=args.initial_capital,
                weekly_selection=args.weekly_selection,
                tier1_threshold=args.tier1_threshold,
                tier2_threshold=args.tier2_threshold,
                tier3_threshold=args.tier3_threshold
            )
            return
        
        # Check if 'all' is specified
        if 'all' in args.quarters:
            print("Running backtest for all quarters")
            run_all_quarters_backtest(
                max_signals=args.max_signals, 
                initial_capital=args.initial_capital,
                multiple_runs=args.multiple_runs,
                num_runs=args.num_runs,
                continuous_capital=args.continuous_capital,
                weekly_selection=args.weekly_selection,
                tier1_threshold=args.tier1_threshold,
                tier2_threshold=args.tier2_threshold,
                tier3_threshold=args.tier3_threshold
            )
            return
        
        # Run backtest for each quarter
        for quarter in args.quarters:
            run_comprehensive_backtest(
                [quarter], 
                max_signals=args.max_signals, 
                initial_capital=args.initial_capital,
                multiple_runs=args.multiple_runs,
                num_runs=args.num_runs,
                random_seed=args.random_seed,
                continuous_capital=args.continuous_capital,
                weekly_selection=args.weekly_selection,
                tier1_threshold=args.tier1_threshold,
                tier2_threshold=args.tier2_threshold,
                tier3_threshold=args.tier3_threshold
            )
    except Exception as e:
        logger.error(f"Error running comprehensive backtest: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
