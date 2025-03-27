#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fixed Backtest Integration
-------------------------
This module integrates the fixed backtest implementation with the web interface.
"""

import os
import sys
import json
import yaml
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import requests
from typing import List, Dict, Any, Tuple, Optional

# Import our fixed backtest implementation
from fixed_backtest_final import run_backtest
from alpaca_api import AlpacaAPI
from portfolio import Portfolio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/fixed_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_quarter_dates(quarter):
    """
    Get start and end dates for a specific quarter
    
    Args:
        quarter (str): Quarter identifier (e.g., 'Q1_2023')
        
    Returns:
        tuple: (start_date, end_date) in YYYY-MM-DD format
    """
    if '_' not in quarter:
        # Handle format like 'Q12023'
        quarter_num = int(quarter[1])
        year = int(quarter[2:])
    else:
        # Handle format like 'Q1_2023'
        parts = quarter.split('_')
        quarter_num = int(parts[0][1])
        year = int(parts[1])
    
    # Calculate start and end dates based on quarter
    if quarter_num == 1:
        start_date = f"{year}-01-01"
        end_date = f"{year}-03-31"
    elif quarter_num == 2:
        start_date = f"{year}-04-01"
        end_date = f"{year}-06-30"
    elif quarter_num == 3:
        start_date = f"{year}-07-01"
        end_date = f"{year}-09-30"
    elif quarter_num == 4:
        start_date = f"{year}-10-01"
        end_date = f"{year}-12-31"
    else:
        raise ValueError(f"Invalid quarter: {quarter}")
    
    return start_date, end_date

def get_sp500_symbols() -> List[str]:
    """
    Get list of S&P 500 symbols from Wikipedia
    
    Returns:
        List[str]: List of S&P 500 symbols
    """
    logger.info("Fetching S&P 500 symbols from Wikipedia")
    
    try:
        # Fetch S&P 500 symbols from Wikipedia
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the table with S&P 500 companies
        table = soup.find('table', {'class': 'wikitable'})
        symbols = []
        
        # Extract symbols from the table
        for row in table.find_all('tr')[1:]:
            symbol = row.find_all('td')[0].text.strip()
            symbols.append(symbol)
        
        logger.info(f"Found {len(symbols)} S&P 500 symbols")
        return symbols
    
    except Exception as e:
        logger.error(f"Error fetching S&P 500 symbols: {str(e)}")
        # Return a default list of major S&P 500 components if fetching fails
        default_symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V']
        logger.warning(f"Using default symbols: {default_symbols}")
        return default_symbols

def get_midcap_symbols() -> List[str]:
    """
    Get list of mid-cap symbols (S&P 400)
    
    Returns:
        List[str]: List of mid-cap symbols
    """
    logger.info("Fetching mid-cap symbols")
    
    try:
        # Fetch mid-cap symbols from Wikipedia (S&P 400)
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the table with mid-cap companies
        table = soup.find('table', {'class': 'wikitable'})
        symbols = []
        
        # Extract symbols from the table
        for row in table.find_all('tr')[1:]:
            symbol = row.find_all('td')[0].text.strip()
            symbols.append(symbol)
        
        logger.info(f"Found {len(symbols)} mid-cap symbols")
        return symbols
    
    except Exception as e:
        logger.error(f"Error fetching mid-cap symbols: {str(e)}")
        # Return a default list of mid-cap symbols if fetching fails
        default_symbols = ['STLD', 'AXON', 'DECK', 'BLDR', 'CGNX', 'EXAS', 'LSTR', 'MANH', 'NDSN', 'RPM']
        logger.warning(f"Using default mid-cap symbols: {default_symbols}")
        return default_symbols

def get_combined_universe(universe_size: int = 50, midcap_percentage: float = 0.2) -> List[str]:
    """
    Get a combined universe of S&P 500 and mid-cap symbols
    
    Args:
        universe_size (int): Total size of the universe
        midcap_percentage (float): Percentage of mid-cap stocks to include (0.0-1.0)
    
    Returns:
        List[str]: Combined list of symbols
    """
    # Get S&P 500 symbols
    sp500_symbols = get_sp500_symbols()
    
    # Get mid-cap symbols
    midcap_symbols = get_midcap_symbols()
    
    # Calculate number of mid-cap symbols to include
    midcap_count = int(universe_size * midcap_percentage)
    sp500_count = universe_size - midcap_count
    
    # Ensure we don't exceed available symbols
    sp500_count = min(sp500_count, len(sp500_symbols))
    midcap_count = min(midcap_count, len(midcap_symbols))
    
    # Randomly select symbols
    np.random.shuffle(sp500_symbols)
    np.random.shuffle(midcap_symbols)
    
    # Combine symbols
    combined_universe = sp500_symbols[:sp500_count] + midcap_symbols[:midcap_count]
    np.random.shuffle(combined_universe)
    
    logger.info(f"Created combined universe with {len(combined_universe)} symbols "
                f"({sp500_count} S&P 500, {midcap_count} mid-cap)")
    
    return combined_universe

def run_fixed_backtest(quarter, max_signals=5, initial_capital=10000, tier1_threshold=0.8, 
                       tier2_threshold=0.7, tier3_threshold=0.0, universe_size=50, 
                       midcap_percentage=0.2, random_seed=42):
    """
    Run the fixed backtest for a specific quarter
    
    Args:
        quarter (str): Quarter identifier (e.g., 'Q1_2023')
        max_signals (int): Maximum number of signals to use
        initial_capital (float): Initial capital for the backtest
        tier1_threshold (float): Threshold for tier 1 signals
        tier2_threshold (float): Threshold for tier 2 signals
        tier3_threshold (float): Threshold for tier 3 signals (set to 0.0 to ignore tier 3)
        universe_size (int): Number of symbols to include in the universe
        midcap_percentage (float): Percentage of mid-cap stocks to include (0.0-1.0)
        random_seed (int): Random seed for reproducibility
        
    Returns:
        dict: Backtest results
    """
    logger.info(f"Running fixed backtest for {quarter} with {max_signals} max signals and ${initial_capital} initial capital")
    
    # Set random seed
    np.random.seed(random_seed)
    
    # Get start and end dates for the quarter
    start_date, end_date = get_quarter_dates(quarter)
    
    # Load configuration
    try:
        with open('sp500_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        config = {
            'alpaca': {
                'api_key': os.environ.get('ALPACA_API_KEY', ''),
                'api_secret': os.environ.get('ALPACA_API_SECRET', ''),
                'base_url': 'https://paper-api.alpaca.markets',
                'data_url': 'https://data.alpaca.markets'
            }
        }
    
    # Initialize API
    api = AlpacaAPI(
        api_key=config['alpaca']['api_key'],
        api_secret=config['alpaca']['api_secret'],
        base_url=config['alpaca']['base_url'],
        data_url=config['alpaca']['data_url']
    )
    
    # Get universe of symbols - now includes mid-cap stocks
    universe = get_combined_universe(universe_size=universe_size, midcap_percentage=midcap_percentage)
    
    # Run backtest
    results = run_backtest(
        api=api,
        strategy_name=f"sp500_strategy_{quarter}",
        universe=universe,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        max_signals=max_signals,
        min_score=tier3_threshold,  # Use tier3 as minimum score (0.0 to ignore tier 3)
        tier1_threshold=tier1_threshold,
        tier2_threshold=tier2_threshold,
        tier3_threshold=tier3_threshold
    )
    
    # Extract performance metrics
    performance = results['performance']
    portfolio = results['portfolio']
    
    # Create a summary for compatibility with the web interface
    summary = {
        'win_rate': performance['win_rate'],
        'profit_factor': performance['profit_factor'] if 'profit_factor' in performance else 0,
        'total_return': performance['return_pct'],
        'annualized_return': performance['annualized_return'],
        'sharpe_ratio': performance['sharpe_ratio'],
        'max_drawdown': performance['max_drawdown'],
        'final_equity': performance['final_equity'],
        'initial_capital': initial_capital,
        'start_date': start_date,
        'end_date': end_date,
        'quarter': quarter
    }
    
    # Create signals list for compatibility with the web interface
    signals = []
    for symbol, position in portfolio.open_positions.items():
        signals.append({
            'symbol': symbol,
            'entry_price': position.entry_price,
            'entry_time': position.entry_time.strftime('%Y-%m-%d'),
            'direction': position.direction,
            'position_size': position.position_size,
            'tier': position.tier if hasattr(position, 'tier') else None
        })
    
    # Add closed positions
    for position in portfolio.closed_positions:
        signals.append({
            'symbol': position.symbol,
            'entry_price': position.entry_price,
            'exit_price': position.exit_price,
            'entry_time': position.entry_time.strftime('%Y-%m-%d'),
            'exit_time': position.exit_time.strftime('%Y-%m-%d'),
            'direction': position.direction,
            'pnl': position.pnl,
            'pnl_pct': position.pnl_pct,
            'exit_reason': position.exit_reason,
            'tier': position.tier if hasattr(position, 'tier') else None
        })
    
    # Save results to file
    os.makedirs("backtest_results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"backtest_results/fixed_backtest_{quarter}_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'summary': summary,
            'signals': signals,
            'equity_curve': [
                {
                    'timestamp': entry['timestamp'].strftime('%Y-%m-%d'),
                    'equity': entry['equity'],
                    'pct_change': entry['pct_change']
                }
                for entry in portfolio.equity_curve
            ]
        }, f, default=str)
    
    logger.info(f"Backtest completed. Results saved to {results_file}")
    logger.info(f"Final portfolio value: ${performance['final_equity']:.2f}")
    logger.info(f"Return: {performance['return_pct']:.2f}%")
    logger.info(f"Sharpe ratio: {performance['sharpe_ratio']:.2f}")
    logger.info(f"Max drawdown: {performance['max_drawdown']:.2f}%")
    
    return summary, signals, results_file, performance['final_equity']  # Return final equity for continuous capital

def run_multiple_fixed_backtests(quarters, max_signals=5, initial_capital=10000, num_runs=5, 
                                tier1_threshold=0.8, tier2_threshold=0.7, tier3_threshold=0.0, 
                                universe_size=50, midcap_percentage=0.2, random_seed=42,
                                continuous_capital=False):
    """
    Run multiple fixed backtests and average the results
    
    Args:
        quarters (list): List of quarter identifiers (e.g., ['Q1_2023', 'Q2_2023'])
        max_signals (int): Maximum number of signals to use
        initial_capital (float): Initial capital for the backtest
        num_runs (int): Number of backtest runs to perform
        tier1_threshold (float): Threshold for tier 1 signals
        tier2_threshold (float): Threshold for tier 2 signals
        tier3_threshold (float): Threshold for tier 3 signals
        universe_size (int): Number of symbols to include in the universe
        midcap_percentage (float): Percentage of mid-cap stocks to include (0.0-1.0)
        random_seed (int): Base random seed for reproducibility
        continuous_capital (bool): Whether to use continuous capital across quarters
        
    Returns:
        dict: Averaged backtest results
    """
    logger.info(f"Running {num_runs} fixed backtests for quarters: {quarters}")
    
    # Lists to store results
    all_summaries = []
    all_signals = []
    
    # Run multiple backtests
    for run_idx in range(num_runs):
        # Set a unique random seed for each run
        current_seed = random_seed + run_idx
        
        logger.info(f"Run {run_idx + 1}/{num_runs} (Seed: {current_seed})")
        
        # Initialize capital for this run
        current_capital = initial_capital
        run_summaries = []
        
        # Run backtest for each quarter
        for quarter_idx, quarter in enumerate(quarters):
            logger.info(f"Processing quarter {quarter} with initial capital ${current_capital:.2f}")
            
            # Run backtest for this quarter with the current seed and capital
            summary, signals, _, final_equity = run_fixed_backtest(
                quarter=quarter,
                max_signals=max_signals,
                initial_capital=current_capital,
                tier1_threshold=tier1_threshold,
                tier2_threshold=tier2_threshold,
                tier3_threshold=tier3_threshold,
                universe_size=universe_size,
                midcap_percentage=midcap_percentage,
                random_seed=current_seed
            )
            
            if summary:
                run_summaries.append(summary)
            
            if signals:
                all_signals.append(signals)
            
            # Update capital for next quarter if using continuous capital
            if continuous_capital and quarter_idx < len(quarters) - 1:
                current_capital = final_equity
                logger.info(f"Updated capital for next quarter: ${current_capital:.2f}")
        
        # Add this run's summaries to all summaries
        if run_summaries:
            all_summaries.extend(run_summaries)
    
    # Calculate averaged metrics
    if all_summaries:
        # Group summaries by quarter
        quarter_summaries = {}
        for summary in all_summaries:
            quarter = summary['quarter']
            if quarter not in quarter_summaries:
                quarter_summaries[quarter] = []
            quarter_summaries[quarter].append(summary)
        
        # Calculate average for each quarter
        avg_summaries = {}
        for quarter, summaries in quarter_summaries.items():
            # Initialize the averaged summary with the structure of the first summary
            avg_summary = {k: 0 for k in summaries[0].keys() if isinstance(summaries[0][k], (int, float))}
            
            # Add non-numeric fields
            for k in summaries[0].keys():
                if not isinstance(summaries[0][k], (int, float)):
                    avg_summary[k] = summaries[0][k]
            
            # Calculate averages for numeric fields
            for metric in avg_summary.keys():
                if isinstance(avg_summary[metric], (int, float)):
                    values = [s.get(metric, 0) for s in summaries]
                    avg_summary[metric] = sum(values) / len(values)
            
            # Calculate standard deviations for key metrics
            std_devs = {}
            for metric in ['win_rate', 'total_return', 'sharpe_ratio', 'max_drawdown']:
                if metric in avg_summary:
                    values = [s.get(metric, 0) for s in summaries]
                    std_devs[f"{metric}_std"] = np.std(values)
            
            # Add standard deviations to the summary
            avg_summary.update(std_devs)
            
            # Store in the averaged summaries dictionary
            avg_summaries[quarter] = avg_summary
        
        # Save averaged results to file
        os.makedirs("backtest_results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"backtest_results/fixed_backtest_avg_{'_'.join(quarters)}_{num_runs}_runs_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                'summaries': avg_summaries,
                'num_runs': num_runs,
                'continuous_capital': continuous_capital,
                'quarters': quarters
            }, f, default=str)
        
        logger.info(f"Multiple backtests completed. Averaged results saved to {results_file}")
        
        # Log average metrics for each quarter
        for quarter, avg_summary in avg_summaries.items():
            logger.info(f"Quarter {quarter}:")
            logger.info(f"  Average return: {avg_summary['total_return']:.2f}% (±{avg_summary.get('total_return_std', 0):.2f}%)")
            logger.info(f"  Average win rate: {avg_summary['win_rate']:.2f}% (±{avg_summary.get('win_rate_std', 0):.2f}%)")
        
        return avg_summaries, results_file
    
    return None, None

def main():
    """Main function to run the fixed backtest integration"""
    parser = argparse.ArgumentParser(description='Run fixed backtest integration')
    parser.add_argument('quarters', nargs='*', default=['Q1_2023'], help='Quarters to run backtest for (e.g., Q1_2023)')
    parser.add_argument('--max_signals', type=int, default=5, help='Maximum number of signals to use')
    parser.add_argument('--initial_capital', type=float, default=10000, help='Initial capital for the backtest')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of backtest runs to perform')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--continuous_capital', action='store_true', help='Use continuous capital across quarters')
    parser.add_argument('--tier1_threshold', type=float, default=0.8, help='Threshold for tier 1 signals')
    parser.add_argument('--tier2_threshold', type=float, default=0.7, help='Threshold for tier 2 signals')
    parser.add_argument('--tier3_threshold', type=float, default=0.0, help='Threshold for tier 3 signals')
    parser.add_argument('--universe_size', type=int, default=50, help='Number of symbols to include in the universe')
    parser.add_argument('--midcap_percentage', type=float, default=0.2, help='Percentage of mid-cap stocks to include (0.0-1.0)')
    
    args = parser.parse_args()
    
    # Handle 'all' quarters
    if 'all' in args.quarters:
        # Define all quarters to run
        all_quarters = [f"Q{q}_{y}" for y in range(2022, 2024) for q in range(1, 5)]
        quarters = all_quarters
    else:
        quarters = args.quarters
    
    logger.info(f"Running fixed backtest integration for quarters: {quarters}")
    
    # Run multiple backtests if num_runs > 1
    if args.num_runs > 1:
        run_multiple_fixed_backtests(
            quarters=quarters,
            max_signals=args.max_signals,
            initial_capital=args.initial_capital,
            num_runs=args.num_runs,
            tier1_threshold=args.tier1_threshold,
            tier2_threshold=args.tier2_threshold,
            tier3_threshold=args.tier3_threshold,
            universe_size=args.universe_size,
            midcap_percentage=args.midcap_percentage,
            random_seed=args.random_seed,
            continuous_capital=args.continuous_capital
        )
    else:
        # Run single backtest for each quarter
        current_capital = args.initial_capital
        
        for quarter_idx, quarter in enumerate(quarters):
            summary, signals, _, final_equity = run_fixed_backtest(
                quarter=quarter,
                max_signals=args.max_signals,
                initial_capital=current_capital,
                tier1_threshold=args.tier1_threshold,
                tier2_threshold=args.tier2_threshold,
                tier3_threshold=args.tier3_threshold,
                universe_size=args.universe_size,
                midcap_percentage=args.midcap_percentage,
                random_seed=args.random_seed
            )
            
            # Update capital for next quarter if using continuous capital
            if args.continuous_capital and quarter_idx < len(quarters) - 1:
                current_capital = final_equity
                logger.info(f"Updated capital for next quarter: ${current_capital:.2f}")

if __name__ == "__main__":
    main()
