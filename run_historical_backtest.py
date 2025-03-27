import os
import sys
import yaml
import logging
import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our backtest module
from fixed_backtest_v2 import run_backtest
from fixed_signal_generator import generate_signals

def load_config(config_file='sp500_config.yaml'):
    """Load configuration from YAML file"""
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    else:
        logger.error(f"Config file {config_file} not found")
        return {}

def run_historical_backtest(start_date, end_date, initial_capital=100000, min_score=0.6, max_signals=20):
    """Run a backtest with real historical data"""
    logger.info(f"Running historical backtest from {start_date} to {end_date} with ${initial_capital} initial capital")
    
    # Load configuration
    config = load_config()
    
    # Run backtest
    results = run_backtest(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        config_path='sp500_config.yaml',
        min_score=min_score,
        max_signals=max_signals
    )
    
    # Display results
    if results and 'performance' in results:
        perf = results['performance']
        print("\n===== BACKTEST RESULTS =====")
        print(f"Period: {start_date} to {end_date}")
        print(f"Initial Capital: ${perf.get('initial_capital', 0):.2f}")
        print(f"Final Value: ${perf.get('final_value', 0):.2f}")
        print(f"Return: {perf.get('return', 0):.2f}%")
        print(f"Annualized Return: {perf.get('annualized_return', 0):.2f}%")
        print(f"Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown: {perf.get('max_drawdown', 0):.2f}%")
        print(f"Win Rate: {perf.get('win_rate', 0):.2f}%")
        print(f"Total Trades: {perf.get('total_trades', 0)}")
        print(f"Winning Trades: {perf.get('winning_trades', 0)}")
        print(f"Losing Trades: {perf.get('losing_trades', 0)}")
        print("============================\n")
        
        # Check if portfolio has open positions
        if 'portfolio' in results:
            portfolio = results['portfolio']
            print(f"Open Positions: {len(portfolio.open_positions)}")
            print(f"Closed Positions: {len(portfolio.closed_positions)}")
            print(f"Final Cash: ${portfolio.cash:.2f}")
            
            # Print sample of trades if available
            if hasattr(portfolio, 'trade_history') and portfolio.trade_history:
                print("\n===== SAMPLE TRADES =====")
                for i, trade in enumerate(portfolio.trade_history[:5]):
                    print(f"Trade {i+1}: {trade['action']} {trade['symbol']} - {trade['shares']} shares @ ${trade['price']:.2f}")
                print("=========================\n")
        
        return results
    else:
        print("Backtest failed or returned no results")
        return None

def run_parameter_optimization(start_date, end_date, initial_capital=100000):
    """Run parameter optimization to find the best combination of parameters"""
    logger.info("Starting parameter optimization...")
    
    # Define parameter ranges to test
    min_scores = [0.5, 0.6, 0.7]
    max_signals_list = [10, 15, 20, 25]
    position_sizes = [0.02, 0.03, 0.04, 0.05]
    stop_losses = [0.05, 0.07, 0.1]
    take_profits = [0.15, 0.2, 0.25, 0.3]
    
    # Load configuration
    config = load_config()
    
    # Create a results dataframe
    results_df = pd.DataFrame(columns=[
        'min_score', 'max_signals', 'position_size', 'stop_loss', 'take_profit',
        'return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'total_trades'
    ])
    
    # Track best parameters
    best_return = -float('inf')
    best_params = {}
    
    # Total number of parameter combinations
    total_combinations = len(min_scores) * len(max_signals_list) * len(position_sizes) * len(stop_losses) * len(take_profits)
    logger.info(f"Testing {total_combinations} parameter combinations...")
    
    # Create progress bar
    pbar = tqdm(total=total_combinations)
    
    # Test each parameter combination
    for min_score in min_scores:
        for max_signals in max_signals_list:
            for position_size in position_sizes:
                for stop_loss in stop_losses:
                    for take_profit in take_profits:
                        # Update config with test parameters
                        test_config = config.copy()
                        if 'portfolio' not in test_config:
                            test_config['portfolio'] = {}
                        test_config['portfolio']['position_size'] = position_size
                        test_config['portfolio']['stop_loss'] = stop_loss
                        test_config['portfolio']['take_profit'] = take_profit
                        
                        # Save temporary config
                        temp_config_path = 'temp_config.yaml'
                        with open(temp_config_path, 'w') as f:
                            yaml.dump(test_config, f)
                        
                        # Run backtest with these parameters
                        results = run_backtest(
                            start_date=start_date,
                            end_date=end_date,
                            initial_capital=initial_capital,
                            config_path=temp_config_path,
                            min_score=min_score,
                            max_signals=max_signals
                        )
                        
                        # Extract performance metrics
                        if results and 'performance' in results:
                            perf = results['performance']
                            
                            # Add to results dataframe
                            results_df = pd.concat([results_df, pd.DataFrame({
                                'min_score': [min_score],
                                'max_signals': [max_signals],
                                'position_size': [position_size],
                                'stop_loss': [stop_loss],
                                'take_profit': [take_profit],
                                'return': [perf.get('return', 0)],
                                'sharpe_ratio': [perf.get('sharpe_ratio', 0)],
                                'max_drawdown': [perf.get('max_drawdown', 0)],
                                'win_rate': [perf.get('win_rate', 0)],
                                'total_trades': [perf.get('total_trades', 0)]
                            })], ignore_index=True)
                            
                            # Check if this is the best combination so far
                            current_return = perf.get('return', 0)
                            if current_return > best_return:
                                best_return = current_return
                                best_params = {
                                    'min_score': min_score,
                                    'max_signals': max_signals,
                                    'position_size': position_size,
                                    'stop_loss': stop_loss,
                                    'take_profit': take_profit,
                                    'return': current_return,
                                    'sharpe_ratio': perf.get('sharpe_ratio', 0),
                                    'max_drawdown': perf.get('max_drawdown', 0),
                                    'win_rate': perf.get('win_rate', 0),
                                    'total_trades': perf.get('total_trades', 0)
                                }
                        
                        # Clean up temporary config
                        if os.path.exists(temp_config_path):
                            os.remove(temp_config_path)
                        
                        # Update progress bar
                        pbar.update(1)
    
    # Close progress bar
    pbar.close()
    
    # Save results to CSV
    results_df.to_csv('parameter_optimization_results.csv', index=False)
    logger.info("Parameter optimization results saved to parameter_optimization_results.csv")
    
    # Display best parameters
    print("\n===== BEST PARAMETERS =====")
    print(f"Min Score: {best_params.get('min_score')}")
    print(f"Max Signals: {best_params.get('max_signals')}")
    print(f"Position Size: {best_params.get('position_size')}")
    print(f"Stop Loss: {best_params.get('stop_loss')}")
    print(f"Take Profit: {best_params.get('take_profit')}")
    print(f"Return: {best_params.get('return')}%")
    print(f"Sharpe Ratio: {best_params.get('sharpe_ratio')}")
    print(f"Max Drawdown: {best_params.get('max_drawdown')}%")
    print(f"Win Rate: {best_params.get('win_rate')}%")
    print(f"Total Trades: {best_params.get('total_trades')}")
    print("===========================\n")
    
    return best_params

if __name__ == "__main__":
    # Run a historical backtest for Q1 2023
    print("Running historical backtest for Q1 2023...")
    run_historical_backtest(
        start_date='2023-01-01',
        end_date='2023-03-31',
        initial_capital=100000,
        min_score=0.6,
        max_signals=20
    )
    
    # Ask if user wants to run parameter optimization
    run_optimization = input("\nDo you want to run parameter optimization? (y/n): ")
    if run_optimization.lower() == 'y':
        print("\nRunning parameter optimization...")
        run_parameter_optimization(
            start_date='2023-01-01',
            end_date='2023-03-31',
            initial_capital=100000
        )
    else:
        print("\nSkipping parameter optimization.")
