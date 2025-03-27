import logging
import os
import yaml
from datetime import datetime
from final_sp500_strategy import run_backtest

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path='sp500_config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def run_quarterly_backtests(year=2023, initial_capital=10000):
    """
    Run backtests for each quarter of the specified year.
    
    Args:
        year (int): Year to run backtests for
        initial_capital (float): Initial capital for the first quarter
    """
    # Define quarters
    quarters = [
        {'name': 'Q1', 'start': f'{year}-01-01', 'end': f'{year}-03-31'},
        {'name': 'Q2', 'start': f'{year}-04-01', 'end': f'{year}-06-30'},
        {'name': 'Q3', 'start': f'{year}-07-01', 'end': f'{year}-09-30'},
        {'name': 'Q4', 'start': f'{year}-10-01', 'end': f'{year}-12-31'}
    ]
    
    # Create results directory
    results_dir = f'backtest_results_{year}'
    os.makedirs(results_dir, exist_ok=True)
    
    # Run backtest for each quarter
    capital = initial_capital
    quarterly_results = []
    
    for quarter in quarters:
        logger.info(f"Running backtest for {quarter['name']} {year}: {quarter['start']} to {quarter['end']}")
        
        # Run backtest
        results = run_backtest(
            start_date=quarter['start'],
            end_date=quarter['end'],
            mode='backtest',
            initial_capital=capital,
            random_seed=42,  # Use fixed seed for reproducibility
            continuous_capital=True,
            previous_capital=capital,
            config_path='sp500_config.yaml',
            max_signals=40,
            tier1_threshold=0.8,
            tier2_threshold=0.7,
            tier3_threshold=0.6,
            largecap_allocation=0.7,
            midcap_allocation=0.3
        )
        
        # Update capital for next quarter if results were returned
        if results and 'final_portfolio_value' in results:
            capital = results['final_portfolio_value']
            
        # Store results
        quarterly_results.append({
            'quarter': quarter['name'],
            'start_date': quarter['start'],
            'end_date': quarter['end'],
            'results': results
        })
        
        # Save quarterly results to file
        with open(os.path.join(results_dir, f"{quarter['name']}_{year}_results.txt"), 'w') as f:
            f.write(f"Backtest Results for {quarter['name']} {year}\n")
            f.write(f"Period: {quarter['start']} to {quarter['end']}\n")
            f.write(f"Initial Capital: ${capital:.2f}\n")
            
            if results:
                f.write(f"Final Portfolio Value: ${results.get('final_portfolio_value', 0):.2f}\n")
                f.write(f"Return: {results.get('return', 0):.2f}%\n")
                f.write(f"Annualized Return: {results.get('annualized_return', 0):.2f}%\n")
                f.write(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}\n")
                f.write(f"Max Drawdown: {results.get('max_drawdown', 0):.2f}%\n")
                f.write(f"Win Rate: {results.get('win_rate', 0):.2f}%\n")
                f.write(f"Total Trades: {results.get('total_trades', 0)}\n")
            else:
                f.write("No results returned from backtest\n")
    
    # Print summary
    logger.info(f"Completed backtests for all quarters of {year}")
    logger.info(f"Initial Capital: ${initial_capital:.2f}")
    
    if quarterly_results and quarterly_results[-1]['results']:
        final_capital = quarterly_results[-1]['results'].get('final_portfolio_value', initial_capital)
        annual_return = ((final_capital / initial_capital) - 1) * 100
        logger.info(f"Final Capital: ${final_capital:.2f}")
        logger.info(f"Annual Return: {annual_return:.2f}%")
    else:
        logger.info("Could not calculate annual return due to missing results")
    
    return quarterly_results

if __name__ == "__main__":
    logger.info("Starting 2023 quarterly backtests")
    results = run_quarterly_backtests(year=2023, initial_capital=10000)
    logger.info("Completed 2023 quarterly backtests")
