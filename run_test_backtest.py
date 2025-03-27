import os
import yaml
import logging
import datetime
import pandas as pd
import numpy as np
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our backtest module
from fixed_backtest_v2 import run_backtest
from fixed_signal_generator import generate_signals

# Override the generate_signals function for testing
def generate_test_signals(start_date, end_date, config, alpaca=None):
    """Generate synthetic signals for testing"""
    signals = []
    
    # Convert dates to datetime
    start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    
    # Generate dates
    dates = []
    current_dt = start_dt
    while current_dt <= end_dt:
        if current_dt.weekday() < 5:  # Weekdays only
            dates.append(current_dt.strftime('%Y-%m-%d'))
        current_dt += datetime.timedelta(days=1)
    
    # Generate signals for each date
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'AMD', 'INTC', 'IBM']
    
    for date in dates:
        # Generate 2-3 signals per day
        num_signals = np.random.randint(2, 4)
        selected_symbols = np.random.choice(symbols, num_signals, replace=False)
        
        for symbol in selected_symbols:
            # Generate random score between 0.6 and 0.9
            score = np.random.uniform(0.6, 0.9)
            
            # Create signal
            signal = {
                'symbol': symbol,
                'date': date,
                'score': score,
                'direction': 'LONG',
                'price': np.random.uniform(50, 200)
            }
            
            signals.append(signal)
    
    logger.info(f"Generated {len(signals)} test signals")
    return signals

# Monkey patch the generate_signals function for testing
import fixed_backtest_v2
fixed_backtest_v2.generate_signals = generate_test_signals

def test_backtest():
    """Run a test backtest to verify implementation"""
    # Set test parameters
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    initial_capital = 100000
    
    logger.info(f"Running test backtest from {start_date} to {end_date} with ${initial_capital} initial capital")
    
    # Run backtest with test signals
    results = run_backtest(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        config_path='sp500_config.yaml',
        min_score=0.6,
        max_signals=20
    )
    
    # Display results
    if results and 'performance' in results:
        perf = results['performance']
        print("\n===== BACKTEST RESULTS =====")
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
    else:
        print("Backtest failed or returned no results")

if __name__ == "__main__":
    test_backtest()
