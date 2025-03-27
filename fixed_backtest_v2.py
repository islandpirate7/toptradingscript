"""
Fixed version of the backtest function for the multi-strategy trading system.
This file contains the corrected run_backtest function with proper error handling and portfolio tracking.
"""

import logging
import os
from datetime import datetime, timedelta
import yaml
import numpy as np
import random
import pandas as pd
from portfolio import Portfolio
from fixed_signal_generator import generate_signals
from alpaca_api import AlpacaAPI

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def process_signals(signals, portfolio, date, price_data, min_score=0.7, tier1_threshold=0.8, tier2_threshold=0.7, max_signals=30, log_file_handle=None):
    """
    Process signals and execute trades.
    
    Args:
        signals (list): List of signal dictionaries
        portfolio (Portfolio): Portfolio instance
        date (str or datetime): Current trading date
        price_data (dict): Dictionary of current prices for symbols
        min_score (float): Minimum signal score to consider
        tier1_threshold (float): Threshold for tier 1 signals
        tier2_threshold (float): Threshold for tier 2 signals
        max_signals (int): Maximum number of signals to process
        log_file_handle (file): Log file handle for writing logs
        
    Returns:
        int: Number of trades executed
    """
    trades_executed = 0
    date_str = date.strftime('%Y-%m-%d')
    
    # Filter signals for current date
    day_signals = []
    for s in signals:
        signal_date = s.get('date')
        # Convert signal date to string format for comparison
        if hasattr(signal_date, 'strftime'):
            signal_date_str = signal_date.strftime('%Y-%m-%d')
        elif isinstance(signal_date, str):
            signal_date_str = signal_date
        else:
            signal_date_str = str(signal_date).split(' ')[0]
            
        if signal_date_str == date_str:
            day_signals.append(s)
    
    # Add debug logging
    if log_file_handle:
        log_file_handle.write(f"{datetime.now()} - DEBUG - Processing {len(day_signals)} signals for {date_str}\n")
    else:
        print(f"DEBUG: Processing {len(day_signals)} signals for {date_str}")
    
    # Filter by min_score
    day_signals = [s for s in day_signals if s.get('score', 0) >= min_score]
    
    # Add debug logging
    if log_file_handle:
        log_file_handle.write(f"{datetime.now()} - DEBUG - After min_score filter: {len(day_signals)} signals with score >= {min_score}\n")
    else:
        print(f"DEBUG: After min_score filter: {len(day_signals)} signals with score >= {min_score}")
    
    # Sort by score (highest first)
    day_signals = sorted(day_signals, key=lambda x: x.get('score', 0), reverse=True)
    
    # Limit to max_signals
    day_signals = day_signals[:max_signals]
    
    for signal in day_signals:
        symbol = signal.get('symbol')
        score = signal.get('score', 0)
        direction = signal.get('direction', 'long').lower()
        
        # Skip if we don't have price data for this symbol
        if symbol not in price_data:
            if log_file_handle:
                log_file_handle.write(f"{datetime.now()} - DEBUG - No price data for {symbol}, skipping\n")
            continue
            
        # Get current price
        current_price = price_data[symbol].get('close', 0)
        if current_price == 0:
            if log_file_handle:
                log_file_handle.write(f"{datetime.now()} - DEBUG - Invalid price (0) for {symbol}, skipping\n")
            continue
        
        # Determine position size based on signal tier
        if score >= 0.8:
            tier = 1
            position_size_pct = 1.5  # 150% of base position size for tier 1
            if log_file_handle:
                log_file_handle.write(f"{datetime.now()} - DEBUG - Tier 1 signal for {symbol} with score {score:.2f}\n")
            else:
                print(f"DEBUG: Tier 1 signal for {symbol} with score {score:.2f}")
        elif score >= 0.7:
            tier = 2
            position_size_pct = 1  # 100% of base position size for tier 2
            if log_file_handle:
                log_file_handle.write(f"{datetime.now()} - DEBUG - Tier 2 signal for {symbol} with score {score:.2f}\n")
            else:
                print(f"DEBUG: Tier 2 signal for {symbol} with score {score:.2f}")
        else:
            # Skip tier 3 signals
            if log_file_handle:
                log_file_handle.write(f"{datetime.now()} - DEBUG - Skipping tier 3 signal for {symbol} with score {score:.2f}\n")
            else:
                print(f"DEBUG: Skipping tier 3 signal for {symbol} with score {score:.2f}")
            continue
        
        # Calculate position size in shares
        position_value = portfolio.cash * position_size_pct
        shares = int(position_value / current_price) if current_price > 0 else 0
        
        # Skip if position size is too small
        if shares <= 0:
            if log_file_handle:
                log_file_handle.write(f"{datetime.now()} - DEBUG - Skipping {symbol} - position size too small: {shares}\n")
            else:
                print(f"DEBUG: Skipping {symbol} - position size too small: {shares}")
            continue
        
        # Add debug logging
        if log_file_handle:
            log_file_handle.write(f"{datetime.now()} - DEBUG - Attempting to open position for {symbol} - Price: ${current_price:.2f}, Size: {shares} shares, Value: ${position_value:.2f}\n")
        else:
            print(f"DEBUG: Attempting to open position for {symbol} - Price: ${current_price:.2f}, Size: {shares} shares, Value: ${position_value:.2f}")
        
        # Execute trade
        if direction == 'long':
            # Open long position
            result = portfolio.open_position(
                symbol=symbol,
                entry_price=current_price,
                entry_time=date,
                shares=shares,
                direction='long',
                tier=tier
            )
            
            if result:
                trades_executed += 1
        
        elif direction == 'short':
            # Open short position
            result = portfolio.open_position(
                symbol=symbol,
                entry_price=current_price,
                entry_time=date,
                shares=shares,
                direction='short',
                tier=tier
            )
            
            if result:
                trades_executed += 1
    
    return trades_executed

def get_historical_data(api, symbols, start_date, end_date):
    """
    Get historical data for a list of symbols.
    
    Args:
        api (AlpacaAPI): AlpacaAPI instance
        symbols (list): List of symbols
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        
    Returns:
        dict: Dictionary of historical data by date and symbol
    """
    # Get historical data for all symbols
    all_data = {}
    
    for symbol in symbols:
        try:
            # Get historical data for this symbol
            bars = api.get_bars([symbol], '1D', start_date, end_date)
            
            if bars is None or len(bars) == 0:
                logger.warning(f"No data for {symbol}")
                continue
                
            # Process bars
            for index, row in bars.iterrows():
                # Convert timestamp to string date
                date_str = index.strftime('%Y-%m-%d') if hasattr(index, 'strftime') else str(index).split(' ')[0]
                
                if date_str not in all_data:
                    all_data[date_str] = {}
                    
                all_data[date_str][symbol] = {
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume']
                }
        except Exception as e:
            logger.error(f"Error getting data for {symbol}: {e}")
            
    return all_data

def run_backtest(start_date, end_date, initial_capital=100000, config_path='sp500_config.yaml', 
                min_score=0.7, max_signals=30, tier1_threshold=0.8, tier2_threshold=0.7,
                largecap_allocation=0.7, midcap_allocation=0.3):
    """
    Run a backtest with the specified parameters.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        initial_capital (float): Initial capital for the backtest
        config_path (str): Path to configuration file
        min_score (float): Minimum signal score to include
        max_signals (int): Maximum number of signals to return
        tier1_threshold (float): Threshold for tier 1 signals
        tier2_threshold (float): Threshold for tier 2 signals
        largecap_allocation (float): Allocation percentage for large-cap stocks
        midcap_allocation (float): Allocation percentage for mid-cap stocks
        
    Returns:
        dict: Dictionary with backtest results
    """
    try:
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Initialize API
        api_key = config['alpaca']['api_key']
        api_secret = config['alpaca']['api_secret']
        base_url = config['alpaca']['base_url']
        data_url = config['alpaca']['data_url']
        
        api = AlpacaAPI(api_key, api_secret, base_url, data_url)
        
        # Create log file
        log_file = f"backtest_{start_date}_to_{end_date}.log"
        log_file_handle = open(log_file, 'w')
        log_file_handle.write(f"{datetime.now()} - INFO - Starting backtest from {start_date} to {end_date}\n")
        log_file_handle.write(f"{datetime.now()} - INFO - Initial capital: ${initial_capital:,.2f}\n")
        
        # Generate signals
        log_file_handle.write(f"{datetime.now()} - INFO - Generating signals...\n")
        signal_kwargs = {
            'start_date': start_date,
            'end_date': end_date,
            'alpaca': api,
            'min_score': min_score,
            'max_signals': max_signals
        }
        
        signals = generate_signals(**signal_kwargs)
        log_file_handle.write(f"{datetime.now()} - INFO - Generated {len(signals)} signals\n")
        
        # Debug: Log some sample signals
        if signals:
            log_file_handle.write(f"{datetime.now()} - DEBUG - Sample signals:\n")
            for i, signal in enumerate(signals[:5]):  # Log first 5 signals
                log_file_handle.write(f"{datetime.now()} - DEBUG - Signal {i+1}: {signal}\n")
        
        # Process signals
        processed_signals = []
        for signal in signals:
            processed_signals.append({
                'symbol': signal['symbol'],
                'score': signal['score'],
                'direction': signal['direction'],
                'date': signal['date']
            })
        log_file_handle.write(f"{datetime.now()} - INFO - Processed {len(processed_signals)} signals\n")
        
        # Get unique symbols from signals
        symbols = list(set([s['symbol'] for s in processed_signals]))
        log_file_handle.write(f"{datetime.now()} - INFO - Getting historical data for {len(symbols)} symbols\n")
        
        # Get historical data
        historical_data = get_historical_data(api, symbols, start_date, end_date)
        log_file_handle.write(f"{datetime.now()} - INFO - Got historical data for {len(historical_data)} trading days\n")
        
        # Debug: Log sample of historical data
        if historical_data:
            first_date = list(historical_data.keys())[0]
            first_symbol = list(historical_data[first_date].keys())[0]
            log_file_handle.write(f"{datetime.now()} - DEBUG - Sample historical data for {first_symbol} on {first_date}: {historical_data[first_date][first_symbol]}\n")
        
        # Initialize portfolio
        portfolio = Portfolio(initial_capital=initial_capital)
        
        # Set up trading days
        trading_days = sorted(historical_data.keys())
        log_file_handle.write(f"{datetime.now()} - INFO - Running backtest for {len(trading_days)} trading days\n")
        
        # Run backtest
        for date_str in trading_days:
            # Convert string date to datetime
            trading_day = datetime.strptime(date_str, '%Y-%m-%d')
            
            log_file_handle.write(f"{datetime.now()} - INFO - Processing {date_str}\n")
            
            # Get price data for this day
            price_data = historical_data[date_str]
            
            # Process signals for this day
            trades_executed = process_signals(
                signals=processed_signals,
                portfolio=portfolio,
                date=trading_day,
                price_data=price_data,
                min_score=min_score,
                tier1_threshold=tier1_threshold,
                tier2_threshold=tier2_threshold,
                max_signals=max_signals,
                log_file_handle=log_file_handle
            )
            
            log_file_handle.write(f"{datetime.now()} - INFO - Executed {trades_executed} trades for {trading_day}\n")
            
            # Update portfolio with current prices
            updates = portfolio.update_positions(trading_day, price_data)
            log_file_handle.write(f"{datetime.now()} - INFO - Portfolio updated for {trading_day}: {updates['positions_updated']} positions updated, {updates['positions_closed']} positions closed, total P&L: ${updates['total_pnl']:.2f}\n")
            
            # Update equity curve
            equity_after = portfolio.get_equity(price_data)
            portfolio.update_equity_curve(trading_day, equity_after)
            
            # Calculate daily return
            daily_return = (equity_after / portfolio.initial_capital - 1) * 100 if portfolio.initial_capital > 0 else 0
            
            # Log portfolio status at the end of the day
            log_file_handle.write(f"{datetime.now()} - INFO - End of day {trading_day} - Portfolio value: ${equity_after:.2f}, Cash: ${portfolio.cash:.2f}, Open positions: {len(portfolio.open_positions)}, Daily return: {daily_return:.2f}%\n")
            log_file_handle.flush()
            os.fsync(log_file_handle.fileno())
        
        # Close any remaining open positions at the end of the backtest
        if trading_days:
            last_day = trading_days[-1]
            last_day_prices = historical_data[last_day]
            last_trading_day = datetime.strptime(last_day, '%Y-%m-%d')
            
            log_file_handle.write(f"{datetime.now()} - INFO - Closing all remaining positions at the end of the backtest\n")
            
            for symbol in list(portfolio.open_positions.keys()):
                if symbol in last_day_prices:
                    exit_price = last_day_prices[symbol]['close']
                    portfolio.close_position(symbol, exit_price, last_trading_day, "end_of_backtest")
                    log_file_handle.write(f"{datetime.now()} - INFO - Closed position for {symbol} at ${exit_price:.2f} at end of backtest\n")
                else:
                    log_file_handle.write(f"{datetime.now()} - WARNING - No price data for {symbol} on last day, cannot close position\n")
        
        # Calculate performance metrics
        log_file_handle.write(f"{datetime.now()} - INFO - Calculating performance metrics\n")
        
        # Get the last day's price data for final performance calculation
        if trading_days:
            last_day = trading_days[-1]
            last_day_prices = historical_data[last_day]
            performance = portfolio.calculate_performance(last_day_prices)
        else:
            performance = portfolio.calculate_performance()
        
        # Log performance
        log_file_handle.write(f"{datetime.now()} - INFO - Backtest completed\n")
        log_file_handle.write(f"{datetime.now()} - INFO - Final portfolio value: ${performance.get('final_value', 0):.2f}\n")
        log_file_handle.write(f"{datetime.now()} - INFO - Return: {performance.get('return', 0):.2f}%\n")
        log_file_handle.write(f"{datetime.now()} - INFO - Annualized return: {performance.get('annualized_return', 0):.2f}%\n")
        log_file_handle.write(f"{datetime.now()} - INFO - Sharpe ratio: {performance.get('sharpe_ratio', 0):.2f}\n")
        log_file_handle.write(f"{datetime.now()} - INFO - Max drawdown: {performance.get('max_drawdown', 0):.2f}%\n")
        log_file_handle.write(f"{datetime.now()} - INFO - Win rate: {performance.get('win_rate', 0):.2f}%\n")
        log_file_handle.flush()
        os.fsync(log_file_handle.fileno())
        
        # Save results
        results = {
            'portfolio': portfolio,
            'performance': performance,
            'signals': processed_signals,
            'log_file': log_file
        }
        
        return results
    
    except Exception as e:
        print(f"Error in run_backtest: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

# Function to test the backtest
def test_backtest():
    """
    Test the backtest function with sample data
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Sample parameters with a longer date range
    start_date = '2022-07-01'  # Use even longer date range
    end_date = '2023-01-31'
    initial_capital = 100000
    
    # Run the backtest
    results = run_backtest(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        max_signals=30,
        min_score=0.7
    )
    
    # Check if the backtest was successful
    if 'portfolio' in results:
        print(f"Backtest completed successfully!")
        print(f"Final portfolio value: ${results['portfolio'].get_equity():.2f}")
        
        if 'performance' in results and results['performance']:
            print(f"Return: {results['performance'].get('return', 0):.2f}%")
            print(f"Sharpe ratio: {results['performance'].get('sharpe_ratio', 0):.2f}")
            print(f"Max drawdown: {results['performance'].get('max_drawdown', 0):.2f}%")
            
        # Print the number of signals generated
        if 'signals' in results:
            print(f"Number of signals generated: {len(results['signals'])}")
            # Print the first few signals if available
            if results['signals']:
                print("\nSample signals:")
                for i, signal in enumerate(results['signals'][:3]):
                    print(f"Signal {i+1}: {signal['symbol']} - Score: {signal['score']:.2f} - Direction: {signal['direction']}")
            else:
                print("\nNo signals were generated. This could be due to:")
                print("1. The signal_generator.py dropna() removing all rows with NaN values")
                print("2. No symbols meeting the minimum score threshold")
                print("3. Issues with the technical indicator calculations")
                print("\nConsider modifying signal_generator.py to be more lenient with NaN values")
    else:
        print(f"Backtest failed: {results.get('error', 'Unknown error')}")
    
    return results

if __name__ == "__main__":
    test_backtest()
