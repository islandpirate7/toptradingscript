#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fixed Backtest Function
-----------------------
This module contains a fixed version of the run_backtest function that properly
tracks portfolio performance over time.
"""

import os
import logging
import datetime
from datetime import datetime, timedelta
from tqdm import tqdm

def fixed_run_backtest(api, portfolio, start_date, end_date, signals, tier1_threshold=0.8, tier2_threshold=0.6, log_file_handle=None):
    """
    Fixed version of the run_backtest function that properly tracks portfolio performance.
    
    Args:
        api: The API object for fetching market data
        portfolio: The Portfolio object
        start_date: Start date for the backtest (YYYY-MM-DD)
        end_date: End date for the backtest (YYYY-MM-DD)
        signals: List of trading signals
        tier1_threshold: Threshold for tier 1 signals
        tier2_threshold: Threshold for tier 2 signals
        log_file_handle: File handle for logging
        
    Returns:
        dict: Backtest results
    """
    try:
        # Log backtest parameters
        log_file_handle.write(f"{datetime.now()} - INFO - Running fixed backtest from {start_date} to {end_date}\n")
        
        # Process signals and execute trades
        log_file_handle.write(f"{datetime.now()} - INFO - Processing {len(signals)} signals\n")
        
        for signal in signals:
            tier = 3  # Default to tier 3
            if signal['score'] >= tier1_threshold:
                tier = 1
            elif signal['score'] >= tier2_threshold:
                tier = 2
            
            trade_result = portfolio.execute_trade(signal, tier=tier)
            if trade_result['success']:
                log_file_handle.write(f"{datetime.now()} - INFO - Executed {signal['direction']} trade for {signal['symbol']} with score {signal['score']:.2f} (Tier {tier})\n")
                
                # Ensure we have historical data for this symbol
                symbol = signal['symbol']
            else:
                log_file_handle.write(f"{datetime.now()} - INFO - Failed to execute trade for {signal['symbol']}: {trade_result.get('error', 'Unknown error')}\n")
        
        # Convert start_date and end_date to datetime objects
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Get all symbols in the portfolio
        portfolio_symbols = list(portfolio.open_positions.keys())
        log_file_handle.write(f"{datetime.now()} - INFO - Portfolio contains {len(portfolio_symbols)} symbols: {', '.join(portfolio_symbols)}\n")
        
        # Get historical data for all symbols in the portfolio for the entire period
        log_file_handle.write(f"{datetime.now()} - INFO - Fetching historical data for portfolio symbols\n")
        
        # Fetch historical data for each symbol
        all_historical_data = {}
        for symbol in portfolio_symbols:
            try:
                log_file_handle.write(f"{datetime.now()} - INFO - Fetching historical data for {symbol}\n")
                
                # Try to get bars from Alpaca API
                bars = api.get_bars(symbol, start_date, end_date, 'day', limit=1000)
                
                if len(bars) > 0:
                    all_historical_data[symbol] = bars
                    log_file_handle.write(f"{datetime.now()} - INFO - Retrieved {len(bars)} bars for {symbol}\n")
                else:
                    # If we don't get any bars, create synthetic data
                    log_file_handle.write(f"{datetime.now()} - WARNING - No historical data found for {symbol}, creating synthetic data\n")
                    
                    # Create synthetic price data using the entry price
                    if symbol in portfolio.open_positions:
                        entry_price = portfolio.open_positions[symbol].entry_price
                        
                        # Create a list of bars with the same price for each day
                        synthetic_bars = []
                        current_date = start_dt
                        while current_date <= end_dt:
                            if current_date.weekday() < 5:  # Weekday
                                synthetic_bars.append({
                                    't': current_date.strftime('%Y-%m-%d'),
                                    'o': entry_price,
                                    'h': entry_price,
                                    'l': entry_price,
                                    'c': entry_price,
                                    'v': 0
                                })
                            current_date += timedelta(days=1)
                        
                        all_historical_data[symbol] = synthetic_bars
                        log_file_handle.write(f"{datetime.now()} - INFO - Created {len(synthetic_bars)} synthetic bars for {symbol} with price ${entry_price:.2f}\n")
            except Exception as e:
                log_file_handle.write(f"{datetime.now()} - ERROR - Failed to fetch historical data for {symbol}: {str(e)}\n")
            
            # Flush log after each symbol
            log_file_handle.flush()
            os.fsync(log_file_handle.fileno())
        
        # Simulate each trading day
        current_date = start_dt
        
        # Initialize the equity curve with the starting value
        initial_equity = portfolio.get_equity()
        portfolio.equity_curve = [{
            'timestamp': current_date,
            'equity': initial_equity,
            'pct_change': 0
        }]
        log_file_handle.write(f"{datetime.now()} - INFO - Initial portfolio value on {current_date.date()}: ${initial_equity:.2f}\n")
        
        # Get all trading days in the date range (excluding weekends)
        trading_days = []
        temp_date = start_dt
        while temp_date <= end_dt:
            if temp_date.weekday() < 5:  # Weekday (0-4 are Monday-Friday)
                trading_days.append(temp_date)
            temp_date += timedelta(days=1)
        
        log_file_handle.write(f"{datetime.now()} - INFO - Simulating {len(trading_days)} trading days from {start_date} to {end_date}\n")
        
        # Process each trading day
        for day_index, current_date in enumerate(trading_days):
            # Get prices for this date
            daily_prices = {}
            for symbol, data in all_historical_data.items():
                # Find the data point for this date
                for bar in data:
                    # Handle different date formats that might be returned by the API
                    try:
                        if 'T' in bar['t']:
                            bar_date = datetime.strptime(bar['t'].split('T')[0], '%Y-%m-%d')
                        else:
                            bar_date = datetime.strptime(bar['t'], '%Y-%m-%d')
                        
                        if bar_date.date() == current_date.date():
                            daily_prices[symbol] = bar['c']  # Use closing price
                            log_file_handle.write(f"{datetime.now()} - DEBUG - Price for {symbol} on {current_date.date()}: ${bar['c']:.2f}\n")
                            break
                    except (ValueError, KeyError) as e:
                        log_file_handle.write(f"{datetime.now()} - ERROR - Error parsing date for {symbol}: {str(e)}, bar data: {bar}\n")
                        continue
            
            # Update portfolio with today's prices
            if daily_prices:
                # Update positions first (checks for stop loss/take profit)
                updates = portfolio.update_positions(current_date, daily_prices)
                log_file_handle.write(f"{datetime.now()} - INFO - Portfolio updated for {current_date.date()}: {updates['positions_updated']} positions updated, {updates['positions_closed']} positions closed\n")
                
                # Then explicitly update the equity curve
                equity = portfolio.get_equity(daily_prices)
                
                # Calculate percentage change from previous day
                prev_equity = portfolio.equity_curve[-1]['equity']
                pct_change = 0 if prev_equity == 0 else (equity / prev_equity - 1) * 100
                
                # Add to equity curve
                portfolio.equity_curve.append({
                    'timestamp': current_date,
                    'equity': equity,
                    'pct_change': pct_change
                })
                
                log_file_handle.write(f"{datetime.now()} - INFO - Portfolio value on {current_date.date()}: ${equity:.2f} (Change: {pct_change:.2f}%)\n")
                
                # Every 30 days, log a summary of the equity curve
                if day_index % 30 == 0 and day_index > 0:
                    log_file_handle.write(f"{datetime.now()} - INFO - Equity curve after {day_index} trading days: ${equity:.2f} (Total change: {(equity / initial_equity - 1) * 100:.2f}%)\n")
                
                log_file_handle.flush()
                os.fsync(log_file_handle.fileno())
        
        # Calculate performance metrics
        performance = portfolio.calculate_performance()
        
        # Log performance metrics
        log_file_handle.write(f"{datetime.now()} - INFO - Backtest completed\n")
        log_file_handle.write(f"{datetime.now()} - INFO - Final portfolio value: ${performance['final_value']:.2f}\n")
        log_file_handle.write(f"{datetime.now()} - INFO - Return: {performance['return']:.2f}%\n")
        log_file_handle.write(f"{datetime.now()} - INFO - Annualized return: {performance['annualized_return']:.2f}%\n")
        log_file_handle.write(f"{datetime.now()} - INFO - Sharpe ratio: {performance['sharpe_ratio']:.2f}\n")
        log_file_handle.write(f"{datetime.now()} - INFO - Max drawdown: {performance['max_drawdown']:.2f}%\n")
        log_file_handle.write(f"{datetime.now()} - INFO - Win rate: {performance['win_rate']:.2f}%\n")
        
        # Log equity curve points
        log_file_handle.write(f"{datetime.now()} - INFO - Equity curve points: {len(portfolio.equity_curve)}\n")
        if portfolio.equity_curve:
            first_point = portfolio.equity_curve[0]
            log_file_handle.write(f"{datetime.now()} - INFO - First equity point: {first_point['timestamp']} - ${first_point['equity']:.2f}\n")
            if len(portfolio.equity_curve) > 1:
                last_point = portfolio.equity_curve[-1]
                log_file_handle.write(f"{datetime.now()} - INFO - Last equity point: {last_point['timestamp']} - ${last_point['equity']:.2f}\n")
        
        return {
            'success': True,
            'performance': performance,
            'equity_curve': portfolio.equity_curve
        }
    
    except Exception as e:
        log_file_handle.write(f"{datetime.now()} - ERROR - Error in backtest: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}
