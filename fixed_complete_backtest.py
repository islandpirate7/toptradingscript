"""
This file contains a fixed version of the run_backtest function with the original signature.
"""

def run_backtest(start_date, end_date, mode='backtest', initial_capital=10000, 
                 random_seed=None, continuous_capital=False, previous_capital=None,
                 config_path='sp500_config.yaml', max_signals=None, min_score=None,
                 tier1_threshold=None, tier2_threshold=None, tier3_threshold=None,
                 largecap_allocation=0.7, midcap_allocation=0.3, data_provider=None):
    """
    Run a backtest for the specified period.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        mode (str): 'backtest' or 'live'
        initial_capital (float): Initial capital for the backtest
        random_seed (int): Random seed for reproducibility
        continuous_capital (bool): Whether to use continuous capital from previous runs
        previous_capital (float): Previous capital to start with if continuous_capital is True
        config_path (str): Path to the configuration file
        max_signals (int): Maximum number of signals to use
        min_score (float): Minimum score for a signal to be considered
        tier1_threshold (float): Threshold for Tier 1 signals
        tier2_threshold (float): Threshold for Tier 2 signals
        tier3_threshold (float): Threshold for Tier 3 signals
        largecap_allocation (float): Allocation for large-cap stocks (0-1)
        midcap_allocation (float): Allocation for mid-cap stocks (0-1)
        data_provider (BacktestDataProvider, optional): Data provider for historical market data
    
    Returns:
        dict: Results of the backtest
    """
    try:
        # Set up logging
        import logging
        import os
        from datetime import datetime, timedelta
        import yaml
        import numpy as np
        
        # Create a timestamp for the log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Create a log file
        log_file = f"logs/portfolio_fix_{timestamp}.log"
        
        # Save original logging configuration
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers[:]
        original_level = root_logger.level
        
        # Open log file
        log_file_handle = open(log_file, 'w')
        
        try:
            # Log start of backtest
            log_file_handle.write(f"{datetime.now()} - INFO - Starting backtest from {start_date} to {end_date}\n")
            log_file_handle.flush()
            os.fsync(log_file_handle.fileno())
            
            # Load configuration
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Initialize API
            from alpaca_api import AlpacaAPI
            api = AlpacaAPI(config['alpaca']['api_key'], config['alpaca']['api_secret'], config['alpaca']['base_url'])
            
            # Set random seed for reproducibility
            if random_seed is not None:
                np.random.seed(random_seed)
                
            # Determine initial capital
            if continuous_capital and previous_capital is not None:
                capital = previous_capital
                log_file_handle.write(f"{datetime.now()} - INFO - Using continuous capital: ${capital:.2f}\n")
            else:
                capital = initial_capital
                log_file_handle.write(f"{datetime.now()} - INFO - Using initial capital: ${capital:.2f}\n")
            
            # Initialize portfolio with custom thresholds if provided
            from portfolio import Portfolio
            portfolio_kwargs = {'initial_capital': capital}
            
            if tier1_threshold is not None:
                portfolio_kwargs['tier1_threshold'] = tier1_threshold
            if tier2_threshold is not None:
                portfolio_kwargs['tier2_threshold'] = tier2_threshold
            if tier3_threshold is not None:
                portfolio_kwargs['tier3_threshold'] = tier3_threshold
                
            portfolio = Portfolio(**portfolio_kwargs)
            
            # Set allocations
            portfolio.largecap_allocation = largecap_allocation
            portfolio.midcap_allocation = midcap_allocation
            
            # Generate signals for the period
            from signal_generator import generate_signals
            
            # Apply signal filters
            signal_kwargs = {
                'start_date': start_date,
                'end_date': end_date,
                'config': config,
                'alpaca': api
            }
            
            if min_score is not None:
                signal_kwargs['min_score'] = min_score
                
            signals = generate_signals(**signal_kwargs)
            
            # Limit number of signals if specified
            if max_signals is not None and max_signals > 0:
                signals = signals[:max_signals]
            
            # Log number of signals
            log_file_handle.write(f"{datetime.now()} - INFO - Generated {len(signals)} signals\n")
            log_file_handle.flush()
            os.fsync(log_file_handle.fileno())
            
            # Initialize historical data dictionary
            all_historical_data = {}
            
            # Execute trades based on signals
            for signal in signals:
                # Determine tier based on signal score
                if signal['score'] >= portfolio.tier1_threshold:
                    tier = 1
                elif signal['score'] >= portfolio.tier2_threshold:
                    tier = 2
                elif signal['score'] >= portfolio.tier3_threshold:
                    tier = 3
                else:
                    log_file_handle.write(f"{datetime.now()} - INFO - Skipping trade for {signal['symbol']} with score {signal['score']:.2f} - below Tier 3 threshold\n")
                    log_file_handle.flush()
                    os.fsync(log_file_handle.fileno())
                    continue
                
                # Execute the trade
                trade_result = portfolio.execute_trade(signal, tier=tier)
                
                if trade_result['success']:
                    log_file_handle.write(f"{datetime.now()} - INFO - Executed {signal['direction']} trade for {signal['symbol']} with score {signal['score']:.2f} (Tier {tier})\n")
                    
                    # Ensure we have historical data for this symbol
                    symbol = signal['symbol']
                    if symbol not in all_historical_data:
                        log_file_handle.write(f"{datetime.now()} - INFO - Fetching historical data for newly traded symbol {symbol}\n")
                        try:
                            bars = api.get_bars(symbol, start_date, end_date, 'day', limit=1000)
                            if len(bars) > 0:
                                all_historical_data[symbol] = bars
                                log_file_handle.write(f"{datetime.now()} - INFO - Retrieved {len(bars)} bars for {symbol}\n")
                            else:
                                log_file_handle.write(f"{datetime.now()} - WARNING - No historical data found for {symbol}\n")
                        except Exception as e:
                            log_file_handle.write(f"{datetime.now()} - ERROR - Failed to fetch historical data for {symbol}: {str(e)}\n")
                else:
                    log_file_handle.write(f"{datetime.now()} - INFO - Failed to execute trade for {signal['symbol']}: {trade_result['error']}\n")
            
            # Convert start_date and end_date to datetime objects
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            # Get all symbols in the portfolio
            portfolio_symbols = list(portfolio.open_positions.keys())
            
            # Initialize the portfolio value on day 1
            if portfolio_symbols:
                log_file_handle.write(f"{datetime.now()} - INFO - Simulating daily portfolio updates from {start_date} to {end_date}\n")
                log_file_handle.flush()
                os.fsync(log_file_handle.fileno())
                
                # Get historical data for all symbols in the portfolio for the entire period
                log_file_handle.write(f"{datetime.now()} - INFO - Fetching historical data for portfolio symbols\n")
                
                # Create a list of all symbols in the portfolio
                portfolio_symbols = list(portfolio.open_positions.keys())
                log_file_handle.write(f"{datetime.now()} - INFO - Portfolio contains {len(portfolio_symbols)} symbols: {', '.join(portfolio_symbols)}\n")
                
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
                
                # Log performance
                log_file_handle.write(f"{datetime.now()} - INFO - Backtest completed\n")
                log_file_handle.write(f"{datetime.now()} - INFO - Final portfolio value: ${performance['final_value']:.2f}\n")
                log_file_handle.write(f"{datetime.now()} - INFO - Return: {performance['return']:.2f}%\n")
                log_file_handle.write(f"{datetime.now()} - INFO - Annualized return: {performance['annualized_return']:.2f}%\n")
                log_file_handle.write(f"{datetime.now()} - INFO - Sharpe ratio: {performance['sharpe_ratio']:.2f}\n")
                log_file_handle.write(f"{datetime.now()} - INFO - Max drawdown: {performance['max_drawdown']:.2f}%\n")
                log_file_handle.write(f"{datetime.now()} - INFO - Win rate: {performance['win_rate']:.2f}%\n")
                log_file_handle.flush()
                os.fsync(log_file_handle.fileno())
                
                # Save results
                results = {
                    'portfolio': portfolio,
                    'performance': performance,
                    'signals': signals,
                    'log_file': log_file
                }
                
                return results
            
            # If no portfolio symbols, return empty results
            else:
                log_file_handle.write(f"{datetime.now()} - INFO - No positions in portfolio, skipping daily simulation\n")
                results = {
                    'portfolio': portfolio,
                    'performance': {},
                    'signals': signals,
                    'log_file': log_file
                }
                return results
                
        finally:
            # Make sure to close the log file handle
            if 'log_file_handle' in locals() and log_file_handle:
                log_file_handle.flush()
                os.fsync(log_file_handle.fileno())
                log_file_handle.close()
            
            # Restore original logging configuration
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
            
            for handler in original_handlers:
                root_logger.addHandler(handler)
            
            root_logger.setLevel(original_level)
    
    except Exception as e:
        print(f"Error in run_backtest: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}
