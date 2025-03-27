"""
Fixed Backtest Function for Multi-Strategy Trading System
This module contains the fixed version of the run_backtest function with proper error handling.
"""

import os
import logging
from datetime import datetime, timedelta
import yaml
import numpy as np
import pandas as pd
from portfolio import Portfolio

# Set up logging
def setup_logging(strategy_name):
    """
    Set up logging configuration
    
    Args:
        strategy_name (str): Name of the strategy for log file naming
        
    Returns:
        Logger instance
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Set up logging to file
    log_file = f"logs/{strategy_name}_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger

def generate_signals(api, universe, current_date, max_signals=10, min_score=0.6):
    """
    Generate trading signals for the given universe of symbols using real data from Alpaca
    
    Args:
        api: Alpaca API instance
        universe (list): List of symbols to generate signals for
        current_date (str): Current date in YYYY-MM-DD format
        max_signals (int): Maximum number of signals to generate
        min_score (float): Minimum score for signals
        
    Returns:
        List of signal dictionaries
    """
    signals = []
    
    # Convert current_date to datetime
    current_dt = datetime.strptime(current_date, '%Y-%m-%d')
    
    # Calculate start date (10 days before current date)
    start_dt = current_dt - timedelta(days=10)
    
    # Format dates for API
    start_date = start_dt.strftime('%Y-%m-%d')
    end_date = current_date
    
    # Log the signal generation process
    logger = logging.getLogger(__name__)
    logger.info(f"Generating signals for {len(universe)} symbols on {current_date}")
    
    # Process each symbol in the universe
    for symbol in universe:
        try:
            # Get historical data for the symbol
            bars = api.get_bars([symbol], '1D', pd.Timestamp(start_date), pd.Timestamp(end_date))
            
            if bars is None or len(bars) < 5:  # Need at least 5 days of data
                continue
                
            # Extract price data
            if isinstance(bars.index, pd.MultiIndex):
                symbol_bars = bars.loc[symbol] if symbol in bars.index.levels[0] else None
            else:
                symbol_bars = bars
                
            if symbol_bars is None or len(symbol_bars) < 5:
                continue
                
            # Calculate technical indicators
            # 1. RSI (Relative Strength Index)
            close_prices = symbol_bars['close'].values
            delta = np.diff(close_prices)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            
            avg_gain = np.mean(gain[-5:])  # 5-day average gain
            avg_loss = np.mean(loss[-5:])  # 5-day average loss
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                
            # 2. Price momentum (% change over last 5 days)
            momentum = (close_prices[-1] / close_prices[-5] - 1) * 100
            
            # 3. Volume trend
            volume = symbol_bars['volume'].values
            vol_change = (volume[-1] / np.mean(volume[-5:]) - 1) * 100
            
            # Combine indicators into a score (0-1)
            # RSI: 0-100 (higher = more overbought)
            # Momentum: % change (higher = stronger uptrend)
            # Volume: % change (higher = increasing volume)
            
            # Normalize RSI to 0-1 (70+ is overbought, 30- is oversold)
            rsi_score = 0.5
            if rsi > 70:  # Overbought - bearish signal
                rsi_score = (100 - rsi) / 30  # 1.0 at RSI 70, 0.0 at RSI 100
            elif rsi < 30:  # Oversold - bullish signal
                rsi_score = (30 - rsi) / 30  # 0.0 at RSI 30, 1.0 at RSI 0
                
            # Normalize momentum (-10% to +10% range)
            momentum_score = (momentum + 10) / 20
            momentum_score = max(0, min(1, momentum_score))
            
            # Normalize volume change (-50% to +50% range)
            volume_score = (vol_change + 50) / 100
            volume_score = max(0, min(1, volume_score))
            
            # Combined score (weighted average)
            score = 0.4 * rsi_score + 0.4 * momentum_score + 0.2 * volume_score
            
            # Determine direction based on RSI
            direction = 'SHORT' if rsi > 70 else 'LONG'
            
            # Only include signals above the minimum score
            if score >= min_score:
                # Get the latest price
                price = close_prices[-1]
                
                # Create signal
                signal = {
                    'symbol': symbol,
                    'price': price,
                    'score': score,
                    'direction': direction,
                    'rsi': rsi,
                    'momentum': momentum,
                    'volume_change': vol_change
                }
                
                signals.append(signal)
                logger.info(f"Generated signal for {symbol}: Score={score:.2f}, Direction={direction}, RSI={rsi:.2f}")
                
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {str(e)}")
            continue
    
    # Sort signals by score (highest first)
    signals = sorted(signals, key=lambda x: x['score'], reverse=True)
    
    # Limit to max_signals
    return signals[:max_signals]

def run_backtest(
    api, 
    strategy_name, 
    universe, 
    start_date, 
    end_date, 
    initial_capital=10000, 
    max_signals=10, 
    min_score=0.6,
    tier1_threshold=0.8,
    tier2_threshold=0.7,
    tier3_threshold=0.3
):
    """
    Run a backtest for a given strategy
    
    Args:
        api: API instance for data retrieval
        strategy_name (str): Name of the strategy
        universe (list): List of symbols to trade
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        initial_capital (float): Initial capital
        max_signals (int): Maximum number of signals to generate
        min_score (float): Minimum score for signals
        tier1_threshold (float): Threshold for Tier 1 signals
        tier2_threshold (float): Threshold for Tier 2 signals
        tier3_threshold (float): Threshold for Tier 3 signals
        
    Returns:
        Dictionary containing backtest results
    """
    try:
        # Set up logging
        logger = setup_logging(strategy_name)
        
        # Parse dates
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Create a list of dates to backtest
        date_range = []
        current_dt = start_dt
        while current_dt <= end_dt:
            # Skip weekends
            if current_dt.weekday() < 5:  # 0-4 are Monday to Friday
                date_range.append(current_dt.strftime('%Y-%m-%d'))
            current_dt += timedelta(days=1)
        
        logger.info(f"Backtest date range: {date_range[0]} to {date_range[-1]} ({len(date_range)} trading days)")
        
        # Initialize portfolio with custom thresholds
        portfolio = Portfolio(initial_capital=initial_capital)
        
        # Set tier thresholds as attributes
        portfolio.tier1_threshold = tier1_threshold
        portfolio.tier2_threshold = tier2_threshold
        portfolio.tier3_threshold = tier3_threshold
        
        # Initialize the equity curve with the starting value
        initial_equity = portfolio.get_equity()
        portfolio.equity_curve = [{
            'timestamp': start_dt,
            'equity': initial_equity,
            'pct_change': 0
        }]
        logger.info(f"Initial portfolio value: ${initial_equity:.2f}")
        
        # Iterate through each date in the range
        for current_date in date_range:
            logger.info(f"Processing date: {current_date}")
            
            # Generate signals for this date
            signals = generate_signals(api, universe, current_date, max_signals=max_signals, min_score=min_score)
            
            # Log number of signals
            logger.info(f"Generated {len(signals)} signals for {current_date}")
            
            # Execute trades based on signals
            for signal in signals:
                # Skip if we already have a position for this symbol
                if signal['symbol'] in portfolio.open_positions:
                    continue
                
                # Determine tier based on signal score
                if signal['score'] >= portfolio.tier1_threshold:
                    tier = 1
                elif signal['score'] >= portfolio.tier2_threshold:
                    tier = 2
                elif signal['score'] >= portfolio.tier3_threshold:
                    tier = 3
                else:
                    logger.info(f"Skipping trade for {signal['symbol']} with score {signal['score']:.2f} - below Tier 3 threshold")
                    continue
                
                # Execute the trade
                trade_result = portfolio.open_position(
                    symbol=signal['symbol'],
                    entry_price=signal['price'],
                    entry_time=datetime.strptime(current_date, '%Y-%m-%d'),
                    position_size=0.01 if tier == 3 else (0.02 if tier == 2 else 0.03),  # Tier-based position sizing
                    direction=signal['direction'],
                    tier=tier
                )
                
                if trade_result:
                    logger.info(f"Executing trade for {signal['symbol']} at {signal['price']} with position size {trade_result}")
                    logger.info(f"Successfully opened position for {signal['symbol']}")
                else:
                    logger.warning(f"Failed to open position for {signal['symbol']}")
            
            # Get price data for all symbols with open positions
            price_data = {}
            symbols_to_price = list(portfolio.open_positions.keys())
            
            if symbols_to_price:
                try:
                    # Get current prices from Alpaca
                    bars = api.get_bars(symbols_to_price, '1D', pd.Timestamp(current_date), pd.Timestamp(current_date))
                    
                    if bars is not None and not bars.empty:
                        # Extract closing prices
                        if isinstance(bars.index, pd.MultiIndex):
                            for symbol in symbols_to_price:
                                if symbol in bars.index.levels[0]:
                                    symbol_bars = bars.loc[symbol]
                                    if not symbol_bars.empty:
                                        price_data[symbol] = symbol_bars['close'].iloc[-1]
                        else:
                            for symbol in symbols_to_price:
                                if symbol in bars.index:
                                    price_data[symbol] = bars.loc[symbol, 'close']
                    
                    # Log the prices
                    for symbol, price in price_data.items():
                        logger.debug(f"Got price for {symbol}: {price:.2f}")
                    
                except Exception as e:
                    logger.error(f"Failed to get prices: {str(e)}")
                    
                # For any symbols without prices, use the last known price
                for symbol in symbols_to_price:
                    if symbol not in price_data and symbol in portfolio.open_positions:
                        entry_price = portfolio.open_positions[symbol].entry_price
                        price_data[symbol] = entry_price
                        logger.warning(f"Using entry price for {symbol}: {entry_price:.2f}")
            
            # Update portfolio with today's prices
            if price_data:
                logger.info(f"Updating portfolio equity curve...")
                logger.debug(f"Open positions: {list(portfolio.open_positions.keys())}")
                logger.debug(f"Price data: {price_data}")
                
                # Check for positions to close (take profit or stop loss)
                for symbol, position in list(portfolio.open_positions.items()):
                    if symbol in price_data:
                        current_price = price_data[symbol]
                        entry_price = position.entry_price
                        
                        # Calculate unrealized P&L
                        unrealized_pnl, unrealized_pnl_pct = position.get_unrealized_pnl(current_price)
                        
                        # Take profit at 5% or stop loss at -2%
                        if position.direction == 'LONG':
                            if current_price >= entry_price * 1.05:  # 5% profit
                                logger.info(f"Taking profit on {symbol} at {current_price:.2f} (Entry: {entry_price:.2f}, P&L: {unrealized_pnl_pct:.2f}%)")
                                portfolio.close_position(symbol, current_price, datetime.strptime(current_date, '%Y-%m-%d'), "take_profit")
                            elif current_price <= entry_price * 0.98:  # 2% loss
                                logger.info(f"Stopping loss on {symbol} at {current_price:.2f} (Entry: {entry_price:.2f}, P&L: {unrealized_pnl_pct:.2f}%)")
                                portfolio.close_position(symbol, current_price, datetime.strptime(current_date, '%Y-%m-%d'), "stop_loss")
                        else:  # SHORT
                            if current_price <= entry_price * 0.95:  # 5% profit (price went down)
                                logger.info(f"Taking profit on {symbol} SHORT at {current_price:.2f} (Entry: {entry_price:.2f}, P&L: {unrealized_pnl_pct:.2f}%)")
                                portfolio.close_position(symbol, current_price, datetime.strptime(current_date, '%Y-%m-%d'), "take_profit")
                            elif current_price >= entry_price * 1.02:  # 2% loss (price went up)
                                logger.info(f"Stopping loss on {symbol} SHORT at {current_price:.2f} (Entry: {entry_price:.2f}, P&L: {unrealized_pnl_pct:.2f}%)")
                                portfolio.close_position(symbol, current_price, datetime.strptime(current_date, '%Y-%m-%d'), "stop_loss")
                
                # Update portfolio equity
                current_dt = datetime.strptime(current_date, '%Y-%m-%d')
                equity = portfolio.get_equity(price_data)
                
                # Calculate percentage change from previous day
                prev_equity = portfolio.equity_curve[-1]['equity']
                pct_change = 0 if prev_equity == 0 else (equity / prev_equity - 1) * 100
                
                # Add to equity curve
                portfolio.equity_curve.append({
                    'timestamp': current_dt,
                    'equity': equity,
                    'pct_change': pct_change
                })
                
                logger.info(f"Updated equity curve: {portfolio.equity_curve[-1]}")
                logger.info(f"Current equity: ${equity:.2f}")
        
        # Calculate performance metrics
        logger.info("Backtest completed")
        
        # Get final portfolio value
        final_equity = portfolio.get_equity()
        
        # Calculate return
        return_pct = (final_equity / initial_capital - 1) * 100
        
        # Calculate annualized return
        days = (end_dt - start_dt).days
        if days > 0:
            annualized_return = ((1 + return_pct / 100) ** (365 / days) - 1) * 100
        else:
            annualized_return = return_pct
        
        # Calculate Sharpe ratio
        daily_returns = [point['pct_change'] / 100 for point in portfolio.equity_curve[1:]] if len(portfolio.equity_curve) > 1 else []
        if daily_returns:
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Calculate max drawdown
        max_drawdown = 0
        peak = initial_capital
        for point in portfolio.equity_curve:
            equity = point['equity']
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate win rate
        if portfolio.closed_positions:
            winning_trades = sum(1 for p in portfolio.closed_positions if p.pnl > 0)
            win_rate = winning_trades / len(portfolio.closed_positions) * 100
        else:
            win_rate = 0
        
        # Log performance metrics
        logger.info(f"Final portfolio value: ${final_equity:.2f}")
        logger.info(f"Return: {return_pct:.2f}%")
        logger.info(f"Annualized return: {annualized_return:.2f}%")
        logger.info(f"Sharpe ratio: {sharpe_ratio:.2f}")
        logger.info(f"Max drawdown: {max_drawdown:.2f}%")
        logger.info(f"Win rate: {win_rate:.2f}%")
        
        # Create performance dictionary
        performance = {
            'final_equity': final_equity,
            'return_pct': return_pct,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'equity_curve': portfolio.equity_curve
        }
        
        # Save results
        results = {
            'portfolio': portfolio,
            'performance': performance,
            'signals': portfolio.open_positions.keys(),  # Track the signals we've acted on
        }
        
        return results
    
    finally:
        # Make sure to close the log file handle
        if 'logger' in locals() and logger:
            logger.info("Backtest completed")

def get_sp500_symbols():
    """
    Get the current S&P 500 symbols by scraping Wikipedia
    
    Returns:
        List of S&P 500 symbols
    """
    import requests
    from bs4 import BeautifulSoup
    import logging
    
    logger = logging.getLogger(__name__)
    logger.info("Fetching S&P 500 symbols from Wikipedia")
    
    try:
        # URL of Wikipedia's S&P 500 companies page
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        
        # Send a request to the URL
        response = requests.get(url)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the table with S&P 500 companies
            table = soup.find('table', {'class': 'wikitable'})
            
            # Extract symbols from the table
            symbols = []
            for row in table.find_all('tr')[1:]:  # Skip the header row
                cells = row.find_all('td')
                if len(cells) > 0:
                    symbol = cells[0].text.strip()
                    symbols.append(symbol)
            
            logger.info(f"Found {len(symbols)} S&P 500 symbols")
            return symbols
        else:
            logger.error(f"Failed to fetch S&P 500 symbols: HTTP {response.status_code}")
            return []
    
    except Exception as e:
        logger.error(f"Error fetching S&P 500 symbols: {str(e)}")
        # Return a small subset of major S&P 500 companies as fallback
        fallback_symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'UNH', 'JNJ', 'WMT']
        logger.info(f"Using fallback list of {len(fallback_symbols)} symbols")
        return fallback_symbols

def test_backtest():
    """
    Test function for the backtest
    """
    # Import required modules
    from portfolio import Portfolio
    from alpaca_api import AlpacaAPI
    import yaml
    
    # Load configuration
    with open('sp500_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize API
    api = AlpacaAPI(
        api_key=config['alpaca']['api_key'],
        api_secret=config['alpaca']['api_secret'],
        base_url=config['alpaca']['base_url'],
        data_url=config['alpaca']['data_url']
    )
    
    # Define universe of symbols
    universe = get_sp500_symbols()[:10]  # Use the first 10 S&P 500 symbols
    
    # Run backtest
    results = run_backtest(
        api=api,
        strategy_name='sp500_strategy',
        universe=universe,
        start_date='2022-07-01',
        end_date='2023-01-31',
        initial_capital=10000,
        max_signals=5,
        min_score=0.6,
        tier1_threshold=0.8,
        tier2_threshold=0.7,
        tier3_threshold=0.6
    )
    
    # Print results
    print("Backtest completed successfully!")
    print(f"Final portfolio value: ${results['performance'].get('final_equity', 0):.2f}")
    print(f"Return: {results['performance'].get('return_pct', 0):.2f}%")
    print(f"Sharpe ratio: {results['performance'].get('sharpe_ratio', 0):.2f}")
    print(f"Max drawdown: {results['performance'].get('max_drawdown', 0):.2f}%")
    
    # Print open positions
    open_positions = list(results['portfolio'].open_positions.keys())
    print(f"Number of open positions: {len(open_positions)}")
    
    # Print closed positions
    closed_positions = results['portfolio'].closed_positions
    print(f"Number of closed positions: {len(closed_positions)}")
    
    # Print sample positions
    if open_positions:
        print("\nSample open positions:")
        for i, symbol in enumerate(list(open_positions)[:3]):
            position = results['portfolio'].open_positions[symbol]
            print(f"Position {i+1}: {symbol} - Entry: ${position.entry_price:.2f} - Direction: {position.direction}")
    
    if closed_positions:
        print("\nSample closed positions:")
        for i, position in enumerate(closed_positions[:3]):
            print(f"Position {i+1}: {position.symbol} - Entry: ${position.entry_price:.2f} - Exit: ${position.exit_price:.2f} - P&L: ${position.pnl:.2f} ({position.pnl_pct:.2f}%)")

if __name__ == "__main__":
    test_backtest()
