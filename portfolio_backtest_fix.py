#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Portfolio Backtest Fix
---------------------
This script demonstrates how to properly track portfolio performance in a backtest
with a simplified example that doesn't rely on other modules.
"""

import os
import yaml
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Position:
    """Simple position class for demonstration purposes"""
    def __init__(self, symbol, quantity, entry_price, entry_date, direction='LONG'):
        self.symbol = symbol
        self.quantity = quantity
        self.entry_price = entry_price
        self.entry_date = entry_date
        self.direction = direction
        self.exit_price = None
        self.exit_date = None
        self.stop_loss = entry_price * 0.95 if direction == 'LONG' else entry_price * 1.05
        self.take_profit = entry_price * 1.15 if direction == 'LONG' else entry_price * 0.85
        self.is_open = True
        self.pnl = 0
        
    def update(self, current_price, current_date):
        """Update position with current price"""
        if not self.is_open:
            return False
            
        # Check for stop loss or take profit
        if self.direction == 'LONG':
            if current_price <= self.stop_loss:
                self.close(current_price, current_date, 'STOP_LOSS')
                return True
            elif current_price >= self.take_profit:
                self.close(current_price, current_date, 'TAKE_PROFIT')
                return True
        else:  # SHORT
            if current_price >= self.stop_loss:
                self.close(current_price, current_date, 'STOP_LOSS')
                return True
            elif current_price <= self.take_profit:
                self.close(current_price, current_date, 'TAKE_PROFIT')
                return True
                
        return False
        
    def close(self, exit_price, exit_date, reason):
        """Close the position"""
        self.exit_price = exit_price
        self.exit_date = exit_date
        self.is_open = False
        
        # Calculate P&L
        if self.direction == 'LONG':
            self.pnl = (self.exit_price - self.entry_price) * self.quantity
        else:  # SHORT
            self.pnl = (self.entry_price - self.exit_price) * self.quantity
            
        logger.info(f"Closed {self.direction} position for {self.symbol} at ${exit_price:.2f} ({reason}). P&L: ${self.pnl:.2f}")
        return self.pnl
        
    def get_market_value(self, current_price):
        """Get current market value of the position"""
        if not self.is_open:
            return 0
        return self.quantity * current_price
        
    def get_unrealized_pnl(self, current_price):
        """Get unrealized P&L"""
        if not self.is_open:
            return 0
            
        if self.direction == 'LONG':
            return (current_price - self.entry_price) * self.quantity
        else:  # SHORT
            return (self.entry_price - current_price) * self.quantity


class SimplePortfolio:
    """Simple portfolio class for demonstration purposes"""
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.open_positions = {}  # symbol -> Position
        self.closed_positions = []
        self.equity_curve = []
        self.trades = []
        
    def execute_trade(self, symbol, direction, quantity, price, date):
        """Execute a trade"""
        # Check if we have enough cash
        cost = quantity * price
        if direction == 'LONG' and cost > self.cash:
            logger.warning(f"Not enough cash to execute {direction} trade for {symbol}. Required: ${cost:.2f}, Available: ${self.cash:.2f}")
            return False
            
        # Execute the trade
        if direction == 'LONG':
            self.cash -= cost
        else:  # SHORT
            self.cash += cost
            
        # Create a new position
        position = Position(symbol, quantity, price, date, direction)
        self.open_positions[symbol] = position
        
        logger.info(f"Executed {direction} trade for {symbol}: {quantity} shares at ${price:.2f}. Remaining cash: ${self.cash:.2f}")
        return True
        
    def update_positions(self, current_date, price_data):
        """Update all open positions with current prices"""
        positions_updated = 0
        positions_closed = 0
        
        for symbol, position in list(self.open_positions.items()):
            if symbol in price_data:
                current_price = price_data[symbol]
                
                # Log unrealized P&L
                unrealized_pnl = position.get_unrealized_pnl(current_price)
                logger.debug(f"Position {symbol}: Unrealized P&L: ${unrealized_pnl:.2f}")
                
                # Check for stop loss or take profit
                if position.update(current_price, current_date):
                    # Position was closed
                    self.closed_positions.append(position)
                    self.cash += position.get_market_value(current_price)
                    del self.open_positions[symbol]
                    positions_closed += 1
                else:
                    positions_updated += 1
                    
        return {
            'positions_updated': positions_updated,
            'positions_closed': positions_closed
        }
        
    def get_equity(self, price_data=None):
        """Calculate total portfolio value"""
        equity = self.cash
        
        # Add value of open positions
        for symbol, position in self.open_positions.items():
            if price_data and symbol in price_data:
                current_price = price_data[symbol]
                position_value = position.get_market_value(current_price)
                logger.debug(f"Position {symbol}: {position.quantity} shares at ${current_price:.2f} = ${position_value:.2f}")
                equity += position_value
            elif position.entry_price:
                # Use entry price if no current price available
                position_value = position.get_market_value(position.entry_price)
                logger.debug(f"Position {symbol}: {position.quantity} shares at ${position.entry_price:.2f} (entry price) = ${position_value:.2f}")
                equity += position_value
                
        logger.debug(f"Total equity: ${equity:.2f} (Cash: ${self.cash:.2f})")
        return equity
        
    def update_equity_curve(self, timestamp, price_data=None):
        """Update the equity curve with current portfolio value"""
        equity = self.get_equity(price_data)
        
        # Calculate percentage change from previous point
        pct_change = 0
        if self.equity_curve:
            prev_equity = self.equity_curve[-1]['equity']
            if prev_equity > 0:
                pct_change = (equity / prev_equity - 1) * 100
                
        # Add to equity curve
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': equity,
            'pct_change': pct_change
        })
        
        logger.debug(f"Equity curve updated: ${equity:.2f} (Change: {pct_change:.2f}%)")
        return equity
        
    def calculate_performance(self):
        """Calculate performance metrics"""
        if not self.equity_curve:
            return {
                'initial_value': self.initial_capital,
                'final_value': self.initial_capital,
                'return': 0,
                'annualized_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0
            }
            
        # Calculate return
        initial_value = self.initial_capital
        final_value = self.equity_curve[-1]['equity']
        total_return = (final_value / initial_value - 1) * 100
        
        # Calculate annualized return
        if len(self.equity_curve) > 1:
            start_date = self.equity_curve[0]['timestamp']
            end_date = self.equity_curve[-1]['timestamp']
            days = (end_date - start_date).days
            if days > 0:
                annualized_return = ((1 + total_return / 100) ** (365 / days) - 1) * 100
            else:
                annualized_return = 0
        else:
            annualized_return = 0
            
        # Calculate Sharpe ratio
        if len(self.equity_curve) > 1:
            returns = [point['pct_change'] for point in self.equity_curve[1:]]
            if returns and np.std(returns) > 0:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
            
        # Calculate max drawdown
        max_drawdown = 0
        peak = self.initial_capital
        for point in self.equity_curve:
            equity = point['equity']
            if equity > peak:
                peak = equity
            else:
                drawdown = (peak - equity) / peak * 100
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
                    
        # Calculate win rate
        if self.closed_positions:
            winning_trades = sum(1 for pos in self.closed_positions if pos.pnl > 0)
            win_rate = winning_trades / len(self.closed_positions) * 100
        else:
            win_rate = 0
            
        return {
            'initial_value': initial_value,
            'final_value': final_value,
            'return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate
        }


def run_simple_backtest():
    """Run a simple backtest to demonstrate portfolio tracking"""
    # Create log directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create log file
    log_filename = f"logs/portfolio_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("Starting simple backtest demonstration")
    
    # Initialize portfolio
    portfolio = SimplePortfolio(initial_capital=100000)
    logger.info(f"Portfolio initialized with ${portfolio.initial_capital:.2f}")
    
    # Define backtest period
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2021, 1, 31)
    logger.info(f"Backtest period: {start_date.date()} to {end_date.date()}")
    
    # Create synthetic price data for demonstration
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB']
    
    # Generate synthetic price data
    price_data = {}
    for symbol in symbols:
        # Start with a random price between $100 and $500
        base_price = np.random.uniform(100, 500)
        
        # Generate daily prices with some randomness
        daily_prices = []
        current_date = start_date
        current_price = base_price
        
        while current_date <= end_date:
            if current_date.weekday() < 5:  # Weekday
                # Add some random movement (-2% to +2%)
                price_change = np.random.uniform(-0.02, 0.02)
                current_price *= (1 + price_change)
                
                daily_prices.append({
                    'date': current_date,
                    'price': current_price
                })
            
            current_date += timedelta(days=1)
        
        price_data[symbol] = daily_prices
        logger.info(f"Generated synthetic price data for {symbol}: {len(daily_prices)} days, starting at ${base_price:.2f}")
    
    # Execute some initial trades
    portfolio.execute_trade('AAPL', 'LONG', 50, price_data['AAPL'][0]['price'], start_date)
    portfolio.execute_trade('MSFT', 'LONG', 40, price_data['MSFT'][0]['price'], start_date)
    portfolio.execute_trade('GOOGL', 'LONG', 10, price_data['GOOGL'][0]['price'], start_date)
    
    # Initialize equity curve with starting value
    initial_equity = portfolio.get_equity()
    portfolio.equity_curve = [{
        'timestamp': start_date,
        'equity': initial_equity,
        'pct_change': 0
    }]
    logger.info(f"Initial portfolio value: ${initial_equity:.2f}")
    
    # Get all trading days in the date range
    trading_days = []
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() < 5:  # Weekday
            trading_days.append(current_date)
        current_date += timedelta(days=1)
    
    logger.info(f"Simulating {len(trading_days)} trading days")
    
    # Simulate each trading day
    for day_index, current_date in enumerate(trading_days):
        logger.info(f"Processing day {day_index + 1}/{len(trading_days)}: {current_date.date()}")
        
        # Get prices for this date
        daily_prices = {}
        for symbol in symbols:
            for day_data in price_data[symbol]:
                if day_data['date'].date() == current_date.date():
                    daily_prices[symbol] = day_data['price']
                    break
        
        # Add a new trade every 5 days
        if day_index > 0 and day_index % 5 == 0:
            # Choose a random symbol
            symbol = np.random.choice(symbols)
            
            # Execute a trade
            if symbol in daily_prices:
                price = daily_prices[symbol]
                quantity = int(10000 / price)  # Invest about $10,000
                
                # Randomly choose direction
                direction = np.random.choice(['LONG', 'SHORT'])
                
                portfolio.execute_trade(symbol, direction, quantity, price, current_date)
        
        # Update portfolio with today's prices
        updates = portfolio.update_positions(current_date, daily_prices)
        logger.info(f"Portfolio updated: {updates['positions_updated']} positions updated, {updates['positions_closed']} positions closed")
        
        # Update equity curve
        equity = portfolio.update_equity_curve(current_date, daily_prices)
        logger.info(f"Portfolio value: ${equity:.2f}")
    
    # Calculate performance metrics
    performance = portfolio.calculate_performance()
    
    # Print performance metrics
    logger.info("Backtest completed")
    logger.info(f"Final portfolio value: ${performance['final_value']:.2f}")
    logger.info(f"Return: {performance['return']:.2f}%")
    logger.info(f"Annualized return: {performance['annualized_return']:.2f}%")
    logger.info(f"Sharpe ratio: {performance['sharpe_ratio']:.2f}")
    logger.info(f"Max drawdown: {performance['max_drawdown']:.2f}%")
    logger.info(f"Win rate: {performance['win_rate']:.2f}%")
    
    # Print equity curve summary
    logger.info(f"Equity curve points: {len(portfolio.equity_curve)}")
    if portfolio.equity_curve:
        first_point = portfolio.equity_curve[0]
        logger.info(f"First equity point: {first_point['timestamp'].date()} - ${first_point['equity']:.2f}")
        if len(portfolio.equity_curve) > 1:
            last_point = portfolio.equity_curve[-1]
            logger.info(f"Last equity point: {last_point['timestamp'].date()} - ${last_point['equity']:.2f}")
    
    logger.info(f"Log file: {log_filename}")
    
    return {
        'success': True,
        'performance': performance,
        'equity_curve': portfolio.equity_curve,
        'log_file': log_filename
    }


if __name__ == "__main__":
    result = run_simple_backtest()
    
    # Print summary to console
    if result['success']:
        performance = result['performance']
        print("\n=== BACKTEST RESULTS ===")
        print(f"Final portfolio value: ${performance['final_value']:.2f}")
        print(f"Return: {performance['return']:.2f}%")
        print(f"Annualized return: {performance['annualized_return']:.2f}%")
        print(f"Sharpe ratio: {performance['sharpe_ratio']:.2f}")
        print(f"Max drawdown: {performance['max_drawdown']:.2f}%")
        print(f"Win rate: {performance['win_rate']:.2f}%")
        print(f"Log file: {result['log_file']}")
        print("========================")
