#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Backtest Combined Strategy (No TA-Lib)
-------------------------------------
This script runs a backtest for the combined strategy that integrates
both mean reversion and trend following approaches without using TA-Lib.
"""

import os
import json
import yaml
import logging
import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from combined_strategy_no_talib import CombinedStrategy, MarketRegime
from trend_following_strategy import TradeDirection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='combined_strategy_backtest_no_talib.log',
    filemode='w'
)
logger = logging.getLogger(__name__)

class BacktestResults:
    """Store and analyze backtest results"""
    
    def __init__(self, initial_capital, trades, equity_curve, metrics=None):
        """Initialize with backtest results"""
        self.initial_capital = initial_capital
        self.trades = trades
        self.equity_curve = equity_curve
        self.metrics = metrics or self.calculate_metrics()
    
    def calculate_metrics(self):
        """Calculate performance metrics"""
        if not self.trades:
            return {
                'total_return_pct': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'max_drawdown_pct': 0,
                'sharpe_ratio': 0,
                'total_trades': 0,
                'avg_trade_return_pct': 0,
                'avg_win_pct': 0,
                'avg_loss_pct': 0
            }
        
        # Calculate basic metrics
        final_equity = self.equity_curve['equity'].iloc[-1]
        total_return = final_equity - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100
        
        # Calculate win rate
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        
        # Calculate profit factor
        gross_profit = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
        losing_trades = [t for t in self.trades if t['pnl'] <= 0]
        gross_loss = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        # Calculate drawdown
        equity_curve = self.equity_curve['equity']
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max * 100
        max_drawdown_pct = abs(drawdown.min())
        
        # Calculate Sharpe ratio (annualized)
        daily_returns = self.equity_curve['equity'].pct_change().dropna()
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if len(daily_returns) > 0 and daily_returns.std() != 0 else 0
        
        # Calculate average trade metrics
        avg_trade_return_pct = np.mean([t['return_pct'] for t in self.trades]) if self.trades else 0
        avg_win_pct = np.mean([t['return_pct'] for t in winning_trades]) if winning_trades else 0
        avg_loss_pct = np.mean([t['return_pct'] for t in losing_trades]) if losing_trades else 0
        
        # Get regime performance if available
        regime_performance = {}
        for regime in ['trending', 'range_bound', 'mixed']:
            regime_trades = [t for t in self.trades if t.get('regime') == regime]
            if regime_trades:
                regime_wins = [t for t in regime_trades if t['pnl'] > 0]
                regime_performance[regime] = {
                    'trades': len(regime_trades),
                    'win_rate': len(regime_wins) / len(regime_trades) if regime_trades else 0,
                    'total_return': sum(t['pnl'] for t in regime_trades)
                }
        
        # Get strategy performance
        strategy_performance = {}
        for strategy in ['mean_reversion', 'trend_following']:
            strategy_trades = [t for t in self.trades if t.get('strategy') == strategy]
            if strategy_trades:
                strategy_wins = [t for t in strategy_trades if t['pnl'] > 0]
                strategy_performance[strategy] = {
                    'trades': len(strategy_trades),
                    'win_rate': len(strategy_wins) / len(strategy_trades) if strategy_trades else 0,
                    'total_return': sum(t['pnl'] for t in strategy_trades)
                }
        
        return {
            'total_return_pct': total_return_pct,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown_pct': max_drawdown_pct,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': len(self.trades),
            'avg_trade_return_pct': avg_trade_return_pct,
            'avg_win_pct': avg_win_pct,
            'avg_loss_pct': avg_loss_pct,
            'regime_performance': regime_performance,
            'strategy_performance': strategy_performance
        }
    
    def generate_report(self):
        """Generate a report of backtest results"""
        metrics = self.metrics or self.calculate_metrics()
        
        report = f"""
        ===== COMBINED STRATEGY BACKTEST REPORT =====

        Performance Metrics:
        - Total Return: {metrics['total_return_pct']:.2f}%
        - Win Rate: {metrics['win_rate']*100:.2f}%
        - Profit Factor: {metrics['profit_factor']:.2f}
        - Max Drawdown: {metrics['max_drawdown_pct']:.2f}%
        - Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
        - Total Trades: {metrics['total_trades']}
        - Average Trade Return: {metrics['avg_trade_return_pct']:.2f}%
        - Average Win: {metrics['avg_win_pct']:.2f}%
        - Average Loss: {metrics['avg_loss_pct']:.2f}%
        """
        
        # Add regime performance if available
        if 'regime_performance' in metrics and metrics['regime_performance']:
            report += "\nPerformance by Market Regime:\n"
            for regime, perf in metrics['regime_performance'].items():
                report += f"""
        - {regime.capitalize()}:
          - Trades: {perf['trades']}
          - Win Rate: {perf['win_rate']*100:.2f}%
          - Total Return: ${perf['total_return']:.2f}
        """
        
        # Add strategy performance if available
        if 'strategy_performance' in metrics and metrics['strategy_performance']:
            report += "\nPerformance by Strategy:\n"
            for strategy, perf in metrics['strategy_performance'].items():
                report += f"""
        - {strategy.replace('_', ' ').capitalize()}:
          - Trades: {perf['trades']}
          - Win Rate: {perf['win_rate']*100:.2f}%
          - Total Return: ${perf['total_return']:.2f}
        """
                
        return report

class Backtester:
    """Backtester for the combined strategy"""
    
    def __init__(self, config_file):
        """Initialize with configuration file"""
        # Load configuration
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize strategy
        self.strategy = CombinedStrategy(self.config)
        
        # Set up logging
        log_level = self.config['general'].get('log_level', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Get backtest parameters
        self.symbols = self.config['general']['symbols']
        self.timeframe = self.config['general']['timeframe']
        self.initial_capital = self.config['general']['initial_capital']
        
        # Initialize Alpaca client
        self.setup_alpaca_client()
    
    def setup_alpaca_client(self):
        """Set up Alpaca API client"""
        # Load API credentials
        with open('alpaca_credentials.json', 'r') as f:
            credentials = json.load(f)
        
        # Use paper trading credentials
        api_key = credentials['paper']['api_key']
        api_secret = credentials['paper']['api_secret']
        
        # Initialize client
        self.client = StockHistoricalDataClient(api_key, api_secret)
    
    def fetch_historical_data(self, symbols, start_date, end_date, timeframe='1D'):
        """Fetch historical data from Alpaca"""
        logger.info(f"Fetching historical data for {symbols} from {start_date} to {end_date}")
        
        # Convert timeframe string to TimeFrame object
        if timeframe == '1D':
            tf = TimeFrame.Day
        elif timeframe == '1H':
            tf = TimeFrame.Hour
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        # Prepare request
        request_params = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=tf,
            start=dt.datetime.strptime(start_date, '%Y-%m-%d'),
            end=dt.datetime.strptime(end_date, '%Y-%m-%d') + dt.timedelta(days=1)  # Add one day to include end_date
        )
        
        # Fetch data
        try:
            bars = self.client.get_stock_bars(request_params)
            
            # Convert to dictionary of DataFrames
            data = {}
            for symbol in symbols:
                if symbol in bars.data:
                    # Extract the list of bars for this symbol
                    symbol_bars = bars.data[symbol]
                    
                    # Convert to DataFrame
                    df = pd.DataFrame([bar.dict() for bar in symbol_bars])
                    
                    # Set timestamp as index
                    if 'timestamp' in df.columns:
                        df.set_index('timestamp', inplace=True)
                        df.index = pd.to_datetime(df.index)
                        df = df.tz_localize(None)  # Remove timezone info
                    
                    data[symbol] = df
            
            return data
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            raise
    
    def run(self, start_date=None, end_date=None):
        """Run the backtest"""
        # Use config dates if not provided
        start_date = start_date or self.config['general']['backtest_start_date']
        end_date = end_date or self.config['general']['backtest_end_date']
        
        logger.info(f"Running backtest from {start_date} to {end_date}")
        
        # Fetch historical data
        data = self.fetch_historical_data(self.symbols, start_date, end_date, self.timeframe)
        
        # Initialize portfolio
        capital = self.initial_capital
        positions = {}  # symbol -> {entry_price, shares, entry_date, stop_loss, take_profit, strategy, regime}
        trades = []
        equity_curve = []
        
        # Track daily equity
        dates = pd.date_range(start=start_date, end=end_date)
        
        # Loop through each date
        for date in dates:
            date_str = date.strftime('%Y-%m-%d')
            logger.info(f"Processing date: {date_str}")
            
            # Skip weekends and holidays
            if date.weekday() >= 5:  # Saturday or Sunday
                continue
            
            # Calculate current equity
            current_equity = capital
            for symbol, position in positions.items():
                if symbol in data and date in data[symbol].index:
                    current_price = data[symbol].loc[date, 'close']
                    position_value = position['shares'] * current_price
                    current_equity += position_value
            
            # Record equity
            equity_curve.append({
                'date': date,
                'equity': current_equity
            })
            
            # Process exits
            symbols_to_remove = []
            for symbol, position in positions.items():
                if symbol in data and date in data[symbol].index:
                    current_bar = data[symbol].loc[date]
                    
                    # Check for exit conditions
                    exit_price = None
                    exit_reason = None
                    
                    # Stop loss hit
                    if current_bar['low'] <= position['stop_loss']:
                        exit_price = position['stop_loss']
                        exit_reason = 'stop_loss'
                    
                    # Take profit hit
                    elif current_bar['high'] >= position['take_profit']:
                        exit_price = position['take_profit']
                        exit_reason = 'take_profit'
                    
                    # Strategy-specific exit conditions
                    elif position['strategy'] == 'mean_reversion':
                        # Mean reversion exit: price reverts back to mean
                        if (position['direction'] == 'long' and current_bar['close'] >= position['target']) or \
                           (position['direction'] == 'short' and current_bar['close'] <= position['target']):
                            exit_price = current_bar['close']
                            exit_reason = 'target_reached'
                    
                    elif position['strategy'] == 'trend_following':
                        # Trend following exit: trend reversal
                        if (position['direction'] == 'long' and current_bar['close'] < position['trailing_stop']) or \
                           (position['direction'] == 'short' and current_bar['close'] > position['trailing_stop']):
                            exit_price = current_bar['close']
                            exit_reason = 'trailing_stop'
                    
                    # Exit position if conditions met
                    if exit_price is not None:
                        # Calculate P&L
                        if position['direction'] == 'long':
                            pnl = (exit_price - position['entry_price']) * position['shares']
                        else:  # short
                            pnl = (position['entry_price'] - exit_price) * position['shares']
                        
                        # Calculate return percentage
                        position_cost = position['entry_price'] * position['shares']
                        return_pct = (pnl / position_cost) * 100
                        
                        # Record trade
                        trade = {
                            'symbol': symbol,
                            'entry_date': position['entry_date'],
                            'exit_date': date_str,
                            'direction': position['direction'],
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'shares': position['shares'],
                            'pnl': pnl,
                            'return_pct': return_pct,
                            'exit_reason': exit_reason,
                            'strategy': position['strategy'],
                            'regime': position['regime']
                        }
                        trades.append(trade)
                        
                        # Update capital
                        capital += position['shares'] * exit_price
                        
                        # Mark position for removal
                        symbols_to_remove.append(symbol)
                        
                        # Update strategy performance tracking
                        self.strategy.update_regime_performance(trade)
                        
                        logger.info(f"Exited {position['direction']} position in {symbol} at {exit_price:.2f} ({exit_reason}), P&L: ${pnl:.2f}")
            
            # Remove closed positions
            for symbol in symbols_to_remove:
                del positions[symbol]
            
            # Generate signals for each symbol
            for symbol in self.symbols:
                if symbol in data and date in data[symbol].index:
                    # Skip if already have a position in this symbol
                    if symbol in positions:
                        continue
                    
                    # Get historical data up to current date
                    symbol_data = data[symbol].loc[:date].copy()
                    
                    # Skip if not enough data
                    if len(symbol_data) < 50:  # Need enough data for indicators
                        continue
                    
                    # Generate signals
                    signals = self.strategy.generate_signals(symbol_data, symbol)
                    
                    # Process signals
                    for signal in signals:
                        # Skip if signal date is not current date
                        if signal['timestamp'].date() != date.date():
                            continue
                        
                        # Calculate position size
                        shares = self.strategy.calculate_position_size(
                            signal, capital, len(positions)
                        )
                        
                        # Skip if not enough capital or shares
                        if shares <= 0 or shares * signal['price'] > capital:
                            continue
                        
                        # Enter position
                        positions[symbol] = {
                            'entry_price': signal['price'],
                            'shares': shares,
                            'entry_date': date_str,
                            'direction': signal['direction'],
                            'stop_loss': signal['stop_loss'],
                            'take_profit': signal['take_profit'],
                            'target': signal.get('target'),
                            'trailing_stop': signal.get('trailing_stop'),
                            'strategy': signal['strategy'],
                            'regime': signal['regime']
                        }
                        
                        # Update capital
                        capital -= shares * signal['price']
                        
                        logger.info(f"Entered {signal['direction']} position in {symbol} at {signal['price']:.2f}, Shares: {shares}, Strategy: {signal['strategy']}")
                        
                        # Only take one signal per symbol per day
                        break
        
        # Close any remaining positions at the end of the backtest
        for symbol, position in positions.items():
            if symbol in data:
                # Use the last available price
                last_date = data[symbol].index[-1]
                exit_price = data[symbol].loc[last_date, 'close']
                
                # Calculate P&L
                if position['direction'] == 'long':
                    pnl = (exit_price - position['entry_price']) * position['shares']
                else:  # short
                    pnl = (position['entry_price'] - exit_price) * position['shares']
                
                # Calculate return percentage
                position_cost = position['entry_price'] * position['shares']
                return_pct = (pnl / position_cost) * 100
                
                # Record trade
                trade = {
                    'symbol': symbol,
                    'entry_date': position['entry_date'],
                    'exit_date': last_date.strftime('%Y-%m-%d'),
                    'direction': position['direction'],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'shares': position['shares'],
                    'pnl': pnl,
                    'return_pct': return_pct,
                    'exit_reason': 'end_of_backtest',
                    'strategy': position['strategy'],
                    'regime': position['regime']
                }
                trades.append(trade)
                
                # Update capital
                capital += position['shares'] * exit_price
                
                # Update strategy performance tracking
                self.strategy.update_regime_performance(trade)
                
                logger.info(f"Closed {position['direction']} position in {symbol} at {exit_price:.2f} (end of backtest), P&L: ${pnl:.2f}")
        
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(equity_curve)
        equity_df.set_index('date', inplace=True)
        
        # Create backtest results
        results = BacktestResults(
            initial_capital=self.initial_capital,
            trades=trades,
            equity_curve=equity_df
        )
        
        return results

def run_backtest():
    """Run the combined strategy backtest"""
    # Initialize backtester
    backtester = Backtester('configuration_combined_strategy.yaml')
    
    # Run backtest
    results = backtester.run()
    
    # Print report
    report = results.generate_report()
    print(report)
    
    # Save report to file
    with open('combined_strategy_report.md', 'w') as f:
        f.write(report)
    
    # Plot equity curve
    plt.figure(figsize=(12, 6))
    results.equity_curve['equity'].plot()
    plt.title('Combined Strategy Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Equity ($)')
    plt.grid(True)
    plt.savefig('combined_strategy_equity_curve.png')
    plt.close()
    
    # Save trades to CSV
    trades_df = pd.DataFrame(results.trades)
    if not trades_df.empty:
        trades_df.to_csv('combined_strategy_trades.csv', index=False)
    
    # Save metrics to JSON
    with open('combined_strategy_metrics.json', 'w') as f:
        json.dump(results.metrics, f, indent=4)
    
    return results

if __name__ == "__main__":
    run_backtest()
