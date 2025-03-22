#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Backtest Combined Strategy
-------------------------------------
This script runs a backtest for the combined strategy that integrates
both mean reversion and trend following approaches.
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

# Alpaca imports
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from combined_strategy import CombinedStrategy, MarketRegime
from trend_following_strategy import TradeDirection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BacktestResults:
    """Store and analyze backtest results"""
    
    def __init__(self, initial_capital, trades, equity_curve=None):
        """Initialize with backtest results
        
        Args:
            initial_capital (float): Initial capital
            trades (list): List of trade dictionaries
            equity_curve (pd.Series, optional): Equity curve. Defaults to None.
        """
        self.initial_capital = initial_capital
        self.trades = trades
        self.equity_curve = equity_curve
        self.metrics = self.calculate_metrics()
        self.regime_performance, self.strategy_performance = self.analyze_trades()
        
    def calculate_metrics(self):
        """Calculate performance metrics"""
        if not self.trades:
            logging.warning("No trades to calculate metrics for")
            return {}
            
        # Calculate total P&L
        total_pnl = sum(t['pnl'] for t in self.trades)
        
        # Calculate win rate
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] <= 0]
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        
        # Calculate profit factor
        gross_profit = sum(t['pnl'] for t in winning_trades)
        gross_loss = abs(sum(t['pnl'] for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        # Calculate average trade return
        avg_trade_return_pct = np.mean([t['return_pct'] for t in self.trades if 'return_pct' in t]) if self.trades else 0
        
        # Calculate average win and loss
        avg_win_pct = np.mean([t['return_pct'] for t in winning_trades if 'return_pct' in t]) if winning_trades else 0
        avg_loss_pct = np.mean([t['return_pct'] for t in losing_trades if 'return_pct' in t]) if losing_trades else 0
        
        # Calculate max drawdown
        if self.equity_curve is not None:
            max_drawdown_pct = self.calculate_max_drawdown()
        else:
            max_drawdown_pct = 0
            
        # Calculate Sharpe ratio
        if self.equity_curve is not None:
            returns = self.equity_curve.pct_change().dropna()
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
        else:
            sharpe_ratio = 0
            
        # Calculate final equity
        final_equity = self.initial_capital + total_pnl
        total_return_pct = ((final_equity / self.initial_capital) - 1) * 100
            
        metrics = {
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade_return_pct': avg_trade_return_pct,
            'avg_win_pct': avg_win_pct,
            'avg_loss_pct': avg_loss_pct,
            'max_drawdown_pct': max_drawdown_pct,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': len(self.trades),
            'total_return_pct': total_return_pct
        }
        
        return metrics
        
    def analyze_trades(self):
        """Analyze trades by various dimensions (market regime, strategy, etc.)"""
        if not self.trades:
            logging.warning("No trades to analyze")
            return {}, {}
            
        # Analyze by market regime
        regime_trades = {}
        
        # First, check what format the regime data is stored in
        sample_trade = self.trades[0] if self.trades else None
        sample_regime = sample_trade.get('regime') if sample_trade else None
        
        # Initialize regime trades dictionary
        for regime in ['trending', 'range_bound', 'mixed', 'unknown']:
            regime_trades[regime] = []
            
        # Categorize trades by regime
        for trade in self.trades:
            trade_regime = trade.get('regime')
            
            # Handle different formats of regime data
            if trade_regime is None:
                regime_key = 'unknown'
            elif isinstance(trade_regime, str):
                if trade_regime.lower() in ['trending', 'range_bound', 'mixed', 'unknown']:
                    regime_key = trade_regime.lower()
                elif 'trending' in trade_regime.lower():
                    regime_key = 'trending'
                elif 'range_bound' in trade_regime.lower():
                    regime_key = 'range_bound'
                elif 'mixed' in trade_regime.lower():
                    regime_key = 'mixed'
                else:
                    regime_key = 'unknown'
            elif hasattr(trade_regime, 'value'):
                # Handle Enum values
                regime_key = trade_regime.value
            else:
                regime_key = 'unknown'
                
            regime_trades[regime_key].append(trade)
            
        # Calculate performance metrics by regime
        self.regime_performance = {}
        for regime, trades in regime_trades.items():
            if not trades:
                continue
                
            win_rate = len([t for t in trades if t['pnl'] > 0]) / len(trades) if trades else 0
            total_pnl = sum(t['pnl'] for t in trades)
            
            # Calculate average return properly
            returns = [t['return_pct'] for t in trades if 'return_pct' in t]
            avg_return = sum(returns) / len(returns) if returns else 0
            
            self.regime_performance[regime] = {
                'trades': len(trades),
                'win_rate': win_rate * 100,  # Convert to percentage
                'total_pnl': total_pnl,
                'avg_return': avg_return
            }
            
        # Analyze by strategy
        strategy_trades = {}
        for strategy in ['mean_reversion', 'trend_following', 'mixed']:
            strategy_trades[strategy] = [t for t in self.trades if t.get('strategy') == strategy]
            
        self.strategy_performance = {}
        for strategy, trades in strategy_trades.items():
            if not trades:
                continue
                
            win_rate = len([t for t in trades if t['pnl'] > 0]) / len(trades) if trades else 0
            total_pnl = sum(t['pnl'] for t in trades)
            
            # Calculate average return properly
            returns = [t['return_pct'] for t in trades if 'return_pct' in t]
            avg_return = sum(returns) / len(returns) if returns else 0
            
            self.strategy_performance[strategy] = {
                'trades': len(trades),
                'win_rate': win_rate * 100,  # Convert to percentage
                'total_pnl': total_pnl,
                'avg_return': avg_return
            }
        
        # Update metrics with regime and strategy performance
        if hasattr(self, 'metrics') and self.metrics:
            self.metrics['regime_performance'] = {
                regime: {
                    'trades': perf['trades'],
                    'win_rate': perf['win_rate'] / 100,  # Convert back to decimal for consistency
                    'total_return': perf['total_pnl'],
                    'avg_return_pct': perf['avg_return']
                } for regime, perf in self.regime_performance.items()
            }
            
            self.metrics['strategy_performance'] = {
                strategy: {
                    'trades': perf['trades'],
                    'win_rate': perf['win_rate'] / 100,  # Convert back to decimal for consistency
                    'total_return': perf['total_pnl'],
                    'avg_return_pct': perf['avg_return']
                } for strategy, perf in self.strategy_performance.items()
            }
            
        return self.regime_performance, self.strategy_performance
    
    def calculate_max_drawdown(self):
        """Calculate maximum drawdown percentage from equity curve"""
        if self.equity_curve is None or len(self.equity_curve) == 0:
            return 0
            
        # Calculate running maximum
        running_max = self.equity_curve.cummax()
        
        # Calculate drawdown in percentage terms
        drawdown = (self.equity_curve - running_max) / running_max * 100
        
        # Get the maximum drawdown
        max_drawdown_pct = abs(drawdown.min()) if not drawdown.empty else 0
        
        return max_drawdown_pct
    
    def generate_report(self):
        """Generate a comprehensive backtest report"""
        if not hasattr(self, 'metrics') or not self.metrics:
            return "No metrics available for report generation"
            
        # Extract metrics
        total_return = self.metrics.get('total_return_pct', 0)
        win_rate = self.metrics.get('win_rate', 0) * 100  # Convert to percentage
        profit_factor = self.metrics.get('profit_factor', 0)
        max_drawdown = self.metrics.get('max_drawdown_pct', 0)
        sharpe_ratio = self.metrics.get('sharpe_ratio', 0)
        total_trades = self.metrics.get('total_trades', 0)
        avg_trade_return = self.metrics.get('avg_trade_return_pct', 0)
        avg_win = self.metrics.get('avg_win_pct', 0)
        avg_loss = self.metrics.get('avg_loss_pct', 0)
        
        # Generate report
        report = f"""
        ===== COMBINED STRATEGY BACKTEST REPORT =====

        Performance Metrics:
        - Total Return: {total_return:.2f}%
        - Win Rate: {win_rate:.2f}%
        - Profit Factor: {profit_factor:.2f}
        - Max Drawdown: {max_drawdown:.2f}%
        - Sharpe Ratio: {sharpe_ratio:.2f}
        - Total Trades: {total_trades}
        - Average Trade Return: {avg_trade_return:.2f}%
        - Average Win: {avg_win:.2f}%
        - Average Loss: {avg_loss:.2f}%
"""
        
        # Add regime performance if available
        if 'regime_performance' in self.metrics:
            report += "\nPerformance by Market Regime:\n"
            
            for regime, perf in self.metrics['regime_performance'].items():
                if perf['trades'] == 0:
                    continue
                    
                trades = perf['trades']
                win_rate = perf['win_rate'] * 100
                total_return = perf['total_return']
                avg_return = perf.get('avg_return_pct', 0)
                
                regime_pct = (trades / total_trades) * 100 if total_trades > 0 else 0
                
                report += f"""
        - {regime.capitalize()}:
          - Trades: {trades} ({regime_pct:.1f}% of total)
          - Win Rate: {win_rate:.2f}%
          - Total Return: ${total_return:.2f}
          - Average Return: {avg_return:.2f}%
"""
        
        # Add strategy performance if available
        if 'strategy_performance' in self.metrics:
            report += "\nPerformance by Strategy:\n"
            
            for strategy, perf in self.metrics['strategy_performance'].items():
                if perf['trades'] == 0:
                    continue
                    
                trades = perf['trades']
                win_rate = perf['win_rate'] * 100
                total_return = perf['total_return']
                avg_return = perf.get('avg_return_pct', 0)
                
                strategy_pct = (trades / total_trades) * 100 if total_trades > 0 else 0
                
                report += f"""
        - {strategy.replace('_', ' ').capitalize()}:
          - Trades: {trades} ({strategy_pct:.1f}% of total)
          - Win Rate: {win_rate:.2f}%
          - Total Return: ${total_return:.2f}
          - Average Return: {avg_return:.2f}%
"""
        
        return report

class Backtester:
    """Backtester for the combined strategy"""
    
    def __init__(self, config_file):
        """Initialize with configuration file
        
        Args:
            config_file (str): Path to configuration file
        """
        # Load configuration
        self.config = self.load_config(config_file)
        
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
        self.position_size_pct = self.config['general']['position_size_pct']
        self.max_positions = self.config['general'].get('max_positions', 8)
        
        # Parse backtest dates
        self.start_date = pd.to_datetime(self.config['general']['backtest_start_date'])
        self.end_date = pd.to_datetime(self.config['general']['backtest_end_date'])
        
        # Initialize backtest state
        self.capital = self.initial_capital
        self.positions = {}  # symbol -> position dict
        self.trades = []  # completed trades
        self.equity_curve = []  # daily equity values
        
        # Initialize Alpaca API
        self.api = self.initialize_alpaca_api()
        
        logging.info(f"Initialized OPTIMIZED strategy with parameters: BB period={self.config['strategy_configs']['MeanReversion'].get('bb_period', 20)}, "
                    f"BB std={self.config['strategy_configs']['MeanReversion'].get('bb_std', 1.9)}, "
                    f"RSI period={self.config['strategy_configs']['MeanReversion'].get('rsi_period', 14)}, "
                    f"RSI thresholds={self.config['strategy_configs']['MeanReversion'].get('rsi_oversold', 35)}/"
                    f"{self.config['strategy_configs']['MeanReversion'].get('rsi_overbought', 65)}, "
                    f"Require reversal={self.config['strategy_configs']['MeanReversion'].get('require_reversal', True)}, "
                    f"SL/TP ATR multipliers={self.config['strategy_configs']['MeanReversion'].get('stop_loss_atr_multiplier', 1.8)}/"
                    f"{self.config['strategy_configs']['MeanReversion'].get('take_profit_atr_multiplier', 3.0)}, "
                    f"Volume filter={self.config['strategy_configs']['MeanReversion'].get('volume_filter', True)}")
        
        logging.info(f"Initialized Combined Strategy with weights: MR={self.config['strategy_configs']['Combined'].get('mean_reversion_weight', 0.6)}, "
                    f"TF={self.config['strategy_configs']['Combined'].get('trend_following_weight', 0.3)}")
    
    def load_config(self, config_file):
        """Load configuration from YAML file
        
        Args:
            config_file (str): Path to configuration file
            
        Returns:
            dict: Configuration dictionary
        """
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def initialize_alpaca_api(self):
        """Initialize Alpaca API
        
        Returns:
            alpaca.trading.client.TradingClient: Alpaca Trading API client
        """
        # Load credentials from file
        with open('alpaca_credentials.json', 'r') as f:
            credentials = json.load(f)
        
        # Use paper trading credentials
        paper_credentials = credentials['paper']
        api_key = paper_credentials['api_key']
        api_secret = paper_credentials['api_secret']
        
        # Initialize Trading client
        api = TradingClient(api_key, api_secret, paper=True)
        
        return api
    
    def fetch_data(self, symbol, timeframe, start_date, end_date):
        """Fetch historical data from Alpaca
        
        Args:
            symbol (str): Symbol to fetch data for
            timeframe (str): Timeframe to fetch data for
            start_date (datetime or str): Start date
            end_date (datetime or str): End date
            
        Returns:
            pd.DataFrame: Historical data
        """
        # Convert string dates to datetime if needed
        if isinstance(start_date, str):
            start_date = dt.datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = dt.datetime.strptime(end_date, '%Y-%m-%d')
            
        # Convert timeframe to Alpaca format
        if timeframe == '1D':
            timeframe = TimeFrame.Day
        elif timeframe == '1H':
            timeframe = TimeFrame.Hour
        
        # Load credentials
        with open('alpaca_credentials.json', 'r') as f:
            credentials = json.load(f)
        
        # Use paper trading credentials
        paper_credentials = credentials['paper']
        api_key = paper_credentials['api_key']
        api_secret = paper_credentials['api_secret']
        
        # Initialize client
        client = StockHistoricalDataClient(api_key, api_secret)
        
        # Prepare request
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=start_date,
            end=end_date + dt.timedelta(days=1)  # Add one day to include end_date
        )
        
        # Fetch data
        try:
            bars = client.get_stock_bars(request_params)
            
            # Extract data for the symbol
            if symbol in bars.data:
                # Convert bars to DataFrame
                symbol_bars = bars.data[symbol]
                df = pd.DataFrame([bar.dict() for bar in symbol_bars])
                
                # Set timestamp as index
                if 'timestamp' in df.columns:
                    df.set_index('timestamp', inplace=True)
                    df.index = pd.to_datetime(df.index)
                    df = df.tz_localize(None)  # Remove timezone info
                
                logging.info(f"Fetched {len(df)} bars for {symbol}")
                return df
            else:
                logging.warning(f"No data found for {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            logging.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def run_backtest(self, start_date=None, end_date=None):
        """Run backtest
        
        Args:
            start_date (datetime, optional): Start date. Defaults to None.
            end_date (datetime, optional): End date. Defaults to None.
            
        Returns:
            BacktestResults: Backtest results
        """
        # Use provided dates or default to config dates
        start_date = start_date or self.start_date
        end_date = end_date or self.end_date
        
        logging.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Fetch data for all symbols
        data = {}
        for symbol in self.symbols:
            data[symbol] = self.fetch_data(symbol, self.timeframe, start_date, end_date)
        
        # Get all unique dates across all symbols
        all_dates = sorted(set().union(*[df.index.to_list() for df in data.values()]))
        
        logging.info(f"Running backtest across {len(all_dates)} trading days")
        
        # Initialize equity curve
        equity_curve = pd.DataFrame(index=all_dates, columns=['cash', 'positions_value', 'equity'])
        equity_curve['cash'] = self.initial_capital
        equity_curve['positions_value'] = 0
        equity_curve['equity'] = self.initial_capital
        
        # Run backtest day by day
        for date in all_dates:
            # Update positions value
            positions_value = 0
            for symbol, position in list(self.positions.items()):
                # Skip if data not available for this date
                if symbol not in data or date not in data[symbol].index:
                    continue
                
                # Get current price
                current_price = data[symbol].loc[date, 'close']
                
                # Update position value
                position['current_price'] = current_price
                position['current_value'] = position['shares'] * current_price
                position['unrealized_pnl'] = position['current_value'] - position['cost_basis']
                position['unrealized_pnl_pct'] = (position['unrealized_pnl'] / position['cost_basis']) * 100
                
                # Check for stop loss or take profit
                if position['direction'] == 'LONG':
                    if current_price <= position['stop_loss']:
                        self.close_position(symbol, current_price, date, 'stop_loss')
                    elif current_price >= position['take_profit']:
                        self.close_position(symbol, current_price, date, 'take_profit')
                elif position['direction'] == 'SHORT':
                    if current_price >= position['stop_loss']:
                        self.close_position(symbol, current_price, date, 'stop_loss')
                    elif current_price <= position['take_profit']:
                        self.close_position(symbol, current_price, date, 'take_profit')
                
                # Add to positions value if still open
                if symbol in self.positions:
                    positions_value += self.positions[symbol]['current_value']
            
            # Generate signals for each symbol
            for symbol in self.symbols:
                # Skip if data not available for this date
                if symbol not in data or date not in data[symbol].index:
                    continue
                
                # Get data up to current date
                df = data[symbol].loc[:date].copy()
                
                # Skip if not enough data
                if len(df) < 50:  # Need enough data for indicators
                    continue
                
                # Generate signals
                signals = self.strategy.generate_signals(df, symbol)
                
                # Process signals
                for signal in signals:
                    # Skip if already have a position in this symbol
                    if symbol in self.positions:
                        continue
                    
                    # Skip if conflicting direction with existing position
                    if any(p['direction'] != signal['direction'] for s, p in self.positions.items() if s == symbol):
                        continue
                    
                    # Skip if at maximum positions
                    if len(self.positions) >= self.max_positions:
                        continue
                    
                    # Calculate position size
                    price = signal['price']
                    shares = self.strategy.calculate_position_size(signal, self.capital, len(self.positions))
                    
                    # Skip if not enough capital or shares
                    if shares <= 0 or shares * price > self.capital:
                        continue
                    
                    # Open position
                    self.open_position(
                        symbol=symbol,
                        direction=signal['direction'],
                        price=price,
                        shares=shares,
                        stop_loss=signal['stop_loss'],
                        take_profit=signal['take_profit'],
                        date=date,
                        regime=signal.get('regime', 'unknown'),
                        strategy=signal.get('strategy', 'unknown')
                    )
            
            # Update equity curve
            equity_curve.loc[date, 'cash'] = self.capital
            equity_curve.loc[date, 'positions_value'] = positions_value
            equity_curve.loc[date, 'equity'] = self.capital + positions_value
        
        # Close any remaining positions at the end of the backtest
        for symbol, position in list(self.positions.items()):
            if symbol in data and all_dates[-1] in data[symbol].index:
                self.close_position(symbol, data[symbol].loc[all_dates[-1], 'close'], all_dates[-1], 'end_of_backtest')
        
        # Calculate final equity
        final_equity = self.capital + sum(p['current_value'] for p in self.positions.values())
        
        # Calculate performance metrics
        self.results = BacktestResults(
            initial_capital=self.initial_capital,
            trades=self.trades,
            equity_curve=equity_curve['equity']
        )
        
        # Generate and print report
        report = self.results.generate_report()
        print(report)
        
        return self.results
    
    def open_position(self, symbol, direction, price, shares, stop_loss, take_profit, date, regime, strategy):
        """Open a new position
        
        Args:
            symbol (str): Symbol to open position for
            direction (str): Direction of position ('LONG' or 'SHORT')
            price (float): Entry price
            shares (int): Number of shares
            stop_loss (float): Stop loss price
            take_profit (float): Take profit price
            date (datetime): Entry date
            regime (str or Enum): Market regime
            strategy (str): Strategy that generated the signal
        """
        # Calculate cost and update capital
        cost = shares * price
        self.capital -= cost
        
        # Convert regime to string if it's an Enum
        if hasattr(regime, 'value'):
            regime_str = regime.value
        else:
            regime_str = str(regime)
        
        # Create position
        self.positions[symbol] = {
            'direction': direction,
            'entry_price': price,
            'shares': shares,
            'cost_basis': cost,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_date': date,
            'current_price': price,
            'current_value': cost,
            'unrealized_pnl': 0,
            'unrealized_pnl_pct': 0,
            'regime': regime_str,
            'strategy': strategy
        }
        
        logging.info(f"Generated {direction} signal for {symbol} at {price:.2f}, stop={stop_loss:.2f}, target={take_profit:.2f}")
    
    def close_position(self, symbol, price, date, reason):
        """Close an existing position
        
        Args:
            symbol (str): Symbol to close position for
            price (float): Exit price
            date (datetime): Exit date
            reason (str): Reason for closing position
        """
        # Skip if no position
        if symbol not in self.positions:
            return
        
        # Get position
        position = self.positions[symbol]
        
        # Calculate PnL
        if position['direction'] == 'LONG':
            pnl = (price - position['entry_price']) * position['shares']
        else:  # SHORT
            pnl = (position['entry_price'] - price) * position['shares']
        
        # Calculate return percentage
        return_pct = (pnl / position['cost_basis']) * 100
        
        # Update capital
        self.capital += position['shares'] * price
        
        # Record trade
        trade = {
            'symbol': symbol,
            'direction': position['direction'],
            'entry_date': position['entry_date'],
            'exit_date': date,
            'entry_price': position['entry_price'],
            'exit_price': price,
            'shares': position['shares'],
            'pnl': pnl,
            'return_pct': return_pct,
            'regime': position.get('regime', 'unknown'),
            'strategy': position.get('strategy', 'unknown'),
            'reason': reason
        }
        self.trades.append(trade)
        
        # Update strategy regime performance
        self.strategy.update_regime_performance(trade)
        
        # Remove position
        del self.positions[symbol]
        
        logging.info(f"Closed {position['direction']} position in {symbol}: {position['shares']} shares at ${price:.2f}, PnL: ${pnl:.2f} ({return_pct:.2f}%)")

def print_quarterly_comparison(results):
    """Print quarterly comparison table"""
    if not hasattr(results, 'metrics') or not results.metrics:
        logging.warning("No metrics to print quarterly comparison for")
        return
        
    print("\n===== QUARTERLY PERFORMANCE COMPARISON =====")
    print("Quarter    Return %   Win Rate   Profit Factor   Max DD %   Trades")
    print("-" * 65)
    
    # This is a placeholder since we don't have quarterly results in this version
    # We'll just print the overall results
    total_return = results.metrics.get('total_return_pct', 0)
    win_rate = results.metrics.get('win_rate', 0) * 100
    profit_factor = results.metrics.get('profit_factor', 0)
    max_dd = results.metrics.get('max_drawdown_pct', 0)
    total_trades = results.metrics.get('total_trades', 0)
    
    print(f"2023-Full     {total_return:6.2f}%    {win_rate:5.2f}%         {profit_factor:5.2f}    {max_dd:5.2f}%    {total_trades}")

def run_quarterly_backtests():
    """Run backtests for each quarter to analyze performance across different time periods"""
    # Define quarters
    quarters = [
        ("2023-Q1", "2023-01-01", "2023-03-31"),
        ("2023-Q2", "2023-04-01", "2023-06-30"),
        ("2023-Q3", "2023-07-01", "2023-09-30"),
        ("2023-Q4", "2023-10-01", "2023-12-31"),
        ("2024-Q1", "2024-01-01", "2024-03-31"),
        ("2024-YTD", "2024-01-01", "2024-04-30"),
        ("2023-Full", "2023-01-01", "2023-12-31")
    ]
    
    results = {}
    
    for quarter_name, start_date, end_date in quarters:
        # Update config dates
        with open('configuration_combined_strategy.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        config['general']['backtest_start_date'] = start_date
        config['general']['backtest_end_date'] = end_date
        
        # Write updated config
        with open('configuration_combined_strategy_temp.yaml', 'w') as f:
            yaml.dump(config, f)
        
        # Run backtest
        print(f"\n===== Running backtest for {quarter_name} =====")
        backtester = Backtester('configuration_combined_strategy_temp.yaml')
        result = backtester.run_backtest()
        
        # Store results
        if result is not None:
            results[quarter_name] = result.calculate_metrics()
        else:
            print(f"No results for {quarter_name}")
    
    # Clean up temp config
    if os.path.exists('configuration_combined_strategy_temp.yaml'):
        os.remove('configuration_combined_strategy_temp.yaml')
    
    # Compare results
    if results:
        print("\n===== QUARTERLY PERFORMANCE COMPARISON =====")
        print(f"{'Quarter':<10} {'Return %':<10} {'Win Rate':<10} {'Profit Factor':<15} {'Max DD %':<10} {'Trades':<8}")
        print("-" * 65)
        
        for quarter, metrics in results.items():
            print(f"{quarter:<10} {metrics['total_return_pct']:>8.2f}% {metrics['win_rate']*100:>8.2f}% {metrics['profit_factor']:>13.2f} {metrics['max_drawdown_pct']:>8.2f}% {metrics['total_trades']:>6}")
    else:
        print("No quarterly results to display")

def run_backtest():
    """Run the backtest"""
    # Run backtest with new stock set
    print("===== Running full backtest with new stock set =====")
    backtester = Backtester('configuration_combined_strategy_new_stocks.yaml')
    results = backtester.run_backtest()
    
    # Analyze individual trades
    if results:
        print_quarterly_comparison(results)
    
    return results

if __name__ == "__main__":
    run_backtest()
