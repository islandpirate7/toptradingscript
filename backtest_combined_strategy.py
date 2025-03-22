#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Backtest Combined Strategy
-------------------------------------
This script runs a backtest for the combined strategy that integrates
both mean reversion and trend following approaches.
"""

import datetime as dt
import logging
import pandas as pd
import numpy as np
import yaml
import json
import matplotlib.pyplot as plt

# Alpaca imports
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import REST

from combined_strategy import CombinedStrategy, MarketRegime
from trend_following_strategy import TradeDirection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BacktestResults:
    """Results of a backtest"""
    
    def __init__(self, initial_capital, equity_curve, trades, signals_generated, signals_executed, start_date=None, end_date=None, metrics=None, daily_returns=None):
        """Initialize backtest results
        
        Args:
            initial_capital (float): Initial capital
            equity_curve (list): Equity curve
            trades (list): List of trades
            signals_generated (list): List of signals generated
            signals_executed (list): List of signals executed
            start_date (datetime, optional): Start date. Defaults to None.
            end_date (datetime, optional): End date. Defaults to None.
            metrics (dict, optional): Metrics. Defaults to None.
            daily_returns (list, optional): Daily returns. Defaults to None.
        """
        self.initial_capital = initial_capital
        self.equity_curve = equity_curve
        self.trades = trades
        self.signals_generated = signals_generated
        self.signals_executed = signals_executed
        self.start_date = start_date
        self.end_date = end_date
        self.metrics = metrics if metrics is not None else {}
        self.daily_returns = daily_returns if daily_returns is not None else []
        
        # Calculate final equity
        self.final_equity = equity_curve[-1] if equity_curve else initial_capital
        
    def calculate_metrics(self):
        """Calculate performance metrics
        
        Returns:
            dict: Performance metrics
        """
        # Check if we have any trades
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'average_profit': 0,
                'average_loss': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'total_return': 0
            }
            
        # Calculate win rate
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        
        # Calculate profit factor
        gross_profit = sum([t['pnl'] for t in winning_trades]) if winning_trades else 0
        losing_trades = [t for t in self.trades if t['pnl'] <= 0]
        gross_loss = abs(sum([t['pnl'] for t in losing_trades])) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate average profit and loss
        average_profit = gross_profit / len(winning_trades) if winning_trades else 0
        average_loss = gross_loss / len(losing_trades) if losing_trades else 0
        
        # Calculate max drawdown
        max_drawdown = self.calculate_max_drawdown()
        
        # Calculate Sharpe ratio
        sharpe_ratio = self.calculate_sharpe_ratio()
        
        # Calculate total return
        total_return = (self.current_equity - self.initial_capital) / self.initial_capital
        
        return {
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'average_profit': average_profit,
            'average_loss': average_loss,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_return': total_return
        }
    
    def calculate_sharpe_ratio(self):
        """Calculate Sharpe ratio
        
        Returns:
            float: Sharpe ratio
        """
        # Check if we have daily returns
        if not self.daily_returns:
            return 0
            
        # Calculate Sharpe ratio
        mean_return = np.mean(self.daily_returns)
        std_return = np.std(self.daily_returns)
        
        # Annualize
        sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
        
        return sharpe_ratio
    
    def calculate_max_drawdown(self):
        """Calculate maximum drawdown
        
        Returns:
            float: Maximum drawdown
        """
        # Check if we have equity curve
        if not self.equity_curve:
            return 0
            
        # Calculate running maximum
        running_max = np.maximum.accumulate(self.equity_curve)
        
        # Calculate drawdown
        drawdown = (running_max - self.equity_curve) / running_max
        
        # Get maximum drawdown
        max_drawdown = np.max(drawdown)
        
        return max_drawdown
    
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
    
    def __init__(self, config_file=None):
        """Initialize the backtester with configuration
        
        Args:
            config_file (str, optional): Path to configuration file. Defaults to None.
        """
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        if config_file is None:
            config_file = 'configuration_combined_strategy.yaml'
            
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize parameters
        self.symbols = self.config['general']['symbols']
        self.timeframe = self.config['general']['timeframe']
        self.initial_capital = self.config['general']['initial_capital']
        self.max_positions = self.config['general']['max_positions']
        self.max_risk_per_trade = self.config['general']['max_risk_per_trade']
        self.max_portfolio_pct = self.config['general']['max_portfolio_pct']
        self.min_signal_score = self.config['general'].get('min_signal_score', 0.5)
        
        # Initialize strategy
        self.strategy = CombinedStrategy(self.config)
        
        # Initialize Alpaca API
        self.initialize_alpaca_api()
        
        # Initialize data cache
        self.data = {}
        
        # Initialize backtest state
        self.current_equity = self.initial_capital
        self.current_positions = {}
        self.trades = []
        self.signals_generated = []
        self.equity_curve = []
        self.daily_returns = []
    
    def initialize_alpaca_api(self):
        """Initialize Alpaca API client"""
        try:
            # Load credentials
            with open('alpaca_credentials.json', 'r') as f:
                credentials = json.load(f)
            
            # Use paper trading credentials
            paper_credentials = credentials['paper']
            api_key = paper_credentials['api_key']
            api_secret = paper_credentials['api_secret']
            base_url = paper_credentials.get('base_url', 'https://paper-api.alpaca.markets')
            
            # Initialize REST API client
            self.api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
            
            self.logger.info("Alpaca API initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing Alpaca API: {e}")
            raise
    
    def load_config(self, config_file):
        """Load configuration from YAML file
        
        Args:
            config_file (str): Path to configuration file
            
        Returns:
            dict: Configuration dictionary
        """
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def convert_timeframe_string(self, timeframe_str):
        """Convert timeframe string to Alpaca TimeFrame object
        
        Args:
            timeframe_str (str): Timeframe string (e.g., '1D', '1H', '15Min')
            
        Returns:
            TimeFrame: Alpaca TimeFrame object
        """
        if timeframe_str == '1D':
            return TimeFrame.Day
        elif timeframe_str == '1H':
            return TimeFrame.Hour
        elif timeframe_str == '15Min':
            return TimeFrame.Minute(15)
        elif timeframe_str == '5Min':
            return TimeFrame.Minute(5)
        elif timeframe_str == '1Min':
            return TimeFrame.Minute
        else:
            logging.warning(f"Unknown timeframe: {timeframe_str}, defaulting to 1D")
            return TimeFrame.Day
    
    def parse_date(self, date_str):
        """Parse date string to datetime object
        
        Args:
            date_str (str): Date string in format YYYY-MM-DD
            
        Returns:
            datetime: Parsed datetime object
        """
        if isinstance(date_str, str):
            return pd.to_datetime(date_str)
        return date_str
    
    def fetch_data(self, symbol, start_date, end_date):
        """Fetch historical data for a symbol
        
        Args:
            symbol (str): Symbol to fetch data for
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            pd.DataFrame: Historical data
        """
        try:
            # Check if data is already cached
            if symbol in self.data:
                # Filter data for the requested date range
                df = self.data[symbol]
                df = df[(df.index >= start_date) & (df.index <= end_date)]
                return df
            
            # Fetch data from Alpaca
            self.logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
            
            # Adjust timeframe to match Alpaca API
            timeframe_map = {
                '1d': '1D',
                '1h': '1H',
                '15m': '15Min',
                '5m': '5Min',
                '1m': '1Min'
            }
            alpaca_timeframe = timeframe_map.get(self.timeframe, '1D')
            
            # Format dates properly for Alpaca API
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Fetch data
            bars = self.api.get_bars(
                symbol,
                alpaca_timeframe,
                start=start_str,
                end=end_str,
                adjustment='raw'
            ).df
            
            # Log the number of bars fetched
            self.logger.info(f"Fetched {len(bars)} bars for {symbol}")
            
            # Cache data
            self.data[symbol] = bars
            
            return bars
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def run_backtest(self, start_date, end_date):
        """Run backtest
        
        Args:
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            BacktestResults: Backtest results
        """
        # Initialize backtest state
        self.current_equity = self.initial_capital
        self.current_positions = {}
        self.trades = []
        self.signals_generated = []
        self.equity_curve = [self.current_equity]
        self.daily_returns = []
        
        # Convert string dates to datetime if needed
        if isinstance(start_date, str):
            start_date = dt.datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = dt.datetime.strptime(end_date, '%Y-%m-%d')
            
        # Log start of backtest
        self.logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Process each symbol
        for symbol in self.symbols:
            self.logger.info(f"Processing {symbol}")
            
            # Fetch data
            df = self.fetch_data(symbol, start_date, end_date)
            
            if df.empty:
                self.logger.warning(f"No data for {symbol}, skipping")
                continue
                
            # Generate signals
            signals = self.strategy.generate_signals(df, symbol)
            
            # Add signals to list of generated signals
            if signals:
                self.signals_generated.extend(signals)
                self.logger.info(f"Generated {len(signals)} signals for {symbol}")
            else:
                self.logger.warning(f"No signals generated for {symbol}")
                
        # Sort signals by date
        self.signals_generated.sort(key=lambda x: x['date'])
        
        # Process signals day by day
        current_date = start_date
        while current_date <= end_date:
            # Get signals for current date
            day_signals = [s for s in self.signals_generated if self.is_same_day(s['date'], current_date)]
            
            # Process signals
            for signal in day_signals:
                # Check if we should execute the signal
                if self.should_execute_signal(signal):
                    # Execute signal
                    self.execute_signal(signal)
            
            # Update equity curve
            self.update_equity_curve(current_date)
            
            # Move to next day
            current_date += dt.timedelta(days=1)
            
        # Close all positions at the end of the backtest
        self.close_all_positions(end_date)
        
        # Calculate metrics
        metrics = self.calculate_metrics()
        
        # Return results
        return BacktestResults(
            initial_capital=self.initial_capital,
            equity_curve=self.equity_curve,
            trades=self.trades,
            signals_generated=self.signals_generated,
            signals_executed=self.trades,
            start_date=start_date,
            end_date=end_date,
            metrics=metrics,
            daily_returns=self.daily_returns
        )
    
    def should_execute_signal(self, signal):
        """Determine if a signal should be executed
        
        Args:
            signal (dict): Signal dictionary
            
        Returns:
            bool: True if signal should be executed, False otherwise
        """
        # Check if signal has a symbol
        if 'symbol' not in signal:
            self.logger.warning(f"Signal missing 'symbol' key: {signal}")
            return False
            
        symbol = signal['symbol']
        direction = signal['direction']
        
        # Check if we already have a position in this symbol
        if symbol in self.current_positions:
            existing_position = self.current_positions[symbol]
            
            # Check if the direction is the same
            if existing_position['direction'] == direction:
                # Same direction, don't execute
                return False
            else:
                # Opposite direction, close existing position and execute new one
                self.close_position(symbol, signal['price'], signal['date'], f"Reversed position due to new {signal['direction']} signal")
                return True
        
        # Check if we have reached the maximum number of positions
        if len(self.current_positions) >= self.max_positions:
            # Too many positions, don't execute
            return False
            
        # Check if the signal is strong enough
        min_signal_score = self.config['general'].get('min_signal_score', 0.5)
        signal_score = signal.get('score', 1.0)
        
        if signal_score < min_signal_score:
            # Signal not strong enough
            return False
        
        # All checks passed, execute signal
        return True
    
    def execute_signal(self, signal):
        """Execute a trading signal
        
        Args:
            signal (dict): Signal dictionary
        """
        symbol = signal['symbol']
        direction = signal['direction']
        entry_price = signal['price']
        stop_loss = signal.get('stop_loss', entry_price * 0.95 if direction == 'LONG' else entry_price * 1.05)
        take_profit = signal.get('take_profit', entry_price * 1.05 if direction == 'LONG' else entry_price * 0.95)
        date = signal['date']
        
        # Calculate position size
        risk_amount = self.current_equity * (self.max_risk_per_trade / 100)
        
        # Calculate stop loss distance
        if direction == 'LONG':
            stop_distance = entry_price - stop_loss
        else:
            stop_distance = stop_loss - entry_price
            
        # Ensure stop distance is positive
        stop_distance = abs(stop_distance)
        
        # Calculate shares to trade
        if stop_distance > 0:
            shares = int(risk_amount / stop_distance)
        else:
            # If stop distance is zero, use a default position size
            shares = int(risk_amount / entry_price)
            
        # Ensure minimum shares
        shares = max(1, shares)
        
        # Calculate position value
        position_value = shares * entry_price
        
        # Check if position value exceeds max portfolio percentage
        max_position_value = self.current_equity * (self.max_portfolio_pct / 100)
        if position_value > max_position_value:
            # Reduce shares to meet max portfolio percentage
            shares = int(max_position_value / entry_price)
            shares = max(1, shares)
            position_value = shares * entry_price
            
        # Calculate risk per share
        risk_per_share = stop_distance
        
        # Calculate total risk
        total_risk = shares * risk_per_share
        
        # Log position sizing
        strength = signal.get('strength_value', 1.0)
        regime = signal.get('market_regime', 'mixed')
        self.logger.info(f"Position sizing: {shares} shares at ${entry_price:.2f} with risk ${total_risk:.2f} (strength: {strength:.2f}, regime: {regime})")
        
        # Log signal execution
        self.logger.info(f"Generated {direction} signal for {symbol} at {entry_price:.2f}, stop={stop_loss:.2f}, target={take_profit:.2f}")
        
        # Create position
        position = {
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'shares': shares,
            'entry_date': date,
            'current_price': entry_price,
            'max_price': entry_price,
            'min_price': entry_price,
            'strategy': signal.get('strategy', 'combined'),
            'market_regime': signal.get('market_regime', 'mixed'),
            'seasonality': signal.get('seasonality', 'neutral'),
            'cost_basis': shares * entry_price
        }
        
        # Add position to current positions
        self.current_positions[symbol] = position
        
        # Update equity
        self.current_equity -= position_value * 0.001  # Commission
    
    def update_equity_curve(self, date):
        """Update equity curve
        
        Args:
            date (datetime): Date to update equity curve for
        """
        self.equity_curve.append(self.current_equity)
        
        # Calculate daily return
        if len(self.equity_curve) > 1:
            daily_return = (self.equity_curve[-1] / self.equity_curve[-2]) - 1
            self.daily_returns.append(daily_return)
    
    def update_positions(self, date):
        """Update positions on a given date
        
        Args:
            date (datetime): Date to update positions for
        """
        for symbol, position in list(self.current_positions.items()):
            # Skip if data not available for this date
            if symbol not in self.data or date not in self.data[symbol].index:
                continue
            
            # Get current price
            current_price = self.data[symbol].loc[date, 'close']
            
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
    
    def close_position(self, symbol, price, date, reason):
        """Close an existing position
        
        Args:
            symbol (str): Symbol to close position for
            price (float): Exit price
            date (datetime): Exit date
            reason (str): Reason for closing position
        """
        # Skip if no position
        if symbol not in self.current_positions:
            return
        
        # Get position
        position = self.current_positions[symbol]
        
        # Calculate PnL
        if position['direction'] == 'LONG':
            pnl = (price - position['entry_price']) * position['shares']
        else:  # SHORT
            pnl = (position['entry_price'] - price) * position['shares']
        
        # Calculate return percentage
        return_pct = (pnl / position['cost_basis']) * 100
        
        # Update capital
        self.current_equity += position['shares'] * price
        
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
        del self.current_positions[symbol]
        
        logging.info(f"Closed {position['direction']} position in {symbol}: {position['shares']} shares at ${price:.2f}, PnL: ${pnl:.2f} ({return_pct:.2f}%)")
    
    def close_all_positions(self, date):
        """Close all positions on a given date
        
        Args:
            date (datetime): Date to close positions for
        """
        for symbol, position in list(self.current_positions.items()):
            if symbol in self.data and date in self.data[symbol].index:
                self.close_position(symbol, self.data[symbol].loc[date, 'close'], date, 'end_of_backtest')
    
    def calculate_metrics(self):
        """Calculate backtest metrics"""
        # Calculate total return
        total_return = (self.current_equity / self.initial_capital) - 1
        
        # Calculate daily returns
        daily_returns = []
        for i in range(len(self.equity_curve) - 1):
            daily_return = (self.equity_curve[i + 1] / self.equity_curve[i]) - 1
            daily_returns.append(daily_return)
        
        # Calculate Sharpe ratio
        sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        
        # Calculate max drawdown
        max_drawdown = self.calculate_max_drawdown()
        
        metrics = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
        
        return metrics
    
    def calculate_max_drawdown(self):
        """Calculate maximum drawdown"""
        # Calculate running maximum
        equity_values = self.equity_curve
        running_max = np.maximum.accumulate(equity_values)
        
        # Calculate drawdown
        drawdown = (np.array(equity_values) - running_max) / running_max
        
        # Get the maximum drawdown
        max_drawdown = abs(np.min(drawdown)) if not np.isnan(np.min(drawdown)) else 0
        
        return max_drawdown
    
    def print_backtest_summary(self, results):
        """Print a summary of the backtest results
        
        Args:
            results (BacktestResults): Backtest results
        """
        print(results.generate_report())
    
    def get_all_trades(self):
        """Get all trades from the backtest
        
        Returns:
            list: List of all trades
        """
        return self.trades
    
    def is_same_day(self, date1, date2):
        """Check if two dates are on the same day
        
        Args:
            date1 (datetime): First date
            date2 (datetime): Second date
            
        Returns:
            bool: True if dates are on the same day, False otherwise
        """
        return (date1.year == date2.year and 
                date1.month == date2.month and 
                date1.day == date2.day)

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
        result = backtester.run_backtest(backtester.start_date, backtester.end_date)
        
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
            print(f"{quarter:<10} {metrics['total_return']*100:>8.2f}% {metrics['win_rate']*100:>8.2f}% {metrics['profit_factor']:>13.2f} {metrics['max_drawdown']*100:>8.2f}% {metrics['total_trades']:>6}")
    else:
        print("No quarterly results to display")

def run_backtest():
    """Run the backtest"""
    # Run backtest with full date range
    print("===== Running full backtest =====")
    backtester = Backtester('configuration_combined_strategy.yaml')
    results = backtester.run_backtest(backtester.start_date, backtester.end_date)
    
    # Analyze individual trades
    if results:
        results.analyze_trades()
    
    # Run quarterly backtests
    print("\n===== Running quarterly backtests =====")
    run_quarterly_backtests()

if __name__ == "__main__":
    run_backtest()
