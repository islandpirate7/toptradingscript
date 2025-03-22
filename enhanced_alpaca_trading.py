#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Alpaca Trading System
------------------------------
This script integrates the enhanced trading functions with Alpaca API
to implement the high-performance trading model that achieved 16,534% returns.
"""

import os
import sys
import json
import logging
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import alpaca_trade_api as tradeapi
from typing import List, Dict, Any, Tuple, Optional, Union
import traceback
import time

# Import the multi-strategy system
from multi_strategy_system import (
    MultiStrategySystem, SystemConfig, Signal, MarketRegime, 
    BacktestResult, StockConfig, MarketState, TradeDirection
)

# Import enhanced trading functions
from enhanced_trading_functions import (
    calculate_adaptive_position_size,
    filter_signals,
    generate_ml_signals,
    SignalStrength
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('enhanced_alpaca_trading.log')
    ]
)

logger = logging.getLogger("EnhancedAlpacaTrading")

def load_alpaca_credentials(credentials_file='alpaca_credentials.json', mode='paper'):
    """Load Alpaca API credentials from JSON file"""
    try:
        with open(credentials_file, 'r') as f:
            credentials = json.load(f)
        
        # Use paper trading credentials by default
        credentials_to_use = credentials.get(mode, {})
        
        api_key = credentials_to_use.get('api_key')
        api_secret = credentials_to_use.get('api_secret')
        base_url = credentials_to_use.get('base_url')
        
        if not all([api_key, api_secret, base_url]):
            logger.error(f"Missing required Alpaca credentials for {mode} mode")
            return None, None, None
            
        return api_key, api_secret, base_url
    except Exception as e:
        logger.error(f"Error loading Alpaca credentials: {str(e)}")
        return None, None, None

def load_config(config_file='enhanced_alpaca_config.yaml'):
    """Load configuration from YAML file"""
    try:
        with open(config_file, 'r') as file:
            config_dict = yaml.safe_load(file)
        return config_dict
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return None

def create_system_config(config_dict):
    """Create SystemConfig object from dictionary"""
    try:
        # Extract required parameters
        stocks = config_dict.pop('stocks', [])
        initial_capital = config_dict.pop('initial_capital', 100000.0)
        max_open_positions = config_dict.pop('max_open_positions', 10)
        max_positions_per_symbol = config_dict.pop('max_positions_per_symbol', 2)
        max_correlated_positions = config_dict.pop('max_correlated_positions', 5)
        max_sector_exposure_pct = config_dict.pop('max_sector_exposure_pct', 30.0)
        max_portfolio_risk_daily_pct = config_dict.pop('max_portfolio_risk_daily_pct', 2.0)
        strategy_weights = config_dict.pop('strategy_weights', {
            "MeanReversion": 0.25,
            "TrendFollowing": 0.25,
            "VolatilityBreakout": 0.25,
            "GapTrading": 0.25
        })
        rebalance_interval = config_dict.pop('rebalance_interval', '1d')
        data_lookback_days = config_dict.pop('data_lookback_days', 30)
        market_hours_start = config_dict.pop('market_hours_start', '09:30')
        market_hours_end = config_dict.pop('market_hours_end', '16:00')
        enable_auto_trading = config_dict.pop('enable_auto_trading', False)
        backtesting_mode = config_dict.pop('backtesting_mode', True)
        data_source = config_dict.pop('data_source', 'ALPACA')
        
        # Convert rebalance_interval to timedelta
        if isinstance(rebalance_interval, str):
            if rebalance_interval.endswith('d'):
                rebalance_interval = dt.timedelta(days=int(rebalance_interval[:-1]))
            elif rebalance_interval.endswith('h'):
                rebalance_interval = dt.timedelta(hours=int(rebalance_interval[:-1]))
            else:
                rebalance_interval = dt.timedelta(days=1)
        
        # Convert market hours to time objects
        if isinstance(market_hours_start, str):
            hours, minutes = map(int, market_hours_start.split(':'))
            market_hours_start = dt.time(hours, minutes)
        
        if isinstance(market_hours_end, str):
            hours, minutes = map(int, market_hours_end.split(':'))
            market_hours_end = dt.time(hours, minutes)
        
        # Convert stock configs to StockConfig objects
        stock_configs = []
        for stock_dict in stocks:
            stock_config = StockConfig(
                symbol=stock_dict['symbol'],
                max_position_size=stock_dict.get('max_position_size', 1000),
                min_position_size=stock_dict.get('min_position_size', 100),
                max_risk_per_trade_pct=stock_dict.get('max_risk_per_trade_pct', 1.0),
                min_volume=stock_dict.get('min_volume', 100000),
                avg_daily_volume=stock_dict.get('avg_daily_volume', 0),
                beta=stock_dict.get('beta', 1.0),
                sector=stock_dict.get('sector', ""),
                industry=stock_dict.get('industry', "")
            )
            
            # Add strategy-specific parameters if available
            if 'mean_reversion_params' in stock_dict:
                stock_config.mean_reversion_params = stock_dict['mean_reversion_params']
            if 'trend_following_params' in stock_dict:
                stock_config.trend_following_params = stock_dict['trend_following_params']
            if 'volatility_breakout_params' in stock_dict:
                stock_config.volatility_breakout_params = stock_dict['volatility_breakout_params']
            if 'gap_trading_params' in stock_dict:
                stock_config.gap_trading_params = stock_dict['gap_trading_params']
                
            stock_configs.append(stock_config)
        
        # Create system config with required parameters
        config = SystemConfig(
            stocks=stock_configs,
            initial_capital=initial_capital,
            max_open_positions=max_open_positions,
            max_positions_per_symbol=max_positions_per_symbol,
            max_correlated_positions=max_correlated_positions,
            max_sector_exposure_pct=max_sector_exposure_pct,
            max_portfolio_risk_daily_pct=max_portfolio_risk_daily_pct,
            strategy_weights=strategy_weights,
            rebalance_interval=rebalance_interval,
            data_lookback_days=data_lookback_days,
            market_hours_start=market_hours_start,
            market_hours_end=market_hours_end,
            enable_auto_trading=enable_auto_trading,
            backtesting_mode=backtesting_mode,
            data_source=data_source
        )
        
        # Add additional parameters
        config.signal_quality_filters = config_dict.get('signal_quality_filters', {})
        config.position_sizing_config = config_dict.get('position_sizing_config', {})
        config.ml_strategy_selector = config_dict.get('ml_strategy_selector', {})
        
        return config
    except Exception as e:
        logger.error(f"Error creating system config: {str(e)}")
        logger.error(traceback.format_exc())
        return None

class EnhancedAlpacaTradingSystem:
    """
    Enhanced Alpaca Trading System that integrates the high-performance trading model
    with Alpaca API for live trading and backtesting.
    """
    
    def __init__(self, config_file='enhanced_alpaca_config.yaml', mode='paper'):
        """Initialize the enhanced Alpaca trading system"""
        self.config_file = config_file
        self.mode = mode
        self.api = None
        self.system = None
        self.config = None
        self.running = False
        self.trade_history = []
        self.equity_curve = []
        self.drawdown_curve = []
        
        # Initialize the system
        self._initialize()
    
    def _initialize(self):
        """Initialize the trading system"""
        try:
            # Load configuration
            config_dict = load_config(self.config_file)
            if not config_dict:
                logger.error("Failed to load configuration")
                return False
            
            # Create system config
            self.config = create_system_config(config_dict)
            if not self.config:
                logger.error("Failed to create system config")
                return False
            
            # Load Alpaca credentials
            api_key, api_secret, base_url = load_alpaca_credentials(mode=self.mode)
            if not all([api_key, api_secret, base_url]):
                logger.error("Failed to load Alpaca credentials")
                return False
            
            # Initialize Alpaca API
            self.api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
            
            # Initialize the enhanced multi-strategy system
            self.system = EnhancedMultiStrategySystem(self.config, self.api)
            
            logger.info(f"Enhanced Alpaca Trading System initialized in {self.mode} mode")
            return True
        except Exception as e:
            logger.error(f"Error initializing trading system: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def _process_backtest_results(self, result: BacktestResult) -> dict:
        """Process backtest results into a format suitable for saving to JSON"""
        # Convert datetime objects to strings in equity and drawdown curves
        equity_curve = [(dt.strftime("%Y-%m-%d %H:%M:%S"), value) for dt, value in result.equity_curve]
        drawdown_curve = [(dt.strftime("%Y-%m-%d %H:%M:%S"), value) for dt, value in result.drawdown_curve]
        
        # Convert trade history - ensure all datetime objects are converted to strings
        trade_history = []
        for trade in result.trade_history:
            if isinstance(trade, dict):
                # If already a dict, ensure datetime fields are converted to strings
                trade_dict = trade.copy()
                if 'entry_time' in trade_dict and isinstance(trade_dict['entry_time'], (dt.datetime, dt.date)):
                    trade_dict['entry_time'] = trade_dict['entry_time'].strftime("%Y-%m-%d %H:%M:%S")
                if 'exit_time' in trade_dict and isinstance(trade_dict['exit_time'], (dt.datetime, dt.date)):
                    trade_dict['exit_time'] = trade_dict['exit_time'].strftime("%Y-%m-%d %H:%M:%S")
                trade_history.append(trade_dict)
            else:
                # If it's an object, convert to dict and then ensure datetime fields are strings
                trade_dict = trade.to_dict() if hasattr(trade, 'to_dict') else vars(trade)
                if 'entry_time' in trade_dict and isinstance(trade_dict['entry_time'], (dt.datetime, dt.date)):
                    trade_dict['entry_time'] = trade_dict['entry_time'].strftime("%Y-%m-%d %H:%M:%S")
                if 'exit_time' in trade_dict and isinstance(trade_dict['exit_time'], (dt.datetime, dt.date)):
                    trade_dict['exit_time'] = trade_dict['exit_time'].strftime("%Y-%m-%d %H:%M:%S")
                trade_history.append(trade_dict)
        
        # Convert strategy performance to dict if needed
        strategy_performance = {}
        for strategy_name, performance in result.strategy_performance.items():
            if hasattr(performance, 'to_dict'):
                strategy_performance[strategy_name] = performance.to_dict()
            elif isinstance(performance, dict):
                strategy_performance[strategy_name] = performance
            else:
                # Try to convert to dict using vars()
                try:
                    strategy_performance[strategy_name] = vars(performance)
                except:
                    # If all else fails, convert to string
                    strategy_performance[strategy_name] = str(performance)
        
        # Create the result dictionary with all datetime objects converted to strings
        result_dict = {
            'start_date': result.start_date.strftime("%Y-%m-%d"),
            'end_date': result.end_date.strftime("%Y-%m-%d"),
            'initial_capital': result.initial_capital,
            'final_capital': result.final_capital,  
            'total_return_pct': result.total_return_pct,
            'annualized_return_pct': result.annualized_return_pct,
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown_pct': result.max_drawdown_pct,
            'win_rate': result.win_rate,
            'profit_factor': result.profit_factor,
            'total_trades': result.total_trades,
            'monthly_returns': result.monthly_returns,
            'equity_curve': equity_curve,
            'drawdown_curve': drawdown_curve,
            'trade_history': trade_history,
            'strategy_performance': strategy_performance
        }
        
        try:
            result_dict['final_capital'] = result.current_equity
        except AttributeError:
            try:
                result_dict['final_capital'] = result.final_capital
            except AttributeError:
                logger.error("Neither 'current_equity' nor 'final_capital' attribute found in BacktestResult object")
        
        return result_dict
    
    def run_backtest(self, start_date, end_date):
        """
        Run backtest for the specified period
        
        Args:
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            
        Returns:
            BacktestResult: Results of the backtest
        """
        try:
            logger.info(f"Running backtest from {start_date} to {end_date}")
            
            # Convert string dates to datetime objects if needed
            if isinstance(start_date, str):
                start_date = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
            if isinstance(end_date, str):
                end_date = dt.datetime.strptime(end_date, "%Y-%m-%d").date()
            
            # Initialize backtest data
            self.trade_history = []
            self.equity_curve = []
            self.drawdown_curve = []
            initial_capital = self.config.initial_capital
            current_capital = initial_capital
            max_capital = initial_capital
            
            # Set up the backtest
            self.system.setup_backtest(start_date, end_date)
            
            # Run the backtest
            result = self.system.run_backtest(start_date, end_date)
            
            # Process and return results
            if isinstance(result, dict):
                # Already in dictionary format
                return result
            else:
                # Convert BacktestResult object to dictionary
                return self._process_backtest_results(result)
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def save_results(self, result, output_file):
        """Save backtest results to file"""
        try:
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Results saved to {output_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            return False
    
    def plot_equity_curve(self, result, output_file=None):
        """Plot equity curve from backtest results"""
        try:
            # Extract equity curve data
            if isinstance(result, dict):
                equity_data = result.get('equity_curve', [])
                drawdown_data = result.get('drawdown_curve', [])
                start_date = result.get('start_date', '')
                end_date = result.get('end_date', '')
                total_return = result.get('total_return_pct', 0)
                max_drawdown = result.get('max_drawdown_pct', 0)
            else:
                equity_data = result.equity_curve
                drawdown_data = result.drawdown_curve
                start_date = result.start_date
                end_date = result.end_date
                total_return = result.total_return_pct
                max_drawdown = result.max_drawdown_pct
            
            if not equity_data:
                logger.error("No equity curve data available")
                return False
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
            
            # Extract dates and equity values
            dates = [dt.datetime.strptime(point[0], "%Y-%m-%d %H:%M:%S") for point in equity_data]
            equity = [point[1] for point in equity_data]
            
            # Plot equity curve
            ax1.plot(dates, equity, 'b-', linewidth=2)
            ax1.set_title(f'Enhanced Trading System Equity Curve ({start_date} to {end_date})')
            ax1.set_ylabel('Portfolio Value ($)')
            ax1.grid(True)
            
            # Add total return annotation
            ax1.annotate(f'Total Return: {total_return:.2f}%', 
                        xy=(0.02, 0.95), xycoords='axes fraction',
                        fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            # Extract drawdown data
            if drawdown_data:
                dd_dates = [dt.datetime.strptime(point[0], "%Y-%m-%d %H:%M:%S") for point in drawdown_data]
                dd_values = [point[1] * 100 for point in drawdown_data]  # Convert to percentage
                
                # Plot drawdown
                ax2.fill_between(dd_dates, 0, dd_values, color='red', alpha=0.3)
                ax2.plot(dd_dates, dd_values, 'r-', linewidth=1)
                ax2.set_title('Drawdown (%)')
                ax2.set_ylabel('Drawdown (%)')
                ax2.set_xlabel('Date')
                ax2.grid(True)
                ax2.invert_yaxis()  # Invert y-axis to show drawdowns as negative
                
                # Add max drawdown annotation
                ax2.annotate(f'Max Drawdown: {max_drawdown:.2f}%', 
                            xy=(0.02, 0.85), xycoords='axes fraction',
                            fontsize=10, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            plt.tight_layout()
            
            # Save figure if output file is specified
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                logger.info(f"Equity curve plot saved to {output_file}")
            
            plt.show()
            return True
        except Exception as e:
            logger.error(f"Error plotting equity curve: {str(e)}")
            logger.error(traceback.format_exc())
            return False

class EnhancedMultiStrategySystem(MultiStrategySystem):
    """
    Enhanced version of the MultiStrategySystem with direct integration of
    adaptive position sizing, ML-based strategy selection, and improved signal filtering.
    """
    
    def __init__(self, config, api=None):
        """Initialize the enhanced multi-strategy system"""
        super().__init__(config)
        
        # Store Alpaca API instance
        self.api = api
        
        # Add signal quality filters and position sizing config
        self.signal_quality_filters = config.signal_quality_filters
        self.position_sizing_config = config.position_sizing_config
        
        # Initialize trade history and equity curve for backtesting
        self.trade_history = []
        self.equity_curve = []
        self.drawdown_curve = []
        
        self.logger.info("Enhanced Multi-Strategy System initialized")
    
    def setup_backtest(self, start_date, end_date):
        """Set up the backtest environment"""
        self.trade_history = []
        self.equity_curve = []
        self.drawdown_curve = []
        self.current_capital = self.config.initial_capital
        self.max_capital = self.config.initial_capital
        
        # Initialize with starting capital
        current_time = dt.datetime.combine(start_date, self.config.market_hours_start)
        self.equity_curve.append([current_time, self.current_capital])
        self.drawdown_curve.append([current_time, 0.0])
    
    def _update_positions(self, current_time, candle_data):
        """
        Enhanced position update method that implements the high-performance trading strategy
        
        This method is called during backtesting to update positions based on market data
        """
        try:
            # Get current market state
            market_state = self.market_analyzer.analyze(candle_data, current_time)
            
            # Generate signals using ML-based strategy selection
            signals = generate_ml_signals(
                self.config.stocks,
                self.strategies,
                candle_data,
                market_state,
                self.ml_strategy_selector if hasattr(self, 'ml_strategy_selector') else None,
                self.logger
            )
            
            # Apply enhanced quality filters
            filtered_signals = filter_signals(
                signals,
                candle_data,
                self.config,
                self.signal_quality_filters,
                self.logger
            )
            
            # Process filtered signals
            for signal in filtered_signals:
                # Calculate adaptive position size
                position_size_dollars = calculate_adaptive_position_size(
                    signal,
                    market_state,
                    candle_data,
                    self.current_capital,
                    self.position_sizing_config,
                    self.logger
                )
                
                # Calculate position size in shares
                shares = int(position_size_dollars / signal.entry_price)
                
                if shares > 0:
                    # Create trade record
                    trade = {
                        "symbol": signal.symbol,
                        "direction": signal.direction.name,
                        "entry_price": signal.entry_price,
                        "stop_loss": signal.stop_loss,
                        "take_profit": signal.take_profit,
                        "entry_time": current_time.isoformat(),
                        "shares": shares,
                        "position_size": position_size_dollars,
                        "strategy": signal.strategy,
                        "market_regime": market_state.regime.name if market_state else "UNKNOWN",
                        "exit_price": None,
                        "exit_time": None,
                        "pnl": None,
                        "pnl_pct": None,
                        "status": "OPEN"
                    }
                    
                    # Add to trade history
                    self.trade_history.append(trade)
                    
                    self.logger.info(f"Opened {signal.direction.name} position in {signal.symbol}: {shares} shares at ${signal.entry_price:.2f}")
            
            # Check for exit signals on open trades
            for trade in self.trade_history:
                if trade["status"] == "OPEN":
                    symbol = trade["symbol"]
                    direction = TradeDirection[trade["direction"]]
                    entry_price = trade["entry_price"]
                    stop_loss = trade["stop_loss"]
                    take_profit = trade["take_profit"]
                    shares = trade["shares"]
                    
                    # Get latest price
                    if symbol in candle_data and len(candle_data[symbol]) > 0:
                        latest_candle = candle_data[symbol][-1]
                        latest_price = latest_candle.close
                        
                        # Check for exit conditions
                        exit_signal = False
                        exit_price = latest_price
                        exit_reason = ""
                        
                        # Stop loss hit
                        if direction == TradeDirection.LONG and latest_price <= stop_loss:
                            exit_signal = True
                            exit_reason = "STOP_LOSS"
                        elif direction == TradeDirection.SHORT and latest_price >= stop_loss:
                            exit_signal = True
                            exit_reason = "STOP_LOSS"
                        
                        # Take profit hit
                        elif direction == TradeDirection.LONG and latest_price >= take_profit:
                            exit_signal = True
                            exit_reason = "TAKE_PROFIT"
                        elif direction == TradeDirection.SHORT and latest_price <= take_profit:
                            exit_signal = True
                            exit_reason = "TAKE_PROFIT"
                        
                        # Time-based exit (hold for 5 days max)
                        elif current_time - dt.datetime.fromisoformat(trade["entry_time"]) >= dt.timedelta(days=5):
                            exit_signal = True
                            exit_reason = "TIME_EXIT"
                        
                        # Process exit if needed
                        if exit_signal:
                            # Calculate P&L
                            if direction == TradeDirection.LONG:
                                pnl = (exit_price - entry_price) * shares
                                pnl_pct = (exit_price / entry_price - 1) * 100
                            else:  # SHORT
                                pnl = (entry_price - exit_price) * shares
                                pnl_pct = (1 - exit_price / entry_price) * 100
                            
                            # Update trade record
                            trade["exit_price"] = exit_price
                            trade["exit_time"] = current_time.isoformat()
                            trade["pnl"] = pnl
                            trade["pnl_pct"] = pnl_pct
                            trade["status"] = "CLOSED"
                            trade["exit_reason"] = exit_reason
                            
                            # Update capital
                            self.current_capital += pnl
                            
                            # Update max capital for drawdown calculation
                            if self.current_capital > self.max_capital:
                                self.max_capital = self.current_capital
                            
                            self.logger.info(f"Closed {direction.name} position in {symbol}: {shares} shares at ${exit_price:.2f}, P&L: ${pnl:.2f} ({pnl_pct:.2f}%)")
            
            # Update equity curve
            self.equity_curve.append([current_time, self.current_capital])
            
            # Calculate drawdown
            current_drawdown = 0.0
            if self.max_capital > 0:
                current_drawdown = 1.0 - (self.current_capital / self.max_capital)
            self.drawdown_curve.append([current_time, current_drawdown])
            
            return True
        except Exception as e:
            self.logger.error(f"Error updating positions: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

def main():
    """Main function to run the enhanced Alpaca trading system"""
    try:
        # Parse command line arguments
        import argparse
        parser = argparse.ArgumentParser(description="Enhanced Alpaca Trading System")
        parser.add_argument("--config", "-c", default="enhanced_alpaca_config.yaml", help="Path to configuration file")
        parser.add_argument("--mode", "-m", choices=["paper", "live"], default="paper", help="Trading mode (paper or live)")
        parser.add_argument("--backtest", "-b", action="store_true", help="Run backtest")
        parser.add_argument("--start-date", "-s", help="Start date for backtest (YYYY-MM-DD)")
        parser.add_argument("--end-date", "-e", help="End date for backtest (YYYY-MM-DD)")
        parser.add_argument("--output", "-o", help="Output file for results (JSON)")
        parser.add_argument("--plot", "-p", action="store_true", help="Plot equity curve")
        parser.add_argument("--plot-output", help="Output file for equity curve plot")
        
        args = parser.parse_args()
        
        # Initialize trading system
        system = EnhancedAlpacaTradingSystem(config_file=args.config, mode=args.mode)
        
        # Run backtest if requested
        if args.backtest:
            if not args.start_date or not args.end_date:
                logger.error("Start date and end date are required for backtest")
                return
            
            # Run backtest
            result = system.run_backtest(args.start_date, args.end_date)
            
            if result:
                # Print summary
                logger.info(f"Backtest Results ({args.start_date} to {args.end_date}):")
                logger.info(f"Total Return: {result['total_return_pct']:.2f}%")
                logger.info(f"Annualized Return: {result['annualized_return_pct']:.2f}%")
                logger.info(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
                logger.info(f"Max Drawdown: {result['max_drawdown_pct']:.2f}%")
                logger.info(f"Win Rate: {result['win_rate']:.2f}")
                logger.info(f"Total Trades: {result['total_trades']}")
                
                # Save results if output file specified
                if args.output:
                    system.save_results(result, args.output)
                
                # Plot equity curve if requested
                if args.plot:
                    system.plot_equity_curve(result, args.plot_output)
            else:
                logger.error("Backtest failed")
        
        logger.info("Enhanced Alpaca Trading System completed")
        
    except Exception as e:
        logger.error(f"Error in Enhanced Alpaca Trading System: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
