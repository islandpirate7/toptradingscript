#!/usr/bin/env python3
"""
Multi-Strategy Trading System - Main Application

This script provides a command-line interface to run the multi-strategy trading system.
"""

import argparse
import datetime as dt
import json
import sys
import logging
import signal
import time
import yaml
import os
from typing import Dict, List
from dataclasses import dataclass
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from multi_strategy_system import (
    MultiStrategySystem,
    SystemConfig,
    StockConfig,
    TradeDirection,
    MarketRegime,
    BacktestResult,
    CandleData
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("multi_strategy_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MultiStrategyApp")

class MultiStrategyApp:
    """Main application for the multi-strategy trading system"""
    
    def __init__(self):
        """Initialize the application"""
        self.system = None
        self.config = None
        self.running = False
    
    def load_config(self, config_file: str) -> bool:
        """Load configuration from a YAML file"""
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Check for debug mode and set logging level
            if config_data.get('debug_mode', False):
                logger.setLevel(logging.DEBUG)
                logging.getLogger("MultiStrategy").setLevel(logging.DEBUG)
                logging.getLogger().setLevel(logging.DEBUG)
                logger.debug("Debug mode enabled - verbose logging activated")
            
            # Parse stock configurations
            stock_configs = []
            for stock_data in config_data.get('stocks', []):
                stock_config = StockConfig(
                    symbol=stock_data['symbol'],
                    max_position_size=stock_data.get('max_position_size', 1000),
                    min_position_size=stock_data.get('min_position_size', 10),
                    max_risk_per_trade_pct=stock_data.get('max_risk_per_trade_pct', 1.0),
                    min_volume=stock_data.get('min_volume', 5000),
                    avg_daily_volume=stock_data.get('avg_daily_volume', 0),
                    beta=stock_data.get('beta', 1.0),
                    sector=stock_data.get('sector', ''),
                    industry=stock_data.get('industry', ''),
                    mean_reversion_params=stock_data.get('mean_reversion_params', {}),
                    trend_following_params=stock_data.get('trend_following_params', {}),
                    volatility_breakout_params=stock_data.get('volatility_breakout_params', {}),
                    gap_trading_params=stock_data.get('gap_trading_params', {})
                )
                stock_configs.append(stock_config)
            
            # Parse market hours
            market_open_str = config_data.get('market_hours_start', '09:30')
            market_close_str = config_data.get('market_hours_end', '16:00')
            
            market_open = dt.datetime.strptime(market_open_str, '%H:%M').time()
            market_close = dt.datetime.strptime(market_close_str, '%H:%M').time()
            
            # Parse strategy weights
            strategy_weights = config_data.get('strategy_weights', {
                "MeanReversion": 0.25,
                "TrendFollowing": 0.25,
                "VolatilityBreakout": 0.25,
                "GapTrading": 0.25
            })
            
            # Parse rebalance interval
            rebalance_str = config_data.get('rebalance_interval', '1d')
            rebalance_unit = rebalance_str[-1]
            rebalance_value = int(rebalance_str[:-1])
            
            if rebalance_unit == 'd':
                rebalance_interval = dt.timedelta(days=rebalance_value)
            elif rebalance_unit == 'h':
                rebalance_interval = dt.timedelta(hours=rebalance_value)
            else:
                rebalance_interval = dt.timedelta(days=1)
            
            # Create system configuration
            self.config = SystemConfig(
                stocks=stock_configs,
                initial_capital=config_data.get('initial_capital', 100000.0),
                max_open_positions=config_data.get('max_open_positions', 10),
                max_positions_per_symbol=config_data.get('max_positions_per_symbol', 2),
                max_correlated_positions=config_data.get('max_correlated_positions', 5),
                max_sector_exposure_pct=config_data.get('max_sector_exposure_pct', 30.0),
                max_portfolio_risk_daily_pct=config_data.get('max_portfolio_risk_daily_pct', 2.0),
                strategy_weights=strategy_weights,
                rebalance_interval=rebalance_interval,
                data_lookback_days=config_data.get('data_lookback_days', 30),
                market_hours_start=market_open,
                market_hours_end=market_close,
                enable_auto_trading=config_data.get('enable_auto_trading', False),
                backtesting_mode=config_data.get('backtesting_mode', False),
                data_source=config_data.get('data_source', 'YAHOO'),
                api_key=config_data.get('api_key'),
                api_secret=config_data.get('api_secret')
            )
            
            logger.info(f"Configuration loaded with {len(stock_configs)} stocks")
            return True
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return False
    
    def initialize_system(self) -> bool:
        """Initialize the trading system"""
        try:
            if not self.config:
                logger.error("Cannot initialize system: Configuration not loaded")
                return False
            
            self.system = MultiStrategySystem(self.config)
            logger.info("Trading system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing trading system: {str(e)}")
            return False
    
    def start_trading(self) -> bool:
        """Start the trading system"""
        try:
            if not self.system:
                logger.error("Cannot start trading: System not initialized")
                return False
            
            self.system.start()
            self.running = True
            
            # Register signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            logger.info("Trading system started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting trading system: {str(e)}")
            return False
    
    def stop_trading(self) -> bool:
        """Stop the trading system"""
        try:
            if not self.system or not self.running:
                logger.warning("Trading system is not running")
                return False
            
            self.system.stop()
            self.running = False
            
            logger.info("Trading system stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping trading system: {str(e)}")
            return False
    
    def run_backtest(self, start_date_str: str, end_date_str: str) -> BacktestResult:
        """Run a backtest of the strategy"""
        try:
            if not self.system:
                logger.error("Cannot run backtest: System not initialized")
                return None
            
            # Parse dates
            start_date = dt.datetime.strptime(start_date_str, '%Y-%m-%d').date()
            end_date = dt.datetime.strptime(end_date_str, '%Y-%m-%d').date()
            
            # Run backtest
            logger.info(f"Running backtest from {start_date} to {end_date}")
            result = self.system.run_backtest(start_date, end_date)
            
            # Print results
            self._print_backtest_results(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            return None
    
    def _print_backtest_results(self, result: BacktestResult):
        """Print backtest results to the console"""
        if not result:
            return
        
        print("\n===== BACKTEST RESULTS =====")
        print(f"Period: {result.start_date} to {result.end_date}")
        print(f"Initial Capital: ${result.initial_capital:.2f}")
        print(f"Final Capital: ${result.final_capital:.2f}")
        print(f"Total Return: {result.total_return_pct:.2f}%")
        print(f"Annualized Return: {result.annualized_return_pct:.2f}%")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"Maximum Drawdown: {result.max_drawdown_pct:.2f}%")
        print(f"Win Rate: {result.win_rate:.2%}")
        print(f"Profit Factor: {result.profit_factor:.2f}")
        print(f"Total Trades: {result.total_trades}")
        print("\nStrategy Performance:")
        for strategy, performance in result.strategy_performance.items():
            print(f"  {strategy}: Win Rate={performance.win_rate:.2%}, Profit Factor={performance.profit_factor:.2f}, Trades={performance.total_trades}")
        print("===========================\n")
    
    def save_backtest_results(self, result: BacktestResult, output_file: str) -> bool:
        """Save backtest results to a JSON file"""
        try:
            if not result:
                logger.error("Cannot save results: No backtest results available")
                return False
            
            # Convert to serializable format
            output_data = result.to_dict()
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            
            # Write to file
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            logger.info(f"Backtest results saved to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving backtest results: {str(e)}")
            return False
    
    def plot_equity_curve(self, result: BacktestResult, output_file: str = None):
        """Plot equity curve from backtest results"""
        try:
            if not result or not result.equity_curve:
                logger.error("Cannot plot equity curve: No data available")
                return False
            
            # Convert equity curve to dataframe
            df = pd.DataFrame(result.equity_curve, columns=['date', 'equity'])
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Convert drawdown curve to dataframe
            dd_df = pd.DataFrame(result.drawdown_curve, columns=['date', 'drawdown'])
            dd_df['date'] = pd.to_datetime(dd_df['date'])
            dd_df.set_index('date', inplace=True)
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
            
            # Plot equity curve
            ax1.plot(df.index, df['equity'], label='Portfolio Value')
            ax1.set_title('Equity Curve')
            ax1.set_ylabel('Portfolio Value ($)')
            ax1.grid(True)
            ax1.legend()
            
            # Plot drawdown
            ax2.fill_between(dd_df.index, 0, dd_df['drawdown'], color='red', alpha=0.3)
            ax2.set_title('Drawdown')
            ax2.set_ylabel('Drawdown (%)')
            ax2.set_xlabel('Date')
            ax2.grid(True)
            
            plt.tight_layout()
            
            # Save or show
            if output_file:
                plt.savefig(output_file)
                logger.info(f"Equity curve saved to {output_file}")
            else:
                plt.show()
            
            return True
            
        except Exception as e:
            logger.error(f"Error plotting equity curve: {str(e)}")
            return False
    
    def plot_strategy_comparison(self, result: BacktestResult, output_file: str = None):
        """Plot comparison of strategy performance"""
        try:
            if not result or not result.strategy_performance:
                logger.error("Cannot plot strategy comparison: No data available")
                return False
            
            # Extract strategy performance
            strategies = list(result.strategy_performance.keys())
            win_rates = [result.strategy_performance[s].win_rate * 100 for s in strategies]
            profit_factors = [result.strategy_performance[s].profit_factor for s in strategies]
            trades = [result.strategy_performance[s].total_trades for s in strategies]
            total_pnl = [result.strategy_performance[s].total_pnl for s in strategies]
            
            # Create figure with multiple subplots
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))
            
            # Plot win rates
            axs[0, 0].bar(strategies, win_rates)
            axs[0, 0].set_title('Win Rate (%)')
            axs[0, 0].set_ylim(0, 100)
            axs[0, 0].grid(axis='y')
            
            # Plot profit factors
            axs[0, 1].bar(strategies, profit_factors)
            axs[0, 1].set_title('Profit Factor')
            axs[0, 1].grid(axis='y')
            
            # Plot trade counts
            axs[1, 0].bar(strategies, trades)
            axs[1, 0].set_title('Number of Trades')
            axs[1, 0].grid(axis='y')
            
            # Plot total P&L
            axs[1, 1].bar(strategies, total_pnl)
            axs[1, 1].set_title('Total P&L ($)')
            axs[1, 1].grid(axis='y')
            
            plt.tight_layout()
            
            # Save or show
            if output_file:
                plt.savefig(output_file)
                logger.info(f"Strategy comparison saved to {output_file}")
            else:
                plt.show()
            
            return True
            
        except Exception as e:
            logger.error(f"Error plotting strategy comparison: {str(e)}")
            return False
    
    def _signal_handler(self, sig, frame):
        """Handle termination signals"""
        logger.info(f"Received signal {sig}, shutting down...")
        self.stop_trading()
        sys.exit(0)


def main():
    """Main entry point for the application"""
    parser = argparse.ArgumentParser(description="Multi-Strategy Trading System")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run the trading system in live mode")
    run_parser.add_argument("--config", "-c", required=True, help="Path to configuration file")
    
    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run a backtest of the strategy")
    backtest_parser.add_argument("--config", "-c", required=True, help="Path to configuration file")
    backtest_parser.add_argument("--start-date", "-s", required=True, help="Start date (YYYY-MM-DD)")
    backtest_parser.add_argument("--end-date", "-e", required=True, help="End date (YYYY-MM-DD)")
    backtest_parser.add_argument("--output", "-o", help="Output file for results (JSON)")
    backtest_parser.add_argument("--plot", "-p", action="store_true", help="Plot equity curve")
    backtest_parser.add_argument("--plot-output", help="Output file for equity curve plot")
    backtest_parser.add_argument("--strategy-plot", action="store_true", help="Plot strategy comparison")
    backtest_parser.add_argument("--strategy-plot-output", help="Output file for strategy comparison plot")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize application
    app = MultiStrategyApp()
    
    # Load configuration
    if not app.load_config(args.config):
        logger.error("Failed to load configuration, exiting")
        return
    
    # Initialize system
    if not app.initialize_system():
        logger.error("Failed to initialize system, exiting")
        return
    
    # Execute command
    if args.command == "run":
        # Run in live mode
        if not app.start_trading():
            logger.error("Failed to start trading system, exiting")
            return
        
        # Keep the main thread alive
        try:
            while app.running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Interrupted by user, shutting down...")
            app.stop_trading()
    
    elif args.command == "backtest":
        # Run backtest
        result = app.run_backtest(args.start_date, args.end_date)
        
        if result:
            # Save results if output file specified
            if args.output:
                app.save_backtest_results(result, args.output)
            
            # Plot equity curve if requested
            if args.plot:
                app.plot_equity_curve(result, args.plot_output)
            
            # Plot strategy comparison if requested
            if args.strategy_plot:
                app.plot_strategy_comparison(result, args.strategy_plot_output)


if __name__ == "__main__":
    main()