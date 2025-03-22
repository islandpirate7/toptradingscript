#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run Fixed Enhanced Backtest
---------------------------
This script runs an enhanced backtest with improved risk management controls
to address the issues that led to catastrophic losses in the original backtest.
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
import argparse
import traceback
from typing import List, Dict, Any, Tuple, Optional, Union

# Import the enhanced Alpaca trading system
from enhanced_alpaca_trading import (
    EnhancedAlpacaTradingSystem, 
    load_config, 
    create_system_config,
    load_alpaca_credentials
)

# Import the risk management fixes
from backtest_risk_fix import (
    safe_calculate_position_size,
    calculate_portfolio_exposure,
    calculate_drawdown,
    apply_risk_management,
    analyze_strategy_performance,
    should_disable_strategy
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('fixed_backtest.log')
    ]
)

logger = logging.getLogger("FixedBacktest")

class FixedEnhancedAlpacaTradingSystem(EnhancedAlpacaTradingSystem):
    """
    Enhanced Alpaca Trading System with improved risk management
    """
    
    def __init__(self, config_file='enhanced_alpaca_config.yaml', mode='paper'):
        """Initialize the fixed enhanced Alpaca trading system"""
        super().__init__(config_file, mode)
        
        # Override position sizing config with safer values
        self.config.position_sizing_config = {
            "base_risk_per_trade": 0.01,      # 1% base risk (reduced from 2%)
            "max_position_size": 0.05,        # 5% maximum (reduced from 15%)
            "min_position_size": 0.005,       # 0.5% minimum (reduced from 1%)
            "volatility_adjustment": True,
            "signal_strength_adjustment": True
        }
        
        # Add portfolio risk limits
        self.max_portfolio_exposure = 30.0    # Maximum 30% portfolio exposure
        self.max_drawdown_for_trading = 25.0  # Stop new trades at 25% drawdown
        self.disabled_strategies = set()      # Strategies to disable due to poor performance
        
        logger.info("Fixed Enhanced Alpaca Trading System initialized with improved risk controls")
    
    def run_backtest(self, start_date, end_date):
        """
        Run backtest with improved risk management for the specified period
        
        Args:
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            
        Returns:
            dict: Results of the backtest
        """
        try:
            logger.info(f"Running fixed backtest from {start_date} to {end_date}")
            
            # Convert string dates to datetime objects if needed
            if isinstance(start_date, str):
                start_date = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
            if isinstance(end_date, str):
                end_date = dt.datetime.strptime(end_date, "%Y-%m-%d").date()
            
            # Initialize backtest data
            self.trade_history = []
            self.equity_curve = []
            self.drawdown_curve = []
            self.current_capital = self.config.initial_capital
            self.max_capital = self.config.initial_capital
            
            # Set up the backtest
            self.system.setup_backtest(start_date, end_date)
            
            # Override the _update_positions method to use our fixed version
            original_update_positions = self.system._update_positions
            self.system._update_positions = self._fixed_update_positions
            
            # Run the backtest
            result = self.system.run_backtest(start_date, end_date)
            
            # Restore original method
            self.system._update_positions = original_update_positions
            
            # Process and return results
            if isinstance(result, dict):
                # Already in dictionary format
                return result
            else:
                # Convert BacktestResult object to dictionary
                return self._process_backtest_results(result)
        except Exception as e:
            logger.error(f"Error running fixed backtest: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def _fixed_update_positions(self, current_time, candle_data):
        """
        Fixed position update method with improved risk management
        
        This method overrides the original _update_positions method to add
        portfolio-level risk controls and prevent excessive leverage.
        """
        try:
            # Get current market state
            market_state = self.market_analyzer.analyze(candle_data, current_time)
            
            # Generate signals using ML-based strategy selection
            from enhanced_trading_functions import generate_ml_signals
            signals = generate_ml_signals(
                self.config.stocks,
                self.strategies,
                candle_data,
                market_state,
                self.ml_strategy_selector if hasattr(self, 'ml_strategy_selector') else None,
                self.logger
            )
            
            # Filter out signals from disabled strategies
            signals = [s for s in signals if s.strategy not in self.disabled_strategies]
            
            # Apply enhanced quality filters
            from enhanced_trading_functions import filter_signals
            filtered_signals = filter_signals(
                signals,
                candle_data,
                self.config,
                self.signal_quality_filters,
                self.logger
            )
            
            # Calculate current portfolio exposure
            current_exposure = calculate_portfolio_exposure(self.trade_history, self.current_capital)
            
            # Calculate current drawdown
            current_drawdown = calculate_drawdown(self.current_capital, self.max_capital)
            current_drawdown_pct = current_drawdown * 100
            
            logger.info(f"Current portfolio state - Equity: ${self.current_capital:.2f}, Exposure: {current_exposure:.2f}%, Drawdown: {current_drawdown_pct:.2f}%")
            
            # Apply risk management to filter signals
            risk_filtered_signals = apply_risk_management(
                filtered_signals,
                self.trade_history,
                self.current_capital,
                self.max_capital,
                self.config.max_open_positions,
                self.max_portfolio_exposure,
                self.logger
            )
            
            # Stop trading if drawdown exceeds threshold
            if current_drawdown_pct > self.max_drawdown_for_trading:
                logger.warning(f"Drawdown of {current_drawdown_pct:.2f}% exceeds threshold of {self.max_drawdown_for_trading}%, pausing new trades")
                risk_filtered_signals = []
            
            # Process filtered signals
            for signal in risk_filtered_signals:
                # Calculate safe position size
                position_size_dollars = safe_calculate_position_size(
                    signal,
                    market_state,
                    candle_data,
                    self.current_capital,
                    self.config.position_sizing_config,
                    current_exposure,
                    self.max_portfolio_exposure,
                    current_drawdown,
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
                    
                    logger.info(f"Opened {signal.direction.name} position in {signal.symbol}: {shares} shares at ${signal.entry_price:.2f}")
            
            # Check for exit signals on open trades
            for trade in self.trade_history:
                if trade["status"] == "OPEN":
                    symbol = trade["symbol"]
                    direction = trade["direction"]  # This is already a string
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
                        if direction == "LONG" and latest_price <= stop_loss:
                            exit_signal = True
                            exit_reason = "STOP_LOSS"
                        elif direction == "SHORT" and latest_price >= stop_loss:
                            exit_signal = True
                            exit_reason = "STOP_LOSS"
                        
                        # Take profit hit
                        elif direction == "LONG" and latest_price >= take_profit:
                            exit_signal = True
                            exit_reason = "TAKE_PROFIT"
                        elif direction == "SHORT" and latest_price <= take_profit:
                            exit_signal = True
                            exit_reason = "TAKE_PROFIT"
                        
                        # Time-based exit (hold for 5 days max)
                        elif current_time - dt.datetime.fromisoformat(trade["entry_time"]) >= dt.timedelta(days=5):
                            exit_signal = True
                            exit_reason = "TIME_EXIT"
                        
                        # Process exit if needed
                        if exit_signal:
                            # Calculate P&L
                            if direction == "LONG":
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
                            
                            logger.info(f"Closed {direction} position in {symbol}: {shares} shares at ${exit_price:.2f}, P&L: ${pnl:.2f} ({pnl_pct:.2f}%)")
            
            # Analyze strategy performance and disable poor performers
            if len(self.trade_history) > 20:  # Wait until we have enough trades
                strategy_performance = analyze_strategy_performance(self.trade_history)
                
                for strategy, perf in strategy_performance.items():
                    if should_disable_strategy(perf, min_trades=5, min_win_rate=40):
                        if strategy not in self.disabled_strategies:
                            logger.warning(f"Disabling poorly performing strategy: {strategy} with win rate {perf['win_rate']:.2f}% and PnL ${perf['total_pnl']:.2f}")
                            self.disabled_strategies.add(strategy)
            
            # Update equity curve
            self.equity_curve.append([current_time, self.current_capital])
            
            # Calculate drawdown
            current_drawdown = 0.0
            if self.max_capital > 0:
                current_drawdown = 1.0 - (self.current_capital / self.max_capital)
            self.drawdown_curve.append([current_time, current_drawdown])
            
            return True
        except Exception as e:
            logger.error(f"Error updating positions: {str(e)}")
            logger.error(traceback.format_exc())
            return False

def main():
    """Main function to run the fixed enhanced backtest"""
    parser = argparse.ArgumentParser(description="Fixed Enhanced Backtest")
    parser.add_argument("--config", "-c", default="enhanced_alpaca_config.yaml", help="Path to configuration file")
    parser.add_argument("--start_date", "-s", default="2023-01-01", help="Start date for backtest (YYYY-MM-DD)")
    parser.add_argument("--end_date", "-e", default="2023-06-30", help="End date for backtest (YYYY-MM-DD)")
    parser.add_argument("--output", "-o", default="fixed_backtest_results.json", help="Output file for results")
    
    args = parser.parse_args()
    
    try:
        # Initialize the fixed trading system
        trading_system = FixedEnhancedAlpacaTradingSystem(args.config)
        
        # Run the backtest
        result = trading_system.run_backtest(args.start_date, args.end_date)
        
        if result:
            # Save results to file
            trading_system.save_results(result, args.output)
            
            # Plot equity curve
            trading_system.plot_equity_curve(result, "fixed_equity_curve.png")
            
            # Print summary
            print("\nFixed Backtest Results Summary:")
            print(f"Initial Capital: ${result['initial_capital']:.2f}")
            print(f"Final Capital: ${result['final_capital']:.2f}")
            print(f"Total Return: {result['total_return_pct']:.2f}%")
            print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {result['max_drawdown_pct']:.2f}%")
            print(f"Total Trades: {result['total_trades']}")
            
            if 'win_rate' in result:
                print(f"Win Rate: {result['win_rate']:.2f}%")
            
            print(f"\nResults saved to {args.output}")
        else:
            print("Backtest failed to produce results")
    except Exception as e:
        print(f"Error running backtest: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
