#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test Enhanced Trading System
---------------------------
This script tests the enhanced trading system and compares results with the original system.
"""

import os
import sys
import logging
import datetime as dt
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import traceback
from typing import List, Dict, Any

# Import the multi-strategy system
from multi_strategy_system import MultiStrategySystem, SystemConfig, Signal, MarketRegime, BacktestResult, StockConfig
from enhanced_trading_functions import (
    calculate_adaptive_position_size,
    filter_signals,
    generate_ml_signals
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_enhanced_system.log')
    ]
)

logger = logging.getLogger("TestEnhancedSystem")

def load_config():
    """Load configuration from YAML file"""
    try:
        with open('multi_strategy_config.yaml', 'r') as file:
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
        data_source = config_dict.pop('data_source', 'YAHOO')
        
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

def run_original_backtest(system, start_date, end_date):
    """Run backtest with original system"""
    logger.info(f"Running original backtest from {start_date} to {end_date}")
    
    try:
        result = system.run_backtest(start_date, end_date)
        logger.info(f"Original backtest completed with {result.total_trades} trades")
        logger.info(f"Total return: {result.total_return_pct:.2f}%, Sharpe: {result.sharpe_ratio:.2f}")
        return result
    except Exception as e:
        logger.error(f"Error in original backtest: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def run_enhanced_backtest(system, start_date, end_date):
    """Run backtest with enhanced system"""
    logger.info(f"Running enhanced backtest from {start_date} to {end_date}")
    
    try:
        # Store original methods
        original_generate_signals = system._generate_signals
        original_calculate_position_size = system._calculate_position_size
        original_filter_signals = system._filter_signals
        
        # Override methods with enhanced versions
        def enhanced_generate_signals(self):
            """Enhanced signal generation using ML-based strategy selection"""
            if not self.market_state:
                self.logger.warning("Cannot generate signals: Market state not available")
                return
                
            self.logger.info(f"Generating signals for market regime: {self.market_state.regime}")
            
            # Clear previous signals
            self.signals = []
            
            # Generate signals using ML-based strategy selection
            all_signals = generate_ml_signals(
                self.config.stocks,
                self.strategies,
                self.candle_data,
                self.market_state,
                self.ml_strategy_selector,
                self.logger
            )
            
            # Apply quality filters
            filtered_signals = filter_signals(
                all_signals,
                self.candle_data,
                self.config,
                self.signal_quality_filters,
                self.logger
            )
            
            # Add filtered signals to the system
            self.signals.extend(filtered_signals)
            
            # Log signal generation summary
            self.logger.info(f"Generated {len(all_signals)} signals, {len(filtered_signals)} passed quality filters")
        
        def enhanced_calculate_position_size(self, signal):
            """Enhanced adaptive position sizing"""
            return calculate_adaptive_position_size(
                signal=signal,
                market_state=self.market_state,
                candle_data=self.candle_data,
                current_equity=self.current_equity,
                position_sizing_config=self.position_sizing_config,
                logger=self.logger
            )
        
        # Apply the enhanced methods
        system._generate_signals = enhanced_generate_signals.__get__(system)
        system._calculate_position_size = enhanced_calculate_position_size.__get__(system)
        
        # Run backtest with enhanced methods
        result = system.run_backtest(start_date, end_date)
        logger.info(f"Enhanced backtest completed with {result.total_trades} trades")
        logger.info(f"Total return: {result.total_return_pct:.2f}%, Sharpe: {result.sharpe_ratio:.2f}")
        
        # Restore original methods
        system._generate_signals = original_generate_signals
        system._calculate_position_size = original_calculate_position_size
        system._filter_signals = original_filter_signals
        
        return result
    except Exception as e:
        logger.error(f"Error in enhanced backtest: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def compare_results(original_result, enhanced_result):
    """Compare backtest results and generate report"""
    if not original_result or not enhanced_result:
        logger.error("Cannot compare results: One or both results are missing")
        return
    
    try:
        # Create comparison dictionary
        comparison = {
            "total_return_pct": {
                "original": original_result.total_return_pct,
                "enhanced": enhanced_result.total_return_pct,
                "difference": enhanced_result.total_return_pct - original_result.total_return_pct,
                "percent_improvement": ((enhanced_result.total_return_pct - original_result.total_return_pct) / 
                                       abs(original_result.total_return_pct) * 100) if original_result.total_return_pct != 0 else 0
            },
            "annualized_return_pct": {
                "original": original_result.annualized_return_pct,
                "enhanced": enhanced_result.annualized_return_pct,
                "difference": enhanced_result.annualized_return_pct - original_result.annualized_return_pct,
                "percent_improvement": ((enhanced_result.annualized_return_pct - original_result.annualized_return_pct) / 
                                       abs(original_result.annualized_return_pct) * 100) if original_result.annualized_return_pct != 0 else 0
            },
            "sharpe_ratio": {
                "original": original_result.sharpe_ratio,
                "enhanced": enhanced_result.sharpe_ratio,
                "difference": enhanced_result.sharpe_ratio - original_result.sharpe_ratio,
                "percent_improvement": ((enhanced_result.sharpe_ratio - original_result.sharpe_ratio) / 
                                       abs(original_result.sharpe_ratio) * 100) if original_result.sharpe_ratio != 0 else 0
            },
            "max_drawdown_pct": {
                "original": original_result.max_drawdown_pct,
                "enhanced": enhanced_result.max_drawdown_pct,
                "difference": enhanced_result.max_drawdown_pct - original_result.max_drawdown_pct,
                "percent_improvement": ((original_result.max_drawdown_pct - enhanced_result.max_drawdown_pct) / 
                                       abs(original_result.max_drawdown_pct) * 100) if original_result.max_drawdown_pct != 0 else 0
            },
            "win_rate": {
                "original": original_result.win_rate,
                "enhanced": enhanced_result.win_rate,
                "difference": enhanced_result.win_rate - original_result.win_rate,
                "percent_improvement": ((enhanced_result.win_rate - original_result.win_rate) / 
                                       abs(original_result.win_rate) * 100) if original_result.win_rate != 0 else 0
            },
            "profit_factor": {
                "original": original_result.profit_factor,
                "enhanced": enhanced_result.profit_factor,
                "difference": enhanced_result.profit_factor - original_result.profit_factor,
                "percent_improvement": ((enhanced_result.profit_factor - original_result.profit_factor) / 
                                       abs(original_result.profit_factor) * 100) if original_result.profit_factor != 0 else 0
            },
            "total_trades": {
                "original": original_result.total_trades,
                "enhanced": enhanced_result.total_trades,
                "difference": enhanced_result.total_trades - original_result.total_trades,
                "percent_change": ((enhanced_result.total_trades - original_result.total_trades) / 
                                  abs(original_result.total_trades) * 100) if original_result.total_trades != 0 else 0
            }
        }
        
        # Save comparison to file
        with open('backtest_comparison.json', 'w') as f:
            json.dump(comparison, f, indent=4)
        
        # Print comparison
        logger.info("=== Backtest Comparison ===")
        logger.info(f"Total Return: {original_result.total_return_pct:.2f}% -> {enhanced_result.total_return_pct:.2f}% ({comparison['total_return_pct']['percent_improvement']:.2f}% improvement)")
        logger.info(f"Annualized Return: {original_result.annualized_return_pct:.2f}% -> {enhanced_result.annualized_return_pct:.2f}% ({comparison['annualized_return_pct']['percent_improvement']:.2f}% improvement)")
        logger.info(f"Sharpe Ratio: {original_result.sharpe_ratio:.2f} -> {enhanced_result.sharpe_ratio:.2f} ({comparison['sharpe_ratio']['percent_improvement']:.2f}% improvement)")
        logger.info(f"Max Drawdown: {original_result.max_drawdown_pct:.2f}% -> {enhanced_result.max_drawdown_pct:.2f}% ({comparison['max_drawdown_pct']['percent_improvement']:.2f}% improvement)")
        logger.info(f"Win Rate: {original_result.win_rate:.2f}% -> {enhanced_result.win_rate:.2f}% ({comparison['win_rate']['percent_improvement']:.2f}% improvement)")
        logger.info(f"Profit Factor: {original_result.profit_factor:.2f} -> {enhanced_result.profit_factor:.2f} ({comparison['profit_factor']['percent_improvement']:.2f}% improvement)")
        logger.info(f"Total Trades: {original_result.total_trades} -> {enhanced_result.total_trades} ({comparison['total_trades']['percent_change']:.2f}% change)")
        
        # Plot equity curves
        plot_equity_curves(original_result, enhanced_result)
        
        return comparison
    except Exception as e:
        logger.error(f"Error comparing results: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def plot_equity_curves(original_result, enhanced_result):
    """Plot equity curves for comparison"""
    try:
        plt.figure(figsize=(12, 8))
        
        # Convert equity curves to DataFrames
        original_equity = pd.DataFrame(original_result.equity_curve, columns=['date', 'equity'])
        original_equity.set_index('date', inplace=True)
        
        enhanced_equity = pd.DataFrame(enhanced_result.equity_curve, columns=['date', 'equity'])
        enhanced_equity.set_index('date', inplace=True)
        
        # Plot equity curves
        plt.plot(original_equity.index, original_equity['equity'], label='Original System', color='blue')
        plt.plot(enhanced_equity.index, enhanced_equity['equity'], label='Enhanced System', color='green')
        
        # Add labels and title
        plt.title('Equity Curve Comparison: Original vs Enhanced System')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plt.savefig('equity_curve_comparison.png')
        logger.info("Equity curve comparison saved to 'equity_curve_comparison.png'")
        
        # Plot drawdown curves
        plt.figure(figsize=(12, 8))
        
        # Convert drawdown curves to DataFrames
        original_drawdown = pd.DataFrame(original_result.drawdown_curve, columns=['date', 'drawdown'])
        original_drawdown.set_index('date', inplace=True)
        
        enhanced_drawdown = pd.DataFrame(enhanced_result.drawdown_curve, columns=['date', 'drawdown'])
        enhanced_drawdown.set_index('date', inplace=True)
        
        # Plot drawdown curves
        plt.plot(original_drawdown.index, original_drawdown['drawdown'], label='Original System', color='red')
        plt.plot(enhanced_drawdown.index, enhanced_drawdown['drawdown'], label='Enhanced System', color='orange')
        
        # Add labels and title
        plt.title('Drawdown Comparison: Original vs Enhanced System')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plt.savefig('drawdown_comparison.png')
        logger.info("Drawdown comparison saved to 'drawdown_comparison.png'")
        
    except Exception as e:
        logger.error(f"Error plotting equity curves: {str(e)}")
        logger.error(traceback.format_exc())

def main():
    """Main function to test enhanced trading system"""
    logger.info("Starting Enhanced Trading System Test")
    
    try:
        # Load configuration
        config_dict = load_config()
        if not config_dict:
            logger.error("Failed to load configuration")
            return
        
        # Create system config
        config = create_system_config(config_dict)
        if not config:
            logger.error("Failed to create system config")
            return
        
        # Initialize the multi-strategy system
        system = MultiStrategySystem(config)
        
        # Set backtest dates
        start_date = dt.date(2023, 1, 1)
        end_date = dt.date(2023, 12, 31)
        
        # Run original backtest
        original_result = run_original_backtest(system, start_date, end_date)
        
        # Run enhanced backtest
        enhanced_result = run_enhanced_backtest(system, start_date, end_date)
        
        # Compare results
        if original_result and enhanced_result:
            compare_results(original_result, enhanced_result)
        
        logger.info("Enhanced Trading System Test completed")
        
    except Exception as e:
        logger.error(f"Error in Enhanced Trading System Test: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
