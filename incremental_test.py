#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Incremental Testing Script for Enhanced Trading System
-----------------------------------------------------
This script tests each enhancement individually to isolate its impact.
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
from multi_strategy_system import (
    MultiStrategySystem, SystemConfig, Signal, MarketRegime, 
    BacktestResult, StockConfig, MarketState
)

# Import enhanced trading functions
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
        logging.FileHandler('incremental_test.log')
    ]
)

logger = logging.getLogger("IncrementalTest")

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

def fix_sector_performance_error(system):
    """Fix the 'technology' sector error by patching the _determine_sub_regime method"""
    original_method = system.market_analyzer._determine_sub_regime
    
    def patched_method(self, base_regime, adx, vix, trend_direction, 
                      breadth_indicators, intermarket_indicators,
                      sector_performance, sentiment_indicators):
        """Patched method that checks if keys exist before accessing them"""
        if base_regime == MarketRegime.CONSOLIDATION:
            # Check if the required sector keys exist before accessing them
            if 'technology' in sector_performance and 'healthcare' in sector_performance:
                if sector_performance['technology'] > 0 and sector_performance['healthcare'] > 0:
                    return "Bullish Consolidation"
                elif sector_performance['technology'] < 0 and sector_performance['healthcare'] < 0:
                    return "Bearish Consolidation"
                else:
                    return "Neutral Consolidation"
            else:
                return "Neutral Consolidation"
        else:
            # Call the original method for other cases
            return original_method(self, base_regime, adx, vix, trend_direction, 
                                  breadth_indicators, intermarket_indicators,
                                  sector_performance, sentiment_indicators)
    
    # Apply the patch
    system.market_analyzer._determine_sub_regime = patched_method.__get__(system.market_analyzer)
    logger.info("Fixed sector performance error by patching _determine_sub_regime method")
    
    return system

def test_adaptive_position_sizing(system, start_date, end_date):
    """Test only the adaptive position sizing enhancement"""
    logger.info(f"Testing adaptive position sizing from {start_date} to {end_date}")
    
    try:
        # Store original method
        original_calculate_position_size = system._calculate_position_size
        
        # Override method with enhanced version
        def enhanced_calculate_position_size(self, signal):
            """Enhanced adaptive position sizing"""
            try:
                return calculate_adaptive_position_size(
                    signal=signal,
                    market_state=self.market_state,
                    candle_data=self.candle_data,
                    current_equity=self.current_equity,
                    position_sizing_config=self.position_sizing_config,
                    logger=self.logger
                )
            except Exception as e:
                self.logger.error(f"Error in enhanced position sizing: {str(e)}")
                # Fall back to original method
                return original_calculate_position_size(self, signal)
        
        # Apply the enhanced method
        system._calculate_position_size = enhanced_calculate_position_size.__get__(system)
        
        # Run backtest with enhanced position sizing
        result = system.run_backtest(start_date, end_date)
        logger.info(f"Adaptive position sizing test completed with {result.total_trades} trades")
        logger.info(f"Total return: {result.total_return_pct:.2f}%, Sharpe: {result.sharpe_ratio:.2f}")
        
        # Restore original method
        system._calculate_position_size = original_calculate_position_size
        
        return result
    except Exception as e:
        logger.error(f"Error in adaptive position sizing test: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def test_ml_strategy_selection(system, start_date, end_date):
    """Test only the ML-based strategy selection enhancement"""
    logger.info(f"Testing ML-based strategy selection from {start_date} to {end_date}")
    
    try:
        # Store original method
        original_generate_signals = system._generate_signals
        
        # Override method with enhanced version
        def enhanced_generate_signals(self):
            """Enhanced signal generation using ML-based strategy selection"""
            try:
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
                
                # Apply original quality filters
                filtered_signals = self._filter_signals(all_signals)
                
                # Add filtered signals to the system
                self.signals.extend(filtered_signals)
                
                # Log signal generation summary
                self.logger.info(f"Generated {len(all_signals)} signals, {len(filtered_signals)} passed quality filters")
            except Exception as e:
                self.logger.error(f"Error in ML-based strategy selection: {str(e)}")
                # Fall back to original method
                original_generate_signals(self)
        
        # Apply the enhanced method
        system._generate_signals = enhanced_generate_signals.__get__(system)
        
        # Run backtest with ML-based strategy selection
        result = system.run_backtest(start_date, end_date)
        logger.info(f"ML-based strategy selection test completed with {result.total_trades} trades")
        logger.info(f"Total return: {result.total_return_pct:.2f}%, Sharpe: {result.sharpe_ratio:.2f}")
        
        # Restore original method
        system._generate_signals = original_generate_signals
        
        return result
    except Exception as e:
        logger.error(f"Error in ML-based strategy selection test: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def test_signal_filtering(system, start_date, end_date):
    """Test only the enhanced signal filtering"""
    logger.info(f"Testing enhanced signal filtering from {start_date} to {end_date}")
    
    try:
        # Store original method
        original_filter_signals = system._filter_signals
        
        # Override method with enhanced version
        def enhanced_filter_signals(self, signals):
            """Enhanced signal filtering"""
            try:
                return filter_signals(
                    signals=signals,
                    candle_data=self.candle_data,
                    config=self.config,
                    signal_quality_filters=self.signal_quality_filters,
                    logger=self.logger
                )
            except Exception as e:
                self.logger.error(f"Error in enhanced signal filtering: {str(e)}")
                # Fall back to original method
                return original_filter_signals(self, signals)
        
        # Apply the enhanced method
        system._filter_signals = enhanced_filter_signals.__get__(system)
        
        # Run backtest with enhanced signal filtering
        result = system.run_backtest(start_date, end_date)
        logger.info(f"Enhanced signal filtering test completed with {result.total_trades} trades")
        logger.info(f"Total return: {result.total_return_pct:.2f}%, Sharpe: {result.sharpe_ratio:.2f}")
        
        # Restore original method
        system._filter_signals = original_filter_signals
        
        return result
    except Exception as e:
        logger.error(f"Error in enhanced signal filtering test: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def test_all_enhancements(system, start_date, end_date):
    """Test all enhancements together"""
    logger.info(f"Testing all enhancements together from {start_date} to {end_date}")
    
    try:
        # Store original methods
        original_generate_signals = system._generate_signals
        original_calculate_position_size = system._calculate_position_size
        original_filter_signals = system._filter_signals
        
        # Override methods with enhanced versions
        def enhanced_generate_signals(self):
            """Enhanced signal generation using ML-based strategy selection"""
            try:
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
                
                # Apply enhanced quality filters
                filtered_signals = self._filter_signals(all_signals)
                
                # Add filtered signals to the system
                self.signals.extend(filtered_signals)
                
                # Log signal generation summary
                self.logger.info(f"Generated {len(all_signals)} signals, {len(filtered_signals)} passed quality filters")
            except Exception as e:
                self.logger.error(f"Error in ML-based strategy selection: {str(e)}")
                # Fall back to original method
                original_generate_signals(self)
        
        def enhanced_calculate_position_size(self, signal):
            """Enhanced adaptive position sizing"""
            try:
                return calculate_adaptive_position_size(
                    signal=signal,
                    market_state=self.market_state,
                    candle_data=self.candle_data,
                    current_equity=self.current_equity,
                    position_sizing_config=self.position_sizing_config,
                    logger=self.logger
                )
            except Exception as e:
                self.logger.error(f"Error in enhanced position sizing: {str(e)}")
                # Fall back to original method
                return original_calculate_position_size(self, signal)
        
        def enhanced_filter_signals(self, signals):
            """Enhanced signal filtering"""
            try:
                return filter_signals(
                    signals=signals,
                    candle_data=self.candle_data,
                    config=self.config,
                    signal_quality_filters=self.signal_quality_filters,
                    logger=self.logger
                )
            except Exception as e:
                self.logger.error(f"Error in enhanced signal filtering: {str(e)}")
                # Fall back to original method
                return original_filter_signals(self, signals)
        
        # Apply the enhanced methods
        system._generate_signals = enhanced_generate_signals.__get__(system)
        system._calculate_position_size = enhanced_calculate_position_size.__get__(system)
        system._filter_signals = enhanced_filter_signals.__get__(system)
        
        # Run backtest with all enhancements
        result = system.run_backtest(start_date, end_date)
        logger.info(f"All enhancements test completed with {result.total_trades} trades")
        logger.info(f"Total return: {result.total_return_pct:.2f}%, Sharpe: {result.sharpe_ratio:.2f}")
        
        # Restore original methods
        system._generate_signals = original_generate_signals
        system._calculate_position_size = original_calculate_position_size
        system._filter_signals = original_filter_signals
        
        return result
    except Exception as e:
        logger.error(f"Error in all enhancements test: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def compare_results(baseline_result, *enhanced_results, labels=None):
    """Compare baseline result with multiple enhanced results"""
    if not baseline_result or not all(enhanced_results):
        logger.error("Cannot compare results: One or more results are missing")
        return
    
    try:
        if not labels:
            labels = [f"Enhanced {i+1}" for i in range(len(enhanced_results))]
        
        # Create comparison dictionary
        comparison = {
            "total_return_pct": {
                "baseline": baseline_result.total_return_pct
            },
            "annualized_return_pct": {
                "baseline": baseline_result.annualized_return_pct
            },
            "sharpe_ratio": {
                "baseline": baseline_result.sharpe_ratio
            },
            "max_drawdown_pct": {
                "baseline": baseline_result.max_drawdown_pct
            },
            "win_rate": {
                "baseline": baseline_result.win_rate
            },
            "profit_factor": {
                "baseline": baseline_result.profit_factor
            },
            "total_trades": {
                "baseline": baseline_result.total_trades
            }
        }
        
        # Add enhanced results
        for i, result in enumerate(enhanced_results):
            label = labels[i]
            
            comparison["total_return_pct"][label] = result.total_return_pct
            comparison["total_return_pct"][f"{label}_diff"] = result.total_return_pct - baseline_result.total_return_pct
            comparison["total_return_pct"][f"{label}_pct_improvement"] = ((result.total_return_pct - baseline_result.total_return_pct) / 
                                                                        abs(baseline_result.total_return_pct) * 100) if baseline_result.total_return_pct != 0 else 0
            
            comparison["annualized_return_pct"][label] = result.annualized_return_pct
            comparison["annualized_return_pct"][f"{label}_diff"] = result.annualized_return_pct - baseline_result.annualized_return_pct
            comparison["annualized_return_pct"][f"{label}_pct_improvement"] = ((result.annualized_return_pct - baseline_result.annualized_return_pct) / 
                                                                             abs(baseline_result.annualized_return_pct) * 100) if baseline_result.annualized_return_pct != 0 else 0
            
            comparison["sharpe_ratio"][label] = result.sharpe_ratio
            comparison["sharpe_ratio"][f"{label}_diff"] = result.sharpe_ratio - baseline_result.sharpe_ratio
            comparison["sharpe_ratio"][f"{label}_pct_improvement"] = ((result.sharpe_ratio - baseline_result.sharpe_ratio) / 
                                                                    abs(baseline_result.sharpe_ratio) * 100) if baseline_result.sharpe_ratio != 0 else 0
            
            comparison["max_drawdown_pct"][label] = result.max_drawdown_pct
            comparison["max_drawdown_pct"][f"{label}_diff"] = result.max_drawdown_pct - baseline_result.max_drawdown_pct
            comparison["max_drawdown_pct"][f"{label}_pct_improvement"] = ((baseline_result.max_drawdown_pct - result.max_drawdown_pct) / 
                                                                        abs(baseline_result.max_drawdown_pct) * 100) if baseline_result.max_drawdown_pct != 0 else 0
            
            comparison["win_rate"][label] = result.win_rate
            comparison["win_rate"][f"{label}_diff"] = result.win_rate - baseline_result.win_rate
            comparison["win_rate"][f"{label}_pct_improvement"] = ((result.win_rate - baseline_result.win_rate) / 
                                                                abs(baseline_result.win_rate) * 100) if baseline_result.win_rate != 0 else 0
            
            comparison["profit_factor"][label] = result.profit_factor
            comparison["profit_factor"][f"{label}_diff"] = result.profit_factor - baseline_result.profit_factor
            comparison["profit_factor"][f"{label}_pct_improvement"] = ((result.profit_factor - baseline_result.profit_factor) / 
                                                                     abs(baseline_result.profit_factor) * 100) if baseline_result.profit_factor != 0 else 0
            
            comparison["total_trades"][label] = result.total_trades
            comparison["total_trades"][f"{label}_diff"] = result.total_trades - baseline_result.total_trades
            comparison["total_trades"][f"{label}_pct_change"] = ((result.total_trades - baseline_result.total_trades) / 
                                                               abs(baseline_result.total_trades) * 100) if baseline_result.total_trades != 0 else 0
        
        # Save comparison to file
        with open('incremental_test_comparison.json', 'w') as f:
            json.dump(comparison, f, indent=4)
        
        # Print comparison
        logger.info("=== Backtest Comparison ===")
        logger.info(f"Metric: Baseline -> {' / '.join(labels)}")
        logger.info(f"Total Return: {baseline_result.total_return_pct:.2f}% -> {' / '.join([f'{result.total_return_pct:.2f}%' for result in enhanced_results])}")
        logger.info(f"Annualized Return: {baseline_result.annualized_return_pct:.2f}% -> {' / '.join([f'{result.annualized_return_pct:.2f}%' for result in enhanced_results])}")
        logger.info(f"Sharpe Ratio: {baseline_result.sharpe_ratio:.2f} -> {' / '.join([f'{result.sharpe_ratio:.2f}' for result in enhanced_results])}")
        logger.info(f"Max Drawdown: {baseline_result.max_drawdown_pct:.2f}% -> {' / '.join([f'{result.max_drawdown_pct:.2f}%' for result in enhanced_results])}")
        logger.info(f"Win Rate: {baseline_result.win_rate:.2f}% -> {' / '.join([f'{result.win_rate:.2f}%' for result in enhanced_results])}")
        logger.info(f"Profit Factor: {baseline_result.profit_factor:.2f} -> {' / '.join([f'{result.profit_factor:.2f}' for result in enhanced_results])}")
        logger.info(f"Total Trades: {baseline_result.total_trades} -> {' / '.join([f'{result.total_trades}' for result in enhanced_results])}")
        
        # Plot equity curves
        plot_equity_curves(baseline_result, enhanced_results, labels)
        
        return comparison
    except Exception as e:
        logger.error(f"Error comparing results: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def plot_equity_curves(baseline_result, enhanced_results, labels):
    """Plot equity curves for comparison"""
    try:
        plt.figure(figsize=(12, 8))
        
        # Convert baseline equity curve to DataFrame
        baseline_equity = pd.DataFrame(baseline_result.equity_curve, columns=['date', 'equity'])
        baseline_equity.set_index('date', inplace=True)
        
        # Plot baseline equity curve
        plt.plot(baseline_equity.index, baseline_equity['equity'], label='Baseline', color='blue')
        
        # Plot enhanced equity curves
        colors = ['green', 'red', 'orange', 'purple']
        for i, result in enumerate(enhanced_results):
            # Convert equity curve to DataFrame
            enhanced_equity = pd.DataFrame(result.equity_curve, columns=['date', 'equity'])
            enhanced_equity.set_index('date', inplace=True)
            
            # Plot equity curve
            plt.plot(enhanced_equity.index, enhanced_equity['equity'], label=labels[i], color=colors[i % len(colors)])
        
        # Add labels and title
        plt.title('Equity Curve Comparison')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plt.savefig('incremental_test_equity_curves.png')
        logger.info("Equity curve comparison saved to 'incremental_test_equity_curves.png'")
        
        # Plot drawdown curves
        plt.figure(figsize=(12, 8))
        
        # Convert baseline drawdown curve to DataFrame
        baseline_drawdown = pd.DataFrame(baseline_result.drawdown_curve, columns=['date', 'drawdown'])
        baseline_drawdown.set_index('date', inplace=True)
        
        # Plot baseline drawdown curve
        plt.plot(baseline_drawdown.index, baseline_drawdown['drawdown'], label='Baseline', color='blue')
        
        # Plot enhanced drawdown curves
        for i, result in enumerate(enhanced_results):
            # Convert drawdown curve to DataFrame
            enhanced_drawdown = pd.DataFrame(result.drawdown_curve, columns=['date', 'drawdown'])
            enhanced_drawdown.set_index('date', inplace=True)
            
            # Plot drawdown curve
            plt.plot(enhanced_drawdown.index, enhanced_drawdown['drawdown'], label=labels[i], color=colors[i % len(colors)])
        
        # Add labels and title
        plt.title('Drawdown Comparison')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plt.savefig('incremental_test_drawdown_curves.png')
        logger.info("Drawdown comparison saved to 'incremental_test_drawdown_curves.png'")
        
    except Exception as e:
        logger.error(f"Error plotting equity curves: {str(e)}")
        logger.error(traceback.format_exc())

def main():
    """Main function to test enhanced trading system incrementally"""
    logger.info("Starting Incremental Testing of Enhanced Trading System")
    
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
        
        # Fix the 'technology' sector error
        system = fix_sector_performance_error(system)
        
        # Set backtest dates
        start_date = dt.date(2023, 1, 1)
        end_date = dt.date(2023, 12, 31)
        
        # Run baseline backtest
        logger.info(f"Running baseline backtest from {start_date} to {end_date}")
        baseline_result = system.run_backtest(start_date, end_date)
        logger.info(f"Baseline backtest completed with {baseline_result.total_trades} trades")
        logger.info(f"Total return: {baseline_result.total_return_pct:.2f}%, Sharpe: {baseline_result.sharpe_ratio:.2f}")
        
        # Test each enhancement individually
        adaptive_sizing_result = test_adaptive_position_sizing(system, start_date, end_date)
        ml_strategy_result = test_ml_strategy_selection(system, start_date, end_date)
        signal_filtering_result = test_signal_filtering(system, start_date, end_date)
        
        # Test all enhancements together
        all_enhancements_result = test_all_enhancements(system, start_date, end_date)
        
        # Compare results
        compare_results(
            baseline_result,
            adaptive_sizing_result,
            ml_strategy_result,
            signal_filtering_result,
            all_enhancements_result,
            labels=["Adaptive Sizing", "ML Strategy", "Signal Filtering", "All Enhancements"]
        )
        
        logger.info("Incremental Testing completed")
        
    except Exception as e:
        logger.error(f"Error in Incremental Testing: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
