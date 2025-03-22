#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trading System Optimizer
-----------------------
This script optimizes the enhanced trading system parameters to improve
Sharpe ratio, win rate, and balance trading frequency.
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
from typing import List, Dict, Any, Tuple
import copy
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed

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

# Import ML strategy selector
from ml_strategy_selector import MLStrategySelector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('system_optimizer.log')
    ]
)

logger = logging.getLogger("SystemOptimizer")

def load_config():
    """Load configuration from YAML file"""
    try:
        with open('multi_strategy_config.yaml', 'r') as file:
            config_dict = yaml.safe_load(file)
        return config_dict
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return None

def save_config(config_dict, filename='optimized_config.yaml'):
    """Save configuration to YAML file"""
    try:
        with open(filename, 'w') as file:
            yaml.dump(config_dict, file, default_flow_style=False)
        logger.info(f"Configuration saved to {filename}")
        return True
    except Exception as e:
        logger.error(f"Error saving configuration: {str(e)}")
        return False

class EnhancedMultiStrategySystem(MultiStrategySystem):
    """
    Enhanced version of the MultiStrategySystem with direct integration of
    adaptive position sizing, ML-based strategy selection, and improved signal filtering.
    """
    
    def __init__(self, config):
        """Initialize the enhanced multi-strategy system"""
        super().__init__(config)
        
        # Initialize ML strategy selector
        self.ml_strategy_selector = MLStrategySelector(
            config=config.ml_strategy_selector,
            logger=self.logger
        )
        
        # Add signal quality filters and position sizing config
        self.signal_quality_filters = config.signal_quality_filters
        self.position_sizing_config = config.position_sizing_config
        
        # Fix the sector performance error
        self._patch_market_analyzer()
        
        self.logger.info("Enhanced Multi-Strategy System initialized")
    
    def _patch_market_analyzer(self):
        """Fix the 'technology' sector error by patching the _determine_sub_regime method"""
        original_method = self.market_analyzer._determine_sub_regime
        
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
        self.market_analyzer._determine_sub_regime = patched_method.__get__(self.market_analyzer)
        self.logger.info("Fixed sector performance error by patching _determine_sub_regime method")
    
    def _generate_signals(self):
        """
        Override the signal generation method to use ML-based strategy selection
        """
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
            super()._generate_signals()
    
    def _calculate_position_size(self, signal):
        """
        Override the position sizing method to use adaptive position sizing
        """
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
            self.logger.error(f"Error in adaptive position sizing: {str(e)}")
            # Fall back to original method
            return super()._calculate_position_size(signal)
    
    def _filter_signals(self, signals):
        """
        Override the signal filtering method to use enhanced filters
        """
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
            return super()._filter_signals(signals)

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

def run_backtest(config_dict, start_date, end_date):
    """
    Run backtest with the given configuration
    
    Args:
        config_dict: Configuration dictionary
        start_date: Start date for backtest
        end_date: End date for backtest
        
    Returns:
        BacktestResult: Backtest result object
    """
    try:
        # Create system config
        config = create_system_config(config_dict)
        if not config:
            logger.error("Failed to create system config")
            return
        
        # Create enhanced system
        system = EnhancedMultiStrategySystem(config)
        
        # Run backtest
        result = system.run_backtest(start_date, end_date)
        
        return result
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def optimize_ml_strategy_selector(base_config, start_date, end_date):
    """
    Optimize ML strategy selector parameters
    
    Args:
        base_config: Base configuration dictionary
        start_date: Start date for backtest
        end_date: End date for backtest
        
    Returns:
        Dict: Optimized ML strategy selector parameters
    """
    logger.info("Optimizing ML strategy selector parameters")
    
    # Parameters to optimize
    param_grid = {
        "ml_lookback_window": [15, 30, 45],
        "ml_min_training_samples": [50, 100, 150],
        "ml_retraining_frequency": [3, 7, 14],
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15]
    }
    
    # Generate parameter combinations
    param_combinations = []
    for ml_lookback_window in param_grid["ml_lookback_window"]:
        for ml_min_training_samples in param_grid["ml_min_training_samples"]:
            for ml_retraining_frequency in param_grid["ml_retraining_frequency"]:
                for n_estimators in param_grid["n_estimators"]:
                    for max_depth in param_grid["max_depth"]:
                        params = {
                            "ml_lookback_window": ml_lookback_window,
                            "ml_min_training_samples": ml_min_training_samples,
                            "ml_retraining_frequency": ml_retraining_frequency,
                            "n_estimators": n_estimators,
                            "max_depth": max_depth
                        }
                        param_combinations.append(params)
    
    # Limit number of combinations to test
    max_combinations = 10
    if len(param_combinations) > max_combinations:
        logger.info(f"Limiting to {max_combinations} parameter combinations")
        param_combinations = param_combinations[:max_combinations]
    
    # Test each parameter combination
    best_sharpe = 0
    best_params = None
    
    for params in param_combinations:
        # Create config copy with updated parameters
        config_copy = copy.deepcopy(base_config)
        
        # Update ML strategy selector parameters
        config_copy["ml_strategy_selector"].update(params)
        
        # Run backtest
        result = run_backtest(config_copy, start_date, end_date)
        
        if result and result.sharpe_ratio > best_sharpe:
            best_sharpe = result.sharpe_ratio
            best_params = params
            
            logger.info(f"New best parameters found: {params}")
            logger.info(f"Sharpe ratio: {best_sharpe:.2f}")
    
    logger.info(f"Optimized ML strategy selector parameters: {best_params}")
    logger.info(f"Best Sharpe ratio: {best_sharpe:.2f}")
    
    return best_params

def optimize_signal_filtering(base_config, start_date, end_date):
    """
    Optimize signal filtering parameters to improve win rate
    
    Args:
        base_config: Base configuration dictionary
        start_date: Start date for backtest
        end_date: End date for backtest
        
    Returns:
        Dict: Optimized signal filtering parameters
    """
    logger.info("Optimizing signal filtering parameters")
    
    # Parameters to optimize
    param_grid = {
        "min_score_threshold": [0.5, 0.6, 0.7, 0.8],
        "max_correlation_threshold": [0.5, 0.7, 0.9],
        "min_volume_percentile": [30, 50, 70],
        "min_regime_weight": [0.2, 0.3, 0.4],
        "max_signals_per_day": [5, 10, 15, 20]
    }
    
    # Generate parameter combinations
    param_combinations = []
    for min_score in param_grid["min_score_threshold"]:
        for max_corr in param_grid["max_correlation_threshold"]:
            for min_vol in param_grid["min_volume_percentile"]:
                for min_regime in param_grid["min_regime_weight"]:
                    for max_signals in param_grid["max_signals_per_day"]:
                        params = {
                            "min_score_threshold": min_score,
                            "max_correlation_threshold": max_corr,
                            "min_volume_percentile": min_vol,
                            "min_regime_weight": min_regime,
                            "max_signals_per_day": max_signals
                        }
                        param_combinations.append(params)
    
    # Limit number of combinations to test
    max_combinations = 10
    if len(param_combinations) > max_combinations:
        logger.info(f"Limiting to {max_combinations} parameter combinations")
        param_combinations = param_combinations[:max_combinations]
    
    # Test each parameter combination
    best_win_rate = 0
    best_sharpe = 0
    best_params = None
    
    for params in param_combinations:
        # Create config copy with updated parameters
        config_copy = copy.deepcopy(base_config)
        
        # Update signal quality filters
        for key, value in params.items():
            config_copy["signal_quality_filters"][key] = value
        
        # Run backtest
        result = run_backtest(config_copy, start_date, end_date)
        
        if result:
            # Calculate combined score (win rate + sharpe)
            combined_score = (result.win_rate * 0.7) + (result.sharpe_ratio * 0.3)
            
            if combined_score > (best_win_rate * 0.7 + best_sharpe * 0.3):
                best_win_rate = result.win_rate
                best_sharpe = result.sharpe_ratio
                best_params = params
                
                logger.info(f"New best parameters found: {params}")
                logger.info(f"Win rate: {best_win_rate:.2f}%, Sharpe: {best_sharpe:.2f}")
    
    logger.info(f"Optimized signal filtering parameters: {best_params}")
    logger.info(f"Best win rate: {best_win_rate:.2f}%, Best Sharpe: {best_sharpe:.2f}")
    
    return best_params

def optimize_position_sizing(base_config, start_date, end_date):
    """
    Optimize position sizing parameters
    
    Args:
        base_config: Base configuration dictionary
        start_date: Start date for backtest
        end_date: End date for backtest
        
    Returns:
        Dict: Optimized position sizing parameters
    """
    logger.info("Optimizing position sizing parameters")
    
    # Parameters to optimize
    param_grid = {
        "base_risk_per_trade": [0.005, 0.01, 0.015, 0.02],
        "max_position_size": [0.05, 0.1, 0.15],
        "min_position_size": [0.002, 0.005, 0.01]
    }
    
    # Generate parameter combinations
    param_combinations = []
    for base_risk in param_grid["base_risk_per_trade"]:
        for max_pos in param_grid["max_position_size"]:
            for min_pos in param_grid["min_position_size"]:
                if min_pos < max_pos:  # Ensure min < max
                    params = {
                        "base_risk_per_trade": base_risk,
                        "max_position_size": max_pos,
                        "min_position_size": min_pos,
                        "volatility_adjustment": True,
                        "signal_strength_adjustment": True
                    }
                    param_combinations.append(params)
    
    # Limit number of combinations to test
    max_combinations = 10
    if len(param_combinations) > max_combinations:
        logger.info(f"Limiting to {max_combinations} parameter combinations")
        param_combinations = param_combinations[:max_combinations]
    
    # Test each parameter combination
    best_sharpe = 0
    best_params = None
    
    for params in param_combinations:
        # Create config copy with updated parameters
        config_copy = copy.deepcopy(base_config)
        
        # Update position sizing parameters
        config_copy["position_sizing_config"] = params
        
        # Run backtest
        result = run_backtest(config_copy, start_date, end_date)
        
        if result and result.sharpe_ratio > best_sharpe:
            best_sharpe = result.sharpe_ratio
            best_params = params
            
            logger.info(f"New best parameters found: {params}")
            logger.info(f"Sharpe ratio: {best_sharpe:.2f}")
    
    logger.info(f"Optimized position sizing parameters: {best_params}")
    logger.info(f"Best Sharpe ratio: {best_sharpe:.2f}")
    
    return best_params

def analyze_sharpe_ratio_factors(config_dict, start_date, end_date):
    """
    Analyze factors affecting Sharpe ratio
    
    Args:
        config_dict: Configuration dictionary
        start_date: Start date for backtest
        end_date: End date for backtest
    """
    logger.info("Analyzing factors affecting Sharpe ratio")
    
    # Create base config
    base_config = copy.deepcopy(config_dict)
    
    # Run baseline backtest
    baseline_result = run_backtest(base_config, start_date, end_date)
    
    if not baseline_result:
        logger.error("Failed to run baseline backtest")
        return
    
    logger.info(f"Baseline Sharpe ratio: {baseline_result.sharpe_ratio:.2f}")
    logger.info(f"Baseline win rate: {baseline_result.win_rate:.2f}%")
    logger.info(f"Baseline total trades: {baseline_result.total_trades}")
    
    # Test impact of reducing trading frequency
    frequency_config = copy.deepcopy(base_config)
    frequency_config["signal_quality_filters"]["max_signals_per_day"] = 5  # Reduce from default
    frequency_config["signal_quality_filters"]["min_score_threshold"] = 0.8  # Increase from default
    
    frequency_result = run_backtest(frequency_config, start_date, end_date)
    
    if frequency_result:
        logger.info(f"Reduced frequency Sharpe ratio: {frequency_result.sharpe_ratio:.2f}")
        logger.info(f"Reduced frequency win rate: {frequency_result.win_rate:.2f}%")
        logger.info(f"Reduced frequency total trades: {frequency_result.total_trades}")
    
    # Test impact of improving signal quality
    quality_config = copy.deepcopy(base_config)
    quality_config["signal_quality_filters"]["min_score_threshold"] = 0.7
    quality_config["signal_quality_filters"]["min_volume_percentile"] = 70
    
    quality_result = run_backtest(quality_config, start_date, end_date)
    
    if quality_result:
        logger.info(f"Improved quality Sharpe ratio: {quality_result.sharpe_ratio:.2f}")
        logger.info(f"Improved quality win rate: {quality_result.win_rate:.2f}%")
        logger.info(f"Improved quality total trades: {quality_result.total_trades}")
    
    # Test impact of position sizing
    sizing_config = copy.deepcopy(base_config)
    sizing_config["position_sizing_config"]["base_risk_per_trade"] = 0.01
    sizing_config["position_sizing_config"]["max_position_size"] = 0.05
    
    sizing_result = run_backtest(sizing_config, start_date, end_date)
    
    if sizing_result:
        logger.info(f"Adjusted sizing Sharpe ratio: {sizing_result.sharpe_ratio:.2f}")
        logger.info(f"Adjusted sizing win rate: {sizing_result.win_rate:.2f}%")
        logger.info(f"Adjusted sizing total trades: {sizing_result.total_trades}")
    
    # Generate report
    logger.info("=== Sharpe Ratio Analysis Report ===")
    
    if frequency_result and frequency_result.sharpe_ratio > baseline_result.sharpe_ratio:
        logger.info("✓ Reducing trading frequency improves Sharpe ratio")
    else:
        logger.info("✗ Reducing trading frequency does not improve Sharpe ratio")
    
    if quality_result and quality_result.sharpe_ratio > baseline_result.sharpe_ratio:
        logger.info("✓ Improving signal quality improves Sharpe ratio")
    else:
        logger.info("✗ Improving signal quality does not improve Sharpe ratio")
    
    if sizing_result and sizing_result.sharpe_ratio > baseline_result.sharpe_ratio:
        logger.info("✓ Adjusting position sizing improves Sharpe ratio")
    else:
        logger.info("✗ Adjusting position sizing does not improve Sharpe ratio")

def main():
    """Main function to optimize the trading system"""
    logger.info("Starting Trading System Optimization")
    
    try:
        # Load configuration
        config_dict = load_config()
        if not config_dict:
            logger.error("Failed to load configuration")
            return
        
        # Define test period
        start_date = dt.date(2023, 1, 1)
        end_date = dt.date(2023, 12, 31)
        
        # Analyze Sharpe ratio factors
        analyze_sharpe_ratio_factors(config_dict, start_date, end_date)
        
        # Optimize ML strategy selector
        ml_params = optimize_ml_strategy_selector(config_dict, start_date, end_date)
        if ml_params:
            config_dict["ml_strategy_selector"].update(ml_params)
        
        # Optimize signal filtering
        filter_params = optimize_signal_filtering(config_dict, start_date, end_date)
        if filter_params:
            for key, value in filter_params.items():
                config_dict["signal_quality_filters"][key] = value
        
        # Optimize position sizing
        position_params = optimize_position_sizing(config_dict, start_date, end_date)
        if position_params:
            config_dict["position_sizing_config"] = position_params
        
        # Save optimized configuration
        save_config(config_dict, "optimized_config.yaml")
        
        # Run final backtest with optimized configuration
        logger.info("Running final backtest with optimized configuration")
        final_result = run_backtest(config_dict, start_date, end_date)
        
        if final_result:
            logger.info("=== Final Optimization Results ===")
            logger.info(f"Total Return: {final_result.total_return_pct:.2f}%")
            logger.info(f"Annualized Return: {final_result.annualized_return_pct:.2f}%")
            logger.info(f"Sharpe Ratio: {final_result.sharpe_ratio:.2f}")
            logger.info(f"Max Drawdown: {final_result.max_drawdown_pct:.2f}%")
            logger.info(f"Win Rate: {final_result.win_rate:.2f}%")
            logger.info(f"Profit Factor: {final_result.profit_factor:.2f}")
            logger.info(f"Total Trades: {final_result.total_trades}")
        
        logger.info("Trading System Optimization completed")
        
    except Exception as e:
        logger.error(f"Error in Trading System Optimization: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
