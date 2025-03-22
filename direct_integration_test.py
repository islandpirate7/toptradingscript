#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Direct Integration Test for Enhanced Trading System
--------------------------------------------------
This script directly integrates the enhanced trading functions into the system
and tests them over different market conditions.
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
        logging.FileHandler('direct_integration_test.log')
    ]
)

logger = logging.getLogger("DirectIntegrationTest")

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

def run_test_over_different_periods(config, periods):
    """
    Run tests over different market periods to evaluate performance in various conditions
    
    Args:
        config: System configuration
        periods: List of (start_date, end_date, description) tuples
        
    Returns:
        Dict: Results for each period
    """
    results = {}
    
    for start_date, end_date, description in periods:
        logger.info(f"Testing period: {description} ({start_date} to {end_date})")
        
        # Create original system
        original_system = MultiStrategySystem(config)
        
        # Create enhanced system
        enhanced_system = EnhancedMultiStrategySystem(config)
        
        # Run backtests
        original_result = original_system.run_backtest(start_date, end_date)
        enhanced_result = enhanced_system.run_backtest(start_date, end_date)
        
        # Compare results
        comparison = compare_results(
            original_result, 
            enhanced_result,
            period_description=description
        )
        
        results[description] = comparison
    
    return results

def compare_results(original_result, enhanced_result, period_description=""):
    """
    Compare original and enhanced system results
    
    Args:
        original_result: Backtest result from original system
        enhanced_result: Backtest result from enhanced system
        period_description: Description of the test period
        
    Returns:
        Dict: Comparison metrics
    """
    try:
        # Create comparison dictionary
        comparison = {
            "period": period_description,
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
        filename = f"comparison_{period_description.lower().replace(' ', '_')}.json"
        with open(filename, 'w') as f:
            json.dump(comparison, f, indent=4)
        
        # Print comparison
        logger.info(f"=== Backtest Comparison for {period_description} ===")
        logger.info(f"Total Return: {original_result.total_return_pct:.2f}% -> {enhanced_result.total_return_pct:.2f}% ({comparison['total_return_pct']['percent_improvement']:.2f}% improvement)")
        logger.info(f"Annualized Return: {original_result.annualized_return_pct:.2f}% -> {enhanced_result.annualized_return_pct:.2f}% ({comparison['annualized_return_pct']['percent_improvement']:.2f}% improvement)")
        logger.info(f"Sharpe Ratio: {original_result.sharpe_ratio:.2f} -> {enhanced_result.sharpe_ratio:.2f} ({comparison['sharpe_ratio']['percent_improvement']:.2f}% improvement)")
        logger.info(f"Max Drawdown: {original_result.max_drawdown_pct:.2f}% -> {enhanced_result.max_drawdown_pct:.2f}% ({comparison['max_drawdown_pct']['percent_improvement']:.2f}% improvement)")
        logger.info(f"Win Rate: {original_result.win_rate:.2f}% -> {enhanced_result.win_rate:.2f}% ({comparison['win_rate']['percent_improvement']:.2f}% improvement)")
        logger.info(f"Profit Factor: {original_result.profit_factor:.2f} -> {enhanced_result.profit_factor:.2f} ({comparison['profit_factor']['percent_improvement']:.2f}% improvement)")
        logger.info(f"Total Trades: {original_result.total_trades} -> {enhanced_result.total_trades} ({comparison['total_trades']['percent_change']:.2f}% change)")
        
        # Plot equity curves
        plot_equity_curves(original_result, enhanced_result, period_description)
        
        return comparison
    except Exception as e:
        logger.error(f"Error comparing results: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def plot_equity_curves(original_result, enhanced_result, period_description=""):
    """
    Plot equity curves for comparison
    
    Args:
        original_result: Backtest result from original system
        enhanced_result: Backtest result from enhanced system
        period_description: Description of the test period
    """
    try:
        plt.figure(figsize=(12, 8))
        
        # Convert original equity curve to DataFrame
        original_equity = pd.DataFrame(original_result.equity_curve, columns=['date', 'equity'])
        original_equity.set_index('date', inplace=True)
        
        # Plot original equity curve
        plt.plot(original_equity.index, original_equity['equity'], label='Original', color='blue')
        
        # Convert enhanced equity curve to DataFrame
        enhanced_equity = pd.DataFrame(enhanced_result.equity_curve, columns=['date', 'equity'])
        enhanced_equity.set_index('date', inplace=True)
        
        # Plot enhanced equity curve
        plt.plot(enhanced_equity.index, enhanced_equity['equity'], label='Enhanced', color='green')
        
        # Add labels and title
        plt.title(f'Equity Curve Comparison - {period_description}')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        filename = f"equity_curve_{period_description.lower().replace(' ', '_')}.png"
        plt.savefig(filename)
        logger.info(f"Equity curve comparison saved to '{filename}'")
        
        # Plot drawdown curves
        plt.figure(figsize=(12, 8))
        
        # Convert original drawdown curve to DataFrame
        original_drawdown = pd.DataFrame(original_result.drawdown_curve, columns=['date', 'drawdown'])
        original_drawdown.set_index('date', inplace=True)
        
        # Plot original drawdown curve
        plt.plot(original_drawdown.index, original_drawdown['drawdown'], label='Original', color='blue')
        
        # Convert enhanced drawdown curve to DataFrame
        enhanced_drawdown = pd.DataFrame(enhanced_result.drawdown_curve, columns=['date', 'drawdown'])
        enhanced_drawdown.set_index('date', inplace=True)
        
        # Plot enhanced drawdown curve
        plt.plot(enhanced_drawdown.index, enhanced_drawdown['drawdown'], label='Enhanced', color='green')
        
        # Add labels and title
        plt.title(f'Drawdown Comparison - {period_description}')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        filename = f"drawdown_{period_description.lower().replace(' ', '_')}.png"
        plt.savefig(filename)
        logger.info(f"Drawdown comparison saved to '{filename}'")
        
    except Exception as e:
        logger.error(f"Error plotting equity curves: {str(e)}")
        logger.error(traceback.format_exc())

def main():
    """Main function to test enhanced trading system"""
    logger.info("Starting Direct Integration Test of Enhanced Trading System")
    
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
        
        # Define test periods for different market conditions
        test_periods = [
            # Full year
            (dt.date(2023, 1, 1), dt.date(2023, 12, 31), "Full Year 2023"),
            
            # Bull market period
            (dt.date(2023, 1, 1), dt.date(2023, 7, 31), "Bull Market H1 2023"),
            
            # Volatile period
            (dt.date(2023, 8, 1), dt.date(2023, 10, 31), "Volatile Q3 2023"),
            
            # Year-end rally
            (dt.date(2023, 11, 1), dt.date(2023, 12, 31), "Year-End Rally 2023"),
            
            # Short-term test (1 month)
            (dt.date(2023, 6, 1), dt.date(2023, 6, 30), "June 2023")
        ]
        
        # Run tests over different periods
        results = run_test_over_different_periods(config, test_periods)
        
        # Save overall results
        with open('direct_integration_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info("Direct Integration Test completed")
        
    except Exception as e:
        logger.error(f"Error in Direct Integration Test: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
