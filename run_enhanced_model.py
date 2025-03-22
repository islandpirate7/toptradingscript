#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run Enhanced Trading Model
--------------------------
This script runs the enhanced trading model that produced the impressive results
in the direct integration test.
"""

import os
import sys
import logging
import datetime as dt
import json
import traceback
import yaml
from typing import Dict, Any, List, Tuple

# Import the multi-strategy system
from multi_strategy_system import (
    MultiStrategySystem, SystemConfig, StockConfig, MarketRegime
)

# Import the enhanced trading components
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
        logging.FileHandler('enhanced_model.log')
    ]
)

logger = logging.getLogger("EnhancedModel")

def load_config(config_file='multi_strategy_config.yaml'):
    """Load configuration from YAML file"""
    try:
        with open(config_file, 'r') as file:
            config_dict = yaml.safe_load(file)
        logger.info(f"Loaded configuration from {config_file}")
        return config_dict
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
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

def run_backtest(config, start_date, end_date):
    """Run backtest with the enhanced trading system"""
    try:
        # Create enhanced system
        system = EnhancedMultiStrategySystem(config)
        
        # Run backtest
        logger.info(f"Running backtest from {start_date} to {end_date}")
        result = system.run_backtest(start_date, end_date)
        
        # Save results
        if result:
            results_file = f"enhanced_results_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json"
            
            # Convert datetime objects to strings for JSON serialization
            def serialize_datetime(obj):
                if isinstance(obj, (dt.datetime, dt.date)):
                    return obj.isoformat()
                raise TypeError(f"Type {type(obj)} not serializable")
            
            with open(results_file, 'w') as f:
                json.dump(result.to_dict(), f, indent=2, default=serialize_datetime)
            
            logger.info(f"Results saved to {results_file}")
            
            # Log performance metrics
            logger.info(f"Total Return: {result.total_return_pct:.2f}%")
            logger.info(f"Annualized Return: {result.annualized_return_pct:.2f}%")
            logger.info(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
            logger.info(f"Max Drawdown: {result.max_drawdown_pct:.2f}%")
            logger.info(f"Win Rate: {result.win_rate:.2f}%")
            logger.info(f"Profit Factor: {result.profit_factor:.2f}")
            logger.info(f"Total Trades: {result.total_trades}")
        
        return result
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def main():
    """Main function to run the enhanced trading model"""
    logger.info("Starting Enhanced Trading Model")
    
    try:
        # Load configuration
        config_dict = load_config('optimized_config.yaml')
        if not config_dict:
            logger.error("Failed to load configuration")
            return
        
        # Create system config
        config = create_system_config(config_dict)
        if not config:
            logger.error("Failed to create system config")
            return
        
        # Define test periods
        test_periods = [
            # Full year
            (dt.date(2023, 1, 1), dt.date(2023, 12, 31), "Full Year 2023"),
            
            # Bull market period
            (dt.date(2023, 1, 1), dt.date(2023, 7, 31), "Bull Market H1 2023"),
            
            # Volatile period
            (dt.date(2023, 8, 1), dt.date(2023, 10, 31), "Volatile Q3 2023"),
            
            # Year-end rally
            (dt.date(2023, 11, 1), dt.date(2023, 12, 31), "Year-End Rally 2023")
        ]
        
        # Run backtests for each period
        results = {}
        for start_date, end_date, description in test_periods:
            logger.info(f"Testing period: {description} ({start_date} to {end_date})")
            result = run_backtest(config, start_date, end_date)
            if result:
                results[description] = result.to_dict()
        
        # Save overall results
        with open('enhanced_model_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=lambda obj: obj.isoformat() if isinstance(obj, (dt.datetime, dt.date)) else None)
        
        logger.info("Enhanced Trading Model completed")
        
    except Exception as e:
        logger.error(f"Error in Enhanced Trading Model: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
