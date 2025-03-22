#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Backtest with Aggressive Settings
--------------------------------
This script runs a backtest with aggressive settings
to generate more trading signals for March 2024.
"""

import os
import yaml
import logging
import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import json

from backtest_combined_strategy import Backtester
from combined_strategy import CombinedStrategy
from mean_reversion_strategy_optimized import MeanReversionStrategyOptimized

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AggressiveBacktester(Backtester):
    """Backtester with aggressive settings to generate more signals"""
    
    def __init__(self, config_file):
        """Initialize with aggressive settings"""
        super().__init__(config_file)
        
        # Override strategy with aggressive settings
        self.apply_aggressive_settings()
    
    def apply_aggressive_settings(self):
        """Apply aggressive settings to generate more signals"""
        logger.info("Applying aggressive settings to generate more signals")
        
        # Modify mean reversion strategy parameters
        if hasattr(self.strategy, 'mean_reversion'):
            # Lower thresholds for more signals
            self.strategy.mean_reversion.bb_std = 1.5
            self.strategy.mean_reversion.rsi_oversold = 45
            self.strategy.mean_reversion.rsi_overbought = 55
            self.strategy.mean_reversion.require_reversal = False
            self.strategy.mean_reversion.use_volume_filter = False
            self.strategy.mean_reversion.min_bb_penetration = 0.1
            
            logger.info(f"Modified mean reversion parameters: BB std={self.strategy.mean_reversion.bb_std}, "
                      f"RSI thresholds={self.strategy.mean_reversion.rsi_oversold}/{self.strategy.mean_reversion.rsi_overbought}, "
                      f"Require reversal={self.strategy.mean_reversion.require_reversal}, "
                      f"Volume filter={self.strategy.mean_reversion.use_volume_filter}")
        
        # Modify trend following strategy parameters if available
        if hasattr(self.strategy, 'trend_following'):
            # Lower thresholds for more signals
            if hasattr(self.strategy.trend_following, 'adx_threshold'):
                self.strategy.trend_following.adx_threshold = 15
                logger.info(f"Modified trend following parameters: ADX threshold={self.strategy.trend_following.adx_threshold}")
        
        # Lower the minimum signal score threshold
        self.strategy.min_signal_score = 0.45
        logger.info(f"Lowered minimum signal score threshold to {self.strategy.min_signal_score}")
        
        # Increase seasonality boost and penalty factors
        if hasattr(self.strategy, 'seasonal_boost'):
            self.strategy.seasonal_boost = 0.5
            self.strategy.seasonal_penalty = 0.5
            logger.info(f"Increased seasonality factors: boost={self.strategy.seasonal_boost}, penalty={self.strategy.seasonal_penalty}")

def run_march_2024_backtest(config_file):
    """Run a backtest for March 2024 with aggressive settings
    
    Args:
        config_file (str): Path to configuration file
    """
    start_date = dt.datetime(2024, 3, 1)
    end_date = dt.datetime(2024, 3, 31)
    
    # Use aggressive backtester
    logger.info(f"Running aggressive backtest for March 2024")
    logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    backtester = AggressiveBacktester(config_file)
    results = backtester.run_backtest(start_date, end_date)
    
    return results

def run_q1_2024_backtest(config_file):
    """Run a backtest for Q1 2024 with aggressive settings
    
    Args:
        config_file (str): Path to configuration file
    """
    start_date = dt.datetime(2024, 1, 1)
    end_date = dt.datetime(2024, 3, 31)
    
    # Use aggressive backtester
    logger.info(f"Running aggressive backtest for Q1 2024")
    logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    backtester = AggressiveBacktester(config_file)
    results = backtester.run_backtest(start_date, end_date)
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run aggressive backtest')
    parser.add_argument('--config', type=str, default='configuration_combined_strategy_march_seasonal.yaml',
                        help='Path to configuration file')
    parser.add_argument('--period', type=str, choices=['march', 'q1', 'all'], default='march',
                        help='Period to backtest (march, q1, or all)')
    
    args = parser.parse_args()
    
    # Run backtest
    if args.period == 'march':
        run_march_2024_backtest(args.config)
    elif args.period == 'q1':
        run_q1_2024_backtest(args.config)
    elif args.period == 'all':
        run_march_2024_backtest(args.config)
        run_q1_2024_backtest(args.config)
