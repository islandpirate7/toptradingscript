#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Signal Filter Optimizer
----------------------
This script focuses on optimizing signal filtering parameters to improve
win rate and balance trading frequency.
"""

import os
import sys
import logging
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import traceback
from typing import List, Dict, Any, Tuple
import copy
import itertools

# Import the multi-strategy system
from multi_strategy_system import (
    MultiStrategySystem, SystemConfig, Signal, MarketRegime, 
    BacktestResult, StockConfig, MarketState
)

# Import enhanced trading functions
from enhanced_trading_functions import filter_signals

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('signal_filter_optimizer.log')
    ]
)

logger = logging.getLogger("SignalFilterOptimizer")

def load_config():
    """Load configuration from YAML file"""
    try:
        with open('multi_strategy_config.yaml', 'r') as file:
            config_dict = yaml.safe_load(file)
        return config_dict
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return None

def save_config(config_dict, filename='optimized_filters_config.yaml'):
    """Save configuration to YAML file"""
    try:
        with open(filename, 'w') as file:
            yaml.dump(config_dict, file, default_flow_style=False)
        logger.info(f"Configuration saved to {filename}")
        return True
    except Exception as e:
        logger.error(f"Error saving configuration: {str(e)}")
        return False

class SignalGenerator:
    """Class to generate synthetic signals for filter optimization"""
    
    def __init__(self, config_dict):
        """Initialize the signal generator"""
        self.config = config_dict
        self.stock_configs = []
        
        # Extract stock configs
        for stock_dict in config_dict.get('stocks', []):
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
            self.stock_configs.append(stock_config)
    
    def generate_synthetic_signals(self, n_signals=100):
        """
        Generate synthetic signals for testing filters
        
        Args:
            n_signals: Number of signals to generate
            
        Returns:
            List: List of synthetic signals
        """
        logger.info(f"Generating {n_signals} synthetic signals")
        
        signals = []
        strategies = ["MeanReversion", "TrendFollowing", "VolatilityBreakout", "GapTrading"]
        
        for i in range(n_signals):
            # Select random stock
            stock_config = np.random.choice(self.stock_configs)
            
            # Create signal
            signal = Signal()
            signal.symbol = stock_config.symbol
            signal.entry_price = np.random.uniform(10, 200)
            signal.stop_loss = signal.entry_price * (1 - np.random.uniform(0.01, 0.05))
            signal.take_profit = signal.entry_price * (1 + np.random.uniform(0.02, 0.1))
            signal.direction = np.random.choice([1, -1])  # 1 for long, -1 for short
            signal.timestamp = dt.datetime.now()
            signal.expiration = dt.datetime.now() + dt.timedelta(days=np.random.randint(1, 5))
            
            # Add score (quality metric)
            signal.score = np.random.uniform(0.3, 1.0)
            
            # Add metadata
            signal.metadata = {}
            signal.metadata["strategy_name"] = np.random.choice(strategies)
            signal.metadata["market_regime"] = np.random.randint(1, 9)  # 1-8 for different regimes
            signal.metadata["regime_weight"] = np.random.uniform(0.2, 1.0)
            signal.metadata["predicted_performance"] = np.random.uniform(-0.02, 0.05)
            
            # Add to signals list
            signals.append(signal)
        
        logger.info(f"Generated {len(signals)} synthetic signals")
        return signals
    
    def generate_synthetic_candle_data(self):
        """
        Generate synthetic candle data for testing filters
        
        Returns:
            Dict: Dictionary of candle data by symbol
        """
        logger.info("Generating synthetic candle data")
        
        candle_data = {}
        
        class Candle:
            def __init__(self, open_price, high, low, close, volume, timestamp):
                self.open = open_price
                self.high = high
                self.low = low
                self.close = close
                self.volume = volume
                self.timestamp = timestamp
        
        # Generate candle data for each stock
        for stock_config in self.stock_configs:
            symbol = stock_config.symbol
            
            # Generate 30 days of candle data
            candles = []
            base_price = np.random.uniform(50, 200)
            base_volume = np.random.uniform(100000, 10000000)
            
            for i in range(30):
                timestamp = dt.datetime.now() - dt.timedelta(days=30-i)
                
                # Generate prices with some randomness and trend
                daily_return = np.random.normal(0, 0.015)
                base_price *= (1 + daily_return)
                
                open_price = base_price * (1 + np.random.normal(0, 0.005))
                close = base_price * (1 + np.random.normal(0, 0.005))
                high = max(open_price, close) * (1 + np.random.uniform(0.001, 0.01))
                low = min(open_price, close) * (1 - np.random.uniform(0.001, 0.01))
                
                # Generate volume with some randomness
                volume = base_volume * (1 + np.random.normal(0, 0.2))
                volume = max(10000, volume)  # Ensure minimum volume
                
                candle = Candle(open_price, high, low, close, volume, timestamp)
                candles.append(candle)
            
            candle_data[symbol] = candles
        
        logger.info(f"Generated candle data for {len(candle_data)} symbols")
        return candle_data

def evaluate_filter_performance(signals, filtered_signals):
    """
    Evaluate filter performance based on signal quality
    
    Args:
        signals: Original signals
        filtered_signals: Filtered signals
        
    Returns:
        Dict: Performance metrics
    """
    if not signals:
        return {"error": "No signals provided"}
    
    if not filtered_signals:
        return {"error": "No signals passed the filters"}
    
    # Calculate metrics
    total_signals = len(signals)
    passed_signals = len(filtered_signals)
    rejection_rate = 1 - (passed_signals / total_signals)
    
    # Calculate average score
    original_avg_score = np.mean([getattr(s, 'score', 0) for s in signals])
    filtered_avg_score = np.mean([getattr(s, 'score', 0) for s in filtered_signals])
    score_improvement = filtered_avg_score - original_avg_score
    
    # Calculate average predicted performance
    original_avg_perf = np.mean([s.metadata.get('predicted_performance', 0) for s in signals if hasattr(s, 'metadata')])
    filtered_avg_perf = np.mean([s.metadata.get('predicted_performance', 0) for s in filtered_signals if hasattr(s, 'metadata')])
    perf_improvement = filtered_avg_perf - original_avg_perf
    
    # Calculate sector diversity
    sectors = {}
    for signal in filtered_signals:
        # Find sector for the symbol
        sector = "Unknown"
        for stock in signals:
            if hasattr(stock, 'sector'):
                if stock.symbol == signal.symbol:
                    sector = stock.sector
                    break
        
        sectors[sector] = sectors.get(sector, 0) + 1
    
    sector_concentration = max(sectors.values()) / passed_signals if passed_signals > 0 else 1.0
    
    # Calculate strategy diversity
    strategies = {}
    for signal in filtered_signals:
        if hasattr(signal, 'metadata') and 'strategy_name' in signal.metadata:
            strategy = signal.metadata['strategy_name']
            strategies[strategy] = strategies.get(strategy, 0) + 1
    
    strategy_concentration = max(strategies.values()) / passed_signals if passed_signals > 0 else 1.0 and len(strategies) > 0
    
    # Return metrics
    return {
        "total_signals": total_signals,
        "passed_signals": passed_signals,
        "rejection_rate": rejection_rate,
        "original_avg_score": original_avg_score,
        "filtered_avg_score": filtered_avg_score,
        "score_improvement": score_improvement,
        "original_avg_perf": original_avg_perf,
        "filtered_avg_perf": filtered_avg_perf,
        "perf_improvement": perf_improvement,
        "sector_concentration": sector_concentration,
        "strategy_concentration": strategy_concentration
    }

def optimize_signal_filters(config_dict):
    """
    Optimize signal filtering parameters
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Dict: Optimized signal filtering parameters
    """
    logger.info("Optimizing signal filtering parameters")
    
    # Create signal generator
    signal_generator = SignalGenerator(config_dict)
    
    # Generate synthetic signals and candle data
    signals = signal_generator.generate_synthetic_signals(n_signals=500)
    candle_data = signal_generator.generate_synthetic_candle_data()
    
    # Parameters to optimize
    param_grid = {
        "min_score_threshold": [0.5, 0.6, 0.7, 0.8],
        "max_correlation_threshold": [0.5, 0.7, 0.9],
        "max_signals_per_day": [5, 10, 15, 20],
        "max_sector_exposure": [0.2, 0.3, 0.4],
        "min_price": [5, 10, 15]
    }
    
    # Generate parameter combinations
    param_combinations = []
    for min_score in param_grid["min_score_threshold"]:
        for max_corr in param_grid["max_correlation_threshold"]:
            for max_signals in param_grid["max_signals_per_day"]:
                for max_sector in param_grid["max_sector_exposure"]:
                    for min_price in param_grid["min_price"]:
                        params = {
                            "min_score_threshold": min_score,
                            "max_correlation_threshold": max_corr,
                            "max_signals_per_day": max_signals,
                            "max_sector_exposure": max_sector,
                            "min_price": min_price
                        }
                        param_combinations.append(params)
    
    # Limit number of combinations to test
    max_combinations = 20
    if len(param_combinations) > max_combinations:
        logger.info(f"Limiting to {max_combinations} parameter combinations")
        param_combinations = param_combinations[:max_combinations]
    
    # Test each parameter combination
    best_combined_score = 0
    best_params = None
    best_metrics = None
    
    for params in param_combinations:
        # Create config copy with updated parameters
        signal_quality_filters = copy.deepcopy(params)
        
        # Apply filters
        filtered_signals = filter_signals(
            signals=signals,
            candle_data=candle_data,
            config=None,  # Not needed for this test
            signal_quality_filters=signal_quality_filters,
            logger=logger
        )
        
        # Evaluate performance
        metrics = evaluate_filter_performance(signals, filtered_signals)
        
        if "error" in metrics:
            logger.warning(f"Error evaluating filters: {metrics['error']}")
            continue
        
        # Calculate combined score
        # We want:
        # - High score improvement
        # - High performance improvement
        # - Moderate rejection rate (not too high, not too low)
        # - Low sector and strategy concentration
        
        score_weight = 0.3
        perf_weight = 0.3
        rejection_weight = 0.2
        diversity_weight = 0.2
        
        # Normalize rejection rate to have optimal value around 0.7
        rejection_score = 1.0 - abs(metrics["rejection_rate"] - 0.7)
        
        # Diversity score (lower concentration is better)
        diversity_score = 2.0 - (metrics["sector_concentration"] + metrics["strategy_concentration"])
        
        combined_score = (
            score_weight * metrics["score_improvement"] * 10 +  # Scale up score improvement
            perf_weight * metrics["perf_improvement"] * 20 +    # Scale up performance improvement
            rejection_weight * rejection_score +
            diversity_weight * diversity_score
        )
        
        logger.info(f"Testing parameters: {params}")
        logger.info(f"Metrics: Score Improvement: {metrics['score_improvement']:.4f}, "
                   f"Perf Improvement: {metrics['perf_improvement']:.4f}, "
                   f"Rejection Rate: {metrics['rejection_rate']:.2f}")
        logger.info(f"Combined Score: {combined_score:.4f}")
        
        if combined_score > best_combined_score:
            best_combined_score = combined_score
            best_params = params
            best_metrics = metrics
            
            logger.info(f"New best parameters found: {params}")
            logger.info(f"Combined Score: {best_combined_score:.4f}")
    
    if best_params:
        logger.info("=== Best Signal Filter Parameters ===")
        logger.info(f"Parameters: {best_params}")
        logger.info(f"Rejection Rate: {best_metrics['rejection_rate']:.2f}")
        logger.info(f"Score Improvement: {best_metrics['score_improvement']:.4f}")
        logger.info(f"Performance Improvement: {best_metrics['perf_improvement']:.4f}")
        logger.info(f"Sector Concentration: {best_metrics['sector_concentration']:.2f}")
        logger.info(f"Strategy Concentration: {best_metrics['strategy_concentration']:.2f}")
    else:
        logger.warning("No valid parameter combination found")
    
    return best_params

def main():
    """Main function to optimize signal filters"""
    logger.info("Starting Signal Filter Optimization")
    
    try:
        # Load configuration
        config_dict = load_config()
        if not config_dict:
            logger.error("Failed to load configuration")
            return
        
        # Optimize signal filters
        filter_params = optimize_signal_filters(config_dict)
        
        if filter_params:
            # Update configuration
            config_dict["signal_quality_filters"] = filter_params
            
            # Save optimized configuration
            save_config(config_dict, "optimized_filters_config.yaml")
            
            logger.info("Signal Filter Optimization completed")
        else:
            logger.error("Signal Filter Optimization failed")
        
    except Exception as e:
        logger.error(f"Error in Signal Filter Optimization: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
