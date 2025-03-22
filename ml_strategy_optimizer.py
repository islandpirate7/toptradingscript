#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ML Strategy Selector Optimizer
-----------------------------
This script focuses specifically on optimizing the ML strategy selector
to improve the Sharpe ratio and overall performance.
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
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Import the ML strategy selector
from ml_strategy_selector import MLStrategySelector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ml_strategy_optimizer.log')
    ]
)

logger = logging.getLogger("MLStrategyOptimizer")

def load_config():
    """Load configuration from YAML file"""
    try:
        with open('multi_strategy_config.yaml', 'r') as file:
            config_dict = yaml.safe_load(file)
        return config_dict
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return None

def save_config(config_dict, filename='optimized_ml_config.yaml'):
    """Save configuration to YAML file"""
    try:
        with open(filename, 'w') as file:
            yaml.dump(config_dict, file, default_flow_style=False)
        logger.info(f"Configuration saved to {filename}")
        return True
    except Exception as e:
        logger.error(f"Error saving configuration: {str(e)}")
        return False

class MLStrategyOptimizerData:
    """Class to generate and manage data for ML strategy optimization"""
    
    def __init__(self, config):
        """Initialize the ML strategy optimizer data"""
        self.config = config
        self.strategy_performance_data = {}
        self.market_regime_data = {}
        
    def generate_synthetic_data(self, n_samples=1000):
        """
        Generate synthetic data for ML model optimization
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Dict: Dictionary of strategy performance data by strategy name
        """
        logger.info(f"Generating {n_samples} synthetic data samples for ML optimization")
        
        # Define market regimes
        regimes = [
            {"regime": 1, "name": "TRENDING_BULLISH"},
            {"regime": 2, "name": "TRENDING_BEARISH"},
            {"regime": 3, "name": "RANGE_BOUND"},
            {"regime": 4, "name": "HIGH_VOLATILITY"},
            {"regime": 5, "name": "LOW_VOLATILITY"},
            {"regime": 6, "name": "BEARISH_BREAKDOWN"},
            {"regime": 7, "name": "BULLISH_BREAKOUT"},
            {"regime": 8, "name": "CONSOLIDATION"}
        ]
        
        # Define strategies
        strategies = ["MeanReversion", "TrendFollowing", "VolatilityBreakout", "GapTrading"]
        
        # Generate dates
        start_date = dt.datetime(2022, 1, 1)
        dates = [start_date + dt.timedelta(days=i) for i in range(n_samples)]
        
        # Generate market regime data
        self.market_regime_data = {
            "date": dates,
            "regime": np.random.choice([r["regime"] for r in regimes], size=n_samples),
            "vix": np.random.uniform(10, 40, size=n_samples),
            "adx": np.random.uniform(10, 50, size=n_samples),
            "market_trend": np.random.uniform(-1, 1, size=n_samples),
            "sector_rotation": np.random.uniform(-1, 1, size=n_samples),
            "breadth": np.random.uniform(-1, 1, size=n_samples)
        }
        
        # Generate performance data for each strategy
        self.strategy_performance_data = {}
        
        for strategy in strategies:
            # Different strategies perform better in different regimes
            performance = np.zeros(n_samples)
            
            for i in range(n_samples):
                regime = self.market_regime_data["regime"][i]
                vix = self.market_regime_data["vix"][i]
                adx = self.market_regime_data["adx"][i]
                market_trend = self.market_regime_data["market_trend"][i]
                
                # Base performance with some randomness
                base_perf = np.random.normal(0, 0.02)
                
                # Strategy-specific adjustments
                if strategy == "MeanReversion":
                    # Mean reversion works well in range-bound markets and high volatility
                    if regime == 3:  # RANGE_BOUND
                        base_perf += 0.03
                    elif regime == 4:  # HIGH_VOLATILITY
                        base_perf += 0.02
                    elif regime == 2:  # TRENDING_BEARISH
                        base_perf -= 0.01
                    
                    # Higher VIX is better for mean reversion
                    base_perf += (vix - 20) * 0.001
                    
                    # Lower ADX is better for mean reversion (less trending)
                    base_perf += (30 - adx) * 0.001
                
                elif strategy == "TrendFollowing":
                    # Trend following works well in trending markets
                    if regime == 1:  # TRENDING_BULLISH
                        base_perf += 0.04
                    elif regime == 2:  # TRENDING_BEARISH
                        base_perf += 0.02
                    elif regime == 7:  # BULLISH_BREAKOUT
                        base_perf += 0.05
                    
                    # Higher ADX is better for trend following
                    base_perf += (adx - 20) * 0.001
                    
                    # Market trend direction affects performance
                    base_perf += market_trend * 0.02
                
                elif strategy == "VolatilityBreakout":
                    # Volatility breakout works well in high volatility and breakouts
                    if regime == 4:  # HIGH_VOLATILITY
                        base_perf += 0.03
                    elif regime == 6:  # BEARISH_BREAKDOWN
                        base_perf += 0.04
                    elif regime == 7:  # BULLISH_BREAKOUT
                        base_perf += 0.04
                    
                    # Higher VIX is better for volatility breakout
                    base_perf += (vix - 15) * 0.002
                
                elif strategy == "GapTrading":
                    # Gap trading works well in trending markets with some volatility
                    if regime == 1:  # TRENDING_BULLISH
                        base_perf += 0.02
                    elif regime == 4:  # HIGH_VOLATILITY
                        base_perf += 0.01
                    
                    # Moderate VIX is ideal for gap trading
                    base_perf += -0.001 * ((vix - 20) ** 2)
                
                # Ensure reasonable range of returns
                performance[i] = max(-0.1, min(0.1, base_perf))
            
            # Store performance data
            self.strategy_performance_data[strategy] = performance
        
        logger.info("Synthetic data generation completed")
        return self.strategy_performance_data
    
    def prepare_training_data(self, strategy_name):
        """
        Prepare training data for a specific strategy
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Tuple: X (features) and y (target) for model training
        """
        if not self.strategy_performance_data or not self.market_regime_data:
            logger.error("No data available. Generate synthetic data first.")
            return None, None
        
        if strategy_name not in self.strategy_performance_data:
            logger.error(f"Strategy {strategy_name} not found in performance data")
            return None, None
        
        # Prepare features (X)
        X = np.column_stack([
            self.market_regime_data["regime"],
            self.market_regime_data["vix"],
            self.market_regime_data["adx"],
            self.market_regime_data["market_trend"],
            self.market_regime_data["sector_rotation"],
            self.market_regime_data["breadth"]
        ])
        
        # Prepare target (y)
        y = self.strategy_performance_data[strategy_name]
        
        return X, y

def optimize_ml_model(strategy_name, X, y):
    """
    Optimize ML model for a specific strategy
    
    Args:
        strategy_name: Name of the strategy
        X: Features for training
        y: Target for training
        
    Returns:
        Dict: Optimized model parameters
    """
    logger.info(f"Optimizing ML model for strategy: {strategy_name}")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt']
    }
    
    # Create base model
    base_model = RandomForestRegressor(random_state=42)
    
    # Create grid search
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )
    
    # Fit grid search
    grid_search.fit(X, y)
    
    # Get best parameters
    best_params = grid_search.best_params_
    
    # Train model with best parameters
    best_model = RandomForestRegressor(**best_params, random_state=42)
    best_model.fit(X, y)
    
    # Evaluate model
    y_pred = best_model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    logger.info(f"Best parameters for {strategy_name}: {best_params}")
    logger.info(f"MSE: {mse:.6f}, R²: {r2:.6f}")
    
    # Get feature importance
    feature_names = ["regime", "vix", "adx", "market_trend", "sector_rotation", "breadth"]
    feature_importance = best_model.feature_importances_
    
    # Sort feature importance
    sorted_idx = feature_importance.argsort()[::-1]
    
    logger.info("Feature importance:")
    for i in sorted_idx:
        logger.info(f"{feature_names[i]}: {feature_importance[i]:.4f}")
    
    return best_params

def analyze_strategy_performance_by_regime(data_generator):
    """
    Analyze strategy performance by market regime
    
    Args:
        data_generator: MLStrategyOptimizerData object
    """
    logger.info("Analyzing strategy performance by market regime")
    
    if not data_generator.strategy_performance_data or not data_generator.market_regime_data:
        logger.error("No data available. Generate synthetic data first.")
        return
    
    # Get unique regimes
    regimes = np.unique(data_generator.market_regime_data["regime"])
    
    # Define regime names
    regime_names = {
        1: "TRENDING_BULLISH",
        2: "TRENDING_BEARISH",
        3: "RANGE_BOUND",
        4: "HIGH_VOLATILITY",
        5: "LOW_VOLATILITY",
        6: "BEARISH_BREAKDOWN",
        7: "BULLISH_BREAKOUT",
        8: "CONSOLIDATION"
    }
    
    # Analyze performance by regime
    performance_by_regime = {}
    
    for regime in regimes:
        regime_name = regime_names.get(regime, f"UNKNOWN_{regime}")
        regime_mask = data_generator.market_regime_data["regime"] == regime
        
        performance_by_regime[regime_name] = {}
        
        for strategy, performance in data_generator.strategy_performance_data.items():
            regime_performance = performance[regime_mask]
            
            if len(regime_performance) > 0:
                avg_performance = np.mean(regime_performance)
                std_performance = np.std(regime_performance)
                win_rate = np.sum(regime_performance > 0) / len(regime_performance)
                
                performance_by_regime[regime_name][strategy] = {
                    "avg_performance": avg_performance,
                    "std_performance": std_performance,
                    "win_rate": win_rate
                }
    
    # Print analysis
    logger.info("Strategy performance by market regime:")
    
    for regime, strategies in performance_by_regime.items():
        logger.info(f"\nRegime: {regime}")
        
        for strategy, metrics in strategies.items():
            logger.info(f"  {strategy}:")
            logger.info(f"    Avg Performance: {metrics['avg_performance']:.4f}")
            logger.info(f"    Std Deviation: {metrics['std_performance']:.4f}")
            logger.info(f"    Win Rate: {metrics['win_rate']:.2f}")
    
    # Find best strategy for each regime
    best_strategy_by_regime = {}
    
    for regime, strategies in performance_by_regime.items():
        best_strategy = max(strategies.items(), key=lambda x: x[1]["avg_performance"])
        best_strategy_by_regime[regime] = {
            "strategy": best_strategy[0],
            "avg_performance": best_strategy[1]["avg_performance"]
        }
    
    logger.info("\nBest strategy by market regime:")
    
    for regime, best in best_strategy_by_regime.items():
        logger.info(f"{regime}: {best['strategy']} (Avg Performance: {best['avg_performance']:.4f})")
    
    return performance_by_regime, best_strategy_by_regime

def optimize_ml_strategy_selector(config_dict):
    """
    Optimize ML strategy selector configuration
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Dict: Optimized ML strategy selector configuration
    """
    logger.info("Optimizing ML strategy selector configuration")
    
    # Create data generator
    data_generator = MLStrategyOptimizerData(config_dict)
    
    # Generate synthetic data
    data_generator.generate_synthetic_data(n_samples=2000)
    
    # Analyze strategy performance by regime
    analyze_strategy_performance_by_regime(data_generator)
    
    # Optimize ML models for each strategy
    optimized_params = {}
    
    for strategy in ["MeanReversion", "TrendFollowing", "VolatilityBreakout", "GapTrading"]:
        # Prepare training data
        X, y = data_generator.prepare_training_data(strategy)
        
        if X is not None and y is not None:
            # Optimize model
            best_params = optimize_ml_model(strategy, X, y)
            optimized_params[strategy] = best_params
    
    # Create optimized configuration
    ml_config = config_dict.get("ml_strategy_selector", {})
    
    # Update ML configuration
    ml_config.update({
        "ml_lookback_window": 30,  # Default, can be tuned further
        "ml_min_training_samples": 100,  # Default, can be tuned further
        "ml_retraining_frequency": 7,  # Default, can be tuned further
        "model_params": optimized_params
    })
    
    return ml_config

def main():
    """Main function to optimize the ML strategy selector"""
    logger.info("Starting ML Strategy Selector Optimization")
    
    try:
        # Load configuration
        config_dict = load_config()
        if not config_dict:
            logger.error("Failed to load configuration")
            return
        
        # Optimize ML strategy selector
        ml_config = optimize_ml_strategy_selector(config_dict)
        
        # Update configuration
        config_dict["ml_strategy_selector"] = ml_config
        
        # Save optimized configuration
        save_config(config_dict, "optimized_ml_config.yaml")
        
        logger.info("ML Strategy Selector Optimization completed")
        
    except Exception as e:
        logger.error(f"Error in ML Strategy Selector Optimization: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
