#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Position Sizing Optimizer
------------------------
This script focuses on optimizing position sizing parameters to improve
the Sharpe ratio and overall risk-adjusted returns.
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

# Import the multi-strategy system
from multi_strategy_system import (
    MultiStrategySystem, SystemConfig, Signal, MarketRegime, 
    BacktestResult, StockConfig, MarketState
)

# Import enhanced trading functions
from enhanced_trading_functions import calculate_adaptive_position_size

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('position_sizing_optimizer.log')
    ]
)

logger = logging.getLogger("PositionSizingOptimizer")

def load_config():
    """Load configuration from YAML file"""
    try:
        with open('multi_strategy_config.yaml', 'r') as file:
            config_dict = yaml.safe_load(file)
        return config_dict
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return None

def save_config(config_dict, filename='optimized_position_sizing.yaml'):
    """Save configuration to YAML file"""
    try:
        with open(filename, 'w') as file:
            yaml.dump(config_dict, file, default_flow_style=False)
        logger.info(f"Configuration saved to {filename}")
        return True
    except Exception as e:
        logger.error(f"Error saving configuration: {str(e)}")
        return False

class PositionSizingSimulator:
    """Class to simulate position sizing for optimization"""
    
    def __init__(self, config_dict):
        """Initialize the position sizing simulator"""
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
        Generate synthetic signals for testing position sizing
        
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
        Generate synthetic candle data for testing position sizing
        
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
    
    def generate_synthetic_market_state(self):
        """
        Generate synthetic market state for testing position sizing
        
        Returns:
            MarketState: Synthetic market state
        """
        logger.info("Generating synthetic market state")
        
        # Create market state
        market_state = MarketState()
        
        # Set random regime
        market_state.regime = np.random.choice([
            MarketRegime.TRENDING_BULLISH,
            MarketRegime.TRENDING_BEARISH,
            MarketRegime.RANGE_BOUND,
            MarketRegime.HIGH_VOLATILITY,
            MarketRegime.LOW_VOLATILITY,
            MarketRegime.BEARISH_BREAKDOWN,
            MarketRegime.BULLISH_BREAKOUT,
            MarketRegime.CONSOLIDATION
        ])
        
        # Set random sub-regime
        market_state.sub_regime = np.random.choice([
            "Bullish Trend",
            "Bearish Trend",
            "Bullish Consolidation",
            "Bearish Consolidation",
            "Volatility Expansion",
            "Volatility Contraction"
        ])
        
        # Set random indicators
        market_state.vix = np.random.uniform(10, 40)
        market_state.adx = np.random.uniform(10, 50)
        market_state.market_trend = np.random.uniform(-1, 1)
        
        logger.info(f"Generated market state with regime: {market_state.regime}")
        return market_state

def simulate_portfolio_performance(signals, position_sizing_config, market_state, candle_data, logger):
    """
    Simulate portfolio performance with given position sizing parameters
    
    Args:
        signals: List of trading signals
        position_sizing_config: Position sizing configuration
        market_state: Market state
        candle_data: Candle data
        logger: Logger
        
    Returns:
        Dict: Performance metrics
    """
    logger.info("Simulating portfolio performance")
    
    # Initialize portfolio
    initial_equity = 100000.0
    current_equity = initial_equity
    positions = []
    trades = []
    
    # Process each signal
    for signal in signals:
        # Calculate position size
        position_size = calculate_adaptive_position_size(
            signal=signal,
            market_state=market_state,
            candle_data=candle_data,
            current_equity=current_equity,
            position_sizing_config=position_sizing_config,
            logger=logger
        )
        
        # Calculate number of shares
        shares = int(position_size / signal.entry_price)
        
        if shares > 0:
            # Create position
            position = {
                "symbol": signal.symbol,
                "entry_price": signal.entry_price,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "direction": signal.direction,
                "shares": shares,
                "position_size": position_size,
                "entry_date": signal.timestamp,
                "score": getattr(signal, 'score', 0.5),
                "predicted_performance": signal.metadata.get('predicted_performance', 0) if hasattr(signal, 'metadata') else 0
            }
            
            positions.append(position)
            
            # Simulate trade outcome
            # Use predicted performance as a bias for the random outcome
            predicted_perf = position["predicted_performance"]
            score = position["score"]
            
            # Combine predicted performance and score to create a bias
            bias = (predicted_perf * 2) + (score - 0.5)
            
            # Generate random outcome with bias
            random_outcome = np.random.normal(bias, 0.05)
            
            # Determine if trade is a win or loss
            if random_outcome > 0:
                # Win
                profit_pct = np.random.uniform(0.01, 0.1)  # 1-10% profit
                profit = position_size * profit_pct
                current_equity += profit
                
                trade = {
                    "symbol": position["symbol"],
                    "direction": position["direction"],
                    "entry_price": position["entry_price"],
                    "exit_price": position["entry_price"] * (1 + profit_pct * position["direction"]),
                    "shares": position["shares"],
                    "profit": profit,
                    "profit_pct": profit_pct,
                    "win": True
                }
            else:
                # Loss
                loss_pct = np.random.uniform(0.01, 0.05)  # 1-5% loss
                loss = position_size * loss_pct
                current_equity -= loss
                
                trade = {
                    "symbol": position["symbol"],
                    "direction": position["direction"],
                    "entry_price": position["entry_price"],
                    "exit_price": position["entry_price"] * (1 - loss_pct * position["direction"]),
                    "shares": position["shares"],
                    "profit": -loss,
                    "profit_pct": -loss_pct,
                    "win": False
                }
            
            trades.append(trade)
    
    # Calculate performance metrics
    if not trades:
        return {"error": "No trades executed"}
    
    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t["win"])
    losing_trades = total_trades - winning_trades
    
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    profits = [t["profit"] for t in trades if t["win"]]
    losses = [t["profit"] for t in trades if not t["win"]]
    
    avg_profit = np.mean(profits) if profits else 0
    avg_loss = np.mean(losses) if losses else 0
    
    profit_factor = abs(sum(profits) / sum(losses)) if sum(losses) != 0 else float('inf')
    
    # Calculate returns
    total_return = current_equity - initial_equity
    total_return_pct = (total_return / initial_equity) * 100
    
    # Calculate Sharpe ratio (simplified)
    returns = [t["profit"] / initial_equity for t in trades]
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
    
    # Calculate max drawdown
    equity_curve = [initial_equity]
    for trade in trades:
        equity_curve.append(equity_curve[-1] + trade["profit"])
    
    max_drawdown = 0
    peak = equity_curve[0]
    
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        drawdown = (peak - equity) / peak
        max_drawdown = max(max_drawdown, drawdown)
    
    max_drawdown_pct = max_drawdown * 100
    
    # Return metrics
    return {
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": win_rate,
        "avg_profit": avg_profit,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "total_return": total_return,
        "total_return_pct": total_return_pct,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown_pct": max_drawdown_pct,
        "final_equity": current_equity
    }

def optimize_position_sizing(config_dict):
    """
    Optimize position sizing parameters
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Dict: Optimized position sizing parameters
    """
    logger.info("Optimizing position sizing parameters")
    
    # Create position sizing simulator
    simulator = PositionSizingSimulator(config_dict)
    
    # Generate synthetic data
    signals = simulator.generate_synthetic_signals(n_signals=200)
    candle_data = simulator.generate_synthetic_candle_data()
    market_state = simulator.generate_synthetic_market_state()
    
    # Parameters to optimize
    param_grid = {
        "base_risk_per_trade": [0.005, 0.01, 0.015, 0.02, 0.025],
        "max_position_size": [0.03, 0.05, 0.07, 0.1],
        "min_position_size": [0.002, 0.005, 0.01],
        "volatility_adjustment": [True, False],
        "signal_strength_adjustment": [True, False]
    }
    
    # Generate parameter combinations
    param_combinations = []
    for base_risk in param_grid["base_risk_per_trade"]:
        for max_pos in param_grid["max_position_size"]:
            for min_pos in param_grid["min_position_size"]:
                for vol_adj in param_grid["volatility_adjustment"]:
                    for sig_adj in param_grid["signal_strength_adjustment"]:
                        if min_pos < max_pos:  # Ensure min < max
                            params = {
                                "base_risk_per_trade": base_risk,
                                "max_position_size": max_pos,
                                "min_position_size": min_pos,
                                "volatility_adjustment": vol_adj,
                                "signal_strength_adjustment": sig_adj
                            }
                            param_combinations.append(params)
    
    # Limit number of combinations to test
    max_combinations = 20
    if len(param_combinations) > max_combinations:
        logger.info(f"Limiting to {max_combinations} parameter combinations")
        param_combinations = param_combinations[:max_combinations]
    
    # Test each parameter combination
    best_sharpe = 0
    best_params = None
    best_metrics = None
    
    for params in param_combinations:
        # Simulate portfolio performance
        metrics = simulate_portfolio_performance(
            signals=signals,
            position_sizing_config=params,
            market_state=market_state,
            candle_data=candle_data,
            logger=logger
        )
        
        if "error" in metrics:
            logger.warning(f"Error simulating performance: {metrics['error']}")
            continue
        
        logger.info(f"Testing parameters: {params}")
        logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}, "
                   f"Win Rate: {metrics['win_rate']:.2f}, "
                   f"Return: {metrics['total_return_pct']:.2f}%, "
                   f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        
        # Check if this is the best combination so far
        if metrics['sharpe_ratio'] > best_sharpe:
            best_sharpe = metrics['sharpe_ratio']
            best_params = params
            best_metrics = metrics
            
            logger.info(f"New best parameters found: {params}")
            logger.info(f"Sharpe Ratio: {best_sharpe:.2f}")
    
    if best_params:
        logger.info("=== Best Position Sizing Parameters ===")
        logger.info(f"Parameters: {best_params}")
        logger.info(f"Sharpe Ratio: {best_metrics['sharpe_ratio']:.2f}")
        logger.info(f"Win Rate: {best_metrics['win_rate']:.2f}")
        logger.info(f"Total Return: {best_metrics['total_return_pct']:.2f}%")
        logger.info(f"Max Drawdown: {best_metrics['max_drawdown_pct']:.2f}%")
        logger.info(f"Profit Factor: {best_metrics['profit_factor']:.2f}")
    else:
        logger.warning("No valid parameter combination found")
    
    return best_params

def main():
    """Main function to optimize position sizing"""
    logger.info("Starting Position Sizing Optimization")
    
    try:
        # Load configuration
        config_dict = load_config()
        if not config_dict:
            logger.error("Failed to load configuration")
            return
        
        # Optimize position sizing
        position_params = optimize_position_sizing(config_dict)
        
        if position_params:
            # Update configuration
            config_dict["position_sizing_config"] = position_params
            
            # Save optimized configuration
            save_config(config_dict, "optimized_position_sizing.yaml")
            
            logger.info("Position Sizing Optimization completed")
        else:
            logger.error("Position Sizing Optimization failed")
        
    except Exception as e:
        logger.error(f"Error in Position Sizing Optimization: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
