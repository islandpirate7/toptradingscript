#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hybrid Model Test
---------------
This script implements and tests the hybrid trading model with enhanced features.
"""

import os
import sys
import logging
import datetime as dt
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
import json
import copy

# Import the multi-strategy system
from multi_strategy_system import (
    MultiStrategySystem, SystemConfig, Signal, MarketRegime, 
    BacktestResult, StockConfig, MarketState, TradeDirection,
    SignalStrength, PositionState, CandleData
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('hybrid_model_test.log')
    ]
)

logger = logging.getLogger("HybridModelTest")

def fix_volatility_breakout_strategy():
    """
    Fix the premature return statement in the VolatilityBreakout strategy's generate_signals method
    """
    import re
    
    # Path to the multi_strategy_system.py file
    file_path = 'multi_strategy_system.py'
    
    # Read the file
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Find the VolatilityBreakoutStrategy class and its generate_signals method
    pattern = r'(class VolatilityBreakoutStrategy.*?def generate_signals.*?signals\.append\(signal\))\s*\n\s*return signals\s*\n(.*?)def'
    
    # Replace the premature return with a commented version
    modified_content = re.sub(pattern, r'\1\n            # return signals  # Commented out premature return\n\2def', content, flags=re.DOTALL)
    
    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.write(modified_content)
    
    logger.info("Fixed VolatilityBreakout strategy by commenting out premature return statement")

def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_file, 'r') as file:
        config_dict = yaml.safe_load(file)
    return config_dict

def create_enhanced_system(config_file: str) -> MultiStrategySystem:
    """
    Create an enhanced trading system with the hybrid optimizations
    """
    # Load configuration
    config_dict = load_config(config_file)
    
    # Create a system with the loaded configuration
    system = MultiStrategySystem(config_dict)
    
    # Apply enhancements
    enhance_system(system)
    
    return system

def enhance_system(system: MultiStrategySystem) -> None:
    """
    Apply enhancements to the trading system
    """
    # Store original methods for reference
    original_calculate_position_size = system.calculate_position_size
    original_should_exit_position = system.should_exit_position
    original_generate_signals = system.generate_signals
    
    # Historical performance metrics for strategies
    historical_performance = {
        "MeanReversion": {
            "win_rate": 0.69,
            "profit_factor": 6.33,
            "sharpe_ratio": 24.80,
            "avg_win_pct": 2.1,
            "avg_loss_pct": 1.0
        },
        "TrendFollowing": {
            "win_rate": 0.55,
            "profit_factor": 2.1,
            "sharpe_ratio": 1.8,
            "avg_win_pct": 2.5,
            "avg_loss_pct": 1.2
        },
        "VolatilityBreakout": {
            "win_rate": 0.60,
            "profit_factor": 3.2,
            "sharpe_ratio": 2.1,
            "avg_win_pct": 3.0,
            "avg_loss_pct": 1.5
        },
        "GapTrading": {
            "win_rate": 0.50,
            "profit_factor": 1.8,
            "sharpe_ratio": 1.5,
            "avg_win_pct": 2.8,
            "avg_loss_pct": 1.6
        }
    }
    
    # Add historical performance to the system
    system.historical_performance = historical_performance
    
    # Initialize correlation matrix
    system.correlation_matrix = initialize_correlation_matrix(system)
    
    # Enhanced position sizing
    def enhanced_calculate_position_size(self, signal, available_capital):
        # Get base position size from original method
        base_size = original_calculate_position_size(signal, available_capital)
        
        # Apply Kelly criterion
        kelly_factor = 0.3  # Conservative Kelly factor
        strategy = signal.strategy
        
        if strategy in self.historical_performance:
            metrics = self.historical_performance[strategy]
            win_rate = metrics.get('win_rate', 0.5)
            avg_win_pct = metrics.get('avg_win_pct', 2.0)
            avg_loss_pct = metrics.get('avg_loss_pct', 1.0)
            
            # Calculate Kelly position size
            q = 1 - win_rate
            b = avg_win_pct / avg_loss_pct
            kelly = (win_rate * b - q) / b
            
            # Apply fractional Kelly and ensure it's within reasonable bounds
            kelly = max(0, min(kelly * kelly_factor, 0.2))
            
            kelly_size = kelly * available_capital
            
            # Blend with base size
            base_size = (base_size + kelly_size) / 2
        
        # Apply regime-based sizing
        market_state = self.market_analyzer.get_current_market_state()
        regime = market_state.regime
        direction = signal.direction
        
        # Default multiplier
        multiplier = 1.0
        
        # Adjust based on regime and direction alignment
        if regime == MarketRegime.TRENDING_BULLISH:
            if direction == TradeDirection.LONG:
                multiplier = 1.3
            else:
                multiplier = 0.7
        elif regime == MarketRegime.TRENDING_BEARISH:
            if direction == TradeDirection.SHORT:
                multiplier = 1.3
            else:
                multiplier = 0.7
        elif regime == MarketRegime.HIGH_VOLATILITY:
            multiplier = 0.8  # Reduce size in high volatility
        
        # Apply multiplier
        base_size *= multiplier
        
        # Apply min/max constraints
        max_position_size = 0.045 * available_capital  # From hybrid config
        min_position_size = 0.002 * available_capital  # From hybrid config
        
        return max(min_position_size, min(base_size, max_position_size))
    
    # Enhanced exit position logic
    def enhanced_should_exit_position(self, position_id, current_price, timestamp):
        # Check basic exit conditions from original method
        should_exit, reason = original_should_exit_position(position_id, current_price, timestamp)
        if should_exit:
            return True, reason
        
        # Get position
        if position_id not in self.positions:
            return False, ""
        
        position = self.positions[position_id]
        symbol = position.symbol
        entry_price = position.entry_price
        direction = position.direction
        current_stop = position.stop_loss
        
        # Trailing stop parameters
        activation_pct = 1.0  # 1% profit to activate trailing stop
        distance_pct = 0.5    # 0.5% trailing stop distance
        
        # Check for trailing stop
        # Calculate current profit percentage
        if direction == TradeDirection.LONG:
            profit_pct = (current_price - entry_price) / entry_price * 100
            if profit_pct >= activation_pct:
                # Calculate trailing stop level
                new_stop = current_price * (1 - distance_pct / 100)
                # Only update if it would raise the stop level
                if new_stop > current_stop:
                    self.positions[position_id].stop_loss = new_stop
                    logger.info(f"Updated trailing stop for {symbol} to {new_stop}")
        else:  # SHORT
            profit_pct = (entry_price - current_price) / entry_price * 100
            if profit_pct >= activation_pct:
                # Calculate trailing stop level
                new_stop = current_price * (1 + distance_pct / 100)
                # Only update if it would lower the stop level
                if new_stop < current_stop:
                    self.positions[position_id].stop_loss = new_stop
                    logger.info(f"Updated trailing stop for {symbol} to {new_stop}")
        
        # Check for time-based exit
        entry_time = position.entry_time
        max_holding_period_days = 10  # Maximum holding period in days
        
        holding_period = (timestamp - entry_time).total_seconds() / (24 * 60 * 60)
        if holding_period >= max_holding_period_days:
            return True, f"Maximum holding period of {max_holding_period_days} days reached"
        
        return False, ""
    
    # Enhanced signal generation with quality filtering
    def enhanced_generate_signals(self, timestamp, symbols):
        # Get base signals from original method
        base_signals = original_generate_signals(timestamp, symbols)
        
        # Apply enhanced filtering
        filtered_signals = {}
        
        for symbol, signals in base_signals.items():
            if not signals:
                filtered_signals[symbol] = []
                continue
            
            # Filter signals by quality
            quality_filtered = []
            for signal in signals:
                strategy = signal.strategy
                
                # Skip if we don't have historical data for this strategy
                if strategy not in self.historical_performance:
                    quality_filtered.append(signal)
                    continue
                
                # Get historical metrics
                metrics = self.historical_performance[strategy]
                profit_factor = metrics.get('profit_factor', 0.0)
                sharpe_ratio = metrics.get('sharpe_ratio', 0.0)
                
                # Filter based on minimum thresholds
                min_profit_factor = 1.5
                min_sharpe_ratio = 0.8
                
                if profit_factor >= min_profit_factor and sharpe_ratio >= min_sharpe_ratio:
                    quality_filtered.append(signal)
            
            filtered_signals[symbol] = quality_filtered
        
        return filtered_signals
    
    # Bind enhanced methods to the system
    system.calculate_position_size = enhanced_calculate_position_size.__get__(system, type(system))
    system.should_exit_position = enhanced_should_exit_position.__get__(system, type(system))
    system.generate_signals = enhanced_generate_signals.__get__(system, type(system))
    
    # Fix the sector performance error
    patch_market_analyzer(system)
    
    logger.info("Applied enhancements to the trading system")

def initialize_correlation_matrix(system: MultiStrategySystem) -> Dict[str, Dict[str, float]]:
    """Initialize correlation matrix between symbols"""
    symbols = [stock.symbol for stock in system.config.stocks]
    correlation_matrix = {}
    
    # Create empty matrix
    for symbol in symbols:
        correlation_matrix[symbol] = {}
        for other_symbol in symbols:
            if symbol == other_symbol:
                correlation_matrix[symbol][other_symbol] = 1.0
            else:
                # Default correlation of 0.5 for technology stocks, 0.3 for others
                # This will be updated with real data during operation
                sector = None
                for stock in system.config.stocks:
                    if stock.symbol == symbol:
                        sector = stock.sector
                        break
                
                correlation_matrix[symbol][other_symbol] = 0.5 if sector == "Technology" else 0.3
    
    return correlation_matrix

def patch_market_analyzer(system: MultiStrategySystem):
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
        
        # Rest of the original method remains unchanged
        return original_method(self, base_regime, adx, vix, trend_direction, 
                             breadth_indicators, intermarket_indicators,
                             sector_performance, sentiment_indicators)
    
    # Replace the method
    system.market_analyzer._determine_sub_regime = patched_method.__get__(system.market_analyzer, type(system.market_analyzer))
    
    logger.info("Fixed sector performance error by patching _determine_sub_regime method")

def run_backtest(system: MultiStrategySystem, start_date: dt.datetime, end_date: dt.datetime) -> BacktestResult:
    """Run backtest for a trading system"""
    result = system.run_backtest(start_date, end_date)
    return result

def compare_results(results: Dict[str, BacktestResult]) -> pd.DataFrame:
    """Compare backtest results and return a DataFrame"""
    comparison = {}
    
    for name, result in results.items():
        comparison[name] = {
            "Total Return (%)": result.total_return_pct,
            "Annualized Return (%)": result.annualized_return_pct,
            "Sharpe Ratio": result.sharpe_ratio,
            "Max Drawdown (%)": result.max_drawdown_pct,
            "Win Rate (%)": result.win_rate * 100,
            "Profit Factor": result.profit_factor,
            "Total Trades": result.total_trades,
            "Avg Trade Duration (days)": result.avg_trade_duration_days
        }
    
    # Convert to DataFrame
    comparison_df = pd.DataFrame(comparison)
    return comparison_df

def plot_equity_curves(results: Dict[str, BacktestResult], save_path: str = None):
    """Plot equity curves for all results"""
    plt.figure(figsize=(12, 8))
    
    for name, result in results.items():
        equity_curve = result.equity_curve
        plt.plot(equity_curve.index, equity_curve['equity'], label=name)
    
    plt.title('Equity Curve Comparison')
    plt.xlabel('Date')
    plt.ylabel('Equity ($)')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.tight_layout()
    plt.show()

def generate_html_report(comparison_df: pd.DataFrame, results: Dict[str, BacktestResult], output_file: str):
    """Generate HTML report with comparison results"""
    # Create HTML file
    with open(output_file, 'w') as f:
        f.write('<html><head>')
        f.write('<title>Trading Model Comparison</title>')
        f.write('<style>body { font-family: Arial, sans-serif; padding: 20px; } ')
        f.write('table { border-collapse: collapse; width: 100%; margin-bottom: 20px; } ')
        f.write('th, td { border: 1px solid #ddd; padding: 8px; text-align: left; } ')
        f.write('th { background-color: #f2f2f2; } ')
        f.write('tr:nth-child(even) { background-color: #f9f9f9; } ')
        f.write('h1, h2, h3 { color: #333; } ')
        f.write('</style>')
        f.write('</head><body>')
        
        # Header
        f.write('<h1>Trading Model Comparison Report</h1>')
        f.write(f'<p>Generated on: {dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>')
        
        # Comparison table
        f.write('<h2>Performance Metrics</h2>')
        f.write(comparison_df.to_html(float_format='%.2f'))
        
        # Strategy performance
        f.write('<h2>Strategy Performance</h2>')
        
        for name, result in results.items():
            f.write(f'<h3>{name}</h3>')
            
            # Calculate basic strategy metrics
            strategy_trades = {}
            for trade in result.trades:
                strategy = trade.strategy
                if strategy not in strategy_trades:
                    strategy_trades[strategy] = []
                strategy_trades[strategy].append(trade)
            
            strategy_data = {}
            for strategy, trades in strategy_trades.items():
                if not trades:
                    continue
                
                win_count = sum(1 for t in trades if t.pnl > 0)
                total_count = len(trades)
                win_rate = win_count / total_count if total_count > 0 else 0
                
                total_profit = sum(t.pnl for t in trades if t.pnl > 0)
                total_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
                profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
                
                total_pnl = sum(t.pnl for t in trades)
                
                strategy_data[strategy] = {
                    'Total Trades': total_count,
                    'Win Rate (%)': win_rate * 100,
                    'Profit Factor': profit_factor,
                    'Total PnL ($)': total_pnl
                }
            
            if strategy_data:
                strategy_df = pd.DataFrame(strategy_data).T
                f.write(strategy_df.to_html(float_format='%.2f'))
            else:
                f.write('<p>No strategy-specific data available</p>')
        
        # Close HTML
        f.write('</body></html>')
    
    logger.info(f"HTML report generated: {output_file}")

def main():
    """Main function to test the hybrid model"""
    # Fix the VolatilityBreakout strategy
    fix_volatility_breakout_strategy()
    
    # Define date range for backtest
    start_date = dt.datetime(2023, 1, 1)
    end_date = dt.datetime(2023, 12, 31)
    
    # Create original system
    logger.info("Creating original system...")
    original_system = MultiStrategySystem(load_config('multi_strategy_config.yaml'))
    
    # Create optimized system
    logger.info("Creating optimized system...")
    optimized_system = MultiStrategySystem(load_config('further_optimized_config.yaml'))
    
    # Create hybrid system
    logger.info("Creating hybrid system...")
    hybrid_system = create_enhanced_system('hybrid_optimized_config.yaml')
    
    # Run backtests
    logger.info("Running backtest for original system...")
    original_result = run_backtest(original_system, start_date, end_date)
    
    logger.info("Running backtest for optimized system...")
    optimized_result = run_backtest(optimized_system, start_date, end_date)
    
    logger.info("Running backtest for hybrid system...")
    hybrid_result = run_backtest(hybrid_system, start_date, end_date)
    
    # Collect results
    results = {
        "Original": original_result,
        "Optimized": optimized_result,
        "Hybrid": hybrid_result
    }
    
    # Compare results
    comparison_df = compare_results(results)
    print("\nPerformance Comparison:")
    print(comparison_df)
    
    # Save results to JSON
    results_dict = {}
    for name, result in results.items():
        results_dict[name] = {
            "total_return_pct": result.total_return_pct,
            "annualized_return_pct": result.annualized_return_pct,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown_pct": result.max_drawdown_pct,
            "win_rate": result.win_rate,
            "profit_factor": result.profit_factor,
            "total_trades": result.total_trades
        }
    
    with open('hybrid_results.json', 'w') as f:
        json.dump(results_dict, f, indent=4)
    
    # Generate HTML report
    generate_html_report(comparison_df, results, 'hybrid_comparison.html')
    
    # Plot results
    plot_equity_curves(results, 'hybrid_equity_curves.png')
    
    logger.info("Hybrid model test completed. Results saved to hybrid_results.json and hybrid_comparison.html")

if __name__ == "__main__":
    main()
