#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hybrid Model Direct Test
----------------------
This script implements and tests the hybrid trading model with enhanced features,
directly integrating with the existing system structure.
"""

import os
import sys
import logging
import datetime as dt
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import traceback
from typing import Dict, Any, List, Tuple

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
        logging.FileHandler('hybrid_model_direct_test.log')
    ]
)

logger = logging.getLogger("HybridModelDirectTest")

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
        config.trade_management = config_dict.get('trade_management', {})
        
        return config
    except Exception as e:
        logger.error(f"Error creating system config: {str(e)}")
        logger.error(traceback.format_exc())
        return None

class HybridMultiStrategySystem(MultiStrategySystem):
    """
    Hybrid Multi-Strategy System with enhanced features
    """
    
    def __init__(self, config):
        """Initialize the hybrid multi-strategy system"""
        super().__init__(config)
        
        # Store trade management config
        self.trade_management = config.trade_management if hasattr(config, 'trade_management') else {}
        
        # Historical performance metrics for strategies
        self.historical_performance = {
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
        
        # Initialize correlation matrix
        self.correlation_matrix = self._initialize_correlation_matrix()
        
        # Fix the sector performance error
        self._patch_market_analyzer()
        
        logger.info("Hybrid Multi-Strategy System initialized")
    
    def _initialize_correlation_matrix(self) -> Dict[str, Dict[str, float]]:
        """Initialize correlation matrix between symbols"""
        symbols = [stock.symbol for stock in self.config.stocks]
        correlation_matrix = {}
        
        # Create empty matrix
        for symbol in symbols:
            correlation_matrix[symbol] = {}
            for other_symbol in symbols:
                if symbol == other_symbol:
                    correlation_matrix[symbol][other_symbol] = 1.0
                else:
                    # Default correlation of 0.5 for technology stocks, 0.3 for others
                    sector = None
                    for stock in self.config.stocks:
                        if stock.symbol == symbol:
                            sector = stock.sector
                            break
                    
                    correlation_matrix[symbol][other_symbol] = 0.5 if sector == "Technology" else 0.3
        
        return correlation_matrix
    
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
            
            # Rest of the original method remains unchanged
            return original_method(self, base_regime, adx, vix, trend_direction, 
                                 breadth_indicators, intermarket_indicators,
                                 sector_performance, sentiment_indicators)
        
        # Replace the method
        self.market_analyzer._determine_sub_regime = patched_method.__get__(self.market_analyzer, type(self.market_analyzer))
        
        logger.info("Fixed sector performance error by patching _determine_sub_regime method")
    
    def calculate_position_size(self, signal: Signal, available_capital: float) -> float:
        """Calculate position size with enhanced methods"""
        # Get base position size from parent class
        base_size = super().calculate_position_size(signal, available_capital)
        
        # Get position sizing config
        position_sizing_config = self.config.position_sizing_config
        
        # Apply Kelly criterion if enabled
        if position_sizing_config.get('kelly_criterion_factor'):
            kelly_factor = position_sizing_config.get('kelly_criterion_factor', 0.3)
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
        
        # Apply regime-based sizing if enabled
        if position_sizing_config.get('regime_based_sizing'):
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
        max_position_size = position_sizing_config.get('max_position_size', 0.05) * available_capital
        min_position_size = position_sizing_config.get('min_position_size', 0.002) * available_capital
        
        return max(min_position_size, min(base_size, max_position_size))
    
    def should_exit_position(self, position_id: str, current_price: float, timestamp: dt.datetime) -> Tuple[bool, str]:
        """Enhanced position exit logic with trailing stops and partial profit taking"""
        # Check basic exit conditions from parent class
        should_exit, reason = super().should_exit_position(position_id, current_price, timestamp)
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
        
        # Check for trailing stop
        if self.trade_management.get('trailing_stop_activation_pct'):
            activation_pct = self.trade_management.get('trailing_stop_activation_pct', 1.0)
            distance_pct = self.trade_management.get('trailing_stop_distance_pct', 0.5)
            
            # Calculate current profit percentage
            if direction == TradeDirection.LONG:
                profit_pct = (current_price - entry_price) / entry_price * 100
                if profit_pct >= activation_pct:
                    # Calculate trailing stop level
                    new_stop = current_price * (1 - distance_pct / 100)
                    # Only update if it would raise the stop level
                    if new_stop > current_stop:
                        self.positions[position_id].stop_loss = new_stop
                        self.logger.info(f"Updated trailing stop for {symbol} to {new_stop}")
            else:  # SHORT
                profit_pct = (entry_price - current_price) / entry_price * 100
                if profit_pct >= activation_pct:
                    # Calculate trailing stop level
                    new_stop = current_price * (1 + distance_pct / 100)
                    # Only update if it would lower the stop level
                    if new_stop < current_stop:
                        self.positions[position_id].stop_loss = new_stop
                        self.logger.info(f"Updated trailing stop for {symbol} to {new_stop}")
        
        # Check for partial profit taking
        if self.trade_management.get('partial_profit_taking'):
            # Calculate risk (R)
            risk = abs(entry_price - current_stop)
            if risk > 0:
                # Calculate current profit in terms of R
                if direction == TradeDirection.LONG:
                    current_profit_r = (current_price - entry_price) / risk
                else:  # SHORT
                    current_profit_r = (entry_price - current_price) / risk
                
                # Check profit levels
                profit_levels = {
                    self.trade_management.get('partial_profit_level_1', 1.5): 0.25,
                    self.trade_management.get('partial_profit_level_2', 2.5): 0.25
                }
                
                for profit_level_r, exit_percentage in sorted(profit_levels.items()):
                    if current_profit_r >= profit_level_r:
                        # Reduce position size
                        new_size = position.size * (1 - exit_percentage)
                        old_size = position.size
                        
                        # Calculate profit
                        if direction == TradeDirection.LONG:
                            profit = (current_price - entry_price) * (old_size - new_size)
                        else:
                            profit = (entry_price - current_price) * (old_size - new_size)
                        
                        # Update position size
                        self.positions[position_id].size = new_size
                        
                        # Update realized PnL
                        self.positions[position_id].realized_pnl += profit
                        
                        self.logger.info(f"Took partial profits on {symbol}: {exit_percentage*100}%, Profit: ${profit:.2f}")
                        
                        # If position is completely closed
                        if new_size <= 0:
                            return True, "Took full profits"
        
        # Check for time-based exit
        if self.trade_management.get('time_based_exit'):
            max_days = self.trade_management.get('max_holding_period_days', 10)
            entry_time = position.entry_time
            
            holding_period = (timestamp - entry_time).total_seconds() / (24 * 60 * 60)
            if holding_period >= max_days:
                return True, f"Maximum holding period of {max_days} days reached"
        
        return False, ""
    
    def generate_signals(self, timestamp: dt.datetime, symbols: List[str]) -> Dict[str, List[Signal]]:
        """Generate trading signals with enhanced filtering"""
        # Get base signals from parent class
        base_signals = super().generate_signals(timestamp, symbols)
        
        # Apply enhanced filtering
        filtered_signals = {}
        
        for symbol, signals in base_signals.items():
            if not signals:
                filtered_signals[symbol] = []
                continue
            
            # Get signal quality filter parameters
            min_profit_factor = self.config.signal_quality_filters.get('min_profit_factor', 1.5)
            min_sharpe_ratio = self.config.signal_quality_filters.get('min_sharpe_ratio', 0.8)
            
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
                if profit_factor >= min_profit_factor and sharpe_ratio >= min_sharpe_ratio:
                    quality_filtered.append(signal)
            
            filtered_signals[symbol] = quality_filtered
        
        return filtered_signals

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

def plot_drawdowns(results: Dict[str, BacktestResult], save_path: str = None):
    """Plot drawdown curves for all results"""
    plt.figure(figsize=(12, 8))
    
    for name, result in results.items():
        drawdown_curve = result.equity_curve['drawdown']
        plt.plot(result.equity_curve.index, drawdown_curve, label=name)
    
    plt.title('Drawdown Comparison')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
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
    
    # Load configurations
    original_config_dict = load_config('multi_strategy_config.yaml')
    optimized_config_dict = load_config('further_optimized_config.yaml')
    hybrid_config_dict = load_config('hybrid_optimized_config.yaml')
    
    # Create system configs
    original_config = create_system_config(original_config_dict.copy())
    optimized_config = create_system_config(optimized_config_dict.copy())
    hybrid_config = create_system_config(hybrid_config_dict.copy())
    
    if not all([original_config, optimized_config, hybrid_config]):
        logger.error("Failed to create one or more system configs")
        return
    
    # Create systems
    logger.info("Creating original system...")
    original_system = MultiStrategySystem(original_config)
    
    logger.info("Creating optimized system...")
    optimized_system = MultiStrategySystem(optimized_config)
    
    logger.info("Creating hybrid system...")
    hybrid_system = HybridMultiStrategySystem(hybrid_config)
    
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
    plot_drawdowns(results, 'hybrid_drawdowns.png')
    
    logger.info("Hybrid model test completed. Results saved to hybrid_results.json and hybrid_comparison.html")

if __name__ == "__main__":
    main()
