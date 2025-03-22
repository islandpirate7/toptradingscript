#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compare Hybrid Model
-------------------
This script compares the performance of the original, further optimized, and hybrid trading models.
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

# Import trading systems
from multi_strategy_system import MultiStrategySystem, SystemConfig, BacktestResult
from hybrid_strategy_system import HybridStrategySystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('compare_models.log')
    ]
)

logger = logging.getLogger("CompareModels")

def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_file, 'r') as file:
        config_dict = yaml.safe_load(file)
    return config_dict

def create_system(config_dict: Dict[str, Any], system_class=MultiStrategySystem) -> MultiStrategySystem:
    """Create a trading system from configuration dictionary"""
    # Create system directly with the config dictionary
    system = system_class(config_dict)
    return system

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
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    
    # Create HTML file
    with open(output_file, 'w') as f:
        f.write('<html><head>')
        f.write('<title>Trading Model Comparison</title>')
        f.write('<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">')
        f.write('<style>body { padding: 20px; } .table { margin-top: 20px; }</style>')
        f.write('</head><body>')
        
        # Header
        f.write('<div class="container">')
        f.write('<h1>Trading Model Comparison Report</h1>')
        f.write(f'<p>Generated on: {dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>')
        
        # Comparison table
        f.write('<h2>Performance Metrics</h2>')
        f.write(comparison_df.to_html(classes='table table-striped table-hover', float_format='%.2f'))
        
        # Create equity curve plot
        fig1 = go.Figure()
        for name, result in results.items():
            equity_curve = result.equity_curve
            fig1.add_trace(go.Scatter(
                x=equity_curve.index,
                y=equity_curve['equity'],
                mode='lines',
                name=name
            ))
        
        fig1.update_layout(
            title='Equity Curve Comparison',
            xaxis_title='Date',
            yaxis_title='Equity ($)',
            legend_title='Model',
            height=600
        )
        
        # Create drawdown plot
        fig2 = go.Figure()
        for name, result in results.items():
            drawdown_curve = result.equity_curve['drawdown']
            fig2.add_trace(go.Scatter(
                x=result.equity_curve.index,
                y=drawdown_curve,
                mode='lines',
                name=name
            ))
        
        fig2.update_layout(
            title='Drawdown Comparison',
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            legend_title='Model',
            height=600
        )
        
        # Add plots to HTML
        f.write('<h2>Equity Curves</h2>')
        f.write(fig1.to_html(full_html=False))
        
        f.write('<h2>Drawdowns</h2>')
        f.write(fig2.to_html(full_html=False))
        
        # Strategy performance comparison
        f.write('<h2>Strategy Performance</h2>')
        
        for name, result in results.items():
            f.write(f'<h3>{name}</h3>')
            
            # Get strategy metrics if available
            strategy_metrics = getattr(result, 'additional_metrics', None)
            
            if strategy_metrics:
                strategy_df = pd.DataFrame(strategy_metrics).T
                f.write(strategy_df.to_html(classes='table table-striped table-hover', float_format='%.2f'))
            else:
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
                    
                    total_pnl = sum(t.pnl for t in trades)
                    
                    strategy_data[strategy] = {
                        'Total Trades': total_count,
                        'Win Rate (%)': win_rate * 100,
                        'Total PnL ($)': total_pnl
                    }
                
                if strategy_data:
                    strategy_df = pd.DataFrame(strategy_data).T
                    f.write(strategy_df.to_html(classes='table table-striped table-hover', float_format='%.2f'))
                else:
                    f.write('<p>No strategy-specific data available</p>')
        
        # Close HTML
        f.write('</div>')
        f.write('<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>')
        f.write('</body></html>')

def main():
    """Main function to compare models"""
    # Define date range for backtest
    start_date = dt.datetime(2023, 1, 1)
    end_date = dt.datetime(2023, 12, 31)
    
    # Load configurations
    original_config = load_config('multi_strategy_config.yaml')
    optimized_config = load_config('further_optimized_config.yaml')
    hybrid_config = load_config('hybrid_optimized_config.yaml')
    
    # Create systems
    original_system = create_system(original_config)
    optimized_system = create_system(optimized_config)
    hybrid_system = create_system(hybrid_config, system_class=HybridStrategySystem)
    
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
    
    with open('comparison_results.json', 'w') as f:
        json.dump(results_dict, f, indent=4)
    
    # Generate HTML report
    generate_html_report(comparison_df, results, 'model_comparison.html')
    
    # Plot results
    plot_equity_curves(results, 'equity_curves.png')
    plot_drawdowns(results, 'drawdowns.png')
    
    logger.info("Comparison completed. Results saved to comparison_results.json and model_comparison.html")

if __name__ == "__main__":
    main()
