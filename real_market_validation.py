#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Real Market Data Validation for Hybrid Trading Model
---------------------------------------------------
This script runs the hybrid trading model with real market data and compares
its performance with the original and optimized models.
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
from typing import Dict, Any, List
import traceback

# Import the multi-strategy system
from multi_strategy_system import (
    MultiStrategySystem, SystemConfig, Signal, MarketRegime, 
    BacktestResult, StockConfig, MarketState
)

# Import from multi_strategy_main
from multi_strategy_main import MultiStrategyApp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('real_market_validation.log')
    ]
)

logger = logging.getLogger("RealMarketValidation")

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
    
    # Check if the pattern is found
    if re.search(pattern, content, re.DOTALL):
        # Replace the premature return with a commented version
        modified_content = re.sub(pattern, r'\1\n            # return signals  # Commented out premature return\n\2def', content, flags=re.DOTALL)
        
        # Write the modified content back to the file
        with open(file_path, 'w') as file:
            file.write(modified_content)
        
        logger.info("Fixed VolatilityBreakout strategy by commenting out premature return statement")
    else:
        logger.info("VolatilityBreakout strategy already fixed or pattern not found")

def run_model_with_real_data(config_file, start_date, end_date, model_name):
    """Run a model with real market data"""
    logger.info(f"Running {model_name} model with real market data from {start_date} to {end_date}")
    
    # Create app instance
    app = MultiStrategyApp()
    
    # Load configuration
    if not app.load_config(config_file):
        logger.error(f"Failed to load configuration from {config_file}")
        return None
    
    # Force backtesting mode and Yahoo data source
    app.config.backtesting_mode = True
    app.config.data_source = 'YAHOO'
    
    # Initialize system
    if not app.initialize_system():
        logger.error("Failed to initialize trading system")
        return None
    
    # Run backtest
    try:
        result = app.run_backtest(start_date, end_date)
        
        if result:
            # Save results
            output_file = f"{model_name.lower()}_real_market_results.json"
            app.save_backtest_results(result, output_file)
            
            # Plot equity curve
            plot_file = f"{model_name.lower()}_real_market_equity.png"
            app.plot_equity_curve(result, plot_file)
            
            logger.info(f"{model_name} model backtest completed successfully")
            return result
        else:
            logger.error(f"{model_name} model backtest failed")
            return None
    except Exception as e:
        logger.error(f"Error running {model_name} model backtest: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def compare_models(results, output_file='model_comparison.html'):
    """Compare the performance of different models"""
    if not results:
        logger.error("No results to compare")
        return
    
    # Create comparison DataFrame
    comparison_data = {}
    for name, result in results.items():
        if result:
            comparison_data[name] = {
                "Total Return (%)": result.total_return_pct,
                "Annualized Return (%)": result.annualized_return_pct,
                "Sharpe Ratio": result.sharpe_ratio,
                "Max Drawdown (%)": result.max_drawdown_pct,
                "Win Rate (%)": result.win_rate * 100,
                "Profit Factor": result.profit_factor,
                "Total Trades": result.total_trades
            }
    
    if not comparison_data:
        logger.error("No valid results for comparison")
        return
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Print comparison table
    print("\n===== MODEL COMPARISON =====")
    print(comparison_df)
    print("===========================\n")
    
    # Generate HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Trading Model Comparison with Real Market Data</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            h1, h2, h3 {{ color: #333; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .section {{ margin-bottom: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .highlight {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
            .best-value {{ font-weight: bold; color: green; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Trading Model Comparison with Real Market Data</h1>
            <p>Generated on: {dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <div class="section highlight">
                <h2>Performance Metrics Comparison</h2>
                {comparison_df.to_html(float_format='%.2f')}
            </div>
            
            <div class="section">
                <h2>Strategy Performance by Model</h2>
    """
    
    # Add strategy performance for each model
    for name, result in results.items():
        if not result:
            continue
            
        html_content += f"""
                <h3>{name} Model</h3>
                <table>
                    <tr>
                        <th>Strategy</th>
                        <th>Win Rate (%)</th>
                        <th>Profit Factor</th>
                        <th>Total Trades</th>
                    </tr>
        """
        
        for strategy, perf in result.strategy_performance.items():
            html_content += f"""
                    <tr>
                        <td>{strategy}</td>
                        <td>{perf.win_rate * 100:.2f}</td>
                        <td>{perf.profit_factor:.2f}</td>
                        <td>{perf.total_trades}</td>
                    </tr>
            """
        
        html_content += """
                </table>
        """
    
    # Add conclusion
    html_content += """
            </div>
            
            <div class="section">
                <h2>Conclusion</h2>
                <p>
                    This report compares the performance of the original, optimized, and hybrid trading models
                    using real market data. The metrics above demonstrate how each model performs in real-world
                    market conditions.
                </p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write HTML report
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Comparison report generated: {output_file}")
    
    # Create combined equity curve plot
    plot_combined_equity_curves(results, 'combined_equity_curves.png')

def plot_combined_equity_curves(results, output_file):
    """Plot combined equity curves for all models"""
    plt.figure(figsize=(12, 8))
    
    for name, result in results.items():
        if not result or not result.equity_curve:
            continue
            
        # Convert equity curve to dataframe
        df = pd.DataFrame(result.equity_curve, columns=['date', 'equity'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Normalize to percentage return
        initial_equity = df['equity'].iloc[0]
        df['return_pct'] = (df['equity'] / initial_equity - 1) * 100
        
        # Plot normalized equity curve
        plt.plot(df.index, df['return_pct'], label=f"{name} Model")
    
    plt.title('Comparative Performance of Trading Models')
    plt.xlabel('Date')
    plt.ylabel('Return (%)')
    plt.grid(True)
    plt.legend()
    
    plt.savefig(output_file)
    logger.info(f"Combined equity curves plot saved to {output_file}")

def main():
    """Main function to run the real market validation"""
    # Fix the VolatilityBreakout strategy
    fix_volatility_breakout_strategy()
    
    # Define date range for backtest - use a recent 1-year period
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    
    # Run models with real market data
    results = {}
    
    # Original model
    original_result = run_model_with_real_data(
        'multi_strategy_config.yaml',
        start_date,
        end_date,
        'Original'
    )
    results['Original'] = original_result
    
    # Optimized model
    optimized_result = run_model_with_real_data(
        'further_optimized_config.yaml',
        start_date,
        end_date,
        'Optimized'
    )
    results['Optimized'] = optimized_result
    
    # Hybrid model
    hybrid_result = run_model_with_real_data(
        'hybrid_optimized_config.yaml',
        start_date,
        end_date,
        'Hybrid'
    )
    results['Hybrid'] = hybrid_result
    
    # Compare models
    compare_models(results, 'real_market_comparison.html')
    
    logger.info("Real market validation completed")

if __name__ == "__main__":
    main()
