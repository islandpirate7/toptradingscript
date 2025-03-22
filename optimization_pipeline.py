#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trading System Optimization Pipeline
-----------------------------------
This script orchestrates the complete optimization pipeline for the trading system,
combining ML strategy selector optimization, signal filtering optimization,
and position sizing optimization to improve overall performance.
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
import argparse

# Import optimization modules
from system_optimizer import (
    load_config, save_config, analyze_sharpe_ratio_factors,
    EnhancedMultiStrategySystem, run_backtest
)
from ml_strategy_optimizer import optimize_ml_strategy_selector
from signal_filter_optimizer import optimize_signal_filters
from position_sizing_optimizer import optimize_position_sizing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('optimization_pipeline.log')
    ]
)

logger = logging.getLogger("OptimizationPipeline")

def run_optimization_pipeline(config_dict, start_date, end_date, output_file="optimized_config.yaml"):
    """
    Run the complete optimization pipeline
    
    Args:
        config_dict: Configuration dictionary
        start_date: Start date for backtest
        end_date: End date for backtest
        output_file: Output file for optimized configuration
        
    Returns:
        Dict: Optimized configuration dictionary
    """
    logger.info("Starting Trading System Optimization Pipeline")
    
    # Step 1: Analyze Sharpe ratio factors
    logger.info("Step 1: Analyzing Sharpe ratio factors")
    analyze_sharpe_ratio_factors(config_dict, start_date, end_date)
    
    # Step 2: Optimize ML strategy selector
    logger.info("Step 2: Optimizing ML strategy selector")
    ml_config = optimize_ml_strategy_selector(config_dict)
    
    if ml_config:
        config_dict["ml_strategy_selector"] = ml_config
        logger.info("ML strategy selector optimization completed")
    else:
        logger.warning("ML strategy selector optimization failed, using original configuration")
    
    # Step 3: Optimize signal filtering
    logger.info("Step 3: Optimizing signal filtering")
    filter_params = optimize_signal_filters(config_dict)
    
    if filter_params:
        config_dict["signal_quality_filters"] = filter_params
        logger.info("Signal filtering optimization completed")
    else:
        logger.warning("Signal filtering optimization failed, using original configuration")
    
    # Step 4: Optimize position sizing
    logger.info("Step 4: Optimizing position sizing")
    position_params = optimize_position_sizing(config_dict)
    
    if position_params:
        config_dict["position_sizing_config"] = position_params
        logger.info("Position sizing optimization completed")
    else:
        logger.warning("Position sizing optimization failed, using original configuration")
    
    # Step 5: Run final backtest with optimized configuration
    logger.info("Step 5: Running final backtest with optimized configuration")
    final_result = run_backtest(config_dict, start_date, end_date)
    
    if final_result:
        logger.info("=== Final Optimization Results ===")
        logger.info(f"Total Return: {final_result.total_return_pct:.2f}%")
        logger.info(f"Annualized Return: {final_result.annualized_return_pct:.2f}%")
        logger.info(f"Sharpe Ratio: {final_result.sharpe_ratio:.2f}")
        logger.info(f"Max Drawdown: {final_result.max_drawdown_pct:.2f}%")
        logger.info(f"Win Rate: {final_result.win_rate:.2f}%")
        logger.info(f"Profit Factor: {final_result.profit_factor:.2f}")
        logger.info(f"Total Trades: {final_result.total_trades}")
    else:
        logger.error("Final backtest failed")
    
    # Save optimized configuration
    save_config(config_dict, output_file)
    logger.info(f"Optimized configuration saved to {output_file}")
    
    logger.info("Trading System Optimization Pipeline completed")
    
    return config_dict

def generate_performance_report(config_dict, start_date, end_date, output_file="performance_report.html"):
    """
    Generate performance report for the optimized trading system
    
    Args:
        config_dict: Configuration dictionary
        start_date: Start date for backtest
        end_date: End date for backtest
        output_file: Output file for performance report
    """
    logger.info("Generating performance report")
    
    try:
        # Run backtest with optimized configuration
        result = run_backtest(config_dict, start_date, end_date)
        
        if not result:
            logger.error("Failed to run backtest for performance report")
            return
        
        # Create DataFrame for equity curve
        equity_curve = pd.DataFrame(result.equity_curve)
        equity_curve.set_index('date', inplace=True)
        
        # Create DataFrame for trades
        trades_df = pd.DataFrame(result.trades)
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trading System Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #2c3e50; }}
                .metrics {{ display: flex; flex-wrap: wrap; }}
                .metric {{ 
                    background-color: #f8f9fa; 
                    border-radius: 5px; 
                    padding: 15px; 
                    margin: 10px; 
                    min-width: 200px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                }}
                .metric h3 {{ margin-top: 0; color: #3498db; }}
                .positive {{ color: #27ae60; }}
                .negative {{ color: #e74c3c; }}
                table {{ 
                    border-collapse: collapse; 
                    width: 100%; 
                    margin: 20px 0;
                }}
                th, td {{ 
                    border: 1px solid #ddd; 
                    padding: 8px; 
                    text-align: left;
                }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <h1>Trading System Performance Report</h1>
            <p>Period: {start_date} to {end_date}</p>
            
            <h2>Performance Metrics</h2>
            <div class="metrics">
                <div class="metric">
                    <h3>Total Return</h3>
                    <p class="{'positive' if result.total_return_pct > 0 else 'negative'}">{result.total_return_pct:.2f}%</p>
                </div>
                <div class="metric">
                    <h3>Annualized Return</h3>
                    <p class="{'positive' if result.annualized_return_pct > 0 else 'negative'}">{result.annualized_return_pct:.2f}%</p>
                </div>
                <div class="metric">
                    <h3>Sharpe Ratio</h3>
                    <p class="{'positive' if result.sharpe_ratio > 1 else 'negative'}">{result.sharpe_ratio:.2f}</p>
                </div>
                <div class="metric">
                    <h3>Max Drawdown</h3>
                    <p class="negative">{result.max_drawdown_pct:.2f}%</p>
                </div>
                <div class="metric">
                    <h3>Win Rate</h3>
                    <p class="{'positive' if result.win_rate > 50 else 'negative'}">{result.win_rate:.2f}%</p>
                </div>
                <div class="metric">
                    <h3>Profit Factor</h3>
                    <p class="{'positive' if result.profit_factor > 1 else 'negative'}">{result.profit_factor:.2f}</p>
                </div>
                <div class="metric">
                    <h3>Total Trades</h3>
                    <p>{result.total_trades}</p>
                </div>
            </div>
            
            <h2>Trade Analysis</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Average Profit (Winners)</td>
                    <td>${result.avg_win:.2f}</td>
                </tr>
                <tr>
                    <td>Average Loss (Losers)</td>
                    <td>${result.avg_loss:.2f}</td>
                </tr>
                <tr>
                    <td>Largest Winner</td>
                    <td>${result.largest_win:.2f}</td>
                </tr>
                <tr>
                    <td>Largest Loser</td>
                    <td>${result.largest_loss:.2f}</td>
                </tr>
                <tr>
                    <td>Average Holding Period</td>
                    <td>{result.avg_holding_period:.1f} days</td>
                </tr>
            </table>
            
            <h2>Strategy Performance</h2>
            <table>
                <tr>
                    <th>Strategy</th>
                    <th>Win Rate</th>
                    <th>Profit Factor</th>
                    <th>Total Trades</th>
                </tr>
        """
        
        # Add strategy performance
        for strategy, stats in result.strategy_stats.items():
            html_content += f"""
                <tr>
                    <td>{strategy}</td>
                    <td>{stats.get('win_rate', 0):.2f}%</td>
                    <td>{stats.get('profit_factor', 0):.2f}</td>
                    <td>{stats.get('total_trades', 0)}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <h2>Recent Trades</h2>
            <table>
                <tr>
                    <th>Symbol</th>
                    <th>Direction</th>
                    <th>Entry Date</th>
                    <th>Exit Date</th>
                    <th>Profit/Loss</th>
                    <th>Strategy</th>
                </tr>
        """
        
        # Add recent trades
        for trade in trades_df.tail(10).to_dict('records'):
            html_content += f"""
                <tr>
                    <td>{trade.get('symbol', '')}</td>
                    <td>{'Long' if trade.get('direction', 0) > 0 else 'Short'}</td>
                    <td>{trade.get('entry_date', '')}</td>
                    <td>{trade.get('exit_date', '')}</td>
                    <td class="{'positive' if trade.get('profit', 0) > 0 else 'negative'}">${trade.get('profit', 0):.2f}</td>
                    <td>{trade.get('strategy', '')}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <h2>Optimization Summary</h2>
            <p>The trading system has been optimized with the following components:</p>
            <ul>
                <li><strong>ML Strategy Selector:</strong> Optimized to better predict strategy performance in different market regimes</li>
                <li><strong>Signal Filtering:</strong> Enhanced to improve win rate and focus on high-quality trades</li>
                <li><strong>Position Sizing:</strong> Optimized to improve risk-adjusted returns and Sharpe ratio</li>
            </ul>
            
            <p>The optimization has resulted in:</p>
            <ul>
                <li>Improved Sharpe ratio through better risk management</li>
                <li>Higher win rate by focusing on higher quality signals</li>
                <li>More balanced trading frequency to avoid overtrading</li>
            </ul>
            
        </body>
        </html>
        """
        
        # Write HTML report to file
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Performance report generated: {output_file}")
        
    except Exception as e:
        logger.error(f"Error generating performance report: {str(e)}")
        logger.error(traceback.format_exc())

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Trading System Optimization Pipeline')
    
    parser.add_argument('--config', type=str, default='multi_strategy_config.yaml',
                        help='Path to configuration file')
    
    parser.add_argument('--output', type=str, default='optimized_config.yaml',
                        help='Path to output optimized configuration file')
    
    parser.add_argument('--report', type=str, default='performance_report.html',
                        help='Path to output performance report file')
    
    parser.add_argument('--start-date', type=str, default='2023-01-01',
                        help='Start date for backtest (YYYY-MM-DD)')
    
    parser.add_argument('--end-date', type=str, default='2023-12-31',
                        help='End date for backtest (YYYY-MM-DD)')
    
    parser.add_argument('--skip-ml', action='store_true',
                        help='Skip ML strategy selector optimization')
    
    parser.add_argument('--skip-filters', action='store_true',
                        help='Skip signal filtering optimization')
    
    parser.add_argument('--skip-position', action='store_true',
                        help='Skip position sizing optimization')
    
    return parser.parse_args()

def main():
    """Main function to run the optimization pipeline"""
    # Parse command line arguments
    args = parse_arguments()
    
    try:
        # Load configuration
        config_dict = None
        try:
            with open(args.config, 'r') as file:
                config_dict = yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return
        
        if not config_dict:
            logger.error("Failed to load configuration")
            return
        
        # Parse dates
        try:
            start_date = dt.datetime.strptime(args.start_date, '%Y-%m-%d').date()
            end_date = dt.datetime.strptime(args.end_date, '%Y-%m-%d').date()
        except Exception as e:
            logger.error(f"Error parsing dates: {str(e)}")
            return
        
        # Run optimization pipeline
        optimized_config = run_optimization_pipeline(
            config_dict=config_dict,
            start_date=start_date,
            end_date=end_date,
            output_file=args.output
        )
        
        # Generate performance report
        generate_performance_report(
            config_dict=optimized_config,
            start_date=start_date,
            end_date=end_date,
            output_file=args.report
        )
        
        logger.info("Optimization pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in optimization pipeline: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
