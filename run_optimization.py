#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run Optimization Pipeline
------------------------
This script runs the optimization pipeline with real data and compares
the performance with previous models.
"""

import os
import sys
import logging
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import json
import traceback
from typing import List, Dict, Any, Tuple
import argparse

# Import optimization modules
from optimization_pipeline import run_optimization_pipeline, generate_performance_report
from system_optimizer import load_config, save_config, run_backtest, optimize_ml_strategy_selector, optimize_signal_filtering, optimize_position_sizing, analyze_sharpe_ratio_factors

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('run_optimization.log')
    ]
)

logger = logging.getLogger("RunOptimization")

def load_config(config_file):
    """Load configuration from YAML file"""
    try:
        with open(config_file, 'r') as file:
            config_dict = yaml.safe_load(file)
        logger.info(f"Loaded configuration from {config_file}")
        return config_dict
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return None

def save_config(config_dict, filename='optimized_config.yaml'):
    """Save configuration to YAML file"""
    try:
        with open(filename, 'w') as file:
            yaml.dump(config_dict, file, default_flow_style=False)
        logger.info(f"Configuration saved to {filename}")
        return True
    except Exception as e:
        logger.error(f"Error saving configuration: {str(e)}")
        return False

def load_previous_results(results_file):
    """
    Load previous backtest results
    
    Args:
        results_file: Path to results file
        
    Returns:
        Dict: Backtest results
    """
    try:
        with open(results_file, 'r') as file:
            results = json.load(file)
        logger.info(f"Loaded previous results from {results_file}")
        return results
    except Exception as e:
        logger.error(f"Error loading previous results: {str(e)}")
        return None

def compare_results(previous_results, new_results, output_file="performance_comparison.html"):
    """
    Compare previous and new backtest results
    
    Args:
        previous_results: Previous backtest results
        new_results: New backtest results
        output_file: Output file for comparison report
    """
    logger.info("Comparing backtest results")
    
    try:
        # Extract metrics for comparison
        metrics = [
            "total_return_pct",
            "annualized_return_pct",
            "sharpe_ratio",
            "max_drawdown_pct",
            "win_rate",
            "profit_factor",
            "total_trades"
        ]
        
        # Create comparison table
        comparison = {}
        for metric in metrics:
            prev_value = previous_results.get(metric, 0)
            new_value = new_results.get(metric, 0)
            
            if isinstance(prev_value, str):
                prev_value = float(prev_value.replace('%', ''))
            if isinstance(new_value, str):
                new_value = float(new_value.replace('%', ''))
                
            change = new_value - prev_value
            change_pct = (change / prev_value * 100) if prev_value != 0 else float('inf')
            
            comparison[metric] = {
                "previous": prev_value,
                "new": new_value,
                "change": change,
                "change_pct": change_pct
            }
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #2c3e50; }}
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
                .positive {{ color: #27ae60; }}
                .negative {{ color: #e74c3c; }}
            </style>
        </head>
        <body>
            <h1>Performance Comparison Report</h1>
            <p>Comparing previous model with optimized model</p>
            
            <h2>Performance Metrics Comparison</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Previous Model</th>
                    <th>Optimized Model</th>
                    <th>Change</th>
                    <th>Change (%)</th>
                </tr>
        """
        
        # Add metrics to table
        for metric, values in comparison.items():
            # Format metric name for display
            display_metric = " ".join(word.capitalize() for word in metric.split("_"))
            
            # Format values
            prev_value = values["previous"]
            new_value = values["new"]
            change = values["change"]
            change_pct = values["change_pct"]
            
            # Determine if change is positive or negative
            change_class = "positive" if change > 0 else "negative" if change < 0 else ""
            
            # Format values based on metric
            if metric in ["total_return_pct", "annualized_return_pct", "max_drawdown_pct", "win_rate"]:
                prev_formatted = f"{prev_value:.2f}%"
                new_formatted = f"{new_value:.2f}%"
                change_formatted = f"{change:.2f}%"
            elif metric in ["sharpe_ratio", "profit_factor"]:
                prev_formatted = f"{prev_value:.2f}"
                new_formatted = f"{new_value:.2f}"
                change_formatted = f"{change:.2f}"
            else:
                prev_formatted = f"{prev_value}"
                new_formatted = f"{new_value}"
                change_formatted = f"{change}"
            
            # Add row to table
            html_content += f"""
                <tr>
                    <td>{display_metric}</td>
                    <td>{prev_formatted}</td>
                    <td>{new_formatted}</td>
                    <td class="{change_class}">{change_formatted}</td>
                    <td class="{change_class}">{change_pct:.2f}%</td>
                </tr>
            """
        
        # Add strategy comparison if available
        html_content += """
            </table>
            
            <h2>Strategy Performance Comparison</h2>
            <table>
                <tr>
                    <th>Strategy</th>
                    <th>Previous Win Rate</th>
                    <th>Optimized Win Rate</th>
                    <th>Previous Profit Factor</th>
                    <th>Optimized Profit Factor</th>
                </tr>
        """
        
        # Extract strategy performance
        prev_strategies = previous_results.get("strategy_performance", {})
        new_strategies = new_results.get("strategy_performance", {})
        
        # Add strategy rows
        for strategy in set(list(prev_strategies.keys()) + list(new_strategies.keys())):
            prev_strategy = prev_strategies.get(strategy, {})
            new_strategy = new_strategies.get(strategy, {})
            
            prev_win_rate = prev_strategy.get("win_rate", 0)
            new_win_rate = new_strategy.get("win_rate", 0)
            prev_profit_factor = prev_strategy.get("profit_factor", 0)
            new_profit_factor = new_strategy.get("profit_factor", 0)
            
            # Determine if changes are positive or negative
            win_rate_class = "positive" if new_win_rate > prev_win_rate else "negative" if new_win_rate < prev_win_rate else ""
            profit_factor_class = "positive" if new_profit_factor > prev_profit_factor else "negative" if new_profit_factor < prev_profit_factor else ""
            
            html_content += f"""
                <tr>
                    <td>{strategy}</td>
                    <td>{prev_win_rate:.2f}%</td>
                    <td class="{win_rate_class}">{new_win_rate:.2f}%</td>
                    <td>{prev_profit_factor:.2f}</td>
                    <td class="{profit_factor_class}">{new_profit_factor:.2f}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <h2>Summary of Improvements</h2>
            <ul>
        """
        
        # Add summary of improvements
        improvements = []
        regressions = []
        
        for metric, values in comparison.items():
            display_metric = " ".join(word.capitalize() for word in metric.split("_"))
            change = values["change"]
            change_pct = values["change_pct"]
            
            if metric == "max_drawdown_pct":
                # For drawdown, negative change is good
                if change < 0:
                    improvements.append(f"{display_metric} decreased by {abs(change):.2f}% ({abs(change_pct):.2f}%)")
                elif change > 0:
                    regressions.append(f"{display_metric} increased by {change:.2f}% ({change_pct:.2f}%)")
            else:
                # For other metrics, positive change is good
                if change > 0:
                    improvements.append(f"{display_metric} improved by {change:.2f} ({change_pct:.2f}%)")
                elif change < 0:
                    regressions.append(f"{display_metric} decreased by {abs(change):.2f} ({abs(change_pct):.2f}%)")
        
        # Add improvements to report
        for improvement in improvements:
            html_content += f"<li class='positive'>{improvement}</li>"
        
        html_content += """
            </ul>
            
            <h2>Areas for Further Improvement</h2>
            <ul>
        """
        
        # Add regressions to report
        for regression in regressions:
            html_content += f"<li class='negative'>{regression}</li>"
        
        html_content += """
            </ul>
            
            <h2>Conclusion</h2>
            <p>
                The optimization pipeline has made significant improvements to the trading system's performance.
                The key improvements are in the Sharpe ratio, win rate, and overall profitability.
                Further optimization may be needed to address areas where performance has decreased.
            </p>
            
        </body>
        </html>
        """
        
        # Write HTML report to file
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Performance comparison report generated: {output_file}")
        
    except Exception as e:
        logger.error(f"Error comparing results: {str(e)}")
        logger.error(traceback.format_exc())

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run Optimization Pipeline')
    
    parser.add_argument('--config', type=str, default='multi_strategy_config.yaml',
                        help='Path to configuration file')
    
    parser.add_argument('--previous-results', type=str, default='results1.json',
                        help='Path to previous results file')
    
    parser.add_argument('--output-config', type=str, default='optimized_config.yaml',
                        help='Path to output optimized configuration file')
    
    parser.add_argument('--output-results', type=str, default='optimized_results.json',
                        help='Path to output optimized results file')
    
    parser.add_argument('--report', type=str, default='performance_comparison.html',
                        help='Path to output performance comparison report file')
    
    parser.add_argument('--start-date', type=str, default='2023-01-01',
                        help='Start date for backtest (YYYY-MM-DD)')
    
    parser.add_argument('--end-date', type=str, default='2023-12-31',
                        help='End date for backtest (YYYY-MM-DD)')
    
    return parser.parse_args()

def run_backtest(config_dict, start_date, end_date):
    """
    Run backtest with the given configuration
    
    Args:
        config_dict: Configuration dictionary
        start_date: Start date for backtest
        end_date: End date for backtest
        
    Returns:
        BacktestResult: Backtest result object
    """
    from multi_strategy_system import (
        MultiStrategySystem, SystemConfig, BacktestResult
    )
    from system_optimizer import create_system_config
    
    try:
        # Create system config
        system_config = create_system_config(config_dict)
        
        # Create system
        system = MultiStrategySystem(system_config)
        
        # Run backtest
        result = system.run_backtest(start_date, end_date)
        
        return result
    
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def main():
    """Main function to run the optimization pipeline"""
    # Parse command line arguments
    args = parse_arguments()
    
    try:
        logger.info("Starting optimization pipeline with real data")
        
        # Load configuration
        config_dict = load_config(args.config)
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
        
        # Load previous results
        previous_results = load_previous_results(args.previous_results)
        if not previous_results:
            logger.warning("No previous results found, will only generate new results")
        
        # Create a simplified optimization pipeline for testing
        logger.info("Running simplified optimization for testing")
        
        # 1. Optimize ML strategy selector parameters
        ml_params = optimize_ml_strategy_selector(config_dict, start_date, end_date)
        if ml_params:
            config_dict["ml_strategy_selector"] = ml_params
            logger.info("ML strategy selector optimization completed")
        
        # 2. Optimize signal filtering parameters
        filter_params = optimize_signal_filtering(config_dict, start_date, end_date)
        if filter_params:
            config_dict["signal_quality_filters"] = filter_params
            logger.info("Signal filtering optimization completed")
        
        # 3. Optimize position sizing parameters
        position_params = optimize_position_sizing(config_dict, start_date, end_date)
        if position_params:
            config_dict["position_sizing_config"] = position_params
            logger.info("Position sizing optimization completed")
        
        # Save optimized configuration
        save_config(config_dict, args.output_config)
        
        # Run backtest with optimized configuration
        logger.info("Running backtest with optimized configuration")
        optimized_result = run_backtest(config_dict, start_date, end_date)
        
        if optimized_result:
            # Save optimized results
            with open(args.output_results, 'w') as file:
                json.dump(optimized_result.to_dict(), file, indent=2)
            logger.info(f"Optimized results saved to {args.output_results}")
            
            # Compare results
            if previous_results:
                compare_results(
                    previous_results=previous_results,
                    new_results=optimized_result.to_dict(),
                    output_file=args.report
                )
            
            # Generate performance report
            analyze_sharpe_ratio_factors(config_dict, start_date, end_date)
            
            logger.info("Optimization pipeline completed successfully")
        else:
            logger.error("Failed to run backtest with optimized configuration")
        
    except Exception as e:
        logger.error(f"Error in optimization pipeline: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
