#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compare Trading Models
---------------------
This script compares the performance of the original and optimized trading models
using the exact same time period and parameters.
"""

import os
import sys
import logging
import datetime as dt
import json
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import traceback
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('model_comparison.log')
    ]
)

logger = logging.getLogger("ModelComparison")

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

def run_backtest(config_dict, start_date, end_date, system_name="MultiStrategySystem"):
    """
    Run backtest with the given configuration
    
    Args:
        config_dict: Configuration dictionary
        start_date: Start date for backtest
        end_date: End date for backtest
        system_name: Name of the system class to use
        
    Returns:
        BacktestResult: Backtest result object
    """
    from multi_strategy_system import (
        MultiStrategySystem, SystemConfig, BacktestResult, MarketRegime
    )
    from system_optimizer import create_system_config
    
    try:
        # Create system config
        system_config = create_system_config(config_dict)
        
        # Create system
        if system_name == "MultiStrategySystem":
            system = MultiStrategySystem(system_config)
        elif system_name == "EnhancedMultiStrategySystem":
            # Import enhanced components
            from enhanced_trading_functions import (
                calculate_adaptive_position_size,
                filter_signals,
                generate_ml_signals
            )
            from ml_strategy_selector import MLStrategySelector
            
            # Create enhanced system
            class EnhancedMultiStrategySystem(MultiStrategySystem):
                """
                Enhanced version of the MultiStrategySystem with direct integration of
                adaptive position sizing, ML-based strategy selection, and improved signal filtering.
                """
                
                def __init__(self, config):
                    """Initialize the enhanced multi-strategy system"""
                    super().__init__(config)
                    
                    # Initialize ML strategy selector
                    self.ml_strategy_selector = MLStrategySelector(
                        config=config.ml_strategy_selector,
                        logger=self.logger
                    )
                    
                    # Add signal quality filters and position sizing config
                    self.signal_quality_filters = config.signal_quality_filters
                    self.position_sizing_config = config.position_sizing_config
                    
                    # Fix the sector performance error
                    self._patch_market_analyzer()
                    
                    self.logger.info("Enhanced Multi-Strategy System initialized")
                
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
                        else:
                            # Call the original method for other cases
                            return original_method(self, base_regime, adx, vix, trend_direction, 
                                                  breadth_indicators, intermarket_indicators,
                                                  sector_performance, sentiment_indicators)
                    
                    # Apply the patch
                    self.market_analyzer._determine_sub_regime = patched_method.__get__(self.market_analyzer)
                    self.logger.info("Fixed sector performance error by patching _determine_sub_regime method")
                
                def _generate_signals(self):
                    """
                    Override the signal generation method to use ML-based strategy selection
                    """
                    try:
                        if not self.market_state:
                            self.logger.warning("Cannot generate signals: Market state not available")
                            return
                            
                        self.logger.info(f"Generating signals for market regime: {self.market_state.regime}")
                        
                        # Clear previous signals
                        self.signals = []
                        
                        # Generate signals using ML-based strategy selection
                        all_signals = generate_ml_signals(
                            self.config.stocks,
                            self.strategies,
                            self.candle_data,
                            self.market_state,
                            self.ml_strategy_selector,
                            self.logger
                        )
                        
                        # Apply enhanced quality filters
                        filtered_signals = self._filter_signals(all_signals)
                        
                        # Add filtered signals to the system
                        self.signals.extend(filtered_signals)
                        
                        # Log signal generation summary
                        self.logger.info(f"Generated {len(all_signals)} signals, {len(filtered_signals)} passed quality filters")
                    except Exception as e:
                        self.logger.error(f"Error in ML-based strategy selection: {str(e)}")
                        # Fall back to original method
                        super()._generate_signals()
                
                def _calculate_position_size(self, signal):
                    """
                    Override the position sizing method to use adaptive position sizing
                    """
                    try:
                        return calculate_adaptive_position_size(
                            signal=signal,
                            market_state=self.market_state,
                            candle_data=self.candle_data,
                            current_equity=self.current_equity,
                            position_sizing_config=self.position_sizing_config,
                            logger=self.logger
                        )
                    except Exception as e:
                        self.logger.error(f"Error in adaptive position sizing: {str(e)}")
                        # Fall back to original method
                        return super()._calculate_position_size(signal)
                
                def _filter_signals(self, signals):
                    """
                    Override the signal filtering method to use enhanced filters
                    """
                    try:
                        return filter_signals(
                            signals=signals,
                            candle_data=self.candle_data,
                            config=self.config,
                            signal_quality_filters=self.signal_quality_filters,
                            logger=self.logger
                        )
                    except Exception as e:
                        self.logger.error(f"Error in enhanced signal filtering: {str(e)}")
                        # Fall back to original method
                        return super()._filter_signals(signals)
            
            system = EnhancedMultiStrategySystem(system_config)
        else:
            # Default to original system
            system = MultiStrategySystem(system_config)
        
        # Run backtest
        logger.info(f"Running backtest from {start_date} to {end_date}")
        result = system.run_backtest(start_date, end_date)
        
        return result
    
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def generate_comparison_report(original_results, optimized_results, output_file="model_comparison.html"):
    """
    Generate a comparison report between original and optimized models
    
    Args:
        original_results: Original model results
        optimized_results: Optimized model results
        output_file: Output HTML file
    """
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
            orig_value = original_results.get(metric, 0)
            opt_value = optimized_results.get(metric, 0)
            
            if isinstance(orig_value, str):
                orig_value = float(orig_value.replace('%', ''))
            if isinstance(opt_value, str):
                opt_value = float(opt_value.replace('%', ''))
            
            # Handle None/null values
            if orig_value is None:
                orig_value = 0
            if opt_value is None:
                opt_value = 0
                
            change = opt_value - orig_value
            change_pct = (change / orig_value * 100) if orig_value != 0 else float('inf')
            
            comparison[metric] = {
                "original": orig_value,
                "optimized": opt_value,
                "change": change,
                "change_pct": change_pct
            }
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trading Model Comparison Report</title>
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
                .neutral {{ color: #7f8c8d; }}
            </style>
        </head>
        <body>
            <h1>Trading Model Comparison Report</h1>
            <p>Comparing original model with optimized model for period: {original_results.get('start_date')} to {original_results.get('end_date')}</p>
            
            <h2>Performance Metrics Comparison</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Original Model</th>
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
            orig_value = values["original"]
            opt_value = values["optimized"]
            change = values["change"]
            change_pct = values["change_pct"]
            
            # Determine if change is positive, negative, or neutral
            if change > 0:
                change_class = "positive"
            elif change < 0:
                change_class = "negative"
            else:
                change_class = "neutral"
            
            # For max_drawdown, negative change is good
            if metric == "max_drawdown_pct":
                change_class = "positive" if change < 0 else "negative" if change > 0 else "neutral"
            
            # Format values based on metric
            if metric in ["total_return_pct", "annualized_return_pct", "max_drawdown_pct", "win_rate"]:
                orig_formatted = f"{orig_value:.2f}%"
                opt_formatted = f"{opt_value:.2f}%"
                change_formatted = f"{change:.2f}%"
            elif metric in ["sharpe_ratio", "profit_factor"]:
                orig_formatted = f"{orig_value:.2f}"
                opt_formatted = f"{opt_value:.2f}"
                change_formatted = f"{change:.2f}"
            else:
                orig_formatted = f"{orig_value}"
                opt_formatted = f"{opt_value}"
                change_formatted = f"{change}"
            
            # Add row to table
            html_content += f"""
                <tr>
                    <td>{display_metric}</td>
                    <td>{orig_formatted}</td>
                    <td>{opt_formatted}</td>
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
                    <th>Original Win Rate</th>
                    <th>Optimized Win Rate</th>
                    <th>Original Profit Factor</th>
                    <th>Optimized Profit Factor</th>
                </tr>
        """
        
        # Extract strategy performance
        orig_strategies = original_results.get("strategy_performance", {})
        opt_strategies = optimized_results.get("strategy_performance", {})
        
        # Add strategy rows
        for strategy in set(list(orig_strategies.keys()) + list(opt_strategies.keys())):
            orig_strategy = orig_strategies.get(strategy, {})
            opt_strategy = opt_strategies.get(strategy, {})
            
            orig_win_rate = orig_strategy.get("win_rate", 0)
            opt_win_rate = opt_strategy.get("win_rate", 0)
            orig_profit_factor = orig_strategy.get("profit_factor", 0)
            opt_profit_factor = opt_strategy.get("profit_factor", 0)
            
            # Determine if changes are positive or negative
            win_rate_class = "positive" if opt_win_rate > orig_win_rate else "negative" if opt_win_rate < orig_win_rate else "neutral"
            profit_factor_class = "positive" if opt_profit_factor > orig_profit_factor else "negative" if opt_profit_factor < orig_profit_factor else "neutral"
            
            html_content += f"""
                <tr>
                    <td>{strategy}</td>
                    <td>{orig_win_rate:.2f}%</td>
                    <td class="{win_rate_class}">{opt_win_rate:.2f}%</td>
                    <td>{orig_profit_factor:.2f}</td>
                    <td class="{profit_factor_class}">{opt_profit_factor:.2f}</td>
                </tr>
            """
        
        # Add summary of differences
        html_content += """
            </table>
            
            <h2>Analysis of Results</h2>
        """
        
        # Add analysis based on the comparison
        if original_results.get("total_trades", 0) == 0 and optimized_results.get("total_trades", 0) == 0:
            html_content += """
            <p>
                <strong>No trades were executed by either model during the testing period.</strong> This could be due to:
                <ul>
                    <li>The testing period was too short to generate trading signals</li>
                    <li>Market conditions during this period didn't meet the criteria for any trading strategies</li>
                    <li>Signal quality filters might be too restrictive</li>
                </ul>
            </p>
            <p>
                <strong>Recommendations:</strong>
                <ul>
                    <li>Test with a longer timeframe to capture more market conditions</li>
                    <li>Temporarily reduce the strictness of signal quality filters</li>
                    <li>Add logging to track signals that were generated but rejected</li>
                </ul>
            </p>
            """
        else:
            # Add analysis if trades were executed
            improvements = []
            regressions = []
            
            for metric, values in comparison.items():
                display_metric = " ".join(word.capitalize() for word in metric.split("_"))
                change = values["change"]
                
                if metric == "max_drawdown_pct":
                    # For drawdown, negative change is good
                    if change < 0:
                        improvements.append(f"{display_metric} decreased by {abs(change):.2f}%")
                    elif change > 0:
                        regressions.append(f"{display_metric} increased by {change:.2f}%")
                else:
                    # For other metrics, positive change is good
                    if change > 0:
                        improvements.append(f"{display_metric} improved by {change:.2f}")
                    elif change < 0:
                        regressions.append(f"{display_metric} decreased by {abs(change):.2f}")
            
            if improvements:
                html_content += "<p><strong>Improvements:</strong><ul>"
                for improvement in improvements:
                    html_content += f"<li class='positive'>{improvement}</li>"
                html_content += "</ul></p>"
            
            if regressions:
                html_content += "<p><strong>Areas for Further Improvement:</strong><ul>"
                for regression in regressions:
                    html_content += f"<li class='negative'>{regression}</li>"
                html_content += "</ul></p>"
        
        html_content += """
            <h2>Conclusion</h2>
            <p>
                The optimization pipeline has made adjustments to the trading system parameters, including:
                <ul>
                    <li>ML Strategy Selector: Modified lookback window and training sample requirements</li>
                    <li>Signal Quality Filters: Adjusted correlation thresholds and signal limits</li>
                    <li>Position Sizing: Implemented signal strength and volatility adjustments</li>
                </ul>
            </p>
            <p>
                For more meaningful performance comparison, consider testing with a longer timeframe 
                or during periods with more market volatility to generate more trading signals.
            </p>
        </body>
        </html>
        """
        
        # Write HTML report to file
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Comparison report generated: {output_file}")
        
    except Exception as e:
        logger.error(f"Error generating comparison report: {str(e)}")
        logger.error(traceback.format_exc())

def plot_equity_curves(original_result, optimized_result, period_description=""):
    """
    Plot equity curves for comparison
    
    Args:
        original_result: Backtest result from original system
        optimized_result: Backtest result from enhanced system
        period_description: Description of the test period
    """
    try:
        # Debug logging to understand data structure
        logger.info(f"Original result type: {type(original_result)}")
        if hasattr(original_result, 'equity_curve'):
            logger.info(f"Original equity curve type: {type(original_result.equity_curve)}")
            if isinstance(original_result.equity_curve, list):
                logger.info(f"Original equity curve is a list with {len(original_result.equity_curve)} items")
                if original_result.equity_curve:
                    logger.info(f"First item type: {type(original_result.equity_curve[0])}")
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Handle the case where equity_curve is a list of tuples (date, equity)
        if hasattr(original_result, 'equity_curve') and isinstance(original_result.equity_curve, list):
            # Extract all dates and equity values from the list of tuples
            original_dates = []
            original_equity = []
            
            for item in original_result.equity_curve:
                if isinstance(item, tuple) and len(item) == 2:
                    date, equity = item
                    original_dates.append(date)
                    original_equity.append(equity)
            
            if original_dates and original_equity:
                plt.plot(original_dates, original_equity, label='Original Model', color='blue')
                logger.info(f"Plotted original model equity curve with {len(original_dates)} points")
            else:
                logger.warning("Could not extract date and equity values from original model")
        else:
            logger.warning("Original model equity curve not available or not in expected format")
        
        # Handle optimized result
        if hasattr(optimized_result, 'equity_curve') and isinstance(optimized_result.equity_curve, list):
            # Extract all dates and equity values from the list of tuples
            optimized_dates = []
            optimized_equity = []
            
            for item in optimized_result.equity_curve:
                if isinstance(item, tuple) and len(item) == 2:
                    date, equity = item
                    optimized_dates.append(date)
                    optimized_equity.append(equity)
            
            if optimized_dates and optimized_equity:
                plt.plot(optimized_dates, optimized_equity, label='Optimized Model', color='green')
                logger.info(f"Plotted optimized model equity curve with {len(optimized_dates)} points")
            else:
                logger.warning("Could not extract date and equity values from optimized model")
        else:
            logger.warning("Optimized model equity curve not available or not in expected format")
        
        # Add labels and title
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.title(f'Equity Curve Comparison - {period_description}')
        plt.grid(True)
        
        # Only add legend if we have data
        if plt.gca().get_legend_handles_labels()[0]:
            plt.legend()
        
        # Format x-axis dates
        plt.gcf().autofmt_xdate()
        
        # Save figure
        plt.savefig(f'equity_curve_comparison_{period_description.lower().replace(" ", "_")}.png')
        logger.info(f"Equity curve comparison saved to equity_curve_comparison_{period_description.lower().replace(' ', '_')}.png")
        
        # Close figure
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting equity curves: {str(e)}")
        logger.error(traceback.format_exc())

def serialize_datetime(obj):
    """Helper function to serialize datetime objects to ISO format strings"""
    if isinstance(obj, (dt.datetime, dt.date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

def main():
    """Main function to compare original and optimized models"""
    logger.info("Starting Model Comparison")
    
    try:
        # Load configurations
        original_config = load_config('multi_strategy_config.yaml')
        optimized_config = load_config('further_optimized_config.yaml')
        
        if not original_config or not optimized_config:
            logger.error("Failed to load configurations")
            return
        
        # Define test period - full year 2023
        start_date = dt.date(2023, 1, 1)
        end_date = dt.date(2023, 12, 31)
        
        # Run backtests
        logger.info(f"Running original model backtest from {start_date} to {end_date}")
        original_result = run_backtest(original_config, start_date, end_date, "MultiStrategySystem")
        
        logger.info(f"Running optimized model backtest from {start_date} to {end_date}")
        optimized_result = run_backtest(optimized_config, start_date, end_date, "EnhancedMultiStrategySystem")
        
        if not original_result or not optimized_result:
            logger.error("Failed to run backtests")
            return
        
        # Convert results to dictionaries
        original_dict = original_result.to_dict()
        optimized_dict = optimized_result.to_dict()
        
        # Save results to files
        with open('original_results.json', 'w') as f:
            json.dump(original_dict, f, indent=2, default=serialize_datetime)
        
        with open('optimized_results.json', 'w') as f:
            json.dump(optimized_dict, f, indent=2, default=serialize_datetime)
        
        # Generate comparison report
        generate_comparison_report(original_dict, optimized_dict, "full_year_comparison.html")
        
        # Plot equity curves
        plot_equity_curves(original_result, optimized_result, "Full Year 2023")
        
        logger.info("Model comparison completed")
        
    except Exception as e:
        logger.error(f"Error in model comparison: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
