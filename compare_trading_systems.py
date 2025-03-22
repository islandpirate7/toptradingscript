#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trading System Comparison
------------------------
This script compares the performance of the original trading system
with the enhanced Alpaca trading system using historical data from 2023.
"""

import os
import json
import yaml
import logging
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from enhanced_alpaca_trading import EnhancedAlpacaTradingSystem
from multi_strategy_system import MultiStrategySystem, SystemConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("comparison.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("comparison")

def load_config(config_file):
    """Load configuration from YAML file"""
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return None

def load_results(results_file):
    """Load results from JSON file"""
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        return results
    except Exception as e:
        logger.error(f"Error loading results: {str(e)}")
        return None

def run_enhanced_backtest(config_file, start_date, end_date):
    """Run backtest with the enhanced trading system"""
    try:
        # Initialize enhanced trading system
        trading_system = EnhancedAlpacaTradingSystem(config_file=config_file, mode='paper')
        
        # Run backtest
        results = trading_system.run_backtest(start_date, end_date)
        
        return results
    except Exception as e:
        logger.error(f"Error running enhanced backtest: {str(e)}")
        return None

def plot_comparison(original_results, enhanced_results, output_file=None):
    """Plot comparison between original and enhanced trading systems"""
    try:
        # Create figure with multiple subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract data
        original_equity = original_results.get('equity_curve', [])
        enhanced_equity = enhanced_results.get('equity_curve', [])
        
        original_dates = [dt.datetime.strptime(d, '%Y-%m-%d') if isinstance(d, str) else d 
                         for d in original_results.get('dates', [])]
        enhanced_dates = [dt.datetime.fromisoformat(point[0]) if isinstance(point[0], str) else point[0] 
                         for point in enhanced_equity]
        
        original_values = original_results.get('equity_values', [])
        enhanced_values = [point[1] for point in enhanced_equity]
        
        # Normalize equity curves to start at 100 for better comparison
        if original_values and original_values[0] != 0:
            original_values = [v / original_values[0] * 100 for v in original_values]
        
        if enhanced_values and enhanced_values[0] != 0:
            enhanced_values = [v / enhanced_values[0] * 100 for v in enhanced_values]
        
        # Plot equity curves
        axs[0, 0].plot(original_dates, original_values, 'b-', label='Original System')
        axs[0, 0].plot(enhanced_dates, enhanced_values, 'g-', label='Enhanced System')
        axs[0, 0].set_title('Equity Curve Comparison (Normalized)')
        axs[0, 0].set_ylabel('Portfolio Value (Normalized to 100)')
        axs[0, 0].grid(True)
        axs[0, 0].legend()
        
        # Calculate and plot drawdowns
        if original_values:
            original_peak = np.maximum.accumulate(original_values)
            original_drawdown = (original_values - original_peak) / original_peak * 100
            axs[0, 1].fill_between(original_dates, original_drawdown, 0, color='blue', alpha=0.3, label='Original System')
        
        if enhanced_values:
            enhanced_peak = np.maximum.accumulate(enhanced_values)
            enhanced_drawdown = (enhanced_values - enhanced_peak) / enhanced_peak * 100
            axs[0, 1].fill_between(enhanced_dates, enhanced_drawdown, 0, color='green', alpha=0.3, label='Enhanced System')
        
        axs[0, 1].set_title('Drawdown Comparison')
        axs[0, 1].set_ylabel('Drawdown (%)')
        axs[0, 1].grid(True)
        axs[0, 1].legend()
        
        # Calculate monthly returns
        if original_dates and original_values:
            original_df = pd.DataFrame({'date': original_dates, 'value': original_values})
            original_df['date'] = pd.to_datetime(original_df['date'])
            original_df.set_index('date', inplace=True)
            original_monthly = original_df.resample('M').last()
            original_monthly_returns = original_monthly['value'].pct_change() * 100
            
            axs[1, 0].bar(original_monthly_returns.index, original_monthly_returns.values, 
                         color='blue', alpha=0.6, label='Original System')
        
        if enhanced_dates and enhanced_values:
            enhanced_df = pd.DataFrame({'date': enhanced_dates, 'value': enhanced_values})
            enhanced_df['date'] = pd.to_datetime(enhanced_df['date'])
            enhanced_df.set_index('date', inplace=True)
            enhanced_monthly = enhanced_df.resample('M').last()
            enhanced_monthly_returns = enhanced_monthly['value'].pct_change() * 100
            
            axs[1, 0].bar(enhanced_monthly_returns.index, enhanced_monthly_returns.values, 
                         color='green', alpha=0.6, label='Enhanced System')
        
        axs[1, 0].set_title('Monthly Returns')
        axs[1, 0].set_ylabel('Return (%)')
        axs[1, 0].grid(True)
        axs[1, 0].legend()
        
        # Create comparison table
        metrics = [
            'Total Return (%)',
            'Annualized Return (%)',
            'Sharpe Ratio',
            'Max Drawdown (%)',
            'Win Rate (%)',
            'Profit Factor'
        ]
        
        original_metrics = [
            original_results.get('total_return_pct', 0),
            original_results.get('annualized_return_pct', 0),
            original_results.get('sharpe_ratio', 0),
            original_results.get('max_drawdown_pct', 0),
            original_results.get('win_rate', 0),
            original_results.get('profit_factor', 0)
        ]
        
        enhanced_metrics = [
            enhanced_results.get('total_return_pct', 0),
            enhanced_results.get('annualized_return_pct', 0),
            enhanced_results.get('sharpe_ratio', 0),
            enhanced_results.get('max_drawdown_pct', 0),
            enhanced_results.get('win_rate', 0),
            enhanced_results.get('profit_factor', 0)
        ]
        
        improvement = [(e - o) for o, e in zip(original_metrics, enhanced_metrics)]
        
        # Create table
        axs[1, 1].axis('tight')
        axs[1, 1].axis('off')
        table_data = []
        for i, metric in enumerate(metrics):
            if i in [0, 1, 3, 4]:  # Metrics that are percentages
                table_data.append([
                    metric, 
                    f"{original_metrics[i]:.2f}%", 
                    f"{enhanced_metrics[i]:.2f}%", 
                    f"{improvement[i]:.2f}%"
                ])
            else:
                table_data.append([
                    metric, 
                    f"{original_metrics[i]:.2f}", 
                    f"{enhanced_metrics[i]:.2f}", 
                    f"{improvement[i]:.2f}"
                ])
        
        table = axs[1, 1].table(
            cellText=table_data,
            colLabels=['Metric', 'Original', 'Enhanced', 'Improvement'],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        axs[1, 1].set_title('Performance Metrics Comparison')
        
        # Add overall title
        plt.suptitle('Trading System Comparison: Original vs. Enhanced', fontsize=16)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        # Save figure if output file is specified
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {output_file}")
        
        plt.show()
        return True
    except Exception as e:
        logger.error(f"Error plotting comparison: {str(e)}")
        return False

def save_comparison_results(original_results, enhanced_results, output_file):
    """Save comparison results to JSON file"""
    try:
        # Calculate improvements
        total_return_improvement = enhanced_results.get('total_return_pct', 0) - original_results.get('total_return_pct', 0)
        annualized_return_improvement = enhanced_results.get('annualized_return_pct', 0) - original_results.get('annualized_return_pct', 0)
        sharpe_ratio_improvement = enhanced_results.get('sharpe_ratio', 0) - original_results.get('sharpe_ratio', 0)
        max_drawdown_improvement = original_results.get('max_drawdown_pct', 0) - enhanced_results.get('max_drawdown_pct', 0)
        win_rate_improvement = enhanced_results.get('win_rate', 0) - original_results.get('win_rate', 0)
        profit_factor_improvement = enhanced_results.get('profit_factor', 0) - original_results.get('profit_factor', 0)
        
        # Create comparison results
        comparison_results = {
            "original_system": {
                "total_return_pct": original_results.get('total_return_pct', 0),
                "annualized_return_pct": original_results.get('annualized_return_pct', 0),
                "sharpe_ratio": original_results.get('sharpe_ratio', 0),
                "max_drawdown_pct": original_results.get('max_drawdown_pct', 0),
                "win_rate": original_results.get('win_rate', 0),
                "profit_factor": original_results.get('profit_factor', 0),
                "total_trades": original_results.get('total_trades', 0)
            },
            "enhanced_system": {
                "total_return_pct": enhanced_results.get('total_return_pct', 0),
                "annualized_return_pct": enhanced_results.get('annualized_return_pct', 0),
                "sharpe_ratio": enhanced_results.get('sharpe_ratio', 0),
                "max_drawdown_pct": enhanced_results.get('max_drawdown_pct', 0),
                "win_rate": enhanced_results.get('win_rate', 0),
                "profit_factor": enhanced_results.get('profit_factor', 0),
                "total_trades": enhanced_results.get('total_trades', 0)
            },
            "improvements": {
                "total_return_pct": total_return_improvement,
                "annualized_return_pct": annualized_return_improvement,
                "sharpe_ratio": sharpe_ratio_improvement,
                "max_drawdown_pct": max_drawdown_improvement,
                "win_rate": win_rate_improvement,
                "profit_factor": profit_factor_improvement
            }
        }
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(comparison_results, f, indent=4)
        
        logger.info(f"Comparison results saved to {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error saving comparison results: {str(e)}")
        return False

def main():
    """Main function"""
    # Define backtest period (using 2023 data for Alpaca free tier compatibility)
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    
    # Load original results
    original_results = load_results("results_2023_yahoo.json")
    if not original_results:
        logger.error("Failed to load original results")
        return
    
    # Run enhanced backtest
    enhanced_results = run_enhanced_backtest("enhanced_alpaca_config.yaml", start_date, end_date)
    if not enhanced_results:
        logger.error("Failed to run enhanced backtest")
        return
    
    # Save enhanced results
    with open("enhanced_results_2023.json", 'w') as f:
        json.dump(enhanced_results, f, indent=4)
    logger.info("Enhanced results saved to enhanced_results_2023.json")
    
    # Save comparison results
    save_comparison_results(original_results, enhanced_results, "system_comparison_2023.json")
    
    # Plot comparison
    plot_comparison(original_results, enhanced_results, "system_comparison_2023.png")
    
    # Print summary
    original_return = original_results.get('total_return_pct', 0)
    enhanced_return = enhanced_results.get('total_return_pct', 0)
    improvement = enhanced_return - original_return
    
    logger.info("=== Trading System Comparison ===")
    logger.info(f"Original System Total Return: {original_return:.2f}%")
    logger.info(f"Enhanced System Total Return: {enhanced_return:.2f}%")
    logger.info(f"Improvement: {improvement:.2f}%")
    logger.info(f"Improvement Factor: {enhanced_return / original_return if original_return > 0 else 'N/A'}x")

if __name__ == "__main__":
    main()
