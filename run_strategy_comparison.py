#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Strategy Comparison Runner
-------------------------
This script runs both the original and improved mean reversion strategies
and compares their performance across all quarters of 2023.
"""

import os
import sys
import json
import yaml
import logging
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import subprocess

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import necessary modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from test_optimized_mean_reversion_alpaca import AlpacaBacktest
from improved_mean_reversion import ImprovedMeanReversionBacktest
from compare_strategies import compare_strategies, plot_equity_curves

# Custom JSON encoder for datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, dt.datetime):
            return obj.isoformat()
        return super(DateTimeEncoder, self).default(obj)

def run_original_strategy():
    """Run the original mean reversion strategy for all quarters of 2023"""
    logger.info("Running original mean reversion strategy...")
    
    # Define quarters
    quarters = [
        {"name": "Q1 2023", "start": "2023-01-01", "end": "2023-03-31"},
        {"name": "Q2 2023", "start": "2023-04-01", "end": "2023-06-30"},
        {"name": "Q3 2023", "start": "2023-07-01", "end": "2023-09-30"},
        {"name": "Q4 2023", "start": "2023-10-01", "end": "2023-12-31"}
    ]
    
    # Initialize results dictionary
    results = {
        "strategy_name": "Original Mean Reversion",
        "quarters": {},
        "overall": {
            "initial_capital": 100000,
            "final_capital": 100000,
            "return_pct": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "max_drawdown": 0,
            "total_trades": 0
        },
        "trades": []
    }
    
    # Create an instance of the improved strategy to use its data loading method
    improved_backtest = ImprovedMeanReversionBacktest('configuration_enhanced_mean_reversion.yaml')
    
    # Run backtest for each quarter
    for quarter in quarters:
        try:
            # Initialize the original strategy
            backtest = AlpacaBacktest('configuration_mean_reversion_final.yaml')
            
            # Load data using the improved strategy's method
            symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'AMD']
            symbol_data = improved_backtest.load_historical_data(quarter["start"], quarter["end"], symbols)
            
            # Manually set the symbol_data in the original strategy
            backtest.symbol_data = symbol_data
            
            # Run the backtest
            quarter_results = backtest.run_backtest(quarter["start"], quarter["end"])
            
            # Store results
            results["quarters"][quarter["name"]] = {
                "initial_capital": quarter_results.get("initial_capital", 100000),
                "final_capital": quarter_results.get("final_capital", 100000),
                "return_pct": quarter_results.get("return_pct", 0),
                "win_rate": quarter_results.get("win_rate", 0),
                "profit_factor": quarter_results.get("profit_factor", 0),
                "max_drawdown": quarter_results.get("max_drawdown", 0),
                "total_trades": quarter_results.get("total_trades", 0)
            }
            
            # Add trades to overall list
            results["trades"].extend(quarter_results.get("trades", []))
            
            logger.info(f"Completed {quarter['name']} backtest for original strategy")
        except Exception as e:
            logger.error(f"Error running original strategy for {quarter['name']}: {e}")
    
    # Calculate overall results
    if results["trades"]:
        # Calculate final capital using compound returns
        compound_return = 1.0
        for quarter_name, quarter_data in results["quarters"].items():
            compound_return *= (1 + quarter_data["return_pct"] / 100)
        
        results["overall"]["final_capital"] = results["overall"]["initial_capital"] * compound_return
        results["overall"]["return_pct"] = (compound_return - 1) * 100
        
        # Calculate win rate
        winning_trades = sum(1 for trade in results["trades"] if trade.get("profit_loss", 0) > 0)
        results["overall"]["win_rate"] = (winning_trades / len(results["trades"])) * 100 if results["trades"] else 0
        
        # Calculate profit factor
        gross_profit = sum(trade.get("profit_loss", 0) for trade in results["trades"] if trade.get("profit_loss", 0) > 0)
        gross_loss = abs(sum(trade.get("profit_loss", 0) for trade in results["trades"] if trade.get("profit_loss", 0) < 0))
        results["overall"]["profit_factor"] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Total trades
        results["overall"]["total_trades"] = len(results["trades"])
    
    # Save results to file
    with open('original_strategy_results.json', 'w') as f:
        json.dump(results, f, cls=DateTimeEncoder, indent=4)
    
    return results

def run_improved_strategy():
    """Run the improved mean reversion strategy for all quarters of 2023"""
    logger.info("Running improved mean reversion strategy...")
    
    # Define quarters
    quarters = [
        {"name": "Q1 2023", "start": "2023-01-01", "end": "2023-03-31"},
        {"name": "Q2 2023", "start": "2023-04-01", "end": "2023-06-30"},
        {"name": "Q3 2023", "start": "2023-07-01", "end": "2023-09-30"},
        {"name": "Q4 2023", "start": "2023-10-01", "end": "2023-12-31"}
    ]
    
    # Initialize results dictionary
    results = {
        "strategy_name": "Improved Mean Reversion",
        "quarters": {},
        "overall": {
            "initial_capital": 100000,
            "final_capital": 100000,
            "return_pct": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "max_drawdown": 0,
            "total_trades": 0
        },
        "trades": []
    }
    
    # Run backtest for each quarter
    for quarter in quarters:
        try:
            # Run the improved strategy
            backtest = ImprovedMeanReversionBacktest('configuration_enhanced_mean_reversion.yaml')
            quarter_results = backtest.run_backtest(quarter["start"], quarter["end"])
            
            # Store results
            results["quarters"][quarter["name"]] = {
                "initial_capital": quarter_results.get("initial_capital", 100000),
                "final_capital": quarter_results.get("final_capital", 100000),
                "return_pct": quarter_results.get("return_pct", 0),
                "win_rate": quarter_results.get("win_rate", 0),
                "profit_factor": quarter_results.get("profit_factor", 0),
                "max_drawdown": quarter_results.get("max_drawdown", 0),
                "total_trades": quarter_results.get("total_trades", 0)
            }
            
            # Add trades to overall list
            results["trades"].extend(quarter_results.get("trades", []))
            
            logger.info(f"Completed {quarter['name']} backtest for improved strategy")
        except Exception as e:
            logger.error(f"Error running improved strategy for {quarter['name']}: {e}")
    
    # Calculate overall results
    if results["trades"]:
        # Calculate final capital using compound returns
        compound_return = 1.0
        for quarter_name, quarter_data in results["quarters"].items():
            compound_return *= (1 + quarter_data["return_pct"] / 100)
        
        results["overall"]["final_capital"] = results["overall"]["initial_capital"] * compound_return
        results["overall"]["return_pct"] = (compound_return - 1) * 100
        
        # Calculate win rate
        winning_trades = sum(1 for trade in results["trades"] if trade.get("profit_loss", 0) > 0)
        results["overall"]["win_rate"] = (winning_trades / len(results["trades"])) * 100 if results["trades"] else 0
        
        # Calculate profit factor
        gross_profit = sum(trade.get("profit_loss", 0) for trade in results["trades"] if trade.get("profit_loss", 0) > 0)
        gross_loss = abs(sum(trade.get("profit_loss", 0) for trade in results["trades"] if trade.get("profit_loss", 0) < 0))
        results["overall"]["profit_factor"] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Total trades
        results["overall"]["total_trades"] = len(results["trades"])
    
    # Save results to file
    with open('improved_strategy_results.json', 'w') as f:
        json.dump(results, f, cls=DateTimeEncoder, indent=4)
    
    return results

def main():
    """Main function"""
    try:
        # Run both strategies
        original_results = run_original_strategy()
        improved_results = run_improved_strategy()
        
        # Compare strategies
        compare_strategies(original_results, improved_results, 
                          "Original Mean Reversion", "Improved Mean Reversion")
        
        # Plot equity curves
        plot_equity_curves(original_results, improved_results,
                          "Original Mean Reversion", "Improved Mean Reversion",
                          "strategy_comparison.png")
        
        logger.info("Strategy comparison completed successfully")
    except Exception as e:
        logger.error(f"Error in strategy comparison: {e}")

if __name__ == "__main__":
    main()
