#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to verify the fixes for:
1. Different quarters producing different results
2. Continuous capital option working correctly
"""

import os
import sys
import json
import logging
import pandas as pd
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the real run_backtest function from final_sp500_strategy
from final_sp500_strategy import run_backtest

def test_different_quarters():
    """Test that different quarters produce different results"""
    logger.info("=== Testing Different Quarters ===")
    
    # Define quarters to test
    quarters = {
        'Q3_2023': ('2023-07-01', '2023-09-30'),
        'Q4_2023': ('2023-10-01', '2023-12-31'),
    }
    
    results = {}
    
    # Run backtest for each quarter
    for quarter_name, (start_date, end_date) in quarters.items():
        logger.info(f"Running backtest for {quarter_name}: {start_date} to {end_date}")
        
        summary, signals = run_backtest(
            start_date,
            end_date,
            mode='backtest',
            max_signals=50,  # Using a smaller number for faster testing
            initial_capital=300,
            weekly_selection=True,
            continuous_capital=False
        )
        
        if summary:
            results[quarter_name] = {
                'total_trades': summary.get('total_trades', 0),
                'win_rate': summary.get('win_rate', 0),
                'total_return': summary.get('total_return', 0),
                'final_capital': summary.get('final_capital', 0)
            }
            
            # Log a few sample trades to verify they're different
            if signals and len(signals) > 0:
                logger.info(f"Sample trades for {quarter_name}:")
                for i, trade in enumerate(signals[:3]):
                    logger.info(f"  Trade {i+1}: {trade.get('symbol')} - Entry: {trade.get('entry_date')} - Exit: {trade.get('exit_date')}")
    
    # Compare results to verify they're different
    if len(results) >= 2:
        quarters_list = list(results.keys())
        q1, q2 = quarters_list[0], quarters_list[1]
        
        logger.info(f"\nComparing results between {q1} and {q2}:")
        for metric in ['total_trades', 'win_rate', 'total_return', 'final_capital']:
            v1 = results[q1][metric]
            v2 = results[q2][metric]
            different = v1 != v2
            logger.info(f"  {metric}: {v1} vs {v2} - {'DIFFERENT ✓' if different else 'SAME ✗'}")
        
        # Overall assessment
        metrics_different = sum(1 for metric in ['total_trades', 'win_rate', 'total_return', 'final_capital'] 
                               if results[q1][metric] != results[q2][metric])
        
        if metrics_different >= 3:
            logger.info("\n✅ PASS: Different quarters produce different results")
        else:
            logger.info("\n❌ FAIL: Different quarters produce similar results")
    else:
        logger.error("Not enough results to compare quarters")

def test_continuous_capital():
    """Test that continuous capital option works correctly"""
    logger.info("\n=== Testing Continuous Capital ===")
    
    # Define quarters to test in sequence
    quarters = {
        'Q3_2023': ('2023-07-01', '2023-09-30'),
        'Q4_2023': ('2023-10-01', '2023-12-31'),
    }
    
    # Test with continuous capital OFF
    logger.info("Running with continuous_capital=FALSE:")
    normal_results = {}
    normal_initial_capital = 300
    
    for quarter_name, (start_date, end_date) in quarters.items():
        logger.info(f"  Running {quarter_name} with initial_capital={normal_initial_capital}")
        
        summary, _ = run_backtest(
            start_date,
            end_date,
            mode='backtest',
            max_signals=50,
            initial_capital=normal_initial_capital,
            weekly_selection=True,
            continuous_capital=False
        )
        
        if summary:
            normal_results[quarter_name] = {
                'initial_capital': summary.get('initial_capital', normal_initial_capital),
                'final_capital': summary.get('final_capital', 0)
            }
            logger.info(f"  {quarter_name} - Initial: ${normal_results[quarter_name]['initial_capital']} - Final: ${normal_results[quarter_name]['final_capital']}")
    
    # Test with continuous capital ON
    logger.info("\nRunning with continuous_capital=TRUE:")
    continuous_results = {}
    continuous_initial_capital = 300
    previous_capital = continuous_initial_capital
    
    for quarter_name, (start_date, end_date) in quarters.items():
        logger.info(f"  Running {quarter_name} with initial_capital={previous_capital}")
        
        summary, _ = run_backtest(
            start_date,
            end_date,
            mode='backtest',
            max_signals=50,
            initial_capital=previous_capital,
            weekly_selection=True,
            continuous_capital=True
        )
        
        if summary:
            continuous_results[quarter_name] = {
                'initial_capital': summary.get('initial_capital', previous_capital),
                'final_capital': summary.get('final_capital', 0)
            }
            logger.info(f"  {quarter_name} - Initial: ${continuous_results[quarter_name]['initial_capital']} - Final: ${continuous_results[quarter_name]['final_capital']}")
            
            # Update previous capital for next quarter
            previous_capital = summary.get('final_capital', previous_capital)
    
    # Verify continuous capital is working
    if len(continuous_results) >= 2:
        quarters_list = list(continuous_results.keys())
        q1, q2 = quarters_list[0], quarters_list[1]
        
        # Check if second quarter's initial capital matches first quarter's final capital
        continuous_working = abs(continuous_results[q1]['final_capital'] - continuous_results[q2]['initial_capital']) < 0.01
        
        logger.info(f"\nContinuous capital test:")
        logger.info(f"  {q1} final capital: ${continuous_results[q1]['final_capital']}")
        logger.info(f"  {q2} initial capital: ${continuous_results[q2]['initial_capital']}")
        
        if continuous_working:
            logger.info("\n✅ PASS: Continuous capital is working correctly")
        else:
            logger.info("\n❌ FAIL: Continuous capital is not working correctly")
            
        # Compare with normal mode (should be different)
        if q2 in normal_results:
            normal_diff = normal_results[q2]['initial_capital'] != continuous_results[q2]['initial_capital']
            logger.info(f"\nNormal vs Continuous comparison for {q2}:")
            logger.info(f"  Normal initial capital: ${normal_results[q2]['initial_capital']}")
            logger.info(f"  Continuous initial capital: ${continuous_results[q2]['initial_capital']}")
            logger.info(f"  {'DIFFERENT ✓' if normal_diff else 'SAME ✗'}")
    else:
        logger.error("Not enough results to verify continuous capital")

if __name__ == "__main__":
    logger.info("Starting tests for backtest fixes...")
    
    # Test different quarters
    test_different_quarters()
    
    # Test continuous capital
    test_continuous_capital()
    
    logger.info("\nAll tests completed.")
