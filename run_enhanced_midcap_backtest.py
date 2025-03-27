#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Mid-Cap Backtest Runner

This script runs a backtest with all the fixes we've implemented:
1. Enhanced mid-cap stock selection with dynamic fetching from Alpaca
2. Tier filtering (only trading on tier 1 and tier 2 stocks)
3. Proper date range handling
4. Seasonality verification
"""

import os
import sys
import logging
import yaml
import json
from datetime import datetime, timedelta
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the required modules
from enhanced_midcap_selection import get_midcap_symbols
from integrate_enhanced_midcap import enforce_tier_filtering, integrate_enhanced_midcap

def run_enhanced_midcap_backtest(start_date=None, end_date=None, config_file='sp500_config_enhanced.yaml', 
                      max_signals=None, initial_capital=500, output=None):
    """
    Run a backtest with enhanced mid-cap selection and tier filtering
    
    Args:
        start_date: Start date for the backtest (YYYY-MM-DD)
        end_date: End date for the backtest (YYYY-MM-DD)
        config_file: Configuration file to use
        max_signals: Maximum number of signals to generate per day
        initial_capital: Initial capital for the backtest
        output: Output file for the backtest results
        
    Returns:
        True if the backtest was successful, False otherwise
    """
    try:
        # Set default dates if not provided
        if not start_date:
            start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        logger.info(f"Running backtest from {start_date} to {end_date}")
        logger.info(f"Using configuration file: {config_file}")
        
        # Step 1: Enforce tier filtering to only trade on tier 1 and tier 2 stocks
        logger.info("Step 1: Enforcing tier filtering...")
        if not enforce_tier_filtering(config_file):
            logger.error("Failed to enforce tier filtering")
            return False
        
        # Step 2: Integrate enhanced mid-cap selection
        logger.info("Step 2: Integrating enhanced mid-cap selection...")
        if not integrate_enhanced_midcap(config_file):
            logger.error("Failed to integrate enhanced mid-cap selection")
            return False
        
        # Step 3: Import and run the backtest
        logger.info("Step 3: Running backtest...")
        from final_sp500_strategy import run_backtest
        
        # Set tier thresholds to ensure proper filtering
        tier1_threshold = 0.9  # Only trade on signals with score >= 0.9 for tier 1
        tier2_threshold = 0.8  # Only trade on signals with score >= 0.8 for tier 2
        tier3_threshold = 0.0  # Don't trade on signals below tier 2
        
        # Run the backtest with all the fixes
        result = run_backtest(
            start_date=start_date,
            end_date=end_date,
            mode='backtest',
            max_signals=max_signals,
            initial_capital=initial_capital,
            tier1_threshold=tier1_threshold,
            tier2_threshold=tier2_threshold,
            tier3_threshold=tier3_threshold,
            weekly_selection=True,
            continuous_capital=False
        )
        
        # Step 4: Save the results if an output file is specified
        if output and result:
            logger.info(f"Step 4: Saving results to {output}...")
            try:
                with open(output, 'w') as f:
                    json.dump(result, f, indent=2)
                logger.info(f"Results saved to {output}")
            except Exception as e:
                logger.error(f"Error saving results: {str(e)}")
        
        return True
    except Exception as e:
        logger.error(f"Error running enhanced mid-cap backtest: {str(e)}")
        return False

def main():
    """Main function to run the enhanced mid-cap backtest"""
    parser = argparse.ArgumentParser(description='Run a backtest with enhanced mid-cap selection and tier filtering')
    parser.add_argument('--start-date', type=str, help='Start date for the backtest (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date for the backtest (YYYY-MM-DD)')
    parser.add_argument('--config', type=str, default='sp500_config_enhanced.yaml', help='Configuration file to use')
    parser.add_argument('--max-signals', type=int, help='Maximum number of signals to generate per day')
    parser.add_argument('--initial-capital', type=float, default=500, help='Initial capital for the backtest')
    parser.add_argument('--output', type=str, help='Output file for the backtest results')
    
    args = parser.parse_args()
    
    # Run the enhanced mid-cap backtest
    success = run_enhanced_midcap_backtest(
        start_date=args.start_date,
        end_date=args.end_date,
        config_file=args.config,
        max_signals=args.max_signals,
        initial_capital=args.initial_capital,
        output=args.output
    )
    
    if success:
        logger.info("Backtest completed successfully")
    else:
        logger.error("Backtest failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
