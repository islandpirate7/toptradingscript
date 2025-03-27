#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Debug script for Enhanced Mid-Cap Selection and Tier Filtering

This script helps debug the enhanced mid-cap selection process and tier filtering
by showing detailed information about the selected symbols, their metrics, and 
the selection criteria. It also tests the integration of the enhanced mid-cap 
selection with tier filtering.
"""

import os
import sys
import logging
import importlib.util
import yaml
import json
from datetime import datetime, timedelta
import pandas as pd
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('midcap_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def import_module_from_file(file_path: str, module_name: str):
    """Import a module from a file path"""
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        logger.error(f"Error importing module from {file_path}: {str(e)}")
        return None

def load_config(config_file='sp500_config_enhanced.yaml'):
    """Load the configuration from the YAML file"""
    try:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_file)
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_file}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration from {config_file}: {str(e)}")
        return {}

def debug_midcap_selection(config_file='sp500_config_enhanced.yaml'):
    """Debug the enhanced mid-cap selection process"""
    try:
        # Import the enhanced mid-cap selection module
        enhanced_midcap = import_module_from_file(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'enhanced_midcap_selection.py'),
            'enhanced_midcap_selection'
        )
        
        if not enhanced_midcap:
            logger.error("Failed to import enhanced mid-cap selection module")
            return False
        
        # Load the configuration
        config = load_config(config_file)
        if not config:
            logger.error("Failed to load configuration")
            return False
        
        # Debug the mid-cap selection process
        logger.info("Starting mid-cap selection debug process")
        logger.info(f"Configuration file: {config_file}")
        
        # Check if mid-cap inclusion is enabled
        include_midcap = config.get('strategy', {}).get('include_midcap', False)
        logger.info(f"Mid-cap inclusion enabled: {include_midcap}")
        
        if not include_midcap:
            logger.warning("Mid-cap inclusion is disabled in the configuration")
            logger.info("Enabling mid-cap inclusion for debugging purposes")
            if 'strategy' not in config:
                config['strategy'] = {}
            config['strategy']['include_midcap'] = True
        
        # Check mid-cap configuration
        midcap_config = config.get('strategy', {}).get('midcap_stocks', {})
        logger.info(f"Mid-cap configuration: {midcap_config}")
        
        # Get mid-cap symbols
        logger.info("Calling get_midcap_symbols function")
        midcap_symbols = enhanced_midcap.get_midcap_symbols(config)
        
        if not midcap_symbols:
            logger.error("No mid-cap symbols were selected")
            return False
        
        logger.info(f"Selected {len(midcap_symbols)} mid-cap symbols: {midcap_symbols}")
        
        # Try to get the detailed metrics if available
        try:
            if hasattr(enhanced_midcap, 'get_midcap_metrics'):
                logger.info("Getting detailed metrics for mid-cap symbols")
                metrics = enhanced_midcap.get_midcap_metrics(config)
                
                if isinstance(metrics, pd.DataFrame):
                    logger.info(f"Metrics shape: {metrics.shape}")
                    logger.info(f"Metrics columns: {metrics.columns.tolist()}")
                    
                    # Save metrics to CSV for inspection
                    metrics_file = 'midcap_metrics_debug.csv'
                    metrics.to_csv(metrics_file)
                    logger.info(f"Saved metrics to {metrics_file}")
                    
                    # Show top 10 symbols by combined score
                    if 'combined_score' in metrics.columns:
                        top_symbols = metrics.sort_values('combined_score', ascending=False).head(10)
                        logger.info(f"Top 10 symbols by combined score:\n{top_symbols[['combined_score', 'avg_volume', 'momentum', 'volatility', 'liquidity']]}")
        except Exception as e:
            logger.error(f"Error getting detailed metrics: {str(e)}")
        
        return True
    except Exception as e:
        logger.error(f"Error debugging mid-cap selection: {str(e)}")
        return False

def main():
    """Main function to debug the mid-cap selection process"""
    config_file = 'sp500_config_enhanced.yaml'
    
    # Load configuration
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_file}")
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return
    
    logger.info("Starting mid-cap selection debug process")
    logger.info(f"Configuration file: {config_file}")
    
    # Check if mid-cap inclusion is enabled
    midcap_enabled = config.get('strategy', {}).get('include_midcap', False)
    logger.info(f"Mid-cap inclusion enabled: {midcap_enabled}")
    
    # Get mid-cap configuration
    midcap_config = config.get('strategy', {}).get('midcap_stocks', {})
    logger.info(f"Mid-cap configuration: {midcap_config}")
    
    # Call the get_midcap_symbols function
    logger.info("Calling get_midcap_symbols function")
    from enhanced_midcap_selection import get_midcap_symbols
    midcap_symbols = get_midcap_symbols(config)
    logger.info(f"Selected {len(midcap_symbols)} mid-cap symbols: {midcap_symbols}")
    
    # Enforce tier filtering
    logger.info("Enforcing tier filtering to only trade on tier 1 and tier 2 stocks")
    from integrate_enhanced_midcap import enforce_tier_filtering
    tier_filtering_result = enforce_tier_filtering(config_file)
    logger.info(f"Tier filtering enforcement result: {tier_filtering_result}")
    
    # Re-load configuration to verify tier filtering changes
    try:
        with open(config_file, 'r') as f:
            updated_config = yaml.safe_load(f)
        tier_multipliers = updated_config.get('strategy', {}).get('position_sizing', {}).get('tier_multipliers', {})
        logger.info(f"Updated tier multipliers: {tier_multipliers}")
        
        # Verify that tier 3 and below have zero multipliers
        below_threshold_multiplier = tier_multipliers.get('Below Threshold (<0.8)', None)
        logger.info(f"Below Threshold (<0.8) multiplier: {below_threshold_multiplier}")
        
        if below_threshold_multiplier == 0.0:
            logger.info("Tier filtering is correctly set to only trade on tier 1 and tier 2 stocks")
        else:
            logger.warning("Tier filtering is NOT correctly set - below threshold multiplier should be 0.0")
    except Exception as e:
        logger.error(f"Error verifying tier filtering: {str(e)}")
    
    # Test running a backtest with the enhanced mid-cap selection
    logger.info("Testing backtest with enhanced mid-cap selection")
    
    # Import the backtest function
    try:
        from final_sp500_strategy import run_backtest
        
        # Set up backtest parameters
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Backtest period: {start_date} to {end_date}")
        
        # Run a quick test backtest
        logger.info("Running a quick test backtest...")
        
        # Check if the run_backtest function exists and has the right parameters
        import inspect
        backtest_params = inspect.signature(run_backtest).parameters
        logger.info(f"Backtest function parameters: {list(backtest_params.keys())}")
        
        # Check if max_signals parameter exists
        if 'max_signals' in backtest_params:
            logger.info("max_signals parameter exists in run_backtest function")
        else:
            logger.warning("max_signals parameter does not exist in run_backtest function")
            
        # Check if tier thresholds are available
        if 'tier1_threshold' in backtest_params and 'tier2_threshold' in backtest_params:
            logger.info("tier threshold parameters exist in run_backtest function")
        else:
            logger.warning("tier threshold parameters do not exist in run_backtest function")
        
    except ImportError as e:
        logger.error(f"Error importing run_backtest function: {str(e)}")
    except Exception as e:
        logger.error(f"Error testing backtest: {str(e)}")
    
    logger.info("Mid-cap selection debug process completed")

if __name__ == "__main__":
    main()
