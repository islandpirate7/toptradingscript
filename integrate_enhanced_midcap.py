#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Integration Script for Enhanced Mid-Cap Selection

This script integrates the enhanced mid-cap selection module with the existing trading system.
It patches the get_midcap_symbols function in final_sp500_strategy.py to use our enhanced version.
"""

import os
import sys
import logging
import importlib.util
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('integration.log'),
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

def integrate_enhanced_midcap(config_file='sp500_config_enhanced.yaml'):
    """Integrate the enhanced mid-cap selection with the trading system"""
    try:
        # Import the enhanced mid-cap selection module
        enhanced_midcap = import_module_from_file(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'enhanced_midcap_selection.py'),
            'enhanced_midcap_selection'
        )
        
        if not enhanced_midcap:
            logger.error("Failed to import enhanced mid-cap selection module")
            return False
        
        # Import the final_sp500_strategy module
        strategy_module = import_module_from_file(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'final_sp500_strategy.py'),
            'final_sp500_strategy'
        )
        
        if not strategy_module:
            logger.error("Failed to import final_sp500_strategy module")
            return False
        
        # Replace the get_midcap_symbols function
        original_function = strategy_module.get_midcap_symbols
        strategy_module.get_midcap_symbols = enhanced_midcap.get_midcap_symbols
        
        # Override the CONFIG global variable to use our enhanced config
        try:
            # Load the enhanced configuration
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_file)
            with open(config_path, 'r') as f:
                import yaml
                enhanced_config = yaml.safe_load(f)
                
            # Set the CONFIG global variable in the strategy module
            if hasattr(strategy_module, 'CONFIG'):
                original_config = strategy_module.CONFIG
                strategy_module.CONFIG = enhanced_config
                logger.info(f"Successfully overrode configuration with {config_file}")
                
                # Ensure mid-cap inclusion is enabled
                if 'strategy' in strategy_module.CONFIG and 'include_midcap' not in strategy_module.CONFIG['strategy']:
                    strategy_module.CONFIG['strategy']['include_midcap'] = True
                    logger.info("Explicitly enabled mid-cap inclusion in configuration")
        except Exception as e:
            logger.error(f"Error overriding configuration: {str(e)}")
            # Continue even if config override fails
        
        logger.info("Successfully integrated enhanced mid-cap selection with trading system")
        logger.info(f"Original function: {original_function.__name__} from {original_function.__module__}")
        logger.info(f"New function: {enhanced_midcap.get_midcap_symbols.__name__} from {enhanced_midcap.get_midcap_symbols.__module__}")
        
        return True
    except Exception as e:
        logger.error(f"Error integrating enhanced mid-cap selection: {str(e)}")
        return False

def enforce_tier_filtering(config_file='sp500_config_enhanced.yaml'):
    """
    Enforce tier filtering to only trade on tier 1 and tier 2 stocks
    
    This function ensures that the position sizing multipliers are set correctly
    to only trade on tier 1 and tier 2 stocks (pass 0.0 to signals below tier 2)
    """
    try:
        # Load the configuration
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_file)
        with open(config_path, 'r') as f:
            import yaml
            config = yaml.safe_load(f)
        
        # Ensure the position sizing multipliers are set correctly
        if 'strategy' in config and 'position_sizing' in config['strategy'] and 'tier_multipliers' in config['strategy']['position_sizing']:
            tier_multipliers = config['strategy']['position_sizing']['tier_multipliers']
            
            # Set tier 1 and tier 2 multipliers
            tier_multipliers['Tier 1 (≥0.9)'] = 3.0  # Highest priority
            tier_multipliers['Tier 2 (0.8-0.9)'] = 1.5  # Medium priority
            tier_multipliers['Below Threshold (<0.8)'] = 0.0  # No trading below tier 2
            
            logger.info("Enforced tier filtering: Only trading on tier 1 and tier 2 stocks")
            
            # Save the updated configuration
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            return True
        else:
            logger.error("Could not find position sizing tier multipliers in configuration")
            return False
    except Exception as e:
        logger.error(f"Error enforcing tier filtering: {str(e)}")
        return False

def run_backtest_with_enhanced_midcap(start_date: str, end_date: str, config_file: str = 'sp500_config_enhanced.yaml', max_signals: int = None, 
                                     initial_capital: float = 500, random_seed: int = 42, output: str = None,
                                     tier1_threshold: float = None, tier2_threshold: float = None, tier3_threshold: float = None,
                                     weekly_selection: bool = False, continuous_capital: bool = False):
    """Run a backtest with the enhanced mid-cap selection"""
    try:
        # Enforce tier filtering to only trade on tier 1 and tier 2 stocks
        enforce_tier_filtering(config_file)
        
        # Integrate the enhanced mid-cap selection
        if not integrate_enhanced_midcap(config_file):
            logger.error("Failed to integrate enhanced mid-cap selection")
            return False
        
        # Import the trading_cli module
        trading_cli = import_module_from_file(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trading_cli.py'),
            'trading_cli'
        )
        
        if not trading_cli:
            logger.error("Failed to import trading_cli module")
            return False
        
        # Create arguments for the backtest
        class Args:
            def __init__(self):
                self.start_date = start_date
                self.end_date = end_date
                self.config = config_file
                self.output = output
                self.random_seed = random_seed
                self.initial_capital = initial_capital
                self.max_signals = max_signals
                self.tier1_threshold = tier1_threshold
                self.tier2_threshold = tier2_threshold
                self.tier3_threshold = tier3_threshold
                self.weekly_selection = weekly_selection
                self.continuous_capital = continuous_capital
        
        # Run the backtest
        logger.info(f"Running backtest from {start_date} to {end_date} with enhanced mid-cap selection")
        if max_signals:
            logger.info(f"Maximum signals per day limited to: {max_signals}")
        trading_cli.run_backtest(Args())
        
        logger.info("Backtest completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error running backtest with enhanced mid-cap selection: {str(e)}")
        return False

if __name__ == "__main__":
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run a backtest with enhanced mid-cap selection')
    parser.add_argument('--start-date', type=str, required=True, help='Start date for the backtest (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True, help='End date for the backtest (YYYY-MM-DD)')
    parser.add_argument('--config', type=str, default='sp500_config_enhanced.yaml', help='Configuration file')
    parser.add_argument('--max-signals', type=int, help='Maximum number of trading signals to generate per day')
    parser.add_argument('--initial-capital', type=float, default=500, help='Initial capital for the backtest')
    parser.add_argument('--random-seed', dest='random_seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--output', type=str, help='Output file for backtest results')
    parser.add_argument('--tier1-threshold', type=float, help='Threshold for tier 1 signals')
    parser.add_argument('--tier2-threshold', type=float, help='Threshold for tier 2 signals')
    parser.add_argument('--tier3-threshold', type=float, help='Threshold for tier 3 signals')
    parser.add_argument('--weekly-selection', action='store_true', help='Use weekly symbol selection')
    parser.add_argument('--continuous-capital', action='store_true', help='Use continuous capital adjustment')
    
    args = parser.parse_args()
    
    # Run the backtest
    run_backtest_with_enhanced_midcap(args.start_date, args.end_date, args.config, args.max_signals, 
                                     args.initial_capital, args.random_seed, args.output, 
                                     args.tier1_threshold, args.tier2_threshold, args.tier3_threshold,
                                     args.weekly_selection, args.continuous_capital)
