#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Alpaca Paper Trading Runner
-----------------------------------
This script runs the enhanced Alpaca trading system in paper trading mode,
allowing for testing the system without risking real money.
"""

import os
import sys
import json
import yaml
import logging
import argparse
import datetime as dt
import time
from enhanced_alpaca_trading import EnhancedAlpacaTradingSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enhanced_paper_trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("enhanced_paper_trading")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run Enhanced Alpaca Trading System in Paper Trading Mode")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="enhanced_alpaca_config.yaml",
        help="Path to configuration file (default: enhanced_alpaca_config.yaml)"
    )
    
    parser.add_argument(
        "--run-once", 
        action="store_true",
        help="Run the trading system once and exit (default: False)"
    )
    
    parser.add_argument(
        "--interval", 
        type=int, 
        default=60,
        help="Trading interval in seconds (default: 60)"
    )
    
    parser.add_argument(
        "--enable-trading", 
        action="store_true",
        help="Enable actual paper trading (default: False, will only generate signals)"
    )
    
    return parser.parse_args()

def update_config_for_paper_trading(config_file):
    """Update configuration for paper trading"""
    try:
        # Load configuration
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update configuration for paper trading
        config['backtesting_mode'] = False
        config['enable_auto_trading'] = True
        config['data_source'] = 'ALPACA'
        
        # Save updated configuration
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Updated configuration for paper trading")
        return True
    except Exception as e:
        logger.error(f"Error updating configuration: {str(e)}")
        return False

def run_paper_trading(args):
    """Run paper trading with the enhanced trading system"""
    try:
        logger.info("Starting Enhanced Alpaca Paper Trading")
        
        # Update configuration for paper trading if trading is enabled
        if args.enable_trading:
            if not update_config_for_paper_trading(args.config):
                logger.error("Failed to update configuration for paper trading")
                return False
        
        # Initialize trading system
        trading_system = EnhancedAlpacaTradingSystem(config_file=args.config, mode='paper')
        
        # Check if the system is initialized correctly
        if not trading_system.system:
            logger.error("Failed to initialize trading system")
            return False
        
        # Run once or continuously
        if args.run_once:
            logger.info("Running trading system once")
            
            # Get current time
            current_time = dt.datetime.now()
            
            # Run trading system
            if args.enable_trading:
                trading_system.run_trading_cycle(current_time)
            else:
                # Just generate signals without trading
                signals = trading_system.generate_signals(current_time)
                logger.info(f"Generated {len(signals)} signals")
                
                # Display signals
                for signal in signals:
                    logger.info(f"Signal: {signal.symbol} - {signal.direction} - {signal.strategy} - Score: {signal.score:.2f}")
        else:
            logger.info(f"Running trading system continuously with {args.interval} second interval")
            logger.info(f"Trading enabled: {args.enable_trading}")
            
            # Run continuously
            while True:
                try:
                    # Get current time
                    current_time = dt.datetime.now()
                    
                    # Check if market is open
                    calendar = trading_system.api.get_calendar(
                        start=current_time.strftime("%Y-%m-%d"),
                        end=current_time.strftime("%Y-%m-%d")
                    )
                    
                    if not calendar:
                        logger.info("Market is closed today")
                        # Sleep for an hour and check again
                        time.sleep(3600)
                        continue
                    
                    market_open = dt.datetime.fromisoformat(calendar[0].open.replace('Z', '+00:00')).replace(tzinfo=None)
                    market_close = dt.datetime.fromisoformat(calendar[0].close.replace('Z', '+00:00')).replace(tzinfo=None)
                    
                    # Convert to local time (assuming Alpaca returns times in UTC)
                    market_open = market_open - dt.timedelta(hours=4)  # EST timezone
                    market_close = market_close - dt.timedelta(hours=4)  # EST timezone
                    
                    # Check if market is open
                    if current_time < market_open:
                        logger.info(f"Market is not open yet. Market opens at {market_open.strftime('%H:%M:%S')}")
                        # Sleep until market open
                        sleep_time = (market_open - current_time).total_seconds()
                        if sleep_time > 0:
                            logger.info(f"Sleeping for {sleep_time:.0f} seconds until market open")
                            time.sleep(min(sleep_time, 3600))  # Sleep for at most an hour
                        continue
                    
                    if current_time > market_close:
                        logger.info(f"Market is closed. Market closed at {market_close.strftime('%H:%M:%S')}")
                        # Sleep until next day
                        next_day = current_time + dt.timedelta(days=1)
                        next_day = next_day.replace(hour=9, minute=0, second=0)
                        sleep_time = (next_day - current_time).total_seconds()
                        logger.info(f"Sleeping until tomorrow morning: {next_day.strftime('%Y-%m-%d %H:%M:%S')}")
                        time.sleep(min(sleep_time, 3600))  # Sleep for at most an hour
                        continue
                    
                    # Market is open, run trading system
                    logger.info(f"Market is open. Running trading cycle at {current_time.strftime('%H:%M:%S')}")
                    
                    if args.enable_trading:
                        trading_system.run_trading_cycle(current_time)
                    else:
                        # Just generate signals without trading
                        signals = trading_system.generate_signals(current_time)
                        logger.info(f"Generated {len(signals)} signals")
                        
                        # Display signals
                        for signal in signals:
                            logger.info(f"Signal: {signal.symbol} - {signal.direction} - {signal.strategy} - Score: {signal.score:.2f}")
                    
                    # Sleep for the specified interval
                    logger.info(f"Sleeping for {args.interval} seconds")
                    time.sleep(args.interval)
                    
                except KeyboardInterrupt:
                    logger.info("Trading system stopped by user")
                    break
                except Exception as e:
                    logger.error(f"Error in trading cycle: {str(e)}")
                    # Sleep for a while before retrying
                    time.sleep(60)
        
        return True
    except Exception as e:
        logger.error(f"Error running paper trading: {str(e)}")
        return False

def main():
    """Main function"""
    args = parse_arguments()
    run_paper_trading(args)

if __name__ == "__main__":
    main()
