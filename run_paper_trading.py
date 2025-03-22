#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Paper Trading Script for S&P 500 Strategy
This script runs the S&P 500 strategy in paper trading mode using Alpaca API
"""

import os
import json
import yaml
import time
import logging
import argparse
import traceback
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
from final_sp500_strategy import SP500Strategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"paper_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_file='sp500_config.yaml'):
    """Load configuration from YAML file"""
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Successfully loaded configuration from {config_file}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise

def load_alpaca_credentials(mode='paper'):
    """Load Alpaca API credentials from JSON file"""
    try:
        with open('alpaca_credentials.json', 'r') as f:
            credentials = json.load(f)
        return credentials[mode]
    except Exception as e:
        logger.error(f"Error loading Alpaca credentials: {str(e)}")
        raise

def run_paper_trading(max_signals=20, duration_hours=1, check_interval_minutes=5):
    """
    Run the S&P 500 strategy in paper trading mode
    
    Args:
        max_signals (int): Maximum number of signals to act on
        duration_hours (int): How long to run the paper trading session in hours
        check_interval_minutes (int): How often to check for new signals in minutes
    """
    try:
        # Load configuration
        config = load_config()
        
        # Load Alpaca credentials
        credentials = load_alpaca_credentials('paper')
        
        # Initialize Alpaca API
        api = tradeapi.REST(
            credentials['api_key'],
            credentials['api_secret'],
            credentials['base_url'],
            api_version='v2'
        )
        
        # Initialize strategy
        strategy = SP500Strategy(
            api=api,
            config=config,
            mode='paper'
        )
        
        # Create output directories if they don't exist
        for path_key in ['trades', 'performance']:
            os.makedirs(config['paths'][path_key], exist_ok=True)
        
        # Calculate end time
        end_time = datetime.now() + timedelta(hours=duration_hours)
        
        logger.info(f"Starting paper trading session until {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Max signals: {max_signals}, Check interval: {check_interval_minutes} minutes")
        
        # Main trading loop
        while datetime.now() < end_time:
            try:
                # Run strategy to generate signals
                logger.info("Generating new trading signals...")
                signals = strategy.run_strategy()
                
                # Log signal summary
                logger.info(f"Generated {len(signals)} total signals")
                if signals:
                    top_signals = sorted(signals, key=lambda x: x['score'], reverse=True)[:5]
                    logger.info(f"Top signals: {', '.join([f'{s['symbol']} ({s['score']:.3f})' for s in top_signals])}")
                
                # Limit signals based on max_signals parameter
                if len(signals) > max_signals:
                    logger.info(f"Limiting to top {max_signals} signals")
                    signals = sorted(signals, key=lambda x: x['score'], reverse=True)[:max_signals]
                
                # Execute trades based on signals
                if signals:
                    logger.info(f"Executing {len(signals)} paper trades")
                    strategy.execute_trades(signals)
                
                # Check stop loss conditions
                logger.info("Checking stop loss conditions for existing positions")
                closed_positions = strategy.check_stop_loss_conditions()
                if closed_positions:
                    logger.info(f"Closed {len(closed_positions)} positions due to stop loss")
                
                # Save current portfolio status
                account = api.get_account()
                positions = api.list_positions()
                
                logger.info(f"Current account value: ${float(account.equity):.2f}")
                logger.info(f"Cash: ${float(account.cash):.2f}")
                logger.info(f"Active positions: {len(positions)}")
                
                # Log all open positions
                if positions:
                    for position in positions:
                        current_price = float(position.current_price)
                        entry_price = float(position.avg_entry_price)
                        pl_percent = (current_price - entry_price) / entry_price * 100
                        logger.info(f"Position: {position.symbol}, Qty: {position.qty}, " +
                                   f"P/L: ${float(position.unrealized_pl):.2f} ({pl_percent:.2f}%)")
                
                # Wait for the next check interval
                next_check = datetime.now() + timedelta(minutes=check_interval_minutes)
                logger.info(f"Next check at {next_check.strftime('%H:%M:%S')}")
                time.sleep(check_interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {str(e)}")
                traceback.print_exc()
                # Continue the loop despite errors
                time.sleep(60)  # Wait a minute before retrying
        
        # End of paper trading session
        logger.info("Paper trading session completed")
        
        # Final account summary
        try:
            account = api.get_account()
            positions = api.list_positions()
            
            logger.info("=== FINAL ACCOUNT SUMMARY ===")
            logger.info(f"Account value: ${float(account.equity):.2f}")
            logger.info(f"Cash: ${float(account.cash):.2f}")
            logger.info(f"P/L: ${float(account.equity) - float(account.last_equity):.2f}")
            logger.info(f"Open positions: {len(positions)}")
            
            # Save final positions to CSV
            if positions:
                import pandas as pd
                positions_data = []
                for position in positions:
                    positions_data.append({
                        'symbol': position.symbol,
                        'qty': position.qty,
                        'entry_price': position.avg_entry_price,
                        'current_price': position.current_price,
                        'market_value': position.market_value,
                        'unrealized_pl': position.unrealized_pl,
                        'unrealized_plpc': position.unrealized_plpc
                    })
                
                positions_df = pd.DataFrame(positions_data)
                positions_file = os.path.join(
                    config['paths']['trades'], 
                    f"paper_positions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                )
                positions_df.to_csv(positions_file, index=False)
                logger.info(f"Final positions saved to {positions_file}")
        
        except Exception as e:
            logger.error(f"Error generating final summary: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error in paper trading: {str(e)}")
        traceback.print_exc()

def main():
    """Main function to run paper trading"""
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Run S&P 500 strategy in paper trading mode')
        parser.add_argument('--max_signals', type=int, default=20, 
                           help='Maximum number of signals to act on')
        parser.add_argument('--duration', type=int, default=1, 
                           help='Duration of paper trading session in hours')
        parser.add_argument('--interval', type=int, default=5, 
                           help='Check interval in minutes')
        args = parser.parse_args()
        
        # Run paper trading
        run_paper_trading(
            max_signals=args.max_signals,
            duration_hours=args.duration,
            check_interval_minutes=args.interval
        )
        
    except Exception as e:
        logger.error(f"Error running paper trading: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
