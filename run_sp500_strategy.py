#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run S&P 500 Trading Strategy
This script runs the S&P 500 trading strategy in either paper or live mode
"""

import os
import sys
import time
import logging
import argparse
import schedule
from datetime import datetime, timedelta
from final_sp500_strategy import SP500Strategy

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"strategy_run_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_strategy(use_live=False, max_retries=3):
    """Run the S&P 500 strategy with retry logic"""
    for attempt in range(max_retries):
        try:
            logger.info(f"Starting strategy run (attempt {attempt+1}/{max_retries})")
            strategy = SP500Strategy(use_live=use_live)
            executed_trades = strategy.run_strategy()
            
            logger.info(f"Strategy run completed with {len(executed_trades)} executed trades")
            return executed_trades
        except Exception as e:
            logger.error(f"Error running strategy (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 60  # Exponential backoff
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error("Failed to run strategy after maximum retries")
                return []

def schedule_strategy(use_live=False, run_time="09:35", days=None):
    """Schedule the strategy to run at a specific time on specific days"""
    if days is None:
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    
    logger.info(f"Scheduling strategy to run at {run_time} on {', '.join(days)}")
    
    for day in days:
        if day.lower() == "monday":
            schedule.every().monday.at(run_time).do(run_strategy, use_live=use_live)
        elif day.lower() == "tuesday":
            schedule.every().tuesday.at(run_time).do(run_strategy, use_live=use_live)
        elif day.lower() == "wednesday":
            schedule.every().wednesday.at(run_time).do(run_strategy, use_live=use_live)
        elif day.lower() == "thursday":
            schedule.every().thursday.at(run_time).do(run_strategy, use_live=use_live)
        elif day.lower() == "friday":
            schedule.every().friday.at(run_time).do(run_strategy, use_live=use_live)
    
    logger.info("Starting scheduler. Press Ctrl+C to exit.")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(30)  # Check every 30 seconds
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
    except Exception as e:
        logger.error(f"Scheduler error: {str(e)}")

def run_backtest(start_date, end_date, use_live=False):
    """Run a backtest for a specific period"""
    from final_sp500_strategy import run_backtest as run_sp500_backtest
    
    logger.info(f"Running backtest from {start_date} to {end_date}")
    
    try:
        signals = run_sp500_backtest(start_date, end_date, use_live=use_live)
        
        if signals:
            logger.info(f"Backtest generated {len(signals)} trade signals")
            
            # Count by direction
            long_signals = [s for s in signals if s['direction'] == 'LONG']
            short_signals = [s for s in signals if s['direction'] == 'SHORT']
            
            logger.info(f"LONG signals: {len(long_signals)}")
            logger.info(f"SHORT signals: {len(short_signals)}")
            
            # Save signals
            import pandas as pd
            signals_df = pd.DataFrame(signals)
            os.makedirs('backtest_results', exist_ok=True)
            signals_df.to_csv(f'backtest_results/backtest_signals_{start_date}_to_{end_date}.csv', index=False)
            
            return signals
        else:
            logger.warning("Backtest did not generate any signals")
            return []
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        return []

def test_api_connection():
    """Test connection to Alpaca API"""
    logger.info("Testing API connection")
    
    try:
        from debug_api_connection import main as run_api_debug
        run_api_debug()
        logger.info("API connection test completed")
    except Exception as e:
        logger.error(f"Error testing API connection: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run S&P 500 Trading Strategy')
    parser.add_argument('--live', action='store_true', help='Use live trading')
    parser.add_argument('--schedule', action='store_true', help='Schedule the strategy to run daily')
    parser.add_argument('--run-time', type=str, default="09:35", help='Time to run the strategy (HH:MM)')
    parser.add_argument('--days', type=str, nargs='+', help='Days to run the strategy')
    parser.add_argument('--backtest', action='store_true', help='Run backtest')
    parser.add_argument('--start-date', type=str, help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date for backtest (YYYY-MM-DD)')
    parser.add_argument('--test-api', action='store_true', help='Test API connection')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Test API connection if requested
    if args.test_api:
        test_api_connection()
        sys.exit(0)
    
    # Run backtest if requested
    if args.backtest:
        if not args.start_date or not args.end_date:
            logger.error("Start date and end date are required for backtest")
            sys.exit(1)
        
        run_backtest(args.start_date, args.end_date, use_live=args.live)
        sys.exit(0)
    
    # Schedule the strategy if requested
    if args.schedule:
        schedule_strategy(use_live=args.live, run_time=args.run_time, days=args.days)
    else:
        # Run the strategy once
        run_strategy(use_live=args.live)
