#!/usr/bin/env python
"""
Alpaca Credentials Verification Tool
-----------------------------------
This script verifies that the Alpaca API credentials are working correctly
and provides detailed information about the account status.
"""

import os
import sys
import json
import yaml
import logging
import traceback
import argparse
from datetime import datetime
import alpaca_trade_api as tradeapi

# Import our custom modules
from trading_logger import get_logger
from error_handler import get_error_handler, error_context, APIError

# Get logger
logger = get_logger("credential_verifier")
error_handler = get_error_handler()

def load_alpaca_credentials(credentials_file='alpaca_credentials.json'):
    """Load Alpaca API credentials from JSON file"""
    try:
        with open(credentials_file, 'r') as file:
            credentials = json.load(file)
        return credentials
    except Exception as e:
        error = APIError(
            f"Error loading Alpaca credentials: {str(e)}",
            severity="ERROR",
            details={"file": credentials_file}
        )
        error_handler.handle_error(error)
        return None

def verify_credentials(mode='paper', verbose=False):
    """Verify Alpaca API credentials and account status"""
    with error_context({"operation": "verify_credentials", "mode": mode}):
        logger.info(f"Verifying {mode} trading credentials...")
        
        # Load credentials
        credentials = load_alpaca_credentials()
        if not credentials:
            logger.error("Failed to load credentials")
            return False
        
        if mode not in credentials:
            logger.error(f"No credentials found for mode: {mode}")
            return False
        
        # Extract credentials for the specified mode
        creds = credentials[mode]
        
        try:
            # Initialize Alpaca API
            api = tradeapi.REST(
                creds['api_key'],
                creds['api_secret'],
                creds['base_url'],
                api_version='v2'
            )
            
            # Test connection by getting account information
            account = api.get_account()
            
            # Basic verification passed
            logger.info(f"✅ Successfully connected to Alpaca API ({mode} mode)")
            logger.info(f"Account ID: {account.id}")
            logger.info(f"Account Status: {account.status}")
            
            if verbose:
                # Print detailed account information
                logger.info("\nDetailed Account Information:")
                logger.info(f"Buying Power: ${float(account.buying_power):.2f}")
                logger.info(f"Cash: ${float(account.cash):.2f}")
                logger.info(f"Portfolio Value: ${float(account.portfolio_value):.2f}")
                logger.info(f"Equity: ${float(account.equity):.2f}")
                
                # Check trading permissions
                logger.info("\nTrading Permissions:")
                if account.trading_blocked:
                    logger.warning("⚠️ Trading is currently blocked")
                else:
                    logger.info("✅ Trading is allowed")
                
                if account.account_blocked:
                    logger.warning("⚠️ Account is blocked")
                else:
                    logger.info("✅ Account is not blocked")
                
                # Check day trading status
                logger.info("\nDay Trading Status:")
                if account.pattern_day_trader:
                    logger.warning("⚠️ Account is flagged as a Pattern Day Trader")
                else:
                    logger.info("✅ Account is not flagged as a Pattern Day Trader")
                
                logger.info(f"Day Trade Count: {account.daytrade_count}")
                logger.info(f"Day Trading Buying Power: ${float(account.daytrading_buying_power):.2f}")
                
                # Get positions
                try:
                    positions = api.list_positions()
                    logger.info(f"\nCurrent Positions: {len(positions)}")
                    
                    if positions:
                        for position in positions:
                            market_value = float(position.market_value)
                            unrealized_pl = float(position.unrealized_pl)
                            unrealized_plpc = float(position.unrealized_plpc) * 100
                            
                            logger.info(f"  {position.symbol}: {position.qty} shares, Market Value: ${market_value:.2f}, "
                                      f"Unrealized P/L: ${unrealized_pl:.2f} ({unrealized_plpc:.2f}%)")
                except Exception as e:
                    logger.warning(f"Could not retrieve positions: {str(e)}")
                
                # Get recent orders
                try:
                    orders = api.list_orders(status='all', limit=5)
                    logger.info(f"\nRecent Orders: {len(orders)}")
                    
                    if orders:
                        for order in orders:
                            logger.info(f"  {order.symbol}: {order.qty} shares, Side: {order.side}, "
                                      f"Type: {order.type}, Status: {order.status}")
                except Exception as e:
                    logger.warning(f"Could not retrieve orders: {str(e)}")
            
            # Test market data access
            try:
                # Get a sample stock quote
                aapl = api.get_latest_trade('AAPL')
                logger.info(f"\nMarket Data Access: ✅ (AAPL last price: ${aapl.price:.2f})")
            except Exception as e:
                logger.warning(f"Could not access market data: {str(e)}")
            
            return True
            
        except Exception as e:
            error = APIError(
                f"Error verifying Alpaca credentials: {str(e)}",
                severity="ERROR",
                details={"mode": mode}
            )
            error_handler.handle_error(error)
            logger.error(f"❌ Failed to connect to Alpaca API: {str(e)}")
            return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Verify Alpaca API credentials')
    parser.add_argument('--mode', choices=['paper', 'live'], default='paper',
                        help='Trading mode (paper or live)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Display detailed account information')
    
    args = parser.parse_args()
    
    # Verify credentials
    success = verify_credentials(args.mode, args.verbose)
    
    if success:
        logger.info("Credential verification completed successfully")
        return 0
    else:
        logger.error("Credential verification failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
