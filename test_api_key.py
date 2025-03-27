#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple Alpaca API Key Test
-------------------------
This script tests if the provided Alpaca API keys can successfully connect to the Alpaca API.
"""

import sys
import logging
from alpaca_trade_api.rest import REST

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_api_key(api_key, api_secret, base_url):
    """Test if the API key can connect to Alpaca"""
    logger.info(f"Testing connection with API key: {api_key[:5]}...")
    logger.info(f"Using base URL: {base_url}")
    
    try:
        # Initialize Alpaca API client
        api = REST(
            api_key,
            api_secret,
            base_url=base_url
        )
        
        # Try to get account info
        account = api.get_account()
        logger.info(f"Connection successful!")
        logger.info(f"Account ID: {account.id}")
        logger.info(f"Account status: {account.status}")
        logger.info(f"Account equity: {account.equity}")
        
        return True
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python test_api_key.py <api_key> <api_secret> <base_url>")
        print("Example: python test_api_key.py PKA123456789 SK123456789 https://paper-api.alpaca.markets")
        sys.exit(1)
    
    api_key = sys.argv[1]
    api_secret = sys.argv[2]
    base_url = sys.argv[3]
    
    success = test_api_key(api_key, api_secret, base_url)
    
    if success:
        logger.info("API key test successful")
        sys.exit(0)
    else:
        logger.error("API key test failed")
        sys.exit(1)
