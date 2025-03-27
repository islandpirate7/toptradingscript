import logging
import os
import yaml
import pandas as pd
from datetime import datetime, timedelta
from alpaca_api import AlpacaAPI

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path='sp500_config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def test_alpaca_full():
    """Test both the Alpaca Trading API and Data API with the provided credentials."""
    # Load configuration
    config = load_config()
    
    # Log API key (first few characters)
    api_key = config['alpaca']['api_key']
    logger.info(f"Testing Alpaca APIs with key: {api_key[:5]}...")
    
    # Initialize Alpaca API with both trading and data endpoints
    api = AlpacaAPI(
        api_key=config['alpaca']['api_key'],
        api_secret=config['alpaca']['api_secret'],
        base_url=config['alpaca']['base_url'],
        data_url=config['alpaca']['data_url']
    )
    
    # Test account access (trading API)
    logger.info("Testing trading API (account access)...")
    try:
        account = api.get_account()
        if account:
            logger.info(f"Account access successful!")
            logger.info(f"Account ID: {account.id}")
            logger.info(f"Account status: {account.status}")
            logger.info(f"Buying power: ${float(account.buying_power):.2f}")
            logger.info(f"Cash: ${float(account.cash):.2f}")
            logger.info(f"Portfolio value: ${float(account.portfolio_value):.2f}")
        else:
            logger.error("Failed to retrieve account information")
            return False
    except Exception as e:
        logger.error(f"Error accessing account: {str(e)}")
        return False
    
    # Test positions (trading API)
    logger.info("Testing trading API (positions)...")
    try:
        positions = api.get_positions()
        if positions is not None:
            logger.info(f"Successfully retrieved {len(positions)} positions")
            for position in positions:
                logger.info(f"  {position.symbol}: {position.qty} shares at {position.avg_entry_price}")
        else:
            logger.warning("No positions found or error retrieving positions")
    except Exception as e:
        logger.error(f"Error retrieving positions: {str(e)}")
    
    # Test market data access (data API)
    logger.info("Testing data API (historical data)...")
    
    # Test fetching historical data for the past month
    symbol = "AAPL"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    try:
        logger.info(f"Fetching historical data for {symbol} from {start_str} to {end_str}")
        bars = api.get_bars([symbol], '1D', start_str, end_str)
        
        if bars is not None and not bars.empty:
            logger.info(f"Successfully retrieved {len(bars)} bars for {symbol}")
            logger.info(f"First few bars:\n{bars.head()}")
            return True
        else:
            logger.error("No data returned")
            return False
    except Exception as e:
        logger.error(f"Error fetching historical data: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_alpaca_full()
    if success:
        logger.info("Alpaca API test successful!")
    else:
        logger.error("Alpaca API test failed!")
