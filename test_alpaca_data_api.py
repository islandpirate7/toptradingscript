import logging
import os
import yaml
import pandas as pd
from alpaca_trade_api.rest import REST
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def test_data_api():
    """Test the Alpaca Data API with the provided credentials."""
    # Load configuration
    config = load_config('sp500_config.yaml')
    
    # Log API key (first few characters)
    api_key = config['alpaca']['api_key']
    logger.info(f"Testing Alpaca Data API with key: {api_key[:5]}...")
    
    # Initialize Alpaca API client for DATA API
    api = REST(
        api_key,
        config['alpaca']['api_secret'],
        base_url='https://data.alpaca.markets'  # Use data API endpoint
    )
    
    # Test fetching historical data from 2022
    symbol = "AAPL"
    start_date = "2022-01-01"
    end_date = "2022-01-31"
    
    try:
        logger.info(f"Fetching historical data for {symbol} from {start_date} to {end_date}")
        bars = api.get_bars(
            symbol,
            timeframe="1Day",
            start=start_date,
            end=end_date,
            adjustment='raw'
        ).df
        
        if bars is not None and not bars.empty:
            logger.info(f"Successfully retrieved {len(bars)} bars for {symbol}")
            logger.info(f"First few bars:\n{bars.head()}")
            return True
        else:
            logger.error("No data returned")
            return False
            
    except Exception as e:
        logger.error(f"Error fetching historical data: {str(e)}")
        
        # Try with a different method
        try:
            logger.info("Trying with get_bars_v2 method...")
            bars = api.get_bars_v2(
                symbol,
                start=start_date,
                end=end_date,
                timeframe="1Day",
                adjustment='raw'
            )
            
            df = pd.DataFrame([bar._asdict() for bar in bars])
            if not df.empty:
                logger.info(f"Successfully retrieved {len(df)} bars for {symbol} with v2 API")
                logger.info(f"First few bars:\n{df.head()}")
                return True
            else:
                logger.error("No data returned from v2 API")
                return False
                
        except Exception as e2:
            logger.error(f"Error fetching historical data with v2 API: {str(e2)}")
            return False
    
    return False

if __name__ == "__main__":
    success = test_data_api()
    if success:
        logger.info("Alpaca Data API test successful!")
    else:
        logger.error("Alpaca Data API test failed!")
