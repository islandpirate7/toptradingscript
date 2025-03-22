import alpaca_trade_api as tradeapi
import json
import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_alpaca_credentials(credentials_file='alpaca_credentials.json'):
    """Load Alpaca API credentials from JSON file"""
    try:
        with open(credentials_file, 'r') as f:
            credentials = json.load(f)
        return credentials
    except Exception as e:
        logger.error(f"Error loading Alpaca credentials: {str(e)}")
        return None

def test_data_access():
    """Test if we can access data from different time periods with Alpaca"""
    # Load credentials
    credentials = load_alpaca_credentials()
    if not credentials:
        logger.error("Failed to load credentials")
        return
    
    # Initialize Alpaca API with paper trading credentials
    api = tradeapi.REST(
        key_id=credentials['paper']['api_key'],
        secret_key=credentials['paper']['api_secret'],
        base_url=credentials['paper']['base_url']
    )
    
    # Test symbols
    test_symbols = ['AAPL', 'MSFT', 'BTCUSD']
    
    # Test periods
    test_periods = [
        {"name": "Q1 2023", "start": "2023-01-01", "end": "2023-03-31"},
        {"name": "Q4 2023", "start": "2023-10-01", "end": "2023-12-31"},
        {"name": "Q1 2024", "start": "2024-01-01", "end": "2024-03-31"}
    ]
    
    for period in test_periods:
        logger.info(f"Testing data access for {period['name']}")
        
        for symbol in test_symbols:
            logger.info(f"Fetching data for {symbol}")
            
            try:
                # Determine if it's a crypto symbol
                is_crypto = symbol.endswith('USD')
                
                if is_crypto:
                    # Fetch crypto data
                    bars = api.get_crypto_bars(
                        symbol, 
                        tradeapi.TimeFrame.Day, 
                        start=period['start'],
                        end=period['end']
                    ).df
                else:
                    # Fetch stock data
                    bars = api.get_bars(
                        symbol, 
                        tradeapi.TimeFrame.Day, 
                        start=period['start'],
                        end=period['end'],
                        adjustment='raw'
                    ).df
                
                if len(bars) > 0:
                    logger.info(f"Successfully fetched {len(bars)} bars for {symbol} in {period['name']}")
                    logger.info(f"First date: {bars.index[0].date()}, Last date: {bars.index[-1].date()}")
                else:
                    logger.warning(f"No data available for {symbol} in {period['name']}")
            
            except Exception as e:
                logger.error(f"Error fetching data for {symbol} in {period['name']}: {str(e)}")

if __name__ == "__main__":
    test_data_access()
