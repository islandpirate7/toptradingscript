"""
Test Alpaca API Connection and Data Retrieval

This script tests the connection to Alpaca API and attempts to fetch historical data
for a small set of stocks to verify that everything is working correctly.
"""

import json
import logging
import pandas as pd
import requests
from datetime import datetime, timedelta
import time
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_alpaca_connection():
    """Test connection to Alpaca API"""
    try:
        # Load API credentials
        with open('alpaca_credentials.json', 'r') as f:
            credentials = json.load(f)
            
        # Use paper trading credentials
        api_key = credentials['paper']['api_key']
        api_secret = credentials['paper']['api_secret']
        
        logging.info(f"Using API key: {api_key[:5]}...{api_key[-5:]}")
        
        # Initialize client
        client = StockHistoricalDataClient(api_key, api_secret)
        logging.info("Successfully initialized Alpaca client")
        
        return client, api_key, api_secret
    except Exception as e:
        logging.error(f"Error initializing Alpaca client: {e}")
        return None, None, None

def test_data_retrieval_sdk(client, symbols, start_date, end_date):
    """Test data retrieval using the Alpaca SDK
    
    Args:
        client: Alpaca client
        symbols: List of symbols to test
        start_date: Start date for data retrieval
        end_date: End date for data retrieval
    """
    if client is None:
        logging.error("Cannot test data retrieval: client is None")
        return
    
    logging.info(f"Testing data retrieval using SDK for {len(symbols)} symbols")
    logging.info(f"Date range: {start_date} to {end_date}")
    
    # Try to fetch data for each symbol individually
    for symbol in symbols:
        try:
            logging.info(f"Fetching data for {symbol}...")
            
            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=pd.Timestamp(start_date, tz='America/New_York'),
                end=pd.Timestamp(end_date, tz='America/New_York'),
                adjustment='all'
            )
            
            bars = client.get_stock_bars(request_params)
            
            if bars and symbol in bars:
                df = bars[symbol].df
                if not df.empty:
                    logging.info(f"SUCCESS: Fetched {len(df)} bars for {symbol}")
                    logging.info(f"First date: {df.index[0]}")
                    logging.info(f"Last date: {df.index[-1]}")
                    logging.info(f"Sample data: {df.iloc[0].to_dict()}")
                else:
                    logging.warning(f"Empty data returned for {symbol}")
            else:
                logging.warning(f"No data returned for {symbol}")
                
        except Exception as e:
            logging.error(f"Error fetching data for {symbol}: {e}")
        
        # Add a small delay to avoid rate limiting
        time.sleep(1)

def test_data_retrieval_rest(api_key, api_secret, symbols, start_date, end_date):
    """Test data retrieval using direct REST API calls
    
    Args:
        api_key: Alpaca API key
        api_secret: Alpaca API secret
        symbols: List of symbols to test
        start_date: Start date for data retrieval
        end_date: End date for data retrieval
    """
    if not api_key or not api_secret:
        logging.error("Cannot test REST API: credentials are None")
        return
    
    logging.info(f"Testing data retrieval using REST API for {len(symbols)} symbols")
    logging.info(f"Date range: {start_date} to {end_date}")
    
    # Base URL for the Alpaca Data API
    base_url = "https://data.alpaca.markets/v2"
    
    # Headers for authentication
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret
    }
    
    # Format dates for API
    start = pd.Timestamp(start_date, tz='America/New_York').isoformat()
    end = pd.Timestamp(end_date, tz='America/New_York').isoformat()
    
    # Try to fetch data for each symbol individually
    for symbol in symbols:
        try:
            logging.info(f"Fetching data for {symbol} via REST API...")
            
            # Construct URL for bars endpoint
            url = f"{base_url}/stocks/{symbol}/bars"
            
            # Parameters for the request
            params = {
                "start": start,
                "end": end,
                "timeframe": "1Day",
                "adjustment": "all",
                "limit": 1000
            }
            
            # Make the request
            response = requests.get(url, headers=headers, params=params)
            
            # Check if request was successful
            if response.status_code == 200:
                data = response.json()
                bars = data.get('bars', [])
                
                if bars:
                    logging.info(f"SUCCESS: Fetched {len(bars)} bars for {symbol} via REST API")
                    logging.info(f"First bar: {bars[0]}")
                    logging.info(f"Last bar: {bars[-1]}")
                else:
                    logging.warning(f"No bars returned for {symbol} via REST API")
                    
                # Check if there's a next page token
                next_page_token = data.get('next_page_token')
                if next_page_token:
                    logging.info(f"More data available for {symbol} (next_page_token: {next_page_token})")
            else:
                logging.error(f"Error fetching data for {symbol} via REST API: {response.status_code} - {response.text}")
                
        except Exception as e:
            logging.error(f"Exception fetching data for {symbol} via REST API: {e}")
        
        # Add a small delay to avoid rate limiting
        time.sleep(1)

def main():
    """Main function"""
    print("=== TESTING ALPACA API CONNECTION AND DATA RETRIEVAL ===")
    
    # Test connection
    client, api_key, api_secret = test_alpaca_connection()
    
    if client is None:
        print("\nFAILED: Could not connect to Alpaca API")
        return
    
    # Define test parameters
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    # Test data retrieval using SDK
    print("\n=== TESTING SDK DATA RETRIEVAL ===")
    test_data_retrieval_sdk(client, test_symbols, start_date, end_date)
    
    # Test data retrieval using REST API
    print("\n=== TESTING REST API DATA RETRIEVAL ===")
    test_data_retrieval_rest(api_key, api_secret, test_symbols, start_date, end_date)
    
    print("\nTest completed. Check the logs above for results.")

if __name__ == "__main__":
    main()
