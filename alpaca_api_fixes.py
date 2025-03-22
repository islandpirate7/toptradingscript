"""
Fixes for Alpaca API connection and date format issues.
Copy and paste these methods into your test_optimized_mean_reversion_alpaca.py file.
"""

def initialize_alpaca_api(self):
    """Initialize the Alpaca API client with credentials from alpaca_credentials.json"""
    try:
        # Load credentials from JSON file
        import json
        import os
        
        credentials_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'alpaca_credentials.json')
        
        self.logger.info(f"Looking for credentials at: {credentials_path}")
        
        if os.path.exists(credentials_path):
            with open(credentials_path, 'r') as f:
                credentials = json.load(f)
            
            # Use paper trading credentials by default
            paper_creds = credentials.get('paper', {})
            api_key = paper_creds.get('api_key')
            api_secret = paper_creds.get('api_secret')
            base_url = paper_creds.get('base_url', 'https://paper-api.alpaca.markets')
            
            # Remove /v2 suffix if it's already included to prevent duplication
            if base_url.endswith('/v2'):
                base_url = base_url[:-3]
            
            self.logger.info(f"Using paper trading credentials with base URL: {base_url}")
        else:
            # Fallback to environment variables
            api_key = os.environ.get('ALPACA_API_KEY')
            api_secret = os.environ.get('ALPACA_API_SECRET')
            base_url = 'https://paper-api.alpaca.markets'
            
            if not api_key or not api_secret:
                self.logger.error("Alpaca API credentials not found")
                return False
        
        # Initialize API
        self.api = REST(api_key, api_secret, base_url)
        self.logger.info("Alpaca API initialized successfully")
        
        # Test the API connection
        account = self.api.get_account()
        self.logger.info(f"Connected to Alpaca account: {account.id}")
        self.logger.info(f"Account status: {account.status}")
        self.logger.info(f"Account equity: {account.equity}")
        
        return True
            
    except Exception as e:
        self.logger.error(f"Error initializing Alpaca API: {str(e)}")
        return False

def fetch_historical_data(self, symbol, start_date, end_date):
    """Fetch historical price data from Alpaca"""
    if not self.api:
        self.logger.error("Alpaca API not initialized")
        return None
    
    try:
        # Convert string dates to datetime if needed
        if isinstance(start_date, str):
            start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        
        self.logger.info(f"Fetching data for {symbol} from {start_date.date()} to {end_date.date()}")
        
        # Format dates as YYYY-MM-DD (without time component)
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        self.logger.info(f"Using date strings: start={start_str}, end={end_str}")
        
        # Determine timeframe based on symbol type
        timeframe = TimeFrame.Day
        
        # Fetch data
        bars = self.api.get_bars(
            symbol,
            timeframe,
            start=start_str,
            end=end_str,
            adjustment='raw'
        ).df
        
        if bars.empty:
            self.logger.warning(f"No data returned for {symbol}")
            return None
        
        # Convert to CandleData objects
        candles = []
        for index, row in bars.iterrows():
            candle = CandleData(
                timestamp=index.to_pydatetime(),
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume']
            )
            candles.append(candle)
        
        self.logger.info(f"Fetched {len(candles)} candles for {symbol}")
        return candles
        
    except Exception as e:
        self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
        return None
