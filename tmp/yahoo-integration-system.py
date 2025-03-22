# Integration of Yahoo Finance into the _data_worker method

def _initialize_data_source(self):
    """Initialize the data source based on configuration"""
    if self.config.data_source == "YAHOO":
        from yahoo_finance_data import YahooFinanceDataSource
        
        self.data_source = YahooFinanceDataSource({
            "cache_duration": 3600,  # Cache data for 1 hour
            "max_retries": 3,
            "retry_delay": 5
        })
        self.logger.info("Initialized Yahoo Finance data source")
    else:
        self.logger.warning(f"Unsupported data source: {self.config.data_source}")
        self.data_source = None

def _fetch_market_data(self):
    """Fetch market index and VIX data"""
    try:
        if not self.data_source:
            raise ValueError("No data source initialized")
        
        # Fetch S&P 500 data
        sp500_df = self.data_source.get_market_index_data(
            index_symbol="^GSPC", 
            period="5d",  # Get last 5 days
            interval="1m"  # 1-minute data
        )
        
        if not sp500_df.empty:
            # Convert to candle data objects
            market_candles = self.data_source.convert_to_candle_data(sp500_df)
            
            # Update market data with most recent data
            if market_candles:
                # If we already have data, only add new candles
                if self.market_data:
                    last_timestamp = self.market_data[-1].timestamp
                    new_candles = [c for c in market_candles if c.timestamp > last_timestamp]
                    
                    if new_candles:
                        self.market_data.extend(new_candles)
                        self.logger.info(f"Added {len(new_candles)} new market candles")
                else:
                    # Initialize with all candles
                    self.market_data = market_candles
                    self.logger.info(f"Initialized market data with {len(market_candles)} candles")
                
                # Keep only recent data
                if len(self.market_data) > 10000:
                    self.market_data = self.market_data[-10000:]
        
        # Fetch VIX data
        vix_df = self.data_source.get_vix_data(
            period="5d",  # Get last 5 days
            interval="1m"  # 1-minute data
        )
        
        if not vix_df.empty:
            # Convert to candle data objects
            vix_candles = self.data_source.convert_to_candle_data(vix_df)
            
            # Update VIX data with most recent data
            if vix_candles:
                # If we already have data, only add new candles
                if self.vix_data:
                    last_timestamp = self.vix_data[-1].timestamp
                    new_candles = [c for c in vix_candles if c.timestamp > last_timestamp]
                    
                    if new_candles:
                        self.vix_data.extend(new_candles)
                        self.logger.info(f"Added {len(new_candles)} new VIX candles")
                else:
                    # Initialize with all candles
                    self.vix_data = vix_candles
                    self.logger.info(f"Initialized VIX data with {len(vix_candles)} candles")
                
                # Keep only recent data
                if len(self.vix_data) > 10000:
                    self.vix_data = self.vix_data[-10000:]
                
    except Exception as e:
        self.logger.error(f"Error fetching market data: {str(e)}")

def _fetch_stock_data(self, symbol: str):
    """Fetch historical and real-time data for a stock"""
    try:
        if not self.data_source:
            raise ValueError("No data source initialized")
        
        # Fetch most recent data for the stock
        stock_df = self.data_source.get_latest_data(
            symbol=symbol,
            lookback_days=self.config.data_lookback_days
        )
        
        if not stock_df.empty:
            # Convert to candle data objects
            stock_candles = self.data_source.convert_to_candle_data(stock_df)
            
            # Update stock data with most recent data
            if stock_candles:
                # If we already have data, only add new candles
                if self.candle_data[symbol]:
                    last_timestamp = self.candle_data[symbol][-1].timestamp
                    new_candles = [c for c in stock_candles if c.timestamp > last_timestamp]
                    
                    if new_candles:
                        self.candle_data[symbol].extend(new_candles)
                        self.logger.info(f"Added {len(new_candles)} new candles for {symbol}")
                else:
                    # Initialize with all candles
                    self.candle_data[symbol] = stock_candles
                    self.logger.info(f"Initialized {symbol} data with {len(stock_candles)} candles")
                
                # Keep only recent data
                if len(self.candle_data[symbol]) > 10000:
                    self.candle_data[symbol] = self.candle_data[symbol][-10000:]
        
        # If we don't have enough data, fetch more history
        if len(self.candle_data[symbol]) < 100:  # Arbitrary threshold
            self.logger.info(f"Not enough data for {symbol}, fetching more history")
            
            history_df = self.data_source.get_historical_data(
                symbol=symbol,
                period="1mo",  # 1 month of data
                interval="1m"  # 1-minute data
            )
            
            if not history_df.empty:
                history_candles = self.data_source.convert_to_candle_data(history_df)
                
                # Replace existing data
                self.candle_data[symbol] = history_candles
                self.logger.info(f"Replaced {symbol} data with {len(history_candles)} historical candles")
                
    except Exception as e:
        self.logger.error(f"Error fetching data for {symbol}: {str(e)}")

def _data_worker(self):
    """Worker function to fetch and process data"""
    self.logger.info("Data worker started")
    
    # Initialize data source
    self._initialize_data_source()
    
    while self.is_running:
        try:
            # Fetch market data (S&P 500 and VIX)
            self._fetch_market_data()
            
            # Update market state
            if len(self.market_data) > 0 and len(self.vix_data) > 0:
                self.market_state = self.market_analyzer.analyze_market(self.market_data, self.vix_data)
                self.logger.info(f"Market regime: {self.market_state.regime.value}, ADX: {self.market_state.market_adx:.1f}, VIX: {self.market_state.vix:.1f}")
            
            # Fetch stock data for all symbols
            for symbol in self.candle_data.keys():
                self._fetch_stock_data(symbol)
            
            # Sleep to avoid overwhelming data sources
            time.sleep(60)  # Update data every minute
            
        except Exception as e:
            self.logger.error(f"Error in data worker: {str(e)}")
            time.sleep(120)  # Wait longer on error
