# Yahoo Finance Integration

## Overview

The multi-strategy trading system now includes integration with Yahoo Finance for real-time and historical market data. This integration allows you to test and run your strategies with actual market data without requiring a paid data subscription.

## Setting Up Yahoo Finance Integration

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Your Data Source**:
   In your configuration file (`multi_strategy_config.yaml`), set:
   ```yaml
   data_source: YAHOO
   ```

3. **Place the Integration Files**:
   - Ensure `yahoo_finance_data.py` is in the same directory as your main application
   - The system will automatically use Yahoo Finance when configured

## Usage Notes for Yahoo Finance Data

- **Data Frequency**: Yahoo Finance provides 1-minute intraday data for the past 7 days. For longer periods, the data frequency becomes daily.

- **Rate Limiting**: Yahoo Finance does not officially support an API, so excessive queries may lead to temporary IP blocking. The system implements:
  - Caching to reduce API calls
  - Retry logic for failed requests
  - Controlled polling frequency

- **Market Hours**: The system is configured to work with standard market hours (9:30 AM to 4:00 PM EST). If you're trading in different markets, adjust the `market_hours_start` and `market_hours_end` in your configuration.

- **Data Accuracy**: Yahoo Finance data may have occasional gaps or delays, especially during high market volatility. This is a limitation of the free data source.

## Real-Time vs. Delayed Data

- Yahoo Finance provides delayed data (typically 15-20 minutes) for most exchanges
- For testing purposes, this delay generally won't impact strategy development
- For live trading with real money, consider using a broker's direct API feed

## Working with Extended Hours Data

To include pre-market and after-hours data:

1. Update your configuration:
   ```yaml
   market_hours_start: '09:00'  # Include some pre-market
   market_hours_end: '16:30'    # Include some after-hours
   ```

2. When fetching data, add the `prepost` parameter:
   ```python
   # Modify in yahoo_finance_data.py
   data = ticker.history(period=period, interval=interval, prepost=True)
   ```

## Limitations

1. **Historical Data**: Limited to 7 days of 1-minute data
2. **API Reliability**: No official API means occasional disruptions
3. **Data Delay**: Free tier provides delayed quotes
4. **Volume Data**: May be less accurate than paid sources

For serious trading applications, consider upgrading to a paid data provider like:
- Alpaca Market Data
- IEX Cloud
- Polygon.io

However, for strategy development, testing, and paper trading, Yahoo Finance provides adequate data quality.
