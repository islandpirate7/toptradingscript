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

def test_alpaca_data_api():
    """Test the Alpaca Data API with the fixed date format."""
    # Load configuration
    config = load_config()
    
    # Log API key (first few characters)
    api_key = config['alpaca']['api_key']
    logger.info(f"Testing Alpaca Data API with key: {api_key[:5]}...")
    
    # Initialize Alpaca API with both trading and data endpoints
    api = AlpacaAPI(
        api_key=config['alpaca']['api_key'],
        api_secret=config['alpaca']['api_secret'],
        base_url=config['alpaca']['base_url'],
        data_url=config['alpaca']['data_url']
    )
    
    # Test historical data access with different time periods
    test_periods = [
        {"name": "Recent (7 days)", "start": (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'), "end": datetime.now().strftime('%Y-%m-%d')},
        {"name": "Last Month", "start": (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'), "end": datetime.now().strftime('%Y-%m-%d')},
        {"name": "Last Year Q4", "start": "2022-10-01", "end": "2022-12-31"},
        {"name": "2022 January", "start": "2022-01-01", "end": "2022-01-31"},
        {"name": "2021 Full Year", "start": "2021-01-01", "end": "2021-12-31"}
    ]
    
    symbol = "AAPL"
    results = []
    
    for period in test_periods:
        logger.info(f"Testing period: {period['name']} ({period['start']} to {period['end']})")
        
        try:
            # Get bars for the specified period
            bars = api.get_bars([symbol], '1D', period['start'], period['end'])
            
            if bars is not None and not bars.empty:
                logger.info(f"SUCCESS: Retrieved {len(bars)} bars for {symbol}")
                logger.info(f"First bar: {bars.iloc[0] if len(bars) > 0 else 'No data'}")
                logger.info(f"Last bar: {bars.iloc[-1] if len(bars) > 0 else 'No data'}")
                results.append({
                    "period": period['name'],
                    "success": True,
                    "bars_count": len(bars)
                })
            else:
                logger.error(f"FAILED: No data returned for {period['name']}")
                results.append({
                    "period": period['name'],
                    "success": False,
                    "error": "No data returned"
                })
        except Exception as e:
            logger.error(f"FAILED: Error retrieving data for {period['name']}: {str(e)}")
            results.append({
                "period": period['name'],
                "success": False,
                "error": str(e)
            })
    
    # Print summary
    logger.info("\n===== TEST RESULTS SUMMARY =====")
    success_count = sum(1 for r in results if r["success"])
    logger.info(f"Successful periods: {success_count}/{len(test_periods)}")
    
    for result in results:
        status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
        if result["success"]:
            logger.info(f"{result['period']}: {status} - Retrieved {result['bars_count']} bars")
        else:
            logger.info(f"{result['period']}: {status} - {result.get('error', 'Unknown error')}")
    
    return results

if __name__ == "__main__":
    logger.info("Starting Alpaca Data API test with fixed date format...")
    results = test_alpaca_data_api()
    
    # Determine overall success
    if any(r["success"] for r in results):
        logger.info("At least one test period was successful!")
        logger.info("You can use these successful time periods for backtesting.")
    else:
        logger.error("All test periods failed. Please contact Alpaca support with the error details.")
        logger.error("Your subscription may not include access to historical market data.")
