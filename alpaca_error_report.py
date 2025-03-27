import logging
import os
import yaml
import pandas as pd
import traceback
import json
import requests
from datetime import datetime, timedelta
from alpaca_trade_api.rest import REST, TimeFrame

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path='sp500_config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def generate_error_report():
    """Generate a detailed error report for Alpaca support."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "system_info": {
            "python_version": os.sys.version,
            "platform": os.sys.platform
        },
        "api_info": {},
        "test_results": []
    }
    
    # Load configuration
    try:
        config = load_config()
        api_key = config['alpaca']['api_key']
        api_secret = config['alpaca']['api_secret']
        base_url = config['alpaca']['base_url']
        data_url = config['alpaca']['data_url']
        
        # Mask API credentials for security
        report["api_info"] = {
            "api_key_prefix": api_key[:5] + "..." if api_key else "None",
            "api_secret_prefix": api_secret[:5] + "..." if api_secret else "None",
            "base_url": base_url,
            "data_url": data_url
        }
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        report["config_error"] = str(e)
        report["traceback"] = traceback.format_exc()
        return report
    
    # Test 1: Account access (trading API)
    logger.info("Test 1: Account access (trading API)")
    try:
        trading_api = REST(api_key, api_secret, base_url, api_version='v2')
        account = trading_api.get_account()
        
        report["test_results"].append({
            "test_name": "Account Access",
            "endpoint": f"{base_url}/v2/account",
            "success": True,
            "response": {
                "account_id": account.id,
                "status": account.status,
                "currency": account.currency,
                "buying_power": account.buying_power,
                "cash": account.cash,
                "portfolio_value": account.portfolio_value
            }
        })
        logger.info("Account access test: SUCCESS")
    except Exception as e:
        logger.error(f"Account access test failed: {str(e)}")
        report["test_results"].append({
            "test_name": "Account Access",
            "endpoint": f"{base_url}/v2/account",
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        })
    
    # Test 2: Recent market data (last 7 days)
    logger.info("Test 2: Recent market data (last 7 days)")
    try:
        data_api = REST(api_key, api_secret, data_url, api_version='v2')
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        symbol = "AAPL"
        endpoint = f"{data_url}/v2/stocks/{symbol}/bars"
        
        logger.info(f"Requesting data from {endpoint} for {start_str} to {end_str}")
        
        # Capture the raw HTTP request for debugging
        session = requests.Session()
        session.headers.update({
            'APCA-API-KEY-ID': api_key,
            'APCA-API-SECRET-KEY': api_secret
        })
        
        params = {
            'timeframe': '1Day',
            'start': start_date.isoformat(),
            'end': end_date.isoformat(),
            'adjustment': 'raw'
        }
        
        response = session.get(endpoint, params=params)
        
        report["test_results"].append({
            "test_name": "Recent Market Data",
            "endpoint": endpoint,
            "params": params,
            "success": response.status_code == 200,
            "status_code": response.status_code,
            "response_headers": dict(response.headers),
            "response_body": response.text[:1000] if response.status_code == 200 else response.text
        })
        
        if response.status_code == 200:
            logger.info("Recent market data test: SUCCESS")
        else:
            logger.error(f"Recent market data test failed with status code {response.status_code}")
            logger.error(f"Response: {response.text}")
    except Exception as e:
        logger.error(f"Recent market data test failed: {str(e)}")
        report["test_results"].append({
            "test_name": "Recent Market Data",
            "endpoint": endpoint if 'endpoint' in locals() else f"{data_url}/v2/stocks/AAPL/bars",
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        })
    
    # Test 3: Historical market data (2022)
    logger.info("Test 3: Historical market data (2022)")
    try:
        data_api = REST(api_key, api_secret, data_url, api_version='v2')
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 1, 31)
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        symbol = "AAPL"
        endpoint = f"{data_url}/v2/stocks/{symbol}/bars"
        
        logger.info(f"Requesting data from {endpoint} for {start_str} to {end_str}")
        
        # Capture the raw HTTP request for debugging
        session = requests.Session()
        session.headers.update({
            'APCA-API-KEY-ID': api_key,
            'APCA-API-SECRET-KEY': api_secret
        })
        
        params = {
            'timeframe': '1Day',
            'start': start_date.isoformat(),
            'end': end_date.isoformat(),
            'adjustment': 'raw'
        }
        
        response = session.get(endpoint, params=params)
        
        report["test_results"].append({
            "test_name": "Historical Market Data (2022)",
            "endpoint": endpoint,
            "params": params,
            "success": response.status_code == 200,
            "status_code": response.status_code,
            "response_headers": dict(response.headers),
            "response_body": response.text[:1000] if response.status_code == 200 else response.text
        })
        
        if response.status_code == 200:
            logger.info("Historical market data test: SUCCESS")
        else:
            logger.error(f"Historical market data test failed with status code {response.status_code}")
            logger.error(f"Response: {response.text}")
    except Exception as e:
        logger.error(f"Historical market data test failed: {str(e)}")
        report["test_results"].append({
            "test_name": "Historical Market Data (2022)",
            "endpoint": endpoint if 'endpoint' in locals() else f"{data_url}/v2/stocks/AAPL/bars",
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        })
    
    # Test 4: Check subscription status
    logger.info("Test 4: Check subscription status")
    try:
        # This endpoint might not exist or might be different based on Alpaca's API
        # Using a generic approach to try to get subscription info
        session = requests.Session()
        session.headers.update({
            'APCA-API-KEY-ID': api_key,
            'APCA-API-SECRET-KEY': api_secret
        })
        
        # Try to get account details which might include subscription info
        response = session.get(f"{base_url}/v2/account")
        
        report["test_results"].append({
            "test_name": "Subscription Status",
            "endpoint": f"{base_url}/v2/account",
            "success": response.status_code == 200,
            "status_code": response.status_code,
            "response_headers": dict(response.headers),
            "response_body": response.text[:1000] if response.status_code == 200 else response.text
        })
        
        if response.status_code == 200:
            logger.info("Subscription status test: SUCCESS")
        else:
            logger.error(f"Subscription status test failed with status code {response.status_code}")
            logger.error(f"Response: {response.text}")
    except Exception as e:
        logger.error(f"Subscription status test failed: {str(e)}")
        report["test_results"].append({
            "test_name": "Subscription Status",
            "endpoint": f"{base_url}/v2/account",
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        })
    
    return report

if __name__ == "__main__":
    logger.info("Generating Alpaca API error report...")
    report = generate_error_report()
    
    # Save report to file
    report_file = "alpaca_error_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Error report saved to {report_file}")
    
    # Print summary
    print("\n===== ALPACA API ERROR REPORT SUMMARY =====")
    print(f"Report generated at: {report['timestamp']}")
    print("\nTest Results:")
    
    for test in report["test_results"]:
        status = "✅ SUCCESS" if test["success"] else "❌ FAILED"
        print(f"{test['test_name']}: {status}")
        if not test["success"] and "status_code" in test:
            print(f"  Status Code: {test['status_code']}")
            print(f"  Error: {test.get('response_body', 'No response body')}")
    
    print("\nDetailed report saved to:", report_file)
    print("Please include this file when contacting Alpaca support.")
    print("================================================")
