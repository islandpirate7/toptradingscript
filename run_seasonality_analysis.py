"""
Run Seasonality Analysis

This script runs the seasonality analysis on a set of stocks and generates
trading opportunities based on seasonal patterns.
"""

import os
import sys
import logging
import pandas as pd
import yaml
from datetime import datetime, timedelta
from seasonality_analyzer import SeasonalityAnalyzer, SeasonType
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Default configuration
DEFAULT_CONFIG = {
    'api_credentials_path': 'alpaca_credentials.json',
    'output_dir': 'output',
    'min_win_rate': 0.55,
    'min_trades': 3,
    'min_avg_return': 0.2,
    'lookback_years': 5,
    'season_type': 'MONTH_OF_YEAR',
    'correlation_threshold': 0.7,
    'current_period_days': 30,
    'generate_plots': True,
}

def get_sp500_symbols():
    """Get a list of S&P 500 symbols for testing
    
    Returns:
        list: List of S&P 500 stock symbols
    """
    # This is a simplified list of some S&P 500 stocks
    # In a real implementation, you might want to fetch this from an API or file
    return [
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'BRK.B', 'UNH', 'JNJ',
        'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'KO', 'PEP', 'ABBV',
        'AVGO', 'LLY', 'COST', 'TMO', 'MCD', 'ABT', 'CSCO', 'ACN', 'WMT', 'CRM',
        'PFE', 'BAC', 'DIS', 'ADBE', 'TXN', 'CMCSA', 'NKE', 'NEE', 'VZ', 'PM',
        'INTC', 'DHR', 'AMD', 'QCOM', 'UPS', 'IBM', 'AMGN', 'SBUX', 'INTU', 'LOW'
    ]

def get_dow_symbols():
    """Get a list of Dow Jones Industrial Average symbols
    
    Returns:
        list: List of Dow Jones stock symbols
    """
    return [
        'AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS', 'DOW',
        'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM',
        'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT'
    ]

def get_nasdaq_100_symbols():
    """Get a list of NASDAQ-100 symbols
    
    Returns:
        list: List of NASDAQ-100 stock symbols
    """
    # This is a simplified list of some NASDAQ-100 stocks
    return [
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'AVGO', 'PEP', 'COST',
        'CSCO', 'ADBE', 'CMCSA', 'INTC', 'AMD', 'QCOM', 'INTU', 'SBUX', 'TXN', 'PYPL',
        'NFLX', 'GILD', 'MDLZ', 'TMUS', 'ISRG', 'REGN', 'BKNG', 'ADP', 'ADI', 'VRTX'
    ]

def get_test_symbols():
    """Get a small list of symbols for testing
    
    Returns:
        list: List of test stock symbols
    """
    return [
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'JNJ'
    ]

def ensure_output_directory(output_dir):
    """Ensure the output directory exists
    
    Args:
        output_dir (str): Path to output directory
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")

def save_opportunities_to_yaml(opportunities, output_path):
    """Save trading opportunities to a YAML file
    
    Args:
        opportunities (list): List of trading opportunities
        output_path (str): Path to output file
    """
    # Convert opportunities to a format suitable for YAML
    opportunities_dict = {
        'opportunities': []
    }
    
    for opp in opportunities:
        opportunities_dict['opportunities'].append({
            'symbol': opp['symbol'],
            'season': opp['season'],
            'win_rate': float(opp['win_rate']),
            'avg_return': float(opp['avg_return']),
            'trade_count': int(opp['trade_count']),
            'correlation': float(opp['correlation']),
            'current_price': float(opp['current_price']) if 'current_price' in opp else None,
            'expected_return': float(opp['expected_return']) if 'expected_return' in opp else None,
        })
    
    # Save to YAML file
    with open(output_path, 'w') as f:
        yaml.dump(opportunities_dict, f, default_flow_style=False)
    
    logging.info(f"Saved {len(opportunities)} opportunities to {output_path}")

def save_config_to_yaml(config, output_path):
    """Save configuration for the seasonality strategy to a YAML file
    
    Args:
        config (dict): Configuration dictionary
        output_path (str): Path to output file
    """
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logging.info(f"Saved configuration to {output_path}")

def main():
    """Main function to run seasonality analysis"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run seasonality analysis')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config = DEFAULT_CONFIG.copy()
    if args.config:
        try:
            with open(args.config, 'r') as f:
                user_config = yaml.safe_load(f)
                config.update(user_config)
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
    
    # Create output directory if it doesn't exist
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize analyzer
    analyzer = SeasonalityAnalyzer(config['api_credentials_path'])
    
    # Define expanded universe of stocks
    # Major indices, sectors, and high-volume stocks
    universe = [
        # Major Tech
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX', 'ADBE', 'INTC', 'AMD', 'CRM', 'CSCO',
        # Financial
        'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'AXP', 'V', 'MA', 'BLK', 'SCHW',
        # Healthcare
        'JNJ', 'PFE', 'MRK', 'ABBV', 'ABT', 'UNH', 'CVS', 'GILD', 'AMGN', 'BMY',
        # Consumer
        'PG', 'KO', 'PEP', 'WMT', 'COST', 'MCD', 'SBUX', 'NKE', 'DIS', 'HD', 'LOW', 'TGT',
        # Energy
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PSX', 'VLO',
        # Industrial
        'GE', 'HON', 'MMM', 'CAT', 'DE', 'BA', 'LMT', 'RTX', 'UPS', 'FDX',
        # Communication
        'VZ', 'T', 'TMUS', 'CMCSA',
        # Utilities
        'NEE', 'DUK', 'SO', 'D', 'AEP',
        # Real Estate
        'AMT', 'PLD', 'CCI', 'SPG', 'EQIX',
        # Materials
        'LIN', 'APD', 'ECL', 'DD', 'NEM',
        # ETFs
        'SPY', 'QQQ', 'IWM', 'DIA', 'XLF', 'XLE', 'XLK', 'XLV', 'XLY', 'XLI', 'XLP', 'XLU', 'XLB', 'XLRE'
    ]
    
    # Set the universe
    analyzer.set_universe(universe)
    
    # Fetch historical data
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365 * config['lookback_years'])
    analyzer.fetch_historical_data(start_date, end_date)
    
    # Calculate seasonal patterns
    season_type = SeasonType[config['season_type']]
    analyzer.calculate_seasonal_patterns(season_type)
    
    # Calculate seasonal correlations
    analyzer.calculate_seasonal_correlation(
        current_period_days=config['current_period_days'],
        correlation_threshold=config['correlation_threshold']
    )
    
    # Generate trading opportunities
    opportunities = analyzer.generate_trading_opportunities(
        min_win_rate=config['min_win_rate'],
        min_trades=config['min_trades'],
        min_avg_return=config['min_avg_return']
    )
    
    # Generate plots if enabled
    if config['generate_plots']:
        analyzer.generate_seasonal_plots(output_dir)
    
    # Save opportunities to file
    if opportunities:
        opportunities_file = os.path.join(output_dir, 'seasonal_opportunities.yaml')
        with open(opportunities_file, 'w') as f:
            yaml.dump({'opportunities': opportunities}, f)
        logging.info(f"Saved {len(opportunities)} opportunities to {opportunities_file}")
    else:
        logging.warning("No opportunities found meeting the criteria")
    
    # Save configuration
    config_file = os.path.join(output_dir, 'configuration_seasonality.yaml')
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    logging.info(f"Saved configuration to {config_file}")
    
    logging.info(f"Seasonality analysis completed successfully. Found {len(opportunities)} trading opportunities.")

if __name__ == "__main__":
    main()
