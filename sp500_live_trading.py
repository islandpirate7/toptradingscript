#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Live trading script for S&P 500 stock selection strategy
"""

import os
import json
import logging
import argparse
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import pytz
import requests
from bs4 import BeautifulSoup
import alpaca_trade_api as tradeapi
from combined_strategy import CombinedStrategy
from data_loader import AlpacaDataLoader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_alpaca_credentials(mode='paper'):
    """
    Load Alpaca API credentials from JSON file
    
    Args:
        mode: 'paper' or 'live'
    
    Returns:
        dict: Alpaca API credentials
    """
    try:
        with open('alpaca_credentials.json', 'r') as f:
            credentials = json.load(f)
        
        if mode not in credentials:
            raise ValueError(f"Mode '{mode}' not found in credentials file")
        
        return credentials[mode]
    except Exception as e:
        logger.error(f"Error loading Alpaca credentials: {str(e)}")
        raise

def get_sp500_symbols():
    """
    Get the current list of S&P 500 symbols from Wikipedia
    
    Returns:
        list: S&P 500 symbols
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table', {'class': 'wikitable'})
        
        symbols = []
        for row in table.find_all('tr')[1:]:
            symbol = row.find_all('td')[0].text.strip()
            symbols.append(symbol.replace('.', '-'))
        
        logger.info(f"Successfully fetched {len(symbols)} S&P 500 symbols")
        return symbols
    except Exception as e:
        logger.error(f"Error fetching S&P 500 symbols: {str(e)}")
        # Fall back to a static list of top 100 symbols
        logger.info("Using static list of top 100 S&P 500 symbols")
        return [
            "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "GOOG", "BRK.B", "TSLA", "UNH",
            "LLY", "JPM", "V", "XOM", "AVGO", "PG", "MA", "HD", "COST", "MRK",
            "CVX", "ABBV", "PEP", "KO", "ADBE", "WMT", "CRM", "BAC", "TMO", "MCD",
            "CSCO", "PFE", "NFLX", "CMCSA", "ABT", "ORCL", "TMUS", "AMD", "DIS", "ACN",
            "DHR", "VZ", "NKE", "TXN", "NEE", "WFC", "PM", "INTC", "INTU", "COP",
            "AMGN", "IBM", "RTX", "HON", "QCOM", "UPS", "CAT", "LOW", "SPGI", "BA",
            "GE", "ELV", "DE", "AMAT", "ISRG", "AXP", "BKNG", "MDLZ", "GILD", "ADI",
            "SBUX", "TJX", "MMC", "SYK", "VRTX", "PLD", "MS", "BLK", "SCHW", "C",
            "ZTS", "CB", "AMT", "ADP", "GS", "ETN", "LRCX", "NOW", "MO", "REGN",
            "EOG", "SO", "BMY", "EQIX", "BSX", "CME", "CI", "PANW", "TGT", "SLB"
        ]

def initialize_alpaca_api(credentials):
    """
    Initialize Alpaca API
    
    Args:
        credentials: Alpaca API credentials
    
    Returns:
        tradeapi.REST: Alpaca API client
    """
    api = tradeapi.REST(
        key_id=credentials['api_key'],
        secret_key=credentials['api_secret'],
        base_url=credentials['base_url']
    )
    
    # Test connection
    try:
        account = api.get_account()
        logger.info(f"Connected to Alpaca API: {account.status}")
        logger.info(f"Account value: ${float(account.equity):.2f}")
        logger.info(f"Buying power: ${float(account.buying_power):.2f}")
        logger.info(f"Cash: ${float(account.cash):.2f}")
        return api
    except Exception as e:
        logger.error(f"Error connecting to Alpaca API: {str(e)}")
        raise

def get_current_positions(api):
    """
    Get current positions
    
    Args:
        api: Alpaca API client
    
    Returns:
        dict: Current positions by symbol
    """
    try:
        positions = api.list_positions()
        positions_dict = {p.symbol: {
            'qty': float(p.qty),
            'market_value': float(p.market_value),
            'current_price': float(p.current_price),
            'avg_entry_price': float(p.avg_entry_price),
            'unrealized_pl': float(p.unrealized_pl),
            'unrealized_plpc': float(p.unrealized_plpc)
        } for p in positions}
        
        logger.info(f"Current positions: {len(positions_dict)}")
        return positions_dict
    except Exception as e:
        logger.error(f"Error getting positions: {str(e)}")
        return {}

def select_top_stocks(config_file, top_n=25):
    """
    Select the top stocks based on multi-factor scoring
    
    Args:
        config_file: Path to configuration file
        top_n: Number of top stocks to select
    
    Returns:
        list: Selected stocks with scores
    """
    # Load configuration
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Get S&P 500 symbols
    symbols = get_sp500_symbols()
    
    # Initialize data loader
    credentials = load_alpaca_credentials(mode='paper')
    data_loader = AlpacaDataLoader(
        api_key=credentials['api_key'],
        api_secret=credentials['api_secret'],
        base_url=credentials['base_url']
    )
    
    # Initialize strategy
    strategy = CombinedStrategy(config)
    
    # Calculate lookback period
    lookback_days = config.get('lookback_days', 100)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    
    # Load historical data
    logger.info(f"Loading data for {len(symbols)} symbols from {start_date} to {end_date}")
    historical_data = data_loader.load_historical_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        timeframe='1D'
    )
    
    # Select stocks
    logger.info("Selecting top stocks based on multi-factor scoring")
    market_regime = config.get('market_regime', 'mixed')
    selections = strategy.select_stocks_multi_factor(
        historical_data=historical_data,
        date=datetime.now().strftime('%Y-%m-%d'),
        market_regime=market_regime
    )
    
    # Sort by combined score and take top N
    sorted_selections = sorted(selections, key=lambda x: x['combined_score'], reverse=True)
    top_selections = sorted_selections[:top_n]
    
    logger.info(f"Selected top {len(top_selections)} stocks")
    for i, selection in enumerate(top_selections):
        logger.info(f"{i+1}. {selection['symbol']} - Score: {selection['combined_score']:.4f}, "
                   f"Direction: {selection['direction']}")
    
    return top_selections

def execute_trades(api, selections, positions, max_positions=25, position_size=0.04):
    """
    Execute trades based on selected stocks
    
    Args:
        api: Alpaca API client
        selections: Selected stocks with scores
        positions: Current positions
        max_positions: Maximum number of positions
        position_size: Position size as a fraction of portfolio
    """
    try:
        # Get account information
        account = api.get_account()
        portfolio_value = float(account.equity)
        
        # Calculate position value
        position_value = portfolio_value * position_size
        
        # Get current positions
        current_symbols = set(positions.keys())
        
        # Get symbols to keep
        selection_symbols = {s['symbol'] for s in selections}
        
        # Determine positions to close
        symbols_to_close = current_symbols - selection_symbols
        
        # Close positions that are no longer in our selection
        if symbols_to_close:
            logger.info(f"Closing {len(symbols_to_close)} positions that are no longer selected")
            for symbol in symbols_to_close:
                try:
                    logger.info(f"Closing position in {symbol}")
                    api.close_position(symbol)
                except Exception as e:
                    logger.error(f"Error closing position in {symbol}: {str(e)}")
        
        # Open new positions or adjust existing ones
        logger.info(f"Processing {len(selections)} selected stocks")
        for selection in selections:
            symbol = selection['symbol']
            direction = selection['direction']
            
            try:
                # Skip if we already have the maximum number of positions
                if len(positions) >= max_positions and symbol not in positions:
                    logger.info(f"Maximum positions reached, skipping {symbol}")
                    continue
                
                # Calculate target shares based on position size
                current_price = float(api.get_latest_trade(symbol).price)
                target_shares = int(position_value / current_price)
                
                if target_shares <= 0:
                    logger.warning(f"Target shares for {symbol} is zero, skipping")
                    continue
                
                # Determine side based on direction
                side = 'buy'
                if direction == 'SHORT':
                    side = 'sell'
                
                # Check if we already have a position in this stock
                if symbol in positions:
                    current_position = positions[symbol]
                    current_shares = abs(float(current_position['qty']))
                    
                    # If direction changed, close and reopen
                    if (direction == 'SHORT' and current_position['qty'] > 0) or \
                       (direction == 'LONG' and current_position['qty'] < 0):
                        logger.info(f"Direction changed for {symbol}, closing position")
                        api.close_position(symbol)
                        time.sleep(1)  # Wait for the order to process
                        
                        logger.info(f"Opening new {direction} position in {symbol}: {target_shares} shares")
                        api.submit_order(
                            symbol=symbol,
                            qty=target_shares,
                            side=side,
                            type='market',
                            time_in_force='day'
                        )
                    else:
                        # Adjust position size if needed
                        shares_diff = target_shares - current_shares
                        if abs(shares_diff) / current_shares > 0.1:  # Only adjust if >10% difference
                            if shares_diff > 0:
                                logger.info(f"Increasing position in {symbol} by {shares_diff} shares")
                                api.submit_order(
                                    symbol=symbol,
                                    qty=shares_diff,
                                    side=side,
                                    type='market',
                                    time_in_force='day'
                                )
                            elif shares_diff < 0:
                                logger.info(f"Reducing position in {symbol} by {abs(shares_diff)} shares")
                                api.submit_order(
                                    symbol=symbol,
                                    qty=abs(shares_diff),
                                    side='sell' if side == 'buy' else 'buy',
                                    type='market',
                                    time_in_force='day'
                                )
                else:
                    # Open new position
                    logger.info(f"Opening new {direction} position in {symbol}: {target_shares} shares")
                    api.submit_order(
                        symbol=symbol,
                        qty=target_shares,
                        side=side,
                        type='market',
                        time_in_force='day'
                    )
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error executing trades: {str(e)}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Live trading for S&P 500 stock selection strategy')
    parser.add_argument('--config', type=str, default='configuration_enhanced_multi_factor_500.json',
                        help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['paper', 'live'], default='paper',
                        help='Trading mode: paper or live')
    parser.add_argument('--top_n', type=int, default=25,
                        help='Number of top stocks to select')
    parser.add_argument('--position_size', type=float, default=0.04,
                        help='Position size as a fraction of portfolio')
    args = parser.parse_args()
    
    # Confirm if using live mode
    if args.mode == 'live':
        confirmation = input("You are about to use LIVE trading mode. Are you sure? (yes/no): ")
        if confirmation.lower() != 'yes':
            logger.info("Live trading cancelled")
            return
        logger.warning("USING LIVE TRADING MODE")
    
    # Load credentials and initialize API
    credentials = load_alpaca_credentials(mode=args.mode)
    api = initialize_alpaca_api(credentials)
    
    # Get current positions
    positions = get_current_positions(api)
    
    # Select top stocks
    selections = select_top_stocks(args.config, args.top_n)
    
    # Execute trades
    execute_trades(api, selections, positions, args.top_n, args.position_size)
    
    # Get updated positions
    updated_positions = get_current_positions(api)
    
    # Print summary
    logger.info(f"Trading completed. Current positions: {len(updated_positions)}")

if __name__ == "__main__":
    main()
