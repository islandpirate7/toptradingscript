#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Alpaca Integration Module
------------------------
This module provides integration with the Alpaca API to fetch market data
for a large universe of stocks and execute trades.
"""

import os
import time
import logging
import datetime as dt
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import requests

# Import for Alpaca API
try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.rest import REST, TimeFrame
    from alpaca_trade_api.stream import Stream
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

# Import from multi_strategy_system
from multi_strategy_system import CandleData, Signal, TradeDirection

logger = logging.getLogger(__name__)

class AlpacaIntegration:
    """Integration with Alpaca API for data fetching and trading"""
    
    def __init__(self, api_key: str, api_secret: str, base_url: str = "https://paper-api.alpaca.markets"):
        """Initialize Alpaca integration"""
        if not ALPACA_AVAILABLE:
            raise ImportError("alpaca-trade-api package is required for Alpaca integration")
        
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        
        # Initialize API
        self.api = REST(api_key, api_secret, base_url)
        logger.info(f"Alpaca API initialized with endpoint: {base_url}")
        
        # Check connection
        try:
            account = self.api.get_account()
            logger.info(f"Connected to Alpaca account: {account.id}")
            logger.info(f"Account status: {account.status}")
            logger.info(f"Account equity: ${float(account.equity):.2f}")
            logger.info(f"Account buying power: ${float(account.buying_power):.2f}")
        except Exception as e:
            logger.error(f"Error connecting to Alpaca API: {str(e)}")
            raise
    
    def get_tradable_assets(self, min_price: float = 5.0, min_volume: int = 500000, 
                           status: str = "active", asset_class: str = "us_equity",
                           max_stocks: int = 100) -> List[Dict[str, Any]]:
        """
        Get list of tradable assets from Alpaca that meet our criteria
        
        Args:
            min_price: Minimum price filter
            min_volume: Minimum average volume filter
            status: Asset status filter ('active' or 'inactive')
            asset_class: Asset class filter ('us_equity')
            max_stocks: Maximum number of stocks to return
            
        Returns:
            List of asset dictionaries with symbol, name, exchange info
        """
        try:
            # Get all assets
            assets = self.api.list_assets(status=status, asset_class=asset_class)
            logger.info(f"Retrieved {len(assets)} assets from Alpaca")
            
            # Filter assets
            filtered_assets = []
            for asset in assets:
                # Skip assets with no symbol
                if not asset.symbol:
                    continue
                
                # Skip OTC stocks
                if asset.exchange == "OTC":
                    continue
                
                # Get recent bars to check price and volume
                try:
                    bars = self.api.get_bars(
                        asset.symbol, 
                        TimeFrame.Day, 
                        limit=5,
                        adjustment='raw'
                    ).df
                    
                    if len(bars) == 0:
                        continue
                    
                    # Calculate average volume and latest price
                    avg_volume = bars['volume'].mean()
                    latest_price = bars['close'].iloc[-1]
                    
                    # Apply filters
                    if latest_price >= min_price and avg_volume >= min_volume:
                        filtered_assets.append({
                            'symbol': asset.symbol,
                            'name': asset.name,
                            'exchange': asset.exchange,
                            'price': latest_price,
                            'volume': avg_volume,
                            'tradable': asset.tradable,
                            'marginable': asset.marginable,
                            'shortable': asset.shortable,
                            'easy_to_borrow': asset.easy_to_borrow
                        })
                        
                        # Log progress periodically
                        if len(filtered_assets) % 20 == 0:
                            logger.info(f"Found {len(filtered_assets)} eligible stocks so far...")
                        
                        # Stop if we reached max_stocks
                        if len(filtered_assets) >= max_stocks:
                            break
                except Exception as e:
                    logger.warning(f"Error getting data for {asset.symbol}: {str(e)}")
                    continue
                
                # Add a small delay to avoid API rate limits
                time.sleep(0.1)
            
            logger.info(f"Filtered down to {len(filtered_assets)} tradable assets")
            return filtered_assets
            
        except Exception as e:
            logger.error(f"Error getting tradable assets: {str(e)}")
            return []
    
    def get_market_data(self, start_date: dt.date, end_date: dt.date) -> Tuple[List[CandleData], List[CandleData]]:
        """
        Get market data (SPY and VIX) for the specified date range
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            Tuple of (market_data, vix_data) as lists of CandleData
        """
        market_data = []
        vix_data = []
        
        try:
            # Get SPY data
            spy_bars = self.api.get_bars(
                "SPY", 
                TimeFrame.Day, 
                start=start_date.isoformat(), 
                end=end_date.isoformat(),
                adjustment='raw'
            ).df
            
            for index, row in spy_bars.iterrows():
                timestamp = index.to_pydatetime()
                market_candle = CandleData(
                    timestamp=timestamp,
                    open=float(row['open']),
                    high=float(row['high']),
                    low=float(row['low']),
                    close=float(row['close']),
                    volume=int(row['volume'])
                )
                market_data.append(market_candle)
                
            logger.info(f"Retrieved {len(market_data)} SPY candles from Alpaca")
            
            # For VIX data, we need to use a different source as Alpaca doesn't provide VIX
            # Here we'll use Yahoo Finance as a fallback for VIX
            import yfinance as yf
            
            vix_data_df = yf.download("^VIX", 
                                    start=start_date.strftime('%Y-%m-%d'),
                                    end=(end_date + dt.timedelta(days=1)).strftime('%Y-%m-%d'),
                                    interval="1d")
            
            for index, row in vix_data_df.iterrows():
                timestamp = index.to_pydatetime()
                vix_candle = CandleData(
                    timestamp=timestamp,
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    volume=int(row['Volume'])
                )
                vix_data.append(vix_candle)
                
            logger.info(f"Retrieved {len(vix_data)} VIX candles from Yahoo Finance")
            
        except Exception as e:
            logger.error(f"Error fetching market data: {str(e)}")
            
        return market_data, vix_data
    
    def get_stock_data(self, symbols: List[str], start_date: dt.date, end_date: dt.date) -> Dict[str, List[CandleData]]:
        """
        Get stock data for multiple symbols for the specified date range
        
        Args:
            symbols: List of stock symbols
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            Dictionary of symbol -> list of CandleData
        """
        result = {}
        
        # Process symbols in batches to avoid API rate limits
        batch_size = 100
        for i in range(0, len(symbols), batch_size):
            batch_symbols = symbols[i:i+batch_size]
            logger.info(f"Fetching data for batch {i//batch_size + 1}/{(len(symbols)-1)//batch_size + 1} ({len(batch_symbols)} symbols)")
            
            try:
                # Get data for the batch
                bars = self.api.get_bars(
                    batch_symbols,
                    TimeFrame.Day,
                    start=start_date.isoformat(),
                    end=end_date.isoformat(),
                    adjustment='raw'
                ).df
                
                # Group by symbol
                for symbol in batch_symbols:
                    try:
                        symbol_bars = bars.loc[bars.index.get_level_values('symbol') == symbol]
                        
                        if len(symbol_bars) == 0:
                            logger.warning(f"No data found for {symbol}")
                            result[symbol] = []
                            continue
                        
                        candles = []
                        for idx, row in symbol_bars.iterrows():
                            timestamp = idx[1].to_pydatetime()  # Multi-index (symbol, timestamp)
                            candle = CandleData(
                                timestamp=timestamp,
                                open=float(row['open']),
                                high=float(row['high']),
                                low=float(row['low']),
                                close=float(row['close']),
                                volume=int(row['volume'])
                            )
                            candles.append(candle)
                        
                        result[symbol] = candles
                        
                    except Exception as e:
                        logger.error(f"Error processing data for {symbol}: {str(e)}")
                        result[symbol] = []
                
            except Exception as e:
                logger.error(f"Error fetching batch data: {str(e)}")
                # Add empty lists for all symbols in the batch
                for symbol in batch_symbols:
                    result[symbol] = []
            
            # Add a delay to avoid API rate limits
            time.sleep(1)
        
        # Log summary
        total_candles = sum(len(candles) for candles in result.values())
        logger.info(f"Retrieved {total_candles} candles for {len(symbols)} symbols")
        
        return result
    
    def place_order(self, symbol: str, qty: int, side: str, order_type: str = 'market', 
                   time_in_force: str = 'day', limit_price: float = None, 
                   stop_price: float = None, client_order_id: str = None) -> Dict[str, Any]:
        """
        Place an order with Alpaca
        
        Args:
            symbol: Stock symbol
            qty: Quantity of shares
            side: 'buy' or 'sell'
            order_type: 'market', 'limit', 'stop', 'stop_limit'
            time_in_force: 'day', 'gtc', 'opg', 'cls', 'ioc', 'fok'
            limit_price: Limit price for limit and stop_limit orders
            stop_price: Stop price for stop and stop_limit orders
            client_order_id: Client order ID for tracking
            
        Returns:
            Order information dictionary
        """
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=order_type,
                time_in_force=time_in_force,
                limit_price=limit_price,
                stop_price=stop_price,
                client_order_id=client_order_id
            )
            
            logger.info(f"Placed {side} order for {qty} shares of {symbol}")
            return {
                'id': order.id,
                'client_order_id': order.client_order_id,
                'symbol': order.symbol,
                'side': order.side,
                'qty': order.qty,
                'order_type': order.type,
                'status': order.status
            }
            
        except Exception as e:
            logger.error(f"Error placing order for {symbol}: {str(e)}")
            return {'error': str(e)}
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        try:
            positions = self.api.list_positions()
            result = []
            
            for position in positions:
                result.append({
                    'symbol': position.symbol,
                    'qty': int(position.qty),
                    'market_value': float(position.market_value),
                    'cost_basis': float(position.cost_basis),
                    'unrealized_pl': float(position.unrealized_pl),
                    'unrealized_plpc': float(position.unrealized_plpc),
                    'current_price': float(position.current_price),
                    'lastday_price': float(position.lastday_price),
                    'change_today': float(position.change_today)
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            return []
    
    def get_account(self) -> Dict[str, Any]:
        """Get account information"""
        try:
            account = self.api.get_account()
            return {
                'id': account.id,
                'status': account.status,
                'equity': float(account.equity),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'long_market_value': float(account.long_market_value),
                'short_market_value': float(account.short_market_value),
                'portfolio_value': float(account.portfolio_value),
                'initial_margin': float(account.initial_margin),
                'maintenance_margin': float(account.maintenance_margin),
                'daytrade_count': int(account.daytrade_count),
                'last_equity': float(account.last_equity),
                'last_maintenance_margin': float(account.last_maintenance_margin)
            }
            
        except Exception as e:
            logger.error(f"Error getting account information: {str(e)}")
            return {'error': str(e)}
