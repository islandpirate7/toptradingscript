#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Monitor Trading Results
This script helps monitor the results of the S&P 500 trading strategy
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
import glob

def load_alpaca_credentials(mode='paper'):
    """Load Alpaca API credentials from JSON file"""
    try:
        with open('alpaca_credentials.json', 'r') as f:
            credentials = json.load(f)
        return credentials[mode]
    except Exception as e:
        print(f"Error loading Alpaca credentials: {str(e)}")
        raise

def get_alpaca_api(use_live=False):
    """Get Alpaca API client"""
    mode = 'live' if use_live else 'paper'
    credentials = load_alpaca_credentials(mode=mode)
    
    api = tradeapi.REST(
        key_id=credentials['api_key'],
        secret_key=credentials['api_secret'],
        base_url=credentials['base_url']
    )
    
    return api

def get_account_info(api):
    """Get account information"""
    account = api.get_account()
    
    return {
        'account_id': account.id,
        'status': account.status,
        'equity': float(account.equity),
        'cash': float(account.cash),
        'buying_power': float(account.buying_power),
        'long_market_value': float(account.long_market_value),
        'short_market_value': float(account.short_market_value),
        'initial_margin': float(account.initial_margin),
        'maintenance_margin': float(account.maintenance_margin),
        'last_equity': float(account.last_equity),
        'day_trade_count': int(account.daytrade_count),
        'portfolio_value': float(account.portfolio_value)
    }

def get_positions(api):
    """Get current positions"""
    positions = api.list_positions()
    
    positions_list = []
    for p in positions:
        positions_list.append({
            'symbol': p.symbol,
            'qty': float(p.qty),
            'side': 'LONG' if float(p.qty) > 0 else 'SHORT',
            'market_value': float(p.market_value),
            'cost_basis': float(p.cost_basis),
            'unrealized_pl': float(p.unrealized_pl),
            'unrealized_plpc': float(p.unrealized_plpc),
            'current_price': float(p.current_price),
            'lastday_price': float(p.lastday_price),
            'change_today': float(p.change_today)
        })
    
    return positions_list

def get_recent_orders(api, status='all', limit=100):
    """Get recent orders"""
    orders = api.list_orders(status=status, limit=limit)
    
    orders_list = []
    for o in orders:
        orders_list.append({
            'id': o.id,
            'client_order_id': o.client_order_id,
            'symbol': o.symbol,
            'side': o.side,
            'qty': float(o.qty),
            'filled_qty': float(o.filled_qty) if o.filled_qty else 0,
            'type': o.type,
            'status': o.status,
            'created_at': o.created_at,
            'filled_at': o.filled_at,
            'filled_avg_price': float(o.filled_avg_price) if o.filled_avg_price else 0
        })
    
    return orders_list

def load_trade_history():
    """Load trade history from CSV files"""
    trade_files = glob.glob('trades/trades_*.csv')
    
    if not trade_files:
        print("No trade history found")
        return pd.DataFrame()
    
    # Combine all trade files
    dfs = []
    for file in trade_files:
        try:
            df = pd.read_csv(file)
            df['file'] = os.path.basename(file)
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {file}: {str(e)}")
    
    if not dfs:
        return pd.DataFrame()
    
    # Combine all dataframes
    trades_df = pd.concat(dfs, ignore_index=True)
    
    # Convert timestamp to datetime
    if 'timestamp' in trades_df.columns:
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    
    return trades_df

def analyze_trade_history(trades_df):
    """Analyze trade history"""
    if trades_df.empty:
        print("No trade history to analyze")
        return
    
    # Count trades by direction
    direction_counts = trades_df['direction'].value_counts()
    
    # Count trades by symbol
    symbol_counts = trades_df['symbol'].value_counts().head(10)
    
    # Count trades by day
    if 'timestamp' in trades_df.columns:
        trades_df['date'] = trades_df['timestamp'].dt.date
        date_counts = trades_df['date'].value_counts().sort_index()
    
    # Print summary
    print("\n=== Trade History Analysis ===")
    print(f"Total trades: {len(trades_df)}")
    print("\nTrades by direction:")
    for direction, count in direction_counts.items():
        print(f"  {direction}: {count} ({count/len(trades_df)*100:.2f}%)")
    
    print("\nTop 10 traded symbols:")
    for symbol, count in symbol_counts.items():
        print(f"  {symbol}: {count}")
    
    if 'timestamp' in trades_df.columns:
        print("\nTrades by date:")
        for date, count in date_counts.items():
            print(f"  {date}: {count}")
    
    # Plot trades by direction
    plt.figure(figsize=(10, 6))
    direction_counts.plot(kind='bar')
    plt.title('Trades by Direction')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('trades_by_direction.png')
    
    # Plot trades by date
    if 'timestamp' in trades_df.columns:
        plt.figure(figsize=(12, 6))
        date_counts.plot(kind='bar')
        plt.title('Trades by Date')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('trades_by_date.png')
    
    # Plot top 10 traded symbols
    plt.figure(figsize=(12, 6))
    symbol_counts.plot(kind='bar')
    plt.title('Top 10 Traded Symbols')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('trades_by_symbol.png')
    
    print("\nPlots saved to current directory")

def analyze_positions(positions):
    """Analyze current positions"""
    if not positions:
        print("No current positions")
        return
    
    # Convert to DataFrame for easier analysis
    positions_df = pd.DataFrame(positions)
    
    # Calculate total market value
    total_market_value = positions_df['market_value'].sum()
    
    # Calculate total unrealized P&L
    total_unrealized_pl = positions_df['unrealized_pl'].sum()
    
    # Count positions by side
    side_counts = positions_df['side'].value_counts()
    
    # Calculate market value by side
    market_value_by_side = positions_df.groupby('side')['market_value'].sum()
    
    # Calculate unrealized P&L by side
    unrealized_pl_by_side = positions_df.groupby('side')['unrealized_pl'].sum()
    
    # Print summary
    print("\n=== Current Positions Analysis ===")
    print(f"Total positions: {len(positions)}")
    print(f"Total market value: ${total_market_value:.2f}")
    print(f"Total unrealized P&L: ${total_unrealized_pl:.2f}")
    
    print("\nPositions by side:")
    for side, count in side_counts.items():
        print(f"  {side}: {count} ({count/len(positions)*100:.2f}%)")
    
    print("\nMarket value by side:")
    for side, value in market_value_by_side.items():
        print(f"  {side}: ${value:.2f} ({value/total_market_value*100:.2f}%)")
    
    print("\nUnrealized P&L by side:")
    for side, value in unrealized_pl_by_side.items():
        print(f"  {side}: ${value:.2f}")
    
    # Sort positions by unrealized P&L
    positions_df = positions_df.sort_values('unrealized_pl', ascending=False)
    
    print("\nTop 5 performing positions:")
    for _, row in positions_df.head(5).iterrows():
        print(f"  {row['symbol']} ({row['side']}): ${row['unrealized_pl']:.2f} ({row['unrealized_plpc']*100:.2f}%)")
    
    print("\nBottom 5 performing positions:")
    for _, row in positions_df.tail(5).iterrows():
        print(f"  {row['symbol']} ({row['side']}): ${row['unrealized_pl']:.2f} ({row['unrealized_plpc']*100:.2f}%)")
    
    # Plot positions by side
    plt.figure(figsize=(10, 6))
    side_counts.plot(kind='bar')
    plt.title('Positions by Side')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('positions_by_side.png')
    
    # Plot market value by side
    plt.figure(figsize=(10, 6))
    market_value_by_side.plot(kind='bar')
    plt.title('Market Value by Side')
    plt.ylabel('Market Value ($)')
    plt.tight_layout()
    plt.savefig('market_value_by_side.png')
    
    # Plot top 10 positions by market value
    plt.figure(figsize=(12, 6))
    positions_df.sort_values('market_value', ascending=False).head(10).set_index('symbol')['market_value'].plot(kind='bar')
    plt.title('Top 10 Positions by Market Value')
    plt.ylabel('Market Value ($)')
    plt.tight_layout()
    plt.savefig('top_positions_by_market_value.png')
    
    print("\nPlots saved to current directory")

def monitor_trading_results(use_live=False):
    """Monitor trading results"""
    print(f"=== Trading Results Monitor ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===")
    print(f"Mode: {'LIVE' if use_live else 'PAPER'}")
    
    try:
        # Get Alpaca API
        api = get_alpaca_api(use_live=use_live)
        
        # Get account information
        account_info = get_account_info(api)
        
        # Get current positions
        positions = get_positions(api)
        
        # Get recent orders
        recent_orders = get_recent_orders(api)
        
        # Load trade history
        trades_df = load_trade_history()
        
        # Print account information
        print("\n=== Account Information ===")
        print(f"Account ID: {account_info['account_id']}")
        print(f"Status: {account_info['status']}")
        print(f"Equity: ${account_info['equity']:.2f}")
        print(f"Cash: ${account_info['cash']:.2f}")
        print(f"Buying Power: ${account_info['buying_power']:.2f}")
        print(f"Long Market Value: ${account_info['long_market_value']:.2f}")
        print(f"Short Market Value: ${account_info['short_market_value']:.2f}")
        print(f"Initial Margin: ${account_info['initial_margin']:.2f}")
        print(f"Maintenance Margin: ${account_info['maintenance_margin']:.2f}")
        print(f"Day Trade Count: {account_info['day_trade_count']}")
        print(f"Portfolio Value: ${account_info['portfolio_value']:.2f}")
        
        # Print current positions
        print(f"\n=== Current Positions ({len(positions)}) ===")
        if positions:
            positions_df = pd.DataFrame(positions)
            print(positions_df[['symbol', 'side', 'qty', 'market_value', 'unrealized_pl', 'unrealized_plpc']].to_string(index=False))
        else:
            print("No current positions")
        
        # Print recent orders
        print(f"\n=== Recent Orders ({len(recent_orders)}) ===")
        if recent_orders:
            recent_orders_df = pd.DataFrame(recent_orders)
            print(recent_orders_df[['symbol', 'side', 'qty', 'status', 'created_at']].head(10).to_string(index=False))
            print("...")
        else:
            print("No recent orders")
        
        # Analyze trade history
        analyze_trade_history(trades_df)
        
        # Analyze current positions
        analyze_positions(positions)
        
    except Exception as e:
        print(f"Error monitoring trading results: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor Trading Results')
    parser.add_argument('--live', action='store_true', help='Use live trading')
    
    args = parser.parse_args()
    
    monitor_trading_results(use_live=args.live)
