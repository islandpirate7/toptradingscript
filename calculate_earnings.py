#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os

def calculate_earnings(starting_capital=100000):
    """Calculate earnings for each backtest period"""
    periods = ['20230101_20230331', '20230701_20230930', '20231001_20231231']
    results = []
    
    for period in periods:
        file_path = f'backtest_results_{period}.csv'
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
            
        df = pd.read_csv(file_path)
        
        # Extract quarter from period
        year = period[:4]
        start_month = int(period[4:6])
        quarter = (start_month - 1) // 3 + 1
        period_name = f"Q{quarter} {year}"
        
        # Calculate returns
        total_return_pct = df['return'].sum()
        dollar_return = starting_capital * (total_return_pct / 100)
        
        # Calculate annualized return (multiply by 4 for quarterly)
        annualized_return = dollar_return * 4
        
        # Calculate per-trade metrics
        avg_dollar_per_trade = dollar_return / len(df)
        
        # Calculate direction performance
        long_trades = df[df['direction'] == 'LONG']
        short_trades = df[df['direction'] == 'SHORT']
        neutral_trades = df[df['direction'] == 'NEUTRAL'] if 'NEUTRAL' in df['direction'].unique() else pd.DataFrame()
        
        long_return = long_trades['return'].sum() if len(long_trades) > 0 else 0
        short_return = short_trades['return'].sum() if len(short_trades) > 0 else 0
        neutral_return = neutral_trades['return'].sum() if len(neutral_trades) > 0 else 0
        
        long_dollar = starting_capital * (long_return / 100)
        short_dollar = starting_capital * (short_return / 100)
        neutral_dollar = starting_capital * (neutral_return / 100)
        
        # Calculate score range performance
        df['score_range'] = pd.cut(df['score'], bins=[0, 0.6, 0.7, 0.8, 0.9, 1.0], 
                                  labels=['0.0-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0'])
        
        score_returns = {}
        score_dollars = {}
        
        for score_range in df['score_range'].unique():
            if pd.isna(score_range):
                continue
                
            score_df = df[df['score_range'] == score_range]
            score_return = score_df['return'].sum()
            score_dollar = starting_capital * (score_return / 100)
            
            score_returns[score_range] = score_return
            score_dollars[score_range] = score_dollar
        
        results.append({
            'period': period,
            'period_name': period_name,
            'total_return_pct': total_return_pct,
            'dollar_return': dollar_return,
            'annualized_return': annualized_return,
            'avg_dollar_per_trade': avg_dollar_per_trade,
            'long_return': long_return,
            'short_return': short_return,
            'neutral_return': neutral_return,
            'long_dollar': long_dollar,
            'short_dollar': short_dollar,
            'neutral_dollar': neutral_dollar,
            'score_returns': score_returns,
            'score_dollars': score_dollars
        })
    
    # Print results
    print(f"\n=== Earnings Analysis (Starting Capital: ${starting_capital:,.2f}) ===\n")
    
    for result in results:
        print(f"Period: {result['period_name']} ({result['period']})")
        print(f"Total Return: {result['total_return_pct']:.2f}%")
        print(f"Dollar Return: ${result['dollar_return']:,.2f}")
        print(f"Annualized Return: ${result['annualized_return']:,.2f}/year")
        print(f"Average $ per Trade: ${result['avg_dollar_per_trade']:,.2f}")
        
        print("\nDirection Performance:")
        print(f"  LONG: ${result['long_dollar']:,.2f} ({result['long_return']:.2f}%)")
        print(f"  SHORT: ${result['short_dollar']:,.2f} ({result['short_return']:.2f}%)")
        if 'neutral_return' in result and result['neutral_return'] != 0:
            print(f"  NEUTRAL: ${result['neutral_dollar']:,.2f} ({result['neutral_return']:.2f}%)")
        
        print("\nScore Range Performance:")
        for score_range, dollar in result['score_dollars'].items():
            print(f"  {score_range}: ${dollar:,.2f} ({result['score_returns'][score_range]:.2f}%)")
        
        print("\n" + "-" * 50 + "\n")

if __name__ == "__main__":
    calculate_earnings(100000)
