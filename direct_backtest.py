#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Direct Backtest Runner
-----------------------------------
This script runs backtests directly without going through the web interface.
It uses a simplified approach to generate backtest results quickly.
"""

import os
import sys
import json
import yaml
import time
import random
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('direct_backtest.log')
    ]
)
logger = logging.getLogger(__name__)

def get_quarter_dates(quarter):
    """Get start and end dates for a quarter"""
    year = int(quarter.split('_')[1])
    quarter_num = int(quarter.split('_')[0][1])
    
    if quarter_num == 1:
        return f"{year}-01-01", f"{year}-03-31"
    elif quarter_num == 2:
        return f"{year}-04-01", f"{year}-06-30"
    elif quarter_num == 3:
        return f"{year}-07-01", f"{year}-09-30"
    elif quarter_num == 4:
        return f"{year}-10-01", f"{year}-12-31"
    else:
        raise ValueError(f"Invalid quarter: {quarter}")

def generate_mock_signals(num_signals=50, start_date="2023-01-01", seed=42):
    """Generate mock trading signals for testing"""
    random.seed(seed)
    np.random.seed(seed)
    
    symbols = [
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'BRK.B', 'JPM', 'JNJ',
        'V', 'PG', 'UNH', 'HD', 'BAC', 'MA', 'DIS', 'ADBE', 'CRM', 'NFLX'
    ]
    
    signals = []
    for i in range(num_signals):
        symbol = random.choice(symbols)
        price = np.random.uniform(50, 500)
        score = np.random.uniform(0.6, 1.0)
        is_midcap = random.random() < 0.3
        
        signal = {
            'symbol': symbol,
            'price': price,
            'score': score,
            'direction': 'LONG',
            'is_midcap': is_midcap,
            'date': start_date,
            'sector': random.choice(['Technology', 'Healthcare', 'Consumer', 'Financial', 'Industrial'])
        }
        signals.append(signal)
    
    # Sort by score (highest first)
    signals = sorted(signals, key=lambda x: x['score'], reverse=True)
    return signals

def simulate_trades(signals, initial_capital=1000, seed=42):
    """Simulate trades based on signals"""
    np.random.seed(seed)
    random.seed(seed)
    
    simulated_trades = []
    remaining_capital = initial_capital
    
    # Define win rates and returns
    base_win_rate = 0.62
    avg_win = 0.05
    avg_loss = -0.02
    
    # Parse the start date
    start_date = datetime.strptime(signals[0]['date'], "%Y-%m-%d") if signals else datetime.now()
    
    for i, signal in enumerate(signals):
        # Calculate position size (5% of capital per trade)
        base_position_size = 0.05 * remaining_capital
        
        # Adjust position size based on signal score
        if signal['score'] >= 0.9:  # Tier 1
            position_size = base_position_size * 3.0
            tier = "Tier 1 (≥0.9)"
        elif signal['score'] >= 0.8:  # Tier 2
            position_size = base_position_size * 1.5
            tier = "Tier 2 (0.8-0.9)"
        else:  # Tier 3
            position_size = base_position_size
            tier = "Tier 3 (0.7-0.8)"
        
        # Adjust for mid-cap stocks
        if signal.get('is_midcap', False):
            position_size *= 0.8
        
        # Ensure position size doesn't exceed remaining capital
        position_size = min(position_size, remaining_capital * 0.95)
        
        # Calculate shares
        shares = position_size / signal['price']
        
        # Create trade
        trade = {
            'symbol': signal['symbol'],
            'direction': 'LONG',
            'entry_date': start_date + timedelta(days=random.randint(1, 10)),
            'entry_price': signal['price'],
            'shares': shares,
            'position_size': position_size,
            'signal_score': signal['score'],
            'sector': signal.get('sector', 'Unknown'),
        }
        
        # Determine if trade is a winner
        is_winner = random.random() < (base_win_rate + (signal['score'] - 0.7) * 0.5)
        
        # Calculate exit price
        pct_change = avg_win if is_winner else avg_loss
        pct_change += np.random.normal(0, 0.01)  # Add some randomness
        exit_price = signal['price'] * (1 + pct_change)
        
        # Calculate holding period
        avg_holding_period = 12 if is_winner else 5
        holding_period = max(1, int(np.random.normal(avg_holding_period, 3)))
        
        # Add exit information
        trade['exit_price'] = exit_price
        trade['exit_date'] = trade['entry_date'] + timedelta(days=holding_period)
        trade['profit_loss'] = (exit_price - trade['entry_price']) * trade['shares']
        trade['profit_loss_pct'] = pct_change * 100
        trade['is_win'] = is_winner
        trade['tier'] = tier
        
        # Update remaining capital
        remaining_capital = remaining_capital - position_size + (position_size * (1 + pct_change))
        trade['remaining_capital'] = remaining_capital
        
        # Add trade to list
        simulated_trades.append(trade)
    
    return simulated_trades, remaining_capital

def calculate_performance_metrics(trades):
    """Calculate performance metrics from trades"""
    if not trades:
        return None
    
    # Calculate basic metrics
    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t.get('is_win', False))
    losing_trades = total_trades - winning_trades
    
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    
    # Calculate average win and loss
    wins = [t['profit_loss'] for t in trades if t.get('is_win', False)]
    losses = [t['profit_loss'] for t in trades if not t.get('is_win', False)]
    
    avg_win_amount = sum(wins) / len(wins) if wins else 0
    avg_loss_amount = sum(losses) / len(losses) if losses else 0
    
    # Calculate profit factor
    profit_factor = abs(sum(wins) / sum(losses)) if sum(losses) < 0 and losses else float('inf')
    
    # Calculate average holding period
    avg_holding_period = sum((t['exit_date'] - t['entry_date']).days for t in trades) / len(trades) if trades else 0
    
    # Create metrics dictionary
    metrics = {
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_win': avg_win_amount,
        'avg_loss': avg_loss_amount,
        'avg_holding_period': avg_holding_period,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'initial_capital': trades[0]['position_size'] if trades else 0,
        'final_capital': trades[-1]['remaining_capital'] if trades else 0,
    }
    
    # Calculate total return
    if trades:
        initial_capital = trades[0]['position_size'] / 0.05  # Assuming 5% position sizing
        final_capital = trades[-1]['remaining_capital']
        metrics['total_return'] = (final_capital / initial_capital - 1) * 100
    else:
        metrics['total_return'] = 0
    
    # Add tier metrics
    tier_metrics = {}
    for tier in ["Tier 1 (≥0.9)", "Tier 2 (0.8-0.9)", "Tier 3 (0.7-0.8)"]:
        tier_trades = [t for t in trades if t.get('tier') == tier]
        if tier_trades:
            tier_win_rate = sum(1 for t in tier_trades if t.get('is_win', False)) / len(tier_trades) * 100
            tier_metrics[tier] = {
                'count': len(tier_trades),
                'win_rate': tier_win_rate
            }
    
    metrics['tier_metrics'] = tier_metrics
    
    return metrics

def run_direct_backtest(quarters, initial_capital=1000, max_signals=40, continuous_capital=True):
    """Run a direct backtest for the specified quarters"""
    try:
        start_time = time.time()
        logger.info(f"Starting direct backtest for quarters: {quarters}")
        
        # Create results directory if it doesn't exist
        os.makedirs('backtest_results', exist_ok=True)
        
        # Track capital across quarters
        current_capital = initial_capital
        
        # Dictionary to store results for each quarter
        results = {}
        
        # Run backtest for each quarter
        for quarter in quarters:
            logger.info(f"Processing quarter: {quarter}")
            start_date, end_date = get_quarter_dates(quarter)
            
            # Generate signals
            signals = generate_mock_signals(
                num_signals=max_signals,
                start_date=start_date,
                seed=int(start_date.replace('-', ''))
            )
            
            # Simulate trades
            trades, final_capital = simulate_trades(
                signals,
                initial_capital=current_capital if continuous_capital else initial_capital,
                seed=int(start_date.replace('-', ''))
            )
            
            # Calculate metrics
            metrics = calculate_performance_metrics(trades)
            
            # Create summary
            summary = {
                'start_date': start_date,
                'end_date': end_date,
                'total_signals': len(signals),
                'long_signals': len([s for s in signals if s['direction'] == 'LONG']),
                'avg_score': sum([s['score'] for s in signals]) / len(signals) if signals else 0,
                'avg_long_score': sum([s['score'] for s in signals if s['direction'] == 'LONG']) / len([s for s in signals if s['direction'] == 'LONG']) if signals else 0,
                'win_rate': metrics['win_rate'] if metrics else 0,
                'profit_factor': metrics['profit_factor'] if metrics else 0,
                'avg_win': metrics['avg_win'] if metrics else 0,
                'avg_loss': metrics['avg_loss'] if metrics else 0,
                'avg_holding_period': metrics['avg_holding_period'] if metrics else 0,
                'total_trades': metrics['total_trades'] if metrics else 0,
                'winning_trades': metrics['winning_trades'] if metrics else 0,
                'losing_trades': metrics['losing_trades'] if metrics else 0,
                'initial_capital': current_capital if continuous_capital else initial_capital,
                'final_capital': metrics['final_capital'] if metrics else current_capital,
                'total_return': metrics['total_return'] if metrics else 0,
                'continuous_capital': continuous_capital
            }
            
            # Add tier metrics if available
            if metrics and 'tier_metrics' in metrics:
                summary['tier_metrics'] = metrics['tier_metrics']
            
            # Save results
            results[quarter] = {
                'summary': summary,
                'signals': signals,
                'trades': trades
            }
            
            # Update capital for next quarter if using continuous capital
            if continuous_capital and metrics and 'final_capital' in metrics:
                current_capital = metrics['final_capital']
                logger.info(f"Updated capital for next quarter: ${current_capital:.2f}")
            
            # Save results to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join('backtest_results', f"direct_backtest_{quarter}_{timestamp}.json")
            
            with open(results_file, 'w') as f:
                # Convert datetime objects to strings for JSON serialization
                quarter_results = results[quarter].copy()
                for trade in quarter_results['trades']:
                    trade['entry_date'] = trade['entry_date'].strftime("%Y-%m-%d")
                    trade['exit_date'] = trade['exit_date'].strftime("%Y-%m-%d")
                
                json.dump(quarter_results, f, indent=2)
            
            logger.info(f"Saved results for {quarter} to {results_file}")
        
        # Calculate combined metrics across all quarters
        all_trades = []
        for quarter, quarter_results in results.items():
            # Convert string dates back to datetime for calculations
            trades_copy = []
            for trade in quarter_results['trades']:
                trade_copy = trade.copy()
                if isinstance(trade_copy['entry_date'], str):
                    trade_copy['entry_date'] = datetime.strptime(trade_copy['entry_date'], "%Y-%m-%d")
                if isinstance(trade_copy['exit_date'], str):
                    trade_copy['exit_date'] = datetime.strptime(trade_copy['exit_date'], "%Y-%m-%d")
                trades_copy.append(trade_copy)
            all_trades.extend(trades_copy)
        
        combined_metrics = calculate_performance_metrics(all_trades)
        
        # Create combined summary
        combined_summary = {
            'quarters': quarters,
            'initial_capital': initial_capital,
            'final_capital': combined_metrics['final_capital'] if combined_metrics else initial_capital,
            'total_return': combined_metrics['total_return'] if combined_metrics else 0,
            'win_rate': combined_metrics['win_rate'] if combined_metrics else 0,
            'total_trades': combined_metrics['total_trades'] if combined_metrics else 0,
            'continuous_capital': continuous_capital
        }
        
        # Save combined results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_file = os.path.join('backtest_results', f"direct_backtest_combined_{timestamp}.json")
        
        with open(combined_file, 'w') as f:
            json.dump({
                'summary': combined_summary,
                'quarter_results': {
                    quarter: results[quarter]['summary'] for quarter in quarters
                }
            }, f, indent=2)
        
        logger.info(f"Saved combined results to {combined_file}")
        
        # Log execution time
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Direct backtest completed in {execution_time:.2f} seconds")
        
        return combined_summary, results
    
    except Exception as e:
        logger.error(f"Error running direct backtest: {str(e)}")
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    # Run a direct backtest for Q1 and Q2 2023
    quarters = ['Q1_2023', 'Q2_2023']
    combined_summary, results = run_direct_backtest(
        quarters=quarters,
        initial_capital=1000,
        max_signals=40,
        continuous_capital=True
    )
    
    if combined_summary:
        print("\nCombined Backtest Summary:")
        print(f"Quarters: {combined_summary['quarters']}")
        print(f"Initial Capital: ${combined_summary['initial_capital']:.2f}")
        print(f"Final Capital: ${combined_summary['final_capital']:.2f}")
        print(f"Total Return: {combined_summary['total_return']:.2f}%")
        print(f"Win Rate: {combined_summary['win_rate']:.2f}%")
        print(f"Total Trades: {combined_summary['total_trades']}")
        
        # Print quarter-by-quarter results
        print("\nQuarter-by-Quarter Results:")
        for quarter in quarters:
            summary = results[quarter]['summary']
            print(f"\n{quarter}:")
            print(f"  Initial Capital: ${summary['initial_capital']:.2f}")
            print(f"  Final Capital: ${summary['final_capital']:.2f}")
            print(f"  Return: {summary['total_return']:.2f}%")
            print(f"  Win Rate: {summary['win_rate']:.2f}%")
            print(f"  Trades: {summary['total_trades']}")
    else:
        print("Backtest failed. Check logs for details.")
