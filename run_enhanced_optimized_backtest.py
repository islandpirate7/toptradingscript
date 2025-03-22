#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run backtest for the optimized enhanced Mean Reversion strategy
"""

import os
import sys
import json
import yaml
import logging
import argparse
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union
from enhanced_mean_reversion_backtest import EnhancedMeanReversionBacktest

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration"""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger("EnhancedMeanReversionBacktest")

def prepare_config_for_backtest(config_file: str, debug: bool = False) -> Dict:
    """Load and prepare configuration for backtest"""
    logger = logging.getLogger("EnhancedMeanReversionBacktest")
    
    # Load configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_file}")
    
    # Check if we have symbols defined
    if 'symbols' not in config or not config['symbols']:
        logger.warning("No symbols defined in configuration, adding default symbols")
        
        # Add default symbols
        config['symbols'] = [
            {'symbol': 'AAPL', 'weight': 1.0},
            {'symbol': 'MSFT', 'weight': 1.0},
            {'symbol': 'AMZN', 'weight': 1.0},
            {'symbol': 'GOOGL', 'weight': 1.0},
            {'symbol': 'META', 'weight': 1.0}
        ]
    
    # Ensure we have position sizing configuration
    if 'position_sizing_config' not in config:
        logger.warning("No position sizing configuration found, adding defaults")
        config['position_sizing_config'] = {
            'base_risk_per_trade': 0.01,  # 1% risk per trade
            'max_position_size': 0.1,     # 10% max position size
            'min_position_size': 0.005    # 0.5% min position size
        }
    
    # Ensure we have initial capital
    if 'initial_capital' not in config:
        logger.warning("No initial capital defined, using default of $100,000")
        config['initial_capital'] = 100000
    
    # Save debug configuration if requested
    if debug:
        debug_config_file = config_file.replace('.yaml', '_debug.yaml')
        with open(debug_config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info(f"Saved debug configuration to {debug_config_file}")
    
    return config

def run_backtest(config_file: str, start_date: str, end_date: str, log_level: str = "INFO", debug: bool = False) -> Dict:
    """Run a backtest using the Enhanced Mean Reversion strategy"""
    # Set up logging
    logger = setup_logging(log_level)
    
    # Load and prepare configuration
    config = prepare_config_for_backtest(config_file, debug)
    
    # Parse dates
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    
    # Create backtest instance
    backtest = EnhancedMeanReversionBacktest(config)
    
    # Run backtest
    logger.info(f"Running backtest from {start_date} to {end_date}")
    results = backtest.run(start_date, end_date)
    
    # Check for errors
    if 'error' in results:
        logger.error(f"Backtest failed: {results['error']}")
        return results
    
    # Log results
    logger.info(f"Backtest completed with {len(results.get('trade_history', []))} trades")
    logger.info(f"Initial capital: ${results['initial_capital']:.2f}")
    logger.info(f"Final equity: ${results['final_equity']:.2f}")
    logger.info(f"Total return: {results.get('return', 0):.2%}")
    
    # Log signal statistics
    logger.info(f"Total signals generated: {results.get('total_signals_generated', 0)}")
    
    if 'signals_by_symbol' in results:
        for symbol, count in results['signals_by_symbol'].items():
            logger.info(f"Signals for {symbol}: {count}")
    
    # Log skipped signal reasons
    if 'skipped_signals_reasons' in results:
        logger.info("Skipped signals by reason:")
        for reason, count in results['skipped_signals_reasons'].items():
            logger.info(f"  {reason}: {count}")
    
    # Save results
    results_file = f"backtest_results_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json"
    
    # Convert datetime objects to strings for JSON serialization
    serializable_results = results.copy()
    
    # Convert trade history if present
    if 'trade_history' in serializable_results:
        serializable_results['trade_history'] = [
            {k: v.strftime('%Y-%m-%d') if isinstance(v, datetime.datetime) else v 
             for k, v in trade.__dict__.items() if not k.startswith('_')}
            for trade in serializable_results['trade_history']
        ]
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Saved results to {results_file}")
    
    # Plot equity curve
    if 'equity_curve' in results:
        plot_equity_curve(results['equity_curve'], start_date, end_date)
    
    return results

def plot_equity_curve(equity_curve: Dict[str, float], start_date: datetime.datetime, end_date: datetime.datetime) -> None:
    """Plot equity curve from backtest results"""
    logger = logging.getLogger("EnhancedMeanReversionBacktest")
    
    if not equity_curve:
        logger.warning("No equity curve data to plot")
        return
    
    # Convert to DataFrame
    try:
        # Sort the equity curve by date
        sorted_dates = sorted(equity_curve.keys())
        equity_df = pd.DataFrame({
            'Date': [datetime.datetime.strptime(d, '%Y-%m-%d') for d in sorted_dates],
            'Equity': [equity_curve[d] for d in sorted_dates]
        })
        
        # Set Date as index
        equity_df.set_index('Date', inplace=True)
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(equity_df.index, equity_df['Equity'])
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity ($)')
        plt.grid(True)
        
        # Save plot
        plot_file = f"equity_curve_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.png"
        plt.savefig(plot_file)
        logger.info(f"Saved equity curve plot to {plot_file}")
        
        # Close plot to free memory
        plt.close()
        
    except Exception as e:
        logger.error(f"Error plotting equity curve: {e}")

def analyze_trades(trade_history: List[Dict]) -> Dict:
    """Analyze trade history and calculate performance metrics"""
    logger = logging.getLogger("EnhancedMeanReversionBacktest")
    
    if not trade_history:
        logger.warning("No trades to analyze")
        return {
            'win_rate': 0,
            'profit_factor': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0
        }
    
    # Convert to DataFrame
    try:
        trade_df = pd.DataFrame(trade_history)
        
        # Calculate basic metrics
        winning_trades = trade_df[trade_df['pnl'] > 0]
        losing_trades = trade_df[trade_df['pnl'] <= 0]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        total_count = len(trade_df)
        
        win_rate = win_count / total_count if total_count > 0 else 0
        
        total_profit = winning_trades['pnl'].sum() if not winning_trades.empty else 0
        total_loss = abs(losing_trades['pnl'].sum()) if not losing_trades.empty else 0
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        avg_win = winning_trades['pnl'].mean() if not winning_trades.empty else 0
        avg_loss = losing_trades['pnl'].mean() if not losing_trades.empty else 0
        
        # Calculate drawdown
        trade_df['cumulative_pnl'] = trade_df['pnl'].cumsum()
        trade_df['peak'] = trade_df['cumulative_pnl'].cummax()
        trade_df['drawdown'] = trade_df['peak'] - trade_df['cumulative_pnl']
        max_drawdown = trade_df['drawdown'].max()
        
        # Calculate Sharpe ratio (simplified)
        daily_returns = []
        current_date = None
        daily_pnl = 0
        
        for _, trade in trade_df.sort_values('exit_date').iterrows():
            exit_date = datetime.datetime.strptime(trade['exit_date'], '%Y-%m-%d').date() if isinstance(trade['exit_date'], str) else trade['exit_date'].date()
            
            if current_date is None:
                current_date = exit_date
            
            if exit_date == current_date:
                daily_pnl += trade['pnl']
            else:
                daily_returns.append(daily_pnl)
                daily_pnl = trade['pnl']
                current_date = exit_date
        
        # Add the last day
        if daily_pnl != 0:
            daily_returns.append(daily_pnl)
        
        daily_returns = np.array(daily_returns)
        sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 0 and np.std(daily_returns) > 0 else 0
        
        # Return metrics
        metrics = {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': total_count,
            'winning_trades': win_count,
            'losing_trades': loss_count
        }
        
        # Log metrics
        logger.info(f"Trade analysis:")
        logger.info(f"  Total trades: {total_count}")
        logger.info(f"  Win rate: {win_rate:.2%}")
        logger.info(f"  Profit factor: {profit_factor:.2f}")
        logger.info(f"  Average win: ${avg_win:.2f}")
        logger.info(f"  Average loss: ${avg_loss:.2f}")
        logger.info(f"  Max drawdown: ${max_drawdown:.2f}")
        logger.info(f"  Sharpe ratio: {sharpe_ratio:.2f}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error analyzing trades: {e}")
        return {
            'error': str(e)
        }

def main():
    """Main function to run the backtest from command line"""
    parser = argparse.ArgumentParser(description='Run Enhanced Mean Reversion backtest')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--start-date', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Logging level')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (saves debug configuration)')
    
    args = parser.parse_args()
    
    # Run backtest
    results = run_backtest(args.config, args.start_date, args.end_date, args.log_level, args.debug)
    
    # Analyze trades if we have any
    if 'trade_history' in results and results['trade_history']:
        analyze_trades(results['trade_history'])
    else:
        print("No trades executed during backtest period.")
        if 'skipped_signals_reasons' in results:
            print("\nSkipped signals by reason:")
            for reason, count in results['skipped_signals_reasons'].items():
                print(f"  {reason}: {count}")
        if 'total_signals_generated' in results:
            print(f"\nTotal signals generated: {results['total_signals_generated']}")

if __name__ == "__main__":
    main()
