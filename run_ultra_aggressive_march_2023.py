import datetime as dt
import logging
import pandas as pd
import numpy as np
import yaml
import json
import matplotlib.pyplot as plt
from tabulate import tabulate
from backtest_march_2023_ultra_aggressive import UltraAggressiveBacktester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def run_ultra_aggressive_backtest():
    """Run ultra aggressive backtest for March 2023"""
    # Set date range for March 2023
    start_date = dt.datetime(2023, 3, 1)
    end_date = dt.datetime(2023, 3, 31)
    
    logger.info(f"Running ultra aggressive backtest for March 2023")
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    
    # Initialize backtester with configuration
    config_file = 'configuration_combined_strategy.yaml'
    backtester = UltraAggressiveBacktester(config_file)
    
    # Log multi-factor stock selection status
    logger.info(f"Multi-factor stock selection enabled: {backtester.config.get('general', {}).get('use_multi_factor_selection', False)}")
    
    # Log strategy parameters
    mr_config = backtester.config.get('strategy_configs', {}).get('MeanReversion', {})
    tf_config = backtester.config.get('strategy_configs', {}).get('TrendFollowing', {})
    combined_config = backtester.config.get('strategy_configs', {}).get('Combined', {})
    
    # Log mean reversion strategy parameters
    logger.info(f"Initialized OPTIMIZED strategy with parameters: BB period={mr_config.get('bb_period', 20)}, "
               f"BB std={mr_config.get('bb_std', 1.8)}, "
               f"RSI period={mr_config.get('rsi_period', 14)}, "
               f"RSI thresholds={mr_config.get('rsi_oversold', 30)}/{mr_config.get('rsi_overbought', 70)}, "
               f"Require reversal={mr_config.get('require_reversal', True)}, "
               f"SL/TP ATR multipliers={mr_config.get('stop_loss_atr_multiplier', 1.5)}/{mr_config.get('take_profit_atr_multiplier', 3.0)}, "
               f"Volume filter={mr_config.get('volume_filter', True)}")
    
    # Log combined strategy weights
    logger.info(f"Initialized Combined Strategy with weights: MR={combined_config.get('mean_reversion_weight', 0.5)}, TF={combined_config.get('trend_following_weight', 0.5)}")
    
    # Log combined strategy parameters
    logger.info(f"Initialized COMBINED strategy with parameters: BB period={combined_config.get('bb_period', 20)}, "
               f"BB std={combined_config.get('bb_std', 1.8)}, "
               f"RSI period={combined_config.get('rsi_period', 14)}, "
               f"RSI thresholds={combined_config.get('rsi_oversold', 30)}/{combined_config.get('rsi_overbought', 70)}, "
               f"Require reversal={combined_config.get('require_reversal', True)}, "
               f"SL/TP ATR multipliers={combined_config.get('stop_loss_atr_multiplier', 1.6)}/{combined_config.get('take_profit_atr_multiplier', 2.8)}, "
               f"Volume filter={combined_config.get('volume_filter', True)}")
    
    # Run backtest
    results = backtester.run_backtest(start_date, end_date)
    
    # Print backtest summary
    print("\nBacktest Summary:")
    print(f"Start Date: {start_date.date()}")
    print(f"End Date: {end_date.date()}")
    print(f"Initial Capital: ${backtester.initial_capital:,.2f}")
    
    # Print metrics if available
    if results.metrics:
        print(f"\nPerformance Metrics:")
        print(f"Total Trades: {results.metrics.get('total_trades', 0)}")
        print(f"Win Rate: {results.metrics.get('win_rate', 0):.2%}")
        print(f"Profit Factor: {results.metrics.get('profit_factor', 0):.2f}")
        print(f"Average Profit: ${results.metrics.get('average_profit', 0):,.2f}")
        print(f"Average Loss: ${results.metrics.get('average_loss', 0):,.2f}")
        print(f"Max Drawdown: {results.metrics.get('max_drawdown', 0):.2%}")
        print(f"Sharpe Ratio: {results.metrics.get('sharpe_ratio', 0):.2f}")
        print(f"Total Return: {results.metrics.get('total_return', 0):.2%}")
        
        # Create a DataFrame for daily returns
        if results.daily_returns:
            # Create a DataFrame with dates and returns
            dates = [start_date + dt.timedelta(days=i) for i in range(len(results.daily_returns))]
            daily_returns_df = pd.DataFrame({
                'date': dates,
                'return_pct': results.daily_returns
            })
            daily_returns_df.set_index('date', inplace=True)
            
            # Plot daily returns
            plt.figure(figsize=(12, 6))
            plt.plot(daily_returns_df.index, daily_returns_df['return_pct'].cumsum() * 100)
            plt.title('Cumulative Returns (%)')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Return (%)')
            plt.grid(True)
            plt.savefig('ultra_aggressive_cumulative_returns.png')
            plt.close()
            
            print(f"\nDaily Returns Chart saved as 'ultra_aggressive_cumulative_returns.png'")
    else:
        print("No metrics available in the results")
    
    # Print trade summary
    if results.trades:
        print(f"\nTrade Summary:")
        print(f"Total Trades: {len(results.trades)}")
        
        # Group trades by symbol
        trades_by_symbol = {}
        for trade in results.trades:
            symbol = trade['symbol']
            if symbol not in trades_by_symbol:
                trades_by_symbol[symbol] = []
            trades_by_symbol[symbol].append(trade)
        
        # Group trades by strategy
        trades_by_strategy = {}
        for trade in results.trades:
            strategy = trade.get('strategy', 'unknown')
            if strategy not in trades_by_strategy:
                trades_by_strategy[strategy] = []
            trades_by_strategy[strategy].append(trade)
        
        # Print trades by symbol
        symbol_stats = []
        for symbol, trades in trades_by_symbol.items():
            winning_trades = [t for t in trades if t['pnl'] > 0]
            win_rate = len(winning_trades) / len(trades) if trades else 0
            total_pnl = sum(t['pnl'] for t in trades)
            avg_pnl = total_pnl / len(trades) if trades else 0
            
            print(f"\n{symbol}:")
            print(f"  Trades: {len(trades)}")
            print(f"  Win Rate: {win_rate:.2%}")
            print(f"  Total P&L: ${total_pnl:.2f}")
            
            symbol_stats.append({
                'Symbol': symbol,
                'Trades': len(trades),
                'Win Rate': f"{win_rate:.2%}",
                'Total P&L': f"${total_pnl:.2f}",
                'Avg P&L': f"${avg_pnl:.2f}"
            })
        
        # Create a DataFrame for symbol statistics
        symbol_stats_df = pd.DataFrame(symbol_stats)
        
        # Print strategy statistics
        print("\nStrategy Performance:")
        strategy_stats = []
        for strategy, trades in trades_by_strategy.items():
            winning_trades = [t for t in trades if t['pnl'] > 0]
            win_rate = len(winning_trades) / len(trades) if trades else 0
            total_pnl = sum(t['pnl'] for t in trades)
            avg_pnl = total_pnl / len(trades) if trades else 0
            
            print(f"\n{strategy.capitalize()}:")
            print(f"  Trades: {len(trades)}")
            print(f"  Win Rate: {win_rate:.2%}")
            print(f"  Total P&L: ${total_pnl:.2f}")
            print(f"  Average P&L: ${avg_pnl:.2f}")
            
            strategy_stats.append({
                'Strategy': strategy.capitalize(),
                'Trades': len(trades),
                'Win Rate': win_rate,
                'Total P&L': total_pnl,
                'Avg P&L': avg_pnl
            })
        
        # Create a DataFrame for strategy statistics
        strategy_stats_df = pd.DataFrame(strategy_stats)
        
        # Plot strategy performance
        if len(strategy_stats_df) > 0:
            plt.figure(figsize=(10, 6))
            bars = plt.bar(strategy_stats_df['Strategy'], strategy_stats_df['Total P&L'])
            
            # Color bars based on P&L
            for i, bar in enumerate(bars):
                if strategy_stats_df['Total P&L'].iloc[i] > 0:
                    bar.set_color('green')
                else:
                    bar.set_color('red')
            
            plt.title('P&L by Strategy')
            plt.xlabel('Strategy')
            plt.ylabel('Total P&L ($)')
            plt.grid(axis='y')
            plt.savefig('ultra_aggressive_strategy_pnl.png')
            plt.close()
            
            print(f"\nStrategy P&L Chart saved as 'ultra_aggressive_strategy_pnl.png'")
        
        # Plot win rate by symbol
        if len(symbol_stats_df) > 0:
            # Sort by win rate
            symbol_stats_df['Win Rate Numeric'] = symbol_stats_df['Win Rate'].apply(lambda x: float(x.strip('%')) / 100)
            symbol_stats_df = symbol_stats_df.sort_values('Win Rate Numeric', ascending=False)
            
            plt.figure(figsize=(12, 6))
            bars = plt.bar(symbol_stats_df['Symbol'], symbol_stats_df['Win Rate Numeric'])
            
            # Color bars based on win rate
            for i, bar in enumerate(bars):
                if symbol_stats_df['Win Rate Numeric'].iloc[i] >= 0.5:
                    bar.set_color('green')
                else:
                    bar.set_color('red')
            
            plt.title('Win Rate by Symbol')
            plt.xlabel('Symbol')
            plt.ylabel('Win Rate')
            plt.axhline(y=0.5, color='black', linestyle='--')
            plt.grid(axis='y')
            plt.savefig('ultra_aggressive_symbol_win_rate.png')
            plt.close()
            
            print(f"\nSymbol Win Rate Chart saved as 'ultra_aggressive_symbol_win_rate.png'")
        
        # Save detailed trade log to CSV
        trades_df = pd.DataFrame(results.trades)
        trades_df.to_csv('ultra_aggressive_trades.csv', index=False)
        print(f"\nDetailed trade log saved to 'ultra_aggressive_trades.csv'")
    else:
        print("No trades available in the results")
    
    return results

if __name__ == "__main__":
    run_ultra_aggressive_backtest()
