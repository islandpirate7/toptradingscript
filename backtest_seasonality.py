#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Backtest for the seasonality-based trading strategy.
This script evaluates the performance of trading based on seasonal patterns.
"""

import os
import logging
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from seasonality_analyzer import SeasonalityAnalyzer, SeasonType, Direction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class SeasonalityBacktester:
    """Class to backtest seasonality-based trading strategies"""
    
    def __init__(self, api_credentials_path: str):
        """Initialize the backtester
        
        Args:
            api_credentials_path (str): Path to Alpaca API credentials
        """
        self.analyzer = SeasonalityAnalyzer(api_credentials_path)
        self.results = {}
        
    def load_opportunities(self, opportunities_file: str) -> list:
        """Load trading opportunities from a YAML file
        
        Args:
            opportunities_file (str): Path to opportunities YAML file
            
        Returns:
            list: List of trading opportunities
        """
        try:
            with open(opportunities_file, 'r') as f:
                data = yaml.safe_load(f)
                opportunities = data.get('opportunities', [])
                logging.info(f"Loaded {len(opportunities)} opportunities from {opportunities_file}")
                return opportunities
        except Exception as e:
            logging.error(f"Error loading opportunities: {e}")
            return []
    
    def backtest_opportunity(self, symbol: str, season_name: str, direction: str, 
                           lookback_years: int = 5, hold_days: int = 20) -> dict:
        """Backtest a single seasonal opportunity
        
        Args:
            symbol (str): Stock symbol
            season_name (str): Name of the season (e.g., 'January', 'Monday')
            direction (str): Trading direction ('LONG' or 'SHORT')
            lookback_years (int, optional): Number of years to look back. Defaults to 5.
            hold_days (int, optional): Number of days to hold the position. Defaults to 20.
            
        Returns:
            dict: Backtest results
        """
        # Get historical data
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=365 * lookback_years)
        
        # Fetch data if not already available
        if not hasattr(self.analyzer, 'historical_data') or symbol not in self.analyzer.historical_data:
            self.analyzer.set_universe([symbol])
            self.analyzer.fetch_historical_data(start_date, end_date)
        
        df = self.analyzer.historical_data.get(symbol)
        if df is None or df.empty:
            logging.warning(f"No data available for {symbol}")
            return None
        
        # Identify seasonal periods
        season_periods = []
        
        # Map season_name to the appropriate datetime attribute
        if season_name in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
            day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4}
            season_value = day_map.get(season_name)
            df['season'] = df.index.dayofweek
        elif season_name in ['January', 'February', 'March', 'April', 'May', 'June', 
                           'July', 'August', 'September', 'October', 'November', 'December']:
            month_map = {
                'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
            }
            season_value = month_map.get(season_name)
            df['season'] = df.index.month
        elif season_name in ['Q1', 'Q2', 'Q3', 'Q4']:
            quarter_map = {'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4}
            season_value = quarter_map.get(season_name)
            df['season'] = df.index.quarter
        elif season_name in ['H1', 'H2']:
            half_map = {'H1': 1, 'H2': 2}
            season_value = half_map.get(season_name)
            df['season'] = (df.index.month - 1) // 6 + 1
        else:
            try:
                # Try to interpret as day of month
                season_value = int(season_name)
                df['season'] = df.index.day
            except ValueError:
                logging.error(f"Unrecognized season name: {season_name}")
                return None
        
        # Find all occurrences of the season
        season_starts = df[df['season'] == season_value].index
        
        # Prepare results
        trades = []
        
        for start_date in season_starts:
            try:
                # Define end date (hold_days trading days later)
                end_idx = df.index.get_loc(start_date) + hold_days
                if end_idx >= len(df):
                    continue
                end_date = df.index[end_idx]
                
                # Get prices
                entry_price = df.loc[start_date, 'close']
                exit_price = df.loc[end_date, 'close']
                
                # Calculate returns
                if direction == 'LONG':
                    returns = (exit_price / entry_price) - 1
                else:  # SHORT
                    returns = 1 - (exit_price / entry_price)
                
                # Add trade to results
                trades.append({
                    'entry_date': start_date.strftime('%Y-%m-%d'),
                    'exit_date': end_date.strftime('%Y-%m-%d'),
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'returns': returns,
                    'win': returns > 0
                })
                
            except Exception as e:
                logging.error(f"Error processing trade starting on {start_date}: {e}")
        
        # Calculate performance metrics
        if not trades:
            logging.warning(f"No trades found for {symbol} in {season_name}")
            return None
            
        returns = [t['returns'] for t in trades]
        wins = [t for t in trades if t['win']]
        
        results = {
            'symbol': symbol,
            'season': season_name,
            'direction': direction,
            'trade_count': len(trades),
            'win_rate': len(wins) / len(trades) if trades else 0,
            'avg_return': np.mean(returns) * 100 if returns else 0,  # Convert to percentage
            'total_return': (np.prod([1 + r for r in returns]) - 1) * 100 if returns else 0,  # Convert to percentage
            'max_return': max(returns) * 100 if returns else 0,  # Convert to percentage
            'min_return': min(returns) * 100 if returns else 0,  # Convert to percentage
            'std_return': np.std(returns) * 100 if returns else 0,  # Convert to percentage
            'sharpe_ratio': np.mean(returns) / np.std(returns) if returns and np.std(returns) > 0 else 0,
            'trades': trades
        }
        
        return results
    
    def backtest_opportunities(self, opportunities: list, lookback_years: int = 5, hold_days: int = 20) -> dict:
        """Backtest multiple seasonal opportunities
        
        Args:
            opportunities (list): List of trading opportunities
            lookback_years (int, optional): Number of years to look back. Defaults to 5.
            hold_days (int, optional): Number of days to hold the position. Defaults to 20.
            
        Returns:
            dict: Backtest results for all opportunities
        """
        results = {}
        
        for opp in opportunities:
            symbol = opp['symbol']
            season = opp['season']
            direction = opp['direction']
            
            logging.info(f"Backtesting {direction} strategy for {symbol} in {season}")
            
            result = self.backtest_opportunity(
                symbol=symbol,
                season_name=season,
                direction=direction,
                lookback_years=lookback_years,
                hold_days=hold_days
            )
            
            if result:
                results[f"{symbol}_{season}_{direction}"] = result
                logging.info(f"Backtest results for {symbol} in {season} ({direction}): "
                           f"Win Rate: {result['win_rate']:.2f}, "
                           f"Avg Return: {result['avg_return']:.2f}%, "
                           f"Total Return: {result['total_return']:.2f}%")
        
        self.results = results
        return results
    
    def generate_performance_report(self, output_dir: str) -> None:
        """Generate a performance report for the backtest results
        
        Args:
            output_dir (str): Directory to save the report
        """
        if not self.results:
            logging.warning("No backtest results available")
            return
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Create summary dataframe
        summary_data = []
        
        for key, result in self.results.items():
            summary_data.append({
                'Symbol': result['symbol'],
                'Season': result['season'],
                'Direction': result['direction'],
                'Trade Count': result['trade_count'],
                'Win Rate': result['win_rate'],
                'Avg Return (%)': result['avg_return'],
                'Total Return (%)': result['total_return'],
                'Sharpe Ratio': result['sharpe_ratio']
            })
        
        df_summary = pd.DataFrame(summary_data)
        
        # Sort by total return
        df_summary = df_summary.sort_values('Total Return (%)', ascending=False)
        
        # Save summary to CSV
        summary_path = os.path.join(output_dir, 'seasonality_backtest_summary.csv')
        df_summary.to_csv(summary_path, index=False)
        logging.info(f"Saved backtest summary to {summary_path}")
        
        # Generate performance plots
        for key, result in self.results.items():
            try:
                symbol = result['symbol']
                season = result['season']
                direction = result['direction']
                
                # Create trade returns plot
                trades = result['trades']
                dates = [datetime.strptime(t['entry_date'], '%Y-%m-%d') for t in trades]
                returns = [t['returns'] * 100 for t in trades]  # Convert to percentage
                
                plt.figure(figsize=(12, 6))
                plt.bar(range(len(returns)), returns, color=['green' if r > 0 else 'red' for r in returns])
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                plt.title(f"{symbol} {season} {direction} - Trade Returns")
                plt.xlabel('Trade Number')
                plt.ylabel('Return (%)')
                plt.grid(True, alpha=0.3)
                
                # Save plot
                plot_path = os.path.join(output_dir, f"{symbol}_{season}_{direction}_returns.png")
                plt.savefig(plot_path)
                plt.close()
                
                # Create equity curve
                equity = [1]
                for r in returns:
                    equity.append(equity[-1] * (1 + r/100))
                
                plt.figure(figsize=(12, 6))
                plt.plot(range(len(equity)), equity, color='blue')
                plt.title(f"{symbol} {season} {direction} - Equity Curve")
                plt.xlabel('Trade Number')
                plt.ylabel('Equity (starting at 1)')
                plt.grid(True, alpha=0.3)
                
                # Save plot
                plot_path = os.path.join(output_dir, f"{symbol}_{season}_{direction}_equity.png")
                plt.savefig(plot_path)
                plt.close()
                
                logging.info(f"Generated performance plots for {symbol} {season} {direction}")
                
            except Exception as e:
                logging.error(f"Error generating performance plots for {key}: {e}")
        
        # Generate combined equity curve for all strategies
        try:
            # Start with $10,000 portfolio
            initial_capital = 10000
            equal_allocation = initial_capital / len(self.results)
            
            # Create a timeline of all trade dates
            all_trades = []
            for key, result in self.results.items():
                for trade in result['trades']:
                    all_trades.append({
                        'strategy': key,
                        'entry_date': datetime.strptime(trade['entry_date'], '%Y-%m-%d'),
                        'exit_date': datetime.strptime(trade['exit_date'], '%Y-%m-%d'),
                        'returns': trade['returns']
                    })
            
            # Sort trades by entry date
            all_trades.sort(key=lambda x: x['entry_date'])
            
            # Create equity curve
            dates = [all_trades[0]['entry_date']]
            equity = [initial_capital]
            
            for trade in all_trades:
                # Add trade return to equity
                current_equity = equity[-1]
                trade_return = equal_allocation * trade['returns']
                new_equity = current_equity + trade_return
                
                # Add to equity curve
                dates.append(trade['exit_date'])
                equity.append(new_equity)
            
            # Plot combined equity curve
            plt.figure(figsize=(12, 6))
            plt.plot(dates, equity, color='blue')
            plt.title(f"Combined Seasonality Strategy - Equity Curve")
            plt.xlabel('Date')
            plt.ylabel('Equity ($)')
            plt.grid(True, alpha=0.3)
            
            # Save plot
            plot_path = os.path.join(output_dir, "combined_equity_curve.png")
            plt.savefig(plot_path)
            plt.close()
            
            logging.info(f"Generated combined equity curve")
            
        except Exception as e:
            logging.error(f"Error generating combined equity curve: {e}")
        
        logging.info(f"Completed performance report generation")
        
    def save_results_to_yaml(self, output_file: str) -> None:
        """Save backtest results to a YAML file
        
        Args:
            output_file (str): Path to output file
        """
        if not self.results:
            logging.warning("No backtest results available")
            return
            
        # Create a simplified version of results for YAML
        yaml_results = {}
        
        for key, result in self.results.items():
            # Remove trade details to keep file size manageable
            result_copy = result.copy()
            result_copy.pop('trades', None)
            yaml_results[key] = result_copy
        
        try:
            with open(output_file, 'w') as f:
                yaml.dump({'backtest_results': yaml_results}, f)
            logging.info(f"Saved backtest results to {output_file}")
        except Exception as e:
            logging.error(f"Error saving backtest results: {e}")

def main():
    """Main function to run the seasonality backtest"""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Backtest seasonality strategy')
    parser.add_argument('--opportunities', type=str, default='output/seasonal_opportunities.yaml',
                      help='Path to opportunities YAML file')
    parser.add_argument('--output', type=str, default='output/backtest',
                      help='Output directory for backtest results')
    parser.add_argument('--credentials', type=str, default='alpaca_credentials.json',
                      help='Path to Alpaca API credentials')
    parser.add_argument('--lookback', type=int, default=5,
                      help='Number of years to look back')
    parser.add_argument('--hold-days', type=int, default=20,
                      help='Number of days to hold each position')
    args = parser.parse_args()
    
    # Initialize backtester
    backtester = SeasonalityBacktester(args.credentials)
    
    # Load opportunities
    opportunities = backtester.load_opportunities(args.opportunities)
    
    if not opportunities:
        logging.error("No opportunities found. Exiting.")
        return
    
    # Run backtest
    results = backtester.backtest_opportunities(
        opportunities=opportunities,
        lookback_years=args.lookback,
        hold_days=args.hold_days
    )
    
    # Generate performance report
    backtester.generate_performance_report(args.output)
    
    # Save results to YAML
    results_file = os.path.join(args.output, 'seasonality_backtest_results.yaml')
    backtester.save_results_to_yaml(results_file)
    
    logging.info(f"Seasonality backtest completed successfully")

if __name__ == "__main__":
    main()
