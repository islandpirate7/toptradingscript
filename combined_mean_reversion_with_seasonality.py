#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Combined Mean Reversion with Seasonality Strategy
------------------------------------------------
This script implements a combined strategy that uses both mean reversion
and seasonality to select stocks from a universe of 500 stocks.
"""

import os
import yaml
import logging
import pandas as pd
import numpy as np
import datetime as dt
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from combined_strategy import CombinedStrategy
from mean_reversion_strategy_optimized import MeanReversionStrategyOptimized
from seasonality_enhanced import SeasonalityEnhanced
from alpaca_data_provider import AlpacaDataProvider
from backtest_combined_strategy_simplified import BacktestCombinedStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CombinedMeanReversionWithSeasonality:
    """Combined Mean Reversion with Seasonality Strategy"""
    
    def __init__(self, config_file: str = 'configuration_combined_strategy.yaml'):
        """
        Initialize the combined strategy.
        
        Args:
            config_file (str): Path to configuration file
        """
        self.config_file = config_file
        self.config = self._load_config(config_file)
        
        # Initialize the combined strategy
        self.strategy = CombinedStrategy(self.config)
        
        # Initialize the mean reversion strategy
        self.mean_reversion = MeanReversionStrategyOptimized(self.config)
        
        # Initialize the data provider
        self.data_provider = AlpacaDataProvider(use_paper=True)
        
        # Initialize the backtest engine
        self.backtest = BacktestCombinedStrategy(self.config)
        
        # Get universe of stocks
        self.universe = self.config.get('general', {}).get('symbols', [])
        logger.info(f"Initialized with universe of {len(self.universe)} stocks")
        
        # Check if seasonality is enabled
        self.use_seasonality = self.config.get('seasonality', {}).get('enabled', False)
        if self.use_seasonality:
            logger.info("Seasonality is enabled")
        else:
            logger.warning("Seasonality is disabled in configuration")
    
    def _load_config(self, config_file: str) -> Dict:
        """
        Load configuration from YAML file.
        
        Args:
            config_file (str): Path to configuration file
            
        Returns:
            dict: Configuration dictionary
        """
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def select_stocks_with_seasonality(self, date: dt.datetime, top_n: int = 20) -> List[str]:
        """
        Select top stocks based on seasonality for a given date.
        
        Args:
            date (datetime): Date to select stocks for
            top_n (int): Number of top stocks to select
            
        Returns:
            list: List of selected stock symbols
        """
        if not self.use_seasonality or not hasattr(self.strategy, 'seasonality_analyzer'):
            logger.warning("Seasonality is not enabled or analyzer not initialized")
            return self.universe[:top_n]  # Return first top_n stocks from universe
        
        # Get seasonality scores for all stocks in universe
        scores = []
        for symbol in self.universe:
            score, direction = self.strategy.get_seasonal_score(symbol, date)
            scores.append({
                'symbol': symbol,
                'score': score,
                'direction': direction
            })
        
        # Sort by score (descending) and filter for LONG direction
        long_scores = [s for s in scores if s['direction'] == 'LONG']
        long_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Get top N symbols
        selected_symbols = [s['symbol'] for s in long_scores[:top_n]]
        
        logger.info(f"Selected {len(selected_symbols)} stocks based on seasonality for {date.strftime('%Y-%m-%d')}")
        return selected_symbols
    
    def generate_mean_reversion_signals(self, symbols: List[str], date: dt.datetime) -> List[Dict]:
        """
        Generate mean reversion signals for a list of symbols.
        
        Args:
            symbols (list): List of stock symbols
            date (datetime): Date to generate signals for
            
        Returns:
            list: List of signal dictionaries
        """
        signals = []
        
        # Set end date to the given date and start date to 50 days before
        end_date = date
        start_date = end_date - dt.timedelta(days=50)
        
        for symbol in symbols:
            try:
                # Get historical data
                df = self.data_provider.get_historical_data(
                    symbol, 
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d'),
                    '1D'
                )
                
                if df is None or len(df) < 30:
                    logger.warning(f"Insufficient data for {symbol}, skipping")
                    continue
                
                # Add debug logging to check data
                logger.info(f"Data for {symbol}: {len(df)} rows, date range: {df.index[0]} to {df.index[-1]}")
                
                # Check if we have enough data to calculate indicators
                if len(df) < 30:
                    logger.warning(f"Not enough data for {symbol} to calculate indicators: {len(df)} rows")
                    continue
                
                # Calculate indicators for debugging
                df_with_indicators = self.mean_reversion.calculate_indicators(df.copy())
                
                # Log some sample indicator values for the last few days
                last_rows = min(5, len(df_with_indicators))
                for i in range(1, last_rows + 1):
                    idx = -i
                    row = df_with_indicators.iloc[idx]
                    bb_lower = row.get('bb_lower', 'N/A')
                    bb_upper = row.get('bb_upper', 'N/A')
                    rsi = row.get('rsi', 'N/A')
                    
                    bb_lower_str = f"{bb_lower:.2f}" if isinstance(bb_lower, float) else "N/A"
                    bb_upper_str = f"{bb_upper:.2f}" if isinstance(bb_upper, float) else "N/A"
                    rsi_str = f"{rsi:.2f}" if isinstance(rsi, float) else "N/A"
                    
                    logger.info(f"{symbol} on {row.name.strftime('%Y-%m-%d')}: "
                               f"Close: {row['close']:.2f}, "
                               f"BB Lower: {bb_lower_str}, "
                               f"BB Upper: {bb_upper_str}, "
                               f"RSI: {rsi_str}")
                
                # Generate signal
                signal = self.mean_reversion.generate_signal(df, symbol)
                
                if signal:
                    # Check if the signal date is close to our target date
                    # Convert both to naive datetime objects for comparison
                    signal_date = signal.get('date')
                    logger.info(f"Signal found for {symbol} with date {signal_date}")
                    
                    if signal_date:
                        # Convert to string format and back to datetime to remove timezone info
                        signal_date_str = signal_date.strftime('%Y-%m-%d')
                        signal_date_naive = dt.datetime.strptime(signal_date_str, '%Y-%m-%d')
                        date_naive = dt.datetime(date.year, date.month, date.day)
                        
                        # Check if dates are within 15 days of each other
                        date_diff = abs((date_naive - signal_date_naive).days)
                        logger.info(f"Date difference: {date_diff} days between {date_naive} and {signal_date_naive}")
                        
                        # Allow signals from the last 15 days
                        if date_diff <= 15:
                            signals.append(signal)
                            logger.info(f"Generated signal for {symbol}: {signal['direction']} at {signal['price']}")
                        else:
                            logger.info(f"Signal date {signal_date_naive} too far from target date {date_naive}, skipping")
                    else:
                        logger.warning(f"Signal for {symbol} has no date")
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        logger.info(f"Generated {len(signals)} mean reversion signals for {date.strftime('%Y-%m-%d')}")
        return signals
    
    def prioritize_signals(self, signals: List[Dict], date: dt.datetime) -> List[Dict]:
        """
        Prioritize signals based on seasonality scores.
        
        Args:
            signals (List[Dict]): List of signals
            date (dt.datetime): Date to prioritize signals for
            
        Returns:
            List[Dict]: Prioritized signals
        """
        if not signals:
            return []
        
        # Get month number (1-12)
        month = date.month
        
        # Add seasonality score to each signal
        for signal in signals:
            symbol = signal.get('symbol')
            
            # Get seasonality score for this symbol and month
            score, direction = self.strategy.get_seasonal_score(symbol, date)
            
            # Add seasonality information to signal
            signal['seasonality_score'] = score
            signal['seasonality_direction'] = direction
            
            # Calculate combined weight
            mean_reversion_weight = 0.5  # Default weight for mean reversion
            seasonality_weight = 0.5     # Default weight for seasonality
            
            # Get weights from config if available
            if hasattr(self, 'config') and self.config:
                mean_reversion_weight = self.config.get('strategy_weights', {}).get('mean_reversion', 0.5)
                seasonality_weight = self.config.get('strategy_weights', {}).get('seasonality', 0.5)
            
            # Adjust weight if signal direction matches seasonality direction
            direction_match = signal.get('direction') == direction
            direction_factor = 1.2 if direction_match else 0.8
            
            # Calculate final weight
            weight = (mean_reversion_weight * signal.get('strength_score', 0.5) + 
                     seasonality_weight * score) * direction_factor
            
            signal['weight'] = weight
            
            # Add entry_price field for backtest engine compatibility
            if 'price' in signal and 'entry_price' not in signal:
                signal['entry_price'] = float(signal['price'])
            
            # Ensure all numeric fields are Python floats, not numpy types
            for field in ['entry_price', 'price', 'stop_loss', 'take_profit', 'weight']:
                if field in signal and hasattr(signal[field], 'item'):
                    signal[field] = float(signal[field])
        
        # Sort by adjusted weight (descending)
        signals.sort(key=lambda x: x.get('weight', 0), reverse=True)
        
        return signals
    
    def run_backtest(self, start_date: str, end_date: str, top_n: int = 20, max_positions: int = 5):
        """
        Run a backtest of the combined strategy.
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            top_n (int): Number of top stocks to select based on seasonality
            max_positions (int): Maximum number of positions to hold
            
        Returns:
            BacktestResults: Results of the backtest
        """
        # Convert dates to datetime objects
        start_dt = dt.datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = dt.datetime.strptime(end_date, '%Y-%m-%d')
        
        # Generate a list of trading days
        all_days = pd.date_range(start=start_dt, end=end_dt, freq='B')
        
        # Initialize backtest parameters
        initial_capital = self.config.get('general', {}).get('initial_capital', 100000)
        self.backtest.set_initial_capital(initial_capital)
        self.backtest.set_max_positions(max_positions)
        
        # Run backtest day by day
        for day in tqdm(all_days, desc="Running backtest"):
            # Select top stocks based on seasonality
            selected_stocks = self.select_stocks_with_seasonality(day, top_n)
            
            # Generate mean reversion signals for selected stocks
            signals = self.generate_mean_reversion_signals(selected_stocks, day)
            
            # Prioritize signals based on seasonality
            prioritized_signals = self.prioritize_signals(signals, day)
            
            # Add debug logging
            if prioritized_signals:
                logger.info(f"Sending {len(prioritized_signals)} prioritized signals to backtest for {day}")
                for i, signal in enumerate(prioritized_signals[:3]):  # Log first 3 signals
                    logger.info(f"Signal {i+1}: {signal['symbol']} {signal['direction']} at {signal['entry_price']:.2f}, weight: {signal.get('weight', 0):.2f}")
            else:
                logger.info(f"No prioritized signals for {day}")
            
            # Process signals for this day
            self.backtest.process_signals_for_date(prioritized_signals, day)
        
        # Finalize backtest
        results = self.backtest.finalize()
        
        # Print summary
        logger.info(f"Backtest results from {start_date} to {end_date}:")
        logger.info(f"Final equity: ${results.final_equity:.2f}")
        logger.info(f"Total return: {results.total_return_pct:.2f}%")
        logger.info(f"Annualized return: {results.annualized_return_pct:.2f}%")
        logger.info(f"Sharpe ratio: {results.sharpe_ratio:.2f}")
        logger.info(f"Max drawdown: {results.max_drawdown_pct:.2f}%")
        logger.info(f"Win rate: {results.win_rate:.2f}%")
        logger.info(f"Profit factor: {results.profit_factor:.2f}")
        
        # Plot equity curve
        self._plot_equity_curve(results)
        
        return results
    
    def _plot_equity_curve(self, results):
        """
        Plot equity curve from backtest results.
        
        Args:
            results: Backtest results object
        """
        plt.figure(figsize=(12, 6))
        
        # Plot equity curve
        plt.plot(results.equity_curve.index, results.equity_curve.values, label='Equity Curve')
        
        # Add labels and title
        plt.title('Combined Mean Reversion with Seasonality Strategy Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity ($)')
        plt.grid(True)
        plt.legend()
        
        # Create output directory if it doesn't exist
        os.makedirs('output', exist_ok=True)
        
        # Save figure
        plt.savefig('output/combined_mean_reversion_seasonality_equity_curve.png')
        plt.close()
        
        # Plot drawdowns
        plt.figure(figsize=(12, 6))
        plt.plot(results.drawdowns.index, results.drawdowns.values * 100)
        plt.title('Drawdowns')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        
        # Save figure
        plt.savefig('output/combined_mean_reversion_seasonality_drawdowns.png')
        plt.close()

def main():
    """Main function to run the combined strategy"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run combined mean reversion with seasonality strategy')
    parser.add_argument('--config', type=str, default='configuration_combined_strategy.yaml',
                        help='Path to configuration file')
    parser.add_argument('--start_date', type=str, default='2023-01-01',
                        help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2023-12-31',
                        help='End date for backtest (YYYY-MM-DD)')
    parser.add_argument('--top_n', type=int, default=20,
                        help='Number of top stocks to select based on seasonality')
    parser.add_argument('--max_positions', type=int, default=5,
                        help='Maximum number of positions to hold')
    args = parser.parse_args()
    
    # Initialize strategy
    strategy = CombinedMeanReversionWithSeasonality(args.config)
    
    # Run backtest
    strategy.run_backtest(args.start_date, args.end_date, args.top_n, args.max_positions)

if __name__ == "__main__":
    main()
