"""
Seasonality Analyzer Module

This module analyzes historical price data to identify seasonal patterns and correlations
similar to the Quantum Screener approach. It helps identify high-probability trading
opportunities based on historical seasonal patterns.
"""

import numpy as np
import pandas as pd
import logging
import yaml
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import requests
import time
from enum import Enum
import os
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class TradeDirection(Enum):
    """Enum for trade direction"""
    LONG = "LONG"
    SHORT = "SHORT"

class SeasonType(Enum):
    """Enum for different types of seasonality analysis"""
    DAY_OF_WEEK = "day_of_week"
    DAY_OF_MONTH = "day_of_month"
    MONTH_OF_YEAR = "month_of_year"
    QUARTER = "quarter"
    HALF_YEAR = "half_year"

class SeasonalityAnalyzer:
    """Class to analyze seasonality patterns in stock data"""
    
    def __init__(self, api_credentials_path: str = 'alpaca_credentials.json'):
        """Initialize SeasonalityAnalyzer
        
        Args:
            api_credentials_path (str): Path to Alpaca API credentials JSON file
        """
        self.api_credentials_path = api_credentials_path
        self._initialize_client()
        self.universe = []  # Stock universe to analyze
        self.historical_data = {}  # Cache for historical data
        self.seasonal_patterns = {}  # Detected seasonal patterns
        self.seasonal_correlations = {}  # Correlations between current price movements and seasonal patterns
        
    def _initialize_client(self) -> None:
        """Initialize Alpaca API client
        
        Returns:
            None
        """
        # Load API credentials
        try:
            with open(self.api_credentials_path, 'r') as f:
                credentials = json.load(f)
                
            # Use paper trading credentials by default
            self.api_key = credentials['paper']['api_key']
            self.api_secret = credentials['paper']['api_secret']
            
            logging.info("Successfully initialized Alpaca client")
            
        except Exception as e:
            logging.error(f"Error initializing Alpaca client: {e}")
            raise
    
    def set_stock_universe(self, symbols: List[str]) -> None:
        """Set the universe of stocks to analyze
        
        Args:
            symbols (List[str]): List of stock symbols
        """
        self.universe = symbols
        logging.info(f"Set stock universe with {len(symbols)} symbols")
    
    def load_stock_universe_from_file(self, file_path: str) -> None:
        """Load stock universe from a file
        
        Args:
            file_path (str): Path to file containing stock symbols (one per line)
        """
        try:
            with open(file_path, 'r') as f:
                symbols = [line.strip() for line in f if line.strip()]
            
            self.set_stock_universe(symbols)
            
        except Exception as e:
            logging.error(f"Error loading stock universe from {file_path}: {e}")
            raise
    
    def fetch_historical_data(self, 
                             start_date: str,
                             end_date: str = None,
                             timeframe: str = "1Day") -> Dict[str, pd.DataFrame]:
        """Fetch historical data for all stocks in the universe using REST API
        
        Args:
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str, optional): End date in format 'YYYY-MM-DD'. Defaults to today.
            timeframe (str, optional): Data timeframe. Defaults to "1Day".
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping symbols to their historical data
        """
        if not self.universe:
            logging.warning("Stock universe is empty. Set it before fetching data.")
            return {}
            
        # Set end date to today if not provided
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        logging.info(f"Fetching historical data from {start_date} to {end_date} for {len(self.universe)} symbols")
        
        # Base URL for the Alpaca Data API
        base_url = "https://data.alpaca.markets/v2"
        
        # Headers for authentication
        headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret
        }
        
        # Format dates for API
        start = pd.Timestamp(start_date, tz='America/New_York').isoformat()
        end = pd.Timestamp(end_date, tz='America/New_York').isoformat()
        
        # Process symbols one by one
        for symbol in self.universe:
            try:
                logging.info(f"Fetching data for {symbol}...")
                
                # Construct URL for bars endpoint
                url = f"{base_url}/stocks/{symbol}/bars"
                
                # Parameters for the request
                params = {
                    "start": start,
                    "end": end,
                    "timeframe": timeframe,
                    "adjustment": "all",
                    "limit": 10000  # Maximum limit
                }
                
                # Make the request
                response = requests.get(url, headers=headers, params=params)
                
                # Check if request was successful
                if response.status_code == 200:
                    data = response.json()
                    bars = data.get('bars', [])
                    
                    if bars:
                        # Convert to DataFrame
                        df = pd.DataFrame(bars)
                        
                        # Convert timestamp to datetime and set as index
                        df['t'] = pd.to_datetime(df['t'])
                        df = df.rename(columns={
                            't': 'timestamp',
                            'o': 'open',
                            'h': 'high',
                            'l': 'low',
                            'c': 'close',
                            'v': 'volume',
                            'n': 'trade_count',
                            'vw': 'vwap'
                        })
                        df.set_index('timestamp', inplace=True)
                        
                        # Store in cache
                        self.historical_data[symbol] = df
                        logging.info(f"Fetched {len(df)} bars for {symbol}")
                        
                        # Check if there's a next page token and fetch more data if needed
                        next_page_token = data.get('next_page_token')
                        while next_page_token:
                            logging.info(f"Fetching more data for {symbol} (next_page_token: {next_page_token})")
                            
                            # Update params with next_page_token
                            params['page_token'] = next_page_token
                            
                            # Make the request
                            response = requests.get(url, headers=headers, params=params)
                            
                            if response.status_code == 200:
                                data = response.json()
                                more_bars = data.get('bars', [])
                                
                                if more_bars:
                                    # Convert to DataFrame
                                    more_df = pd.DataFrame(more_bars)
                                    
                                    # Convert timestamp to datetime and set as index
                                    more_df['t'] = pd.to_datetime(more_df['t'])
                                    more_df = more_df.rename(columns={
                                        't': 'timestamp',
                                        'o': 'open',
                                        'h': 'high',
                                        'l': 'low',
                                        'c': 'close',
                                        'v': 'volume',
                                        'n': 'trade_count',
                                        'vw': 'vwap'
                                    })
                                    more_df.set_index('timestamp', inplace=True)
                                    
                                    # Append to existing DataFrame
                                    self.historical_data[symbol] = pd.concat([self.historical_data[symbol], more_df])
                                    logging.info(f"Fetched additional {len(more_df)} bars for {symbol}")
                                
                                # Update next_page_token
                                next_page_token = data.get('next_page_token')
                            else:
                                logging.error(f"Error fetching more data for {symbol}: {response.status_code} - {response.text}")
                                break
                            
                            # Add a small delay to avoid rate limiting
                            time.sleep(0.5)
                    else:
                        logging.warning(f"No bars returned for {symbol}")
                else:
                    logging.error(f"Error fetching data for {symbol}: {response.status_code} - {response.text}")
            
            except Exception as e:
                logging.error(f"Exception fetching data for {symbol}: {e}")
            
            # Add a small delay to avoid rate limiting
            time.sleep(1)
        
        # Check if we got any data
        if not self.historical_data:
            logging.error("Failed to fetch data for any symbols. Check API credentials and connectivity.")
        
        return self.historical_data
    
    def calculate_seasonal_correlation(self, 
                                      symbol: str, 
                                      lookback_years: List[int] = [1, 3, 5, 10, 15, 25],
                                      window_days: int = 90) -> Dict[int, float]:
        """Calculate correlation between current price movement and historical seasonal patterns
        
        Args:
            symbol (str): Stock symbol
            lookback_years (List[int], optional): Years to look back for seasonal patterns. 
                                                 Defaults to [1, 3, 5, 10, 15, 25].
            window_days (int, optional): Window size in days for correlation. Defaults to 90.
            
        Returns:
            Dict[int, float]: Dictionary mapping lookback years to correlation values
        """
        if symbol not in self.historical_data:
            logging.warning(f"No historical data for {symbol}. Fetch data first.")
            return {}
            
        df = self.historical_data[symbol].copy()
        
        # Get current window
        current_window = df['close'].iloc[-window_days:]
        
        # Normalize current window (percentage change from start)
        current_norm = (current_window / current_window.iloc[0]) - 1
        
        correlations = {}
        
        for years in lookback_years:
            try:
                # Calculate start index for the seasonal lookback
                days_to_lookback = 365 * years + window_days
                
                if len(df) <= days_to_lookback:
                    logging.warning(f"Not enough data for {symbol} to look back {years} years")
                    continue
                
                # Get historical window
                historical_start_idx = len(df) - days_to_lookback
                historical_window = df['close'].iloc[historical_start_idx:historical_start_idx + window_days]
                
                # Normalize historical window
                historical_norm = (historical_window / historical_window.iloc[0]) - 1
                
                # Calculate correlation
                if len(current_norm) == len(historical_norm):
                    correlation = current_norm.corr(historical_norm)
                    correlations[years] = correlation
                else:
                    logging.warning(f"Window size mismatch for {symbol} with {years} years lookback")
                
            except Exception as e:
                logging.error(f"Error calculating correlation for {symbol} with {years} years lookback: {e}")
        
        return correlations
    
    def calculate_win_rate(self, 
                          symbol: str, 
                          lookback_years: int,
                          forward_days: int = 90,
                          direction: TradeDirection = None) -> Tuple[float, float, TradeDirection]:
        """Calculate win rate and average return for a given seasonal pattern
        
        Args:
            symbol (str): Stock symbol
            lookback_years (int): Years to look back for seasonal pattern
            forward_days (int, optional): Days to look forward for return calculation. Defaults to 90.
            direction (TradeDirection, optional): Trade direction. If None, it will be determined.
            
        Returns:
            Tuple[float, float, TradeDirection]: Win rate, average return, and optimal direction
        """
        if symbol not in self.historical_data:
            logging.warning(f"No historical data for {symbol}. Fetch data first.")
            return 0.0, 0.0, TradeDirection.LONG
            
        df = self.historical_data[symbol].copy()
        
        # Calculate how many samples we can get
        days_per_year = 252  # Trading days
        total_samples = (len(df) - forward_days) // days_per_year
        
        if total_samples < 1:
            logging.warning(f"Not enough data for {symbol} to calculate win rate")
            return 0.0, 0.0, TradeDirection.LONG
            
        # Limit samples to lookback years
        samples = min(total_samples, lookback_years)
        
        # Calculate returns for each sample
        returns = []
        
        for i in range(samples):
            start_idx = len(df) - (i + 1) * days_per_year
            end_idx = start_idx + forward_days
            
            if start_idx < 0 or end_idx >= len(df):
                continue
                
            start_price = df['close'].iloc[start_idx]
            end_price = df['close'].iloc[end_idx]
            
            # Calculate return
            ret = (end_price - start_price) / start_price
            returns.append(ret)
            
        if not returns:
            logging.warning(f"No valid returns calculated for {symbol}")
            return 0.0, 0.0, TradeDirection.LONG
            
        # Determine optimal direction
        avg_return = np.mean(returns)
        if direction is None:
            direction = TradeDirection.LONG if avg_return > 0 else TradeDirection.SHORT
            
        # Calculate win rate based on direction
        if direction == TradeDirection.LONG:
            wins = sum(1 for ret in returns if ret > 0)
            win_rate = wins / len(returns)
            avg_return = abs(avg_return)  # Make positive for reporting
        else:  # SHORT
            wins = sum(1 for ret in returns if ret < 0)
            win_rate = wins / len(returns)
            avg_return = abs(avg_return)  # Make positive for reporting
            
        return win_rate, avg_return, direction
    
    def analyze_seasonality(self, 
                           forward_days: int = 90,
                           min_correlation: float = 0.7,
                           min_win_rate: float = 0.6) -> List[Dict]:
        """Analyze seasonality for all stocks in the universe
        
        Args:
            forward_days (int, optional): Days to look forward for return calculation. Defaults to 90.
            min_correlation (float, optional): Minimum correlation to consider. Defaults to 0.7.
            min_win_rate (float, optional): Minimum win rate to consider. Defaults to 0.6.
            
        Returns:
            List[Dict]: List of dictionaries with seasonality analysis results
        """
        results = []
        
        for symbol in self.universe:
            if symbol not in self.historical_data:
                continue
                
            # Calculate correlations for different lookback periods
            correlations = self.calculate_seasonal_correlation(symbol, window_days=forward_days)
            
            if not correlations:
                continue
                
            # Find best correlation
            best_year = max(correlations.items(), key=lambda x: x[1])
            best_lookback_years, best_correlation = best_year
            
            if best_correlation < min_correlation:
                continue
                
            # Calculate win rate and average return for best correlation
            win_rate, avg_return, direction = self.calculate_win_rate(
                symbol, best_lookback_years, forward_days
            )
            
            if win_rate < min_win_rate:
                continue
                
            # Calculate trade dates
            today = datetime.now()
            open_date = today.strftime('%m/%d/%Y')
            close_date = (today + timedelta(days=forward_days)).strftime('%m/%d/%Y')
            
            # Store result
            result = {
                'symbol': symbol,
                'best_correlation': best_correlation * 100,  # Convert to percentage
                'correlation_years': best_lookback_years,
                'direction': direction.value,
                'win_rate': win_rate * 100,  # Convert to percentage
                'avg_return': avg_return * 100,  # Convert to percentage
                'open_date': open_date,
                'close_date': close_date,
                'forward_days': forward_days
            }
            
            results.append(result)
            
        # Sort by combined score (correlation * win_rate * avg_return)
        for result in results:
            result['score'] = (result['best_correlation'] * result['win_rate'] * result['avg_return']) / 10000
            
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results
    
    def get_top_opportunities(self, 
                             top_n: int = 10, 
                             forward_days: int = 90,
                             min_correlation: float = 0.7,
                             min_win_rate: float = 0.6) -> pd.DataFrame:
        """Get top seasonal trading opportunities
        
        Args:
            top_n (int, optional): Number of top opportunities to return. Defaults to 10.
            forward_days (int, optional): Days to look forward for return calculation. Defaults to 90.
            min_correlation (float, optional): Minimum correlation to consider. Defaults to 0.7.
            min_win_rate (float, optional): Minimum win rate to consider. Defaults to 0.6.
            
        Returns:
            pd.DataFrame: DataFrame with top opportunities
        """
        results = self.analyze_seasonality(forward_days, min_correlation, min_win_rate)
        
        if not results:
            logging.warning("No opportunities found meeting the criteria")
            return pd.DataFrame()
            
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Select columns and rename
        columns = [
            'symbol', 'best_correlation', 'correlation_years', 'direction', 
            'win_rate', 'avg_return', 'open_date', 'close_date', 'score'
        ]
        
        df = df[columns].head(top_n)
        
        # Rename columns for better readability
        df.columns = [
            'Symbol', 'Best Correlation %', 'Lookback [years]', 'Direction', 
            'Win Rate %', 'Avg Return %', 'Open Date', 'Close Date', 'Score'
        ]
        
        return df
    
    def plot_seasonal_pattern(self, 
                             symbol: str, 
                             lookback_years: int,
                             window_days: int = 90) -> None:
        """Plot current price movement vs historical seasonal pattern
        
        Args:
            symbol (str): Stock symbol
            lookback_years (int): Years to look back for seasonal pattern
            window_days (int, optional): Window size in days. Defaults to 90.
        """
        if symbol not in self.historical_data:
            logging.warning(f"No historical data for {symbol}. Fetch data first.")
            return
            
        df = self.historical_data[symbol].copy()
        
        # Get current window
        current_window = df['close'].iloc[-window_days:]
        
        # Normalize current window (percentage change from start)
        current_norm = (current_window / current_window.iloc[0]) - 1
        
        # Calculate start index for the seasonal lookback
        days_to_lookback = 365 * lookback_years + window_days
        
        if len(df) <= days_to_lookback:
            logging.warning(f"Not enough data for {symbol} to look back {lookback_years} years")
            return
        
        # Get historical window
        historical_start_idx = len(df) - days_to_lookback
        historical_window = df['close'].iloc[historical_start_idx:historical_start_idx + window_days]
        
        # Normalize historical window
        historical_norm = (historical_window / historical_window.iloc[0]) - 1
        
        # Calculate correlation
        correlation = current_norm.corr(historical_norm)
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(current_norm.values, label='Current')
        plt.plot(historical_norm.values, label=f'{lookback_years} Year Ago')
        plt.title(f'{symbol} Seasonal Pattern - Correlation: {correlation:.2f}')
        plt.xlabel('Days')
        plt.ylabel('Normalized Return')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plt.savefig(f'{symbol}_seasonal_pattern.png')
        plt.close()
        
    def save_opportunities_to_file(self, 
                                  file_path: str, 
                                  top_n: int = 10,
                                  forward_days: int = 90) -> None:
        """Save top opportunities to a file
        
        Args:
            file_path (str): Path to save the opportunities
            top_n (int, optional): Number of top opportunities to save. Defaults to 10.
            forward_days (int, optional): Days to look forward for return calculation. Defaults to 90.
        """
        df = self.get_top_opportunities(top_n, forward_days)
        
        if df.empty:
            logging.warning("No opportunities to save")
            return
            
        # Save to CSV
        df.to_csv(file_path, index=False)
        logging.info(f"Saved {len(df)} opportunities to {file_path}")
        
    def generate_trading_universe(self, 
                                 output_file: str,
                                 top_n: int = 10,
                                 forward_days: int = 90,
                                 min_correlation: float = 0.7,
                                 min_win_rate: float = 0.6) -> List[str]:
        """Generate a trading universe based on seasonal opportunities
        
        Args:
            output_file (str): Path to save the trading universe
            top_n (int, optional): Number of top opportunities to include. Defaults to 10.
            forward_days (int, optional): Days to look forward for return calculation. Defaults to 90.
            min_correlation (float, optional): Minimum correlation to consider. Defaults to 0.7.
            min_win_rate (float, optional): Minimum win rate to consider. Defaults to 0.6.
            
        Returns:
            List[str]: List of symbols in the trading universe
        """
        results = self.analyze_seasonality(forward_days, min_correlation, min_win_rate)
        
        if not results:
            logging.warning("No opportunities found meeting the criteria")
            return []
            
        # Get top N symbols
        top_symbols = [result['symbol'] for result in results[:top_n]]
        
        # Save to file
        with open(output_file, 'w') as f:
            for symbol in top_symbols:
                f.write(f"{symbol}\n")
                
        logging.info(f"Generated trading universe with {len(top_symbols)} symbols and saved to {output_file}")
        
        return top_symbols
    
    def generate_configuration(self, 
                              output_file: str,
                              top_n: int = 10,
                              forward_days: int = 90) -> None:
        """Generate a configuration file for the trading strategy based on seasonal opportunities
        
        Args:
            output_file (str): Path to save the configuration
            top_n (int, optional): Number of top opportunities to include. Defaults to 10.
            forward_days (int, optional): Days to look forward for return calculation. Defaults to 90.
        """
        # Get top opportunities
        df = self.get_top_opportunities(top_n, forward_days)
        
        if df.empty:
            logging.warning("No opportunities to generate configuration")
            return
            
        # Create configuration dictionary
        config = {
            'general': {
                'log_level': 'INFO',
                'initial_capital': 100000,
                'max_positions': min(15, len(df)),
                'max_portfolio_risk_pct': 0.02,
                'min_capital_per_trade': 1000,
                'symbols': df['Symbol'].tolist(),
                'timeframe': '1D',
                'position_size_pct': 0.05,
                'backtest_start_date': datetime.now().strftime('%Y-%m-%d'),
                'backtest_end_date': (datetime.now() + timedelta(days=forward_days)).strftime('%Y-%m-%d'),
                'min_signal_score': 0.55
            },
            'strategy_configs': {
                'MeanReversion': {
                    'weight': 0.45,
                    'bb_period': 20,
                    'bb_std': 1.8,
                    'rsi_period': 14,
                    'rsi_lower': 38,
                    'rsi_upper': 62,
                    'require_reversal': True,
                    'stop_loss_atr_multiplier': 1.6,
                    'take_profit_atr_multiplier': 2.8,
                    'atr_period': 14,
                    'volume_filter': True,
                    'volume_threshold': 1.3,
                    'symbol_weights': {}
                },
                'TrendFollowing': {
                    'weight': 0.55,
                    'fast_ma_period': 9,
                    'slow_ma_period': 21,
                    'signal_ma_period': 9,
                    'adx_period': 14,
                    'adx_threshold': 25,
                    'macd_fast_period': 12,
                    'macd_slow_period': 26,
                    'macd_signal_period': 9,
                    'rsi_period': 14,
                    'rsi_lower': 40,
                    'rsi_upper': 60,
                    'atr_period': 14,
                    'stop_loss_atr_multiplier': 1.5,
                    'take_profit_atr_multiplier': 3.0,
                    'volume_filter': True,
                    'volume_threshold': 1.2,
                    'symbol_weights': {}
                },
                'Combined': {
                    'mean_reversion_weight': 0.45,
                    'trend_following_weight': 0.55,
                    'trending_regime_weights': {
                        'mean_reversion': 0.15,
                        'trend_following': 0.85
                    },
                    'range_bound_regime_weights': {
                        'mean_reversion': 0.75,
                        'trend_following': 0.25
                    },
                    'mixed_regime_weights': {
                        'mean_reversion': 0.55,
                        'trend_following': 0.45
                    },
                    'regime_lookback': 20,
                    'volatility_period': 10,
                    'adx_threshold': 25,
                    'min_signal_score': 0.55,
                    'max_signals_per_day': 4,
                    'position_size_pct': 0.05
                }
            },
            'risk_management': {
                'max_portfolio_risk_pct': 0.02,
                'max_position_risk_pct': 0.005,
                'max_sector_exposure_pct': 0.20,
                'max_drawdown_exit_pct': 0.15,
                'trailing_stop_activation_pct': 0.05,
                'trailing_stop_distance_pct': 0.03,
                'profit_taking_levels': [
                    {'level': 0.05, 'size_reduction_pct': 0.25},
                    {'level': 0.10, 'size_reduction_pct': 0.50}
                ],
                'volatility_adjustment': True,
                'volatility_lookback': 20,
                'volatility_scaling_factor': 0.8
            }
        }
        
        # Set symbol-specific weights based on direction and win rate
        for _, row in df.iterrows():
            symbol = row['Symbol']
            direction = row['Direction']
            win_rate = row['Win Rate %'] / 100
            
            # Adjust weights based on direction
            if direction == 'LONG':
                # For long signals, favor trend following for high win rate
                mr_weight = 0.3 + (1 - win_rate) * 0.4  # 0.3 to 0.7 based on win rate
                tf_weight = 0.7 + (win_rate - 0.6) * 0.3  # 0.7 to 1.0 based on win rate
            else:  # SHORT
                # For short signals, favor mean reversion for high win rate
                mr_weight = 0.7 + (win_rate - 0.6) * 0.3  # 0.7 to 1.0 based on win rate
                tf_weight = 0.3 + (1 - win_rate) * 0.4  # 0.3 to 0.7 based on win rate
                
            # Normalize weights
            total = mr_weight + tf_weight
            mr_weight = mr_weight / total
            tf_weight = tf_weight / total
            
            # Set weights
            config['strategy_configs']['MeanReversion']['symbol_weights'][symbol] = mr_weight
            config['strategy_configs']['TrendFollowing']['symbol_weights'][symbol] = tf_weight
            
        # Save configuration to YAML file
        with open(output_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
        logging.info(f"Generated configuration with {len(df)} symbols and saved to {output_file}")

    def set_universe(self, universe: List[str]) -> None:
        """Set the stock universe to analyze
        
        Args:
            universe (List[str]): List of stock symbols
        """
        self.universe = universe
        logging.info(f"Set universe to {len(universe)} symbols")

    def calculate_seasonal_patterns(self, season_type: SeasonType) -> Dict[str, Dict]:
        """Calculate seasonal patterns for all stocks in the universe
        
        Args:
            season_type (SeasonType): Type of seasonality to analyze
            
        Returns:
            Dict[str, Dict]: Dictionary mapping symbols to their seasonal patterns
        """
        if not self.historical_data:
            logging.warning("No historical data available. Fetch data first.")
            return {}
            
        logging.info(f"Calculating {season_type.value} seasonal patterns for {len(self.historical_data)} symbols")
        
        self.seasonal_patterns = {}
        
        for symbol, df in self.historical_data.items():
            try:
                if df.empty:
                    continue
                    
                # Add necessary columns for analysis
                df = df.copy()
                df['returns'] = df['close'].pct_change()
                df['positive_return'] = df['returns'] > 0
                
                # Calculate seasonal patterns based on type
                if season_type == SeasonType.DAY_OF_WEEK:
                    df['season'] = df.index.dayofweek
                    season_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday'}
                elif season_type == SeasonType.DAY_OF_MONTH:
                    df['season'] = df.index.day
                    season_names = {i: str(i) for i in range(1, 32)}
                elif season_type == SeasonType.MONTH_OF_YEAR:
                    df['season'] = df.index.month
                    season_names = {
                        1: 'January', 2: 'February', 3: 'March', 4: 'April',
                        5: 'May', 6: 'June', 7: 'July', 8: 'August',
                        9: 'September', 10: 'October', 11: 'November', 12: 'December'
                    }
                elif season_type == SeasonType.QUARTER:
                    df['season'] = df.index.quarter
                    season_names = {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'}
                elif season_type == SeasonType.HALF_YEAR:
                    df['season'] = (df.index.month - 1) // 6 + 1
                    season_names = {1: 'H1', 2: 'H2'}
                else:
                    logging.error(f"Unsupported season type: {season_type}")
                    continue
                
                # Group by season and calculate statistics
                season_stats = {}
                for season, group in df.groupby('season'):
                    if len(group) < 5:  # Require at least 5 data points
                        continue
                        
                    win_rate = group['positive_return'].mean()
                    avg_return = group['returns'].mean() * 100  # Convert to percentage
                    
                    season_stats[season] = {
                        'win_rate': win_rate,
                        'avg_return': avg_return,
                        'trade_count': len(group),
                        'season_name': season_names.get(season, str(season))
                    }
                
                self.seasonal_patterns[symbol] = {
                    'stats': season_stats,
                    'season_type': season_type.value
                }
                
                logging.info(f"Calculated seasonal patterns for {symbol}")
                
            except Exception as e:
                logging.error(f"Error calculating seasonal patterns for {symbol}: {e}")
        
        logging.info(f"Calculated seasonal patterns for {len(self.seasonal_patterns)} symbols")
        return self.seasonal_patterns
    
    def calculate_seasonal_correlation(self, 
                                      current_period_days: int = 30,
                                      correlation_threshold: float = 0.7) -> Dict[str, Dict]:
        """Calculate correlation between current price movements and historical seasonal patterns
        
        Args:
            current_period_days (int, optional): Number of days to look back for current pattern. Defaults to 30.
            correlation_threshold (float, optional): Minimum correlation to consider. Defaults to 0.7.
            
        Returns:
            Dict[str, Dict]: Dictionary mapping symbols to their correlation data
        """
        if not self.seasonal_patterns:
            logging.warning("No seasonal patterns available. Calculate patterns first.")
            return {}
            
        logging.info(f"Calculating seasonal correlations for {len(self.seasonal_patterns)} symbols")
        
        self.seasonal_correlations = {}
        
        for symbol, pattern_data in self.seasonal_patterns.items():
            try:
                df = self.historical_data.get(symbol)
                if df is None or df.empty:
                    continue
                    
                # Get recent price data
                recent_data = df.tail(current_period_days).copy()
                if len(recent_data) < 10:  # Need at least 10 data points
                    continue
                    
                # Calculate returns
                recent_data['returns'] = recent_data['close'].pct_change()
                recent_data = recent_data.dropna()
                
                # Get current season
                season_type = pattern_data['season_type']
                if season_type == 'day_of_week':
                    current_season = recent_data.index[-1].dayofweek
                elif season_type == 'day_of_month':
                    current_season = recent_data.index[-1].day
                elif season_type == 'month_of_year':
                    current_season = recent_data.index[-1].month
                elif season_type == 'quarter':
                    current_season = recent_data.index[-1].quarter
                elif season_type == 'half_year':
                    current_season = (recent_data.index[-1].month - 1) // 6 + 1
                else:
                    logging.error(f"Unsupported season type: {season_type}")
                    continue
                
                # Get historical pattern for current season
                season_stats = pattern_data['stats'].get(current_season)
                if not season_stats:
                    continue
                    
                # Calculate correlation
                recent_returns = recent_data['returns'].values
                recent_win_rate = (recent_returns > 0).mean()
                recent_avg_return = recent_returns.mean() * 100
                
                # Simple correlation metric (can be improved with actual correlation calculation)
                win_rate_diff = abs(recent_win_rate - season_stats['win_rate'])
                return_diff = abs(recent_avg_return - season_stats['avg_return'])
                
                # Normalize differences
                win_rate_corr = 1 - win_rate_diff
                return_corr = 1 - min(return_diff / 5, 1)  # Cap at 5% difference
                
                # Combined correlation
                correlation = (win_rate_corr + return_corr) / 2
                
                if correlation >= correlation_threshold:
                    season_name = season_stats['season_name']
                    self.seasonal_correlations[symbol] = {
                        'season': current_season,
                        'season_name': season_name,
                        'correlation': correlation,
                        'win_rate': season_stats['win_rate'],
                        'avg_return': season_stats['avg_return'],
                        'trade_count': season_stats['trade_count'],
                        'current_price': df['close'].iloc[-1],
                        'expected_return': df['close'].iloc[-1] * (1 + season_stats['avg_return'] / 100)
                    }
                    
                    logging.info(f"Found correlation of {correlation:.2f} for {symbol} in {season_name}")
                
            except Exception as e:
                logging.error(f"Error calculating seasonal correlation for {symbol}: {e}")
        
        logging.info(f"Found {len(self.seasonal_correlations)} symbols with correlation >= {correlation_threshold}")
        return self.seasonal_correlations
    
    def generate_trading_opportunities(self,
                                      min_win_rate: float = 0.6,
                                      min_trades: int = 5,
                                      min_avg_return: float = 0.5) -> List[Dict]:
        """Generate trading opportunities based on seasonal patterns and correlations
        
        Args:
            min_win_rate (float, optional): Minimum win rate to consider. Defaults to 0.6.
            min_trades (int, optional): Minimum number of trades to consider. Defaults to 5.
            min_avg_return (float, optional): Minimum average return to consider. Defaults to 0.5.
            
        Returns:
            List[Dict]: List of trading opportunities
        """
        if not self.seasonal_correlations:
            logging.warning("No seasonal correlations available. Calculate correlations first.")
            return []
            
        logging.info(f"Generating trading opportunities from {len(self.seasonal_correlations)} correlations")
        
        opportunities = []
        
        for symbol, corr_data in self.seasonal_correlations.items():
            try:
                win_rate = corr_data['win_rate']
                avg_return = corr_data['avg_return']
                trade_count = corr_data['trade_count']
                
                if win_rate >= min_win_rate and trade_count >= min_trades and avg_return >= min_avg_return:
                    opportunities.append({
                        'symbol': symbol,
                        'season': corr_data['season_name'],
                        'win_rate': win_rate,
                        'avg_return': avg_return,
                        'trade_count': trade_count,
                        'correlation': corr_data['correlation'],
                        'current_price': corr_data['current_price'],
                        'expected_return': corr_data['expected_return'],
                        'direction': 'LONG' if avg_return > 0 else 'SHORT'
                    })
                    
                    logging.info(f"Added opportunity for {symbol} with win rate {win_rate:.2f} and avg return {avg_return:.2f}%")
                
            except Exception as e:
                logging.error(f"Error generating opportunity for {symbol}: {e}")
        
        # Sort opportunities by win rate * avg_return (expected value)
        opportunities.sort(key=lambda x: x['win_rate'] * x['avg_return'], reverse=True)
        
        logging.info(f"Generated {len(opportunities)} trading opportunities")
        return opportunities
    
    def generate_seasonal_plots(self, output_dir: str) -> None:
        """Generate plots of seasonal patterns for each symbol
        
        Args:
            output_dir (str): Directory to save plots
        """
        if not self.seasonal_patterns:
            logging.warning("No seasonal patterns available. Calculate patterns first.")
            return
            
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        logging.info(f"Generating seasonal plots for {len(self.seasonal_patterns)} symbols")
        
        for symbol, pattern_data in self.seasonal_patterns.items():
            try:
                stats = pattern_data['stats']
                season_type = pattern_data['season_type']
                
                if not stats:
                    continue
                    
                # Prepare data for plotting
                seasons = []
                win_rates = []
                avg_returns = []
                
                for season, data in sorted(stats.items()):
                    seasons.append(data['season_name'])
                    win_rates.append(data['win_rate'] * 100)  # Convert to percentage
                    avg_returns.append(data['avg_return'])
                
                if not seasons:
                    continue
                    
                # Create plot
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                
                # Win rate plot
                ax1.bar(seasons, win_rates, color='blue', alpha=0.7)
                ax1.axhline(y=50, color='r', linestyle='--')
                ax1.set_ylabel('Win Rate (%)')
                ax1.set_title(f'{symbol} Seasonal Win Rate ({season_type})')
                ax1.grid(True, alpha=0.3)
                
                # Average return plot
                ax2.bar(seasons, avg_returns, color='green', alpha=0.7)
                ax2.axhline(y=0, color='r', linestyle='--')
                ax2.set_ylabel('Average Return (%)')
                ax2.set_title(f'{symbol} Seasonal Average Return ({season_type})')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Save plot
                plot_path = os.path.join(output_dir, f'{symbol}_{season_type}_seasonal.png')
                plt.savefig(plot_path)
                plt.close()
                
                logging.info(f"Generated plot for {symbol} at {plot_path}")
                
            except Exception as e:
                logging.error(f"Error generating plot for {symbol}: {e}")
        
        logging.info(f"Completed generating seasonal plots")

def main():
    """Main function to demonstrate the SeasonalityAnalyzer"""
    # Initialize analyzer
    analyzer = SeasonalityAnalyzer()
    
    # Set stock universe (example: S&P 500 stocks)
    sp500_symbols = [
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V',
        'PG', 'UNH', 'HD', 'BAC', 'MA', 'XOM', 'DIS', 'CSCO', 'VZ', 'ADBE',
        'CRM', 'NFLX', 'CMCSA', 'PFE', 'INTC', 'ABT', 'KO', 'PEP', 'T', 'MRK',
        'WMT', 'CVX', 'TMO', 'ACN', 'COST', 'ABBV', 'AVGO', 'MCD', 'DHR', 'TXN',
        'NEE', 'LLY', 'PM', 'LIN', 'QCOM', 'MDT', 'BMY', 'UNP', 'HON', 'ORCL'
    ]
    analyzer.set_stock_universe(sp500_symbols)
    
    # Fetch historical data (5 years)
    start_date = (datetime.now() - timedelta(days=365 * 5)).strftime('%Y-%m-%d')
    analyzer.fetch_historical_data(start_date)
    
    # Get top opportunities for next 90 days
    top_opportunities = analyzer.get_top_opportunities(top_n=10, forward_days=90)
    
    # Print results
    if not top_opportunities.empty:
        print("\n===== TOP SEASONAL TRADING OPPORTUNITIES =====")
        print(top_opportunities.to_string(index=False))
        
        # Generate configuration file
        analyzer.generate_configuration('configuration_seasonal_strategy.yaml', top_n=10, forward_days=90)
        
        # Plot seasonal patterns for top 3 opportunities
        for symbol in top_opportunities['Symbol'].head(3):
            lookback_years = top_opportunities.loc[top_opportunities['Symbol'] == symbol, 'Lookback [years]'].values[0]
            analyzer.plot_seasonal_pattern(symbol, int(lookback_years), window_days=90)
    else:
        print("No seasonal opportunities found meeting the criteria")

if __name__ == "__main__":
    main()
