#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced seasonality analysis for stock selection and trading.
This module provides advanced seasonality-based methods to improve stock selection.
"""

import pandas as pd
import numpy as np
import logging
import yaml
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sector mappings
SECTOR_MAPPING = {
    # Technology
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 
    'NVDA': 'Technology', 'AMD': 'Technology',
    
    # Consumer Discretionary
    'AMZN': 'Consumer Discretionary', 'TSLA': 'Consumer Discretionary',
    
    # Financial
    'JPM': 'Financial', 'BAC': 'Financial', 'GS': 'Financial',
    
    # Healthcare
    'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare',
    
    # Energy
    'XOM': 'Energy', 'CVX': 'Energy',
    
    # Industrial
    'BA': 'Industrial', 'CAT': 'Industrial',
    
    # Communication Services
    'META': 'Communication Services', 'NFLX': 'Communication Services',
    
    # Consumer Staples
    'WMT': 'Consumer Staples', 'PG': 'Consumer Staples', 'KO': 'Consumer Staples',
    
    # ETFs
    'SPY': 'ETF', 'QQQ': 'ETF', 'IWM': 'ETF'
}

# Seasonal sector performance (based on historical data)
# Format: {month: [best_sectors, worst_sectors]}
SECTOR_SEASONALITY = {
    1: {  # January
        'best': ['Technology', 'Financial', 'Consumer Discretionary'],
        'worst': ['Energy', 'Consumer Staples']
    },
    2: {  # February
        'best': ['Technology', 'Healthcare'],
        'worst': ['Energy', 'Utilities']
    },
    3: {  # March
        'best': ['Consumer Discretionary', 'Technology'],
        'worst': ['Financial', 'Energy']
    },
    4: {  # April
        'best': ['Consumer Staples', 'Healthcare'],
        'worst': ['Technology', 'Communication Services']
    },
    5: {  # May
        'best': ['Healthcare', 'Consumer Staples'],
        'worst': ['Technology', 'Consumer Discretionary']
    },
    6: {  # June
        'best': ['Healthcare', 'Consumer Staples'],
        'worst': ['Technology', 'Financial']
    },
    7: {  # July
        'best': ['Technology', 'Consumer Discretionary'],
        'worst': ['Energy', 'Financial']
    },
    8: {  # August
        'best': ['Healthcare', 'Consumer Staples'],
        'worst': ['Technology', 'Consumer Discretionary']
    },
    9: {  # September
        'best': ['Healthcare', 'Consumer Staples'],
        'worst': ['Technology', 'Consumer Discretionary']
    },
    10: {  # October
        'best': ['Technology', 'Financial'],
        'worst': ['Consumer Staples', 'Utilities']
    },
    11: {  # November
        'best': ['Consumer Discretionary', 'Technology'],
        'worst': ['Healthcare', 'Utilities']
    },
    12: {  # December
        'best': ['Consumer Discretionary', 'Technology'],
        'worst': ['Healthcare', 'Energy']
    }
}

class SeasonalityEnhanced:
    """
    Enhanced seasonality analysis for stock selection and trading.
    
    This class provides methods to analyze seasonal patterns in stock performance
    and improve stock selection based on historical seasonal performance.
    """
    
    def __init__(self, seasonality_file: str, sector_influence: float = 0.3, stock_influence: float = 0.7, config: Dict = None):
        """
        Initialize the enhanced seasonality analyzer.
        
        Args:
            seasonality_file (str): Path to seasonality data file
            sector_influence (float, optional): Weight for sector seasonality (0-1)
            stock_influence (float, optional): Weight for stock-specific seasonality (0-1)
            config (Dict, optional): Configuration dictionary
        """
        self.seasonality_file = seasonality_file
        self.config = config
        
        # Set influence weights
        self.stock_influence = stock_influence
        self.sector_influence = sector_influence
        
        # Override with config values if provided
        if config and 'seasonality' in config:
            self.stock_influence = config['seasonality'].get('stock_specific_influence', stock_influence)
            self.sector_influence = config['seasonality'].get('sector_influence', sector_influence)
        
        # Load seasonality data
        self.seasonality_data = self._load_seasonality_data()
        self.sector_data = SECTOR_MAPPING
        self.sector_seasonality = SECTOR_SEASONALITY
        
        logger.info(f"Initialized enhanced seasonality analyzer with data for {len(self.seasonality_data)} symbols")
        logger.debug(f"Using stock influence: {self.stock_influence}, sector influence: {self.sector_influence}")
    
    def _load_seasonality_data(self):
        """
        Load seasonality data from file.
        
        Returns:
            dict: Dictionary of seasonality data by symbol
        """
        try:
            # Load data from file
            with open(self.seasonality_file, 'r') as f:
                raw_data = yaml.safe_load(f)
            
            logger.info(f"Raw data loaded from {self.seasonality_file}: {type(raw_data)}")
            
            # Check if data is in the expected format
            if isinstance(raw_data, dict) and 'opportunities' in raw_data:
                logger.info(f"Found 'opportunities' key with {len(raw_data['opportunities'])} items")
                
                # Convert data to the format expected by the analyzer
                data = {}
                for opportunity in raw_data['opportunities']:
                    symbol = opportunity.get('symbol')
                    season = opportunity.get('season')
                    
                    if not symbol or not season:
                        continue
                    
                    # Convert season to month number
                    month = self._season_to_month(season)
                    if not month:
                        continue
                    
                    # Initialize symbol data if needed
                    if symbol not in data:
                        data[symbol] = {}
                    
                    # Initialize month data if needed
                    if str(month) not in data[symbol]:
                        data[symbol][str(month)] = {
                            'win_rate': opportunity.get('win_rate', 0.5),
                            'avg_return': opportunity.get('avg_return', 0),
                            'correlation': opportunity.get('correlation', 0),
                            'direction': opportunity.get('direction', 'LONG'),
                            'trade_count': opportunity.get('trade_count', 0)
                        }
                
                logger.info(f"Converted data for {len(data)} symbols from opportunities format")
                
                # Log sample data for debugging
                sample_symbols = list(data.keys())[:3]
                for symbol in sample_symbols:
                    logger.info(f"Sample data for {symbol}: {data[symbol]}")
                
                return data
            elif isinstance(raw_data, dict) and len(raw_data) > 0:
                # Data might already be in the right format
                # Check the first item to see if it has the expected structure
                first_symbol = next(iter(raw_data))
                if isinstance(raw_data[first_symbol], dict):
                    logger.info(f"Data appears to be in the expected format with {len(raw_data)} symbols")
                    return raw_data
                else:
                    logger.warning(f"Data is in an unexpected format: {type(raw_data[first_symbol])}")
                    return {}
            else:
                # Assume data is already in the expected format
                logger.info(f"Data appears to be in direct format with {len(raw_data)} items")
                return raw_data
        except Exception as e:
            logger.error(f"Error loading seasonality data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
    
    def _season_to_month(self, season: str) -> int:
        """
        Convert season name to month number.
        
        Args:
            season (str): Season name (e.g., 'January', 'February', etc.)
            
        Returns:
            int: Month number (1-12) or None if invalid
        """
        season_map = {
            'January': 1, 'February': 2, 'March': 3, 'April': 4,
            'May': 5, 'June': 6, 'July': 7, 'August': 8,
            'September': 9, 'October': 10, 'November': 11, 'December': 12
        }
        return season_map.get(season)
    
    def get_seasonal_score(self, symbol: str, date: datetime) -> float:
        """
        Get the overall seasonal score for a symbol on a specific date.
        
        Args:
            symbol (str): Stock symbol
            date (datetime): Date to check
            
        Returns:
            float: Seasonal score between 0 and 1
        """
        # Extract month and day
        month = date.month
        day = date.day
        
        # Get stock-specific seasonal score
        stock_score = self._get_stock_seasonal_score(symbol, month, day)
        
        # Get sector seasonal score
        sector_score = self._get_sector_seasonal_score(symbol, month)
        
        # Apply monthly weight adjustment if available in config
        monthly_weight_multiplier = 1.0
        if hasattr(self, 'config') and self.config and 'seasonality' in self.config:
            if 'monthly_weights' in self.config['seasonality'] and month in self.config['seasonality']['monthly_weights']:
                monthly_weight = self.config['seasonality']['monthly_weights'].get(month, 1.0)
                # Normalize the weight to be a multiplier around 1.0
                monthly_weight_multiplier = monthly_weight / 0.4  # Assuming 0.4 is the baseline weight
                logger.debug(f"Applied monthly weight multiplier for month {month}: {monthly_weight_multiplier:.2f}")
        
        # Combine scores with configured weights
        combined_score = (stock_score * self.stock_influence + 
                          sector_score * self.sector_influence) * monthly_weight_multiplier
        
        # Ensure score is within 0-1 range
        combined_score = max(0, min(1, combined_score))
        
        return combined_score
    
    def _get_stock_seasonal_score(self, symbol: str, month: int, day: int) -> float:
        """
        Get seasonal score for a specific stock based on historical performance.
        
        Args:
            symbol (str): Stock symbol
            month (int): Month (1-12)
            day (int): Day of month
            
        Returns:
            float: Seasonal score between 0 and 1
        """
        if symbol not in self.seasonality_data:
            return 0.5  # Neutral if no data
        
        month_str = str(month)
        if month_str not in self.seasonality_data[symbol]:
            return 0.5  # Neutral if no data for this month
        
        # Get monthly performance data
        monthly_data = self.seasonality_data[symbol][month_str]
        
        # Calculate score based on win rate and average return
        win_rate = monthly_data.get('win_rate', 0.5)
        avg_return = monthly_data.get('avg_return', 0)
        
        # Normalize avg_return to a 0-1 scale (assuming max return of 5%)
        normalized_return = min(max(avg_return / 0.05 + 0.5, 0), 1)
        
        # Combine win rate and normalized return
        score = (win_rate * 0.6) + (normalized_return * 0.4)
        
        return score
    
    def _get_sector_seasonal_score(self, symbol: str, month: int) -> float:
        """
        Get seasonal score based on sector performance for the month.
        
        Args:
            symbol (str): Stock symbol
            month (int): Month (1-12)
            
        Returns:
            float: Sector seasonal score between 0 and 1
        """
        # Get sector for the symbol
        sector = self.sector_data.get(symbol, None)
        if not sector or month not in self.sector_seasonality:
            return 0.5  # Neutral if no sector data
        
        # Check if sector is in best or worst list for the month
        if sector in self.sector_seasonality[month]['best']:
            return 0.8  # High score for best sectors
        elif sector in self.sector_seasonality[month]['worst']:
            return 0.2  # Low score for worst sectors
        else:
            return 0.5  # Neutral for other sectors
    
    def get_top_seasonal_stocks(self, symbols: List[str], date: datetime, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Get top N stocks with the best seasonal scores for a given date.
        
        Args:
            symbols (list): List of stock symbols to consider
            date (datetime): Date to check
            top_n (int): Number of top stocks to return
            
        Returns:
            list: List of tuples (symbol, score) for top seasonal stocks
        """
        scores = []
        for symbol in symbols:
            score = self.get_seasonal_score(symbol, date)
            scores.append((symbol, score))
        
        # Sort by score in descending order
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_n]
    
    def filter_signals_by_seasonality(self, signals: List[Dict], date: datetime, threshold: float = 0.6) -> List[Dict]:
        """
        Filter trading signals based on seasonality scores.
        
        Args:
            signals (list): List of signal dictionaries
            date (datetime): Current date
            threshold (float): Minimum seasonality score to keep a signal
            
        Returns:
            list: Filtered list of signals
        """
        filtered_signals = []
        
        for signal in signals:
            symbol = signal.get('symbol')
            if not symbol:
                continue
                
            score = self.get_seasonal_score(symbol, date)
            
            # Add seasonality score to the signal
            signal['seasonality_score'] = score
            
            # Keep signals with score above threshold
            if score >= threshold:
                filtered_signals.append(signal)
        
        return filtered_signals
    
    def adjust_signal_weights(self, signals: List[Dict], date: datetime) -> List[Dict]:
        """
        Adjust signal weights based on seasonality scores.
        
        Args:
            signals (list): List of signal dictionaries
            date (datetime): Current date
            
        Returns:
            list: List of signals with adjusted weights
        """
        for signal in signals:
            symbol = signal.get('symbol')
            if not symbol:
                continue
                
            score = self.get_seasonal_score(symbol, date)
            
            # Add seasonality score to the signal
            signal['seasonality_score'] = score
            
            # Adjust weight based on seasonality score
            # Amplify weight for high scores, reduce for low scores
            if 'weight' in signal:
                original_weight = signal['weight']
                # Scale factor: 0.5 to 1.5 based on score
                scale_factor = 0.5 + score
                signal['weight'] = original_weight * scale_factor
        
        return signals
    
    def analyze_seasonal_performance(self, trades: pd.DataFrame, output_dir: str = 'output/seasonality') -> Dict:
        """
        Analyze trading performance based on seasonality alignment.
        
        Args:
            trades (pd.DataFrame): DataFrame of trades
            output_dir (str): Directory to save analysis results
            
        Returns:
            dict: Dictionary of performance metrics
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Add month to trades
        trades['month'] = pd.to_datetime(trades['timestamp']).dt.month
        
        # Calculate seasonality score for each trade
        trades['seasonality_score'] = trades.apply(
            lambda x: self.get_seasonal_score(x['symbol'], pd.to_datetime(x['timestamp'])), 
            axis=1
        )
        
        # Group trades by seasonality score
        high_seasonality = trades[trades['seasonality_score'] >= 0.7]
        medium_seasonality = trades[(trades['seasonality_score'] >= 0.4) & (trades['seasonality_score'] < 0.7)]
        low_seasonality = trades[trades['seasonality_score'] < 0.4]
        
        # Calculate performance metrics for each group
        metrics = {
            'high_seasonality': self._calculate_group_metrics(high_seasonality),
            'medium_seasonality': self._calculate_group_metrics(medium_seasonality),
            'low_seasonality': self._calculate_group_metrics(low_seasonality)
        }
        
        # Plot comparison
        self._plot_seasonality_comparison(metrics, output_dir)
        
        return metrics
    
    def calculate_seasonal_score(self, symbol: str, date: datetime) -> float:
        """
        Calculate the seasonal score for a symbol on a specific date.
        This is an alias for get_seasonal_score to maintain compatibility with CombinedStrategy.
        
        Args:
            symbol (str): Stock symbol
            date (datetime): Date to check
            
        Returns:
            float: Seasonal score between 0 and 1
        """
        return self.get_seasonal_score(symbol, date)
    
    def _calculate_group_metrics(self, trades: pd.DataFrame) -> Dict:
        """
        Calculate performance metrics for a group of trades.
        
        Args:
            trades (pd.DataFrame): DataFrame of trades
            
        Returns:
            dict: Dictionary of performance metrics
        """
        if len(trades) == 0:
            return {
                'win_rate': 0,
                'avg_return': 0,
                'profit_factor': 0,
                'trade_count': 0
            }
        
        # Calculate win rate
        if 'return' in trades.columns:
            wins = len(trades[trades['return'] > 0])
            win_rate = wins / len(trades) if len(trades) > 0 else 0
            
            # Calculate average return
            avg_return = trades['return'].mean() if len(trades) > 0 else 0
            
            # Calculate profit factor
            profits = trades[trades['return'] > 0]['return'].sum()
            losses = abs(trades[trades['return'] < 0]['return'].sum())
            profit_factor = profits / losses if losses != 0 else 0
        else:
            win_rate = 0
            avg_return = 0
            profit_factor = 0
        
        return {
            'win_rate': win_rate,
            'avg_return': avg_return,
            'profit_factor': profit_factor,
            'trade_count': len(trades)
        }
    
    def _plot_seasonality_comparison(self, metrics: Dict, output_dir: str):
        """
        Plot a comparison of performance across different seasonality levels.
        
        Args:
            metrics (dict): Dictionary of metrics by seasonality level
            output_dir (str): Directory to save the plot
        """
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Prepare data
        categories = list(metrics.keys())
        win_rates = [metrics[cat]['win_rate'] for cat in categories]
        avg_returns = [metrics[cat]['avg_return'] * 100 for cat in categories]  # Convert to percentage
        profit_factors = [metrics[cat]['profit_factor'] for cat in categories]
        trade_counts = [metrics[cat]['trade_count'] for cat in categories]
        
        # Create subplots
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # Win rate
        axs[0, 0].bar(categories, win_rates, color='skyblue')
        axs[0, 0].set_title('Win Rate by Seasonality')
        axs[0, 0].set_ylim(0, 1)
        for i, v in enumerate(win_rates):
            axs[0, 0].text(i, v + 0.05, f"{v:.2f}", ha='center')
        
        # Average return
        axs[0, 1].bar(categories, avg_returns, color='lightgreen')
        axs[0, 1].set_title('Average Return (%) by Seasonality')
        for i, v in enumerate(avg_returns):
            axs[0, 1].text(i, v + 0.5, f"{v:.2f}%", ha='center')
        
        # Profit factor
        axs[1, 0].bar(categories, profit_factors, color='salmon')
        axs[1, 0].set_title('Profit Factor by Seasonality')
        for i, v in enumerate(profit_factors):
            axs[1, 0].text(i, v + 0.2, f"{v:.2f}", ha='center')
        
        # Trade count
        axs[1, 1].bar(categories, trade_counts, color='plum')
        axs[1, 1].set_title('Trade Count by Seasonality')
        for i, v in enumerate(trade_counts):
            axs[1, 1].text(i, v + 2, str(v), ha='center')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/seasonality_performance.png")
        plt.close()
        
        # Save metrics to file
        with open(f"{output_dir}/seasonality_metrics.yaml", 'w') as f:
            yaml.dump(metrics, f, default_flow_style=False)
    
    def get_data_points_count(self, symbol: str, date: datetime) -> int:
        """
        Get the number of data points available for a symbol on a specific date.
        
        Args:
            symbol (str): Stock symbol
            date (datetime): Date to check
            
        Returns:
            int: Number of data points (years of data)
        """
        month = date.month
        day = date.day
        
        month_str = f"{month:02d}"
        
        # Check if we have data for this symbol and month
        if symbol not in self.seasonality_data:
            return 0
            
        if month_str not in self.seasonality_data[symbol]:
            return 0
            
        # Get monthly data
        monthly_data = self.seasonality_data[symbol][month_str]
        
        # Count data points (days with data)
        if 'days' in monthly_data and isinstance(monthly_data['days'], dict):
            # Count days with data
            return len(monthly_data['days'])
        elif 'opportunities' in monthly_data and isinstance(monthly_data['opportunities'], list):
            # Count opportunities
            return len(monthly_data['opportunities'])
        else:
            return 0
            
    def get_seasonal_consistency(self, symbol: str, date: datetime) -> float:
        """
        Calculate the consistency of seasonal patterns for a symbol on a specific date.
        
        Args:
            symbol (str): Stock symbol
            date (datetime): Date to check
            
        Returns:
            float: Consistency score between 0 and 1, or None if not available
        """
        month = date.month
        day = date.day
        
        month_str = f"{month:02d}"
        day_str = f"{day:02d}"
        
        # Check if we have data for this symbol and month
        if symbol not in self.seasonality_data:
            return None
            
        if month_str not in self.seasonality_data[symbol]:
            return None
            
        # Get monthly data
        monthly_data = self.seasonality_data[symbol][month_str]
        
        # Calculate consistency based on available data structure
        if 'days' in monthly_data and isinstance(monthly_data['days'], dict):
            # If we have day-specific data
            if day_str in monthly_data['days']:
                day_data = monthly_data['days'][day_str]
                
                # If we have win rate data
                if 'win_rate' in day_data:
                    # Win rate is a direct measure of consistency
                    return day_data['win_rate']
                    
                # If we have up/down counts
                if 'up' in day_data and 'down' in day_data:
                    total = day_data['up'] + day_data['down']
                    if total > 0:
                        # Calculate consistency as the dominance of the majority direction
                        return max(day_data['up'], day_data['down']) / total
            
            # If no day-specific data, use month average
            if 'avg_win_rate' in monthly_data:
                return monthly_data['avg_win_rate']
                
        elif 'opportunities' in monthly_data and isinstance(monthly_data['opportunities'], list):
            # For opportunities data structure, find relevant opportunity
            for opp in monthly_data['opportunities']:
                # Check if this opportunity covers our day
                if 'start_day' in opp and 'end_day' in opp:
                    start_day = int(opp['start_day'])
                    end_day = int(opp['end_day'])
                    
                    if start_day <= day <= end_day:
                        # If we have win rate data
                        if 'win_rate' in opp:
                            return opp['win_rate']
                        
                        # If we have up/down counts
                        if 'up_years' in opp and 'down_years' in opp:
                            total = opp['up_years'] + opp['down_years']
                            if total > 0:
                                # Calculate consistency as the dominance of the majority direction
                                return max(opp['up_years'], opp['down_years']) / total
        
        # Default to None if no consistency data available
        return None


# Example usage
if __name__ == "__main__":
    # Initialize seasonality analyzer
    seasonality = SeasonalityEnhanced('data/seasonality.yaml')
    
    # Get seasonal score for a symbol on a specific date
    score = seasonality.get_seasonal_score('AAPL', datetime(2023, 12, 15))
    print(f"Seasonal score for AAPL on 2023-12-15: {score:.2f}")
    
    # Get top seasonal stocks for a specific date
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'AMD']
    top_stocks = seasonality.get_top_seasonal_stocks(symbols, datetime(2023, 12, 15))
    print("Top seasonal stocks for 2023-12-15:")
    for symbol, score in top_stocks:
        print(f"  {symbol}: {score:.2f}")
