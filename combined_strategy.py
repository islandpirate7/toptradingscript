#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Combined Strategy Implementation
-------------------------------------
This module implements a combined strategy that integrates both mean reversion
and trend following approaches, adapting to different market conditions.
"""

import numpy as np
import pandas as pd
import logging
import datetime as dt
from datetime import datetime
from typing import List, Dict, Optional
from enum import Enum
from dataclasses import dataclass
import yaml

# Try to import talib, but provide fallback if not available
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logging.warning("TA-Lib not available, using pandas-based technical indicators")

# Import our individual strategies
from mean_reversion_strategy_optimized import MeanReversionStrategyOptimized
from trend_following_strategy import TrendFollowingStrategy, TradeDirection, Signal
from seasonality_enhanced import SeasonalityEnhanced

# Configure logging
logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    TRENDING = "trending"
    RANGE_BOUND = "range_bound"
    MIXED = "mixed"
    UNKNOWN = "unknown"

class CombinedStrategy:
    """Combined strategy that integrates mean reversion and trend following approaches"""
    
    def __init__(self, config):
        """Initialize the combined strategy with configuration"""
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize symbol data
        self.symbol_data = {}
        
        # Initialize strategy parameters
        self.config = config
        
        # Initialize seasonality parameters
        self.use_seasonality = config.get('seasonality', {}).get('enabled', False)
        self.use_seasonality_filter = config.get('seasonality', {}).get('use_filter', False)
        self.seasonality_file = config.get('seasonality', {}).get('data_file', 'output/seasonal_opportunities_converted.yaml')
        self.seasonality_min_score = config.get('seasonality', {}).get('min_score_threshold', 0.6)
        self.seasonality_weight_adjustment = config.get('seasonality', {}).get('weight_adjustment', True)
        self.seasonality_sector_influence = config.get('seasonality', {}).get('sector_influence', 0.3)
        self.seasonality_stock_influence = config.get('seasonality', {}).get('stock_specific_influence', 0.7)
        self.seasonality_top_n = config.get('seasonality', {}).get('top_n_selection', 10)
        self.seasonal_boost = config.get('seasonality', {}).get('boost_factor', 0.2)
        self.seasonal_penalty = config.get('seasonality', {}).get('penalty_factor', 0.2)
        
        # Initialize multi-factor parameters
        self.use_multi_factor = config.get('stock_selection', {}).get('enable_multi_factor', False)
        self.logger.info(f"Multi-factor stock selection enabled: {self.use_multi_factor}")
        
        # Initialize weights for combined strategy
        self.mr_weight = config.get('strategy_configs', {}).get('Combined', {}).get('mean_reversion_weight', 0.5)
        self.tf_weight = config.get('strategy_configs', {}).get('Combined', {}).get('trend_following_weight', 0.5)
        
        # Initialize strategies
        self.mean_reversion = MeanReversionStrategyOptimized(config)
        self.trend_following = TrendFollowingStrategy(config)
        
        # Initialize seasonality analyzer
        self.seasonality_analyzer = None
        if self.config['seasonality'].get('enabled', True):
            try:
                from seasonality_enhanced import SeasonalityEnhanced
                
                # Initialize seasonality analyzer with config
                self.seasonality_analyzer = SeasonalityEnhanced(
                    seasonality_file=self.config['seasonality'].get('data_file', 'output/seasonal_opportunities_converted.yaml'),
                    sector_influence=self.config['seasonality'].get('sector_influence', 0.3),
                    stock_influence=self.config['seasonality'].get('stock_specific_influence', 0.7),
                    config=self.config
                )
                self.logger.info("Seasonality analyzer initialized successfully")
            except Exception as e:
                self.logger.error(f"Error initializing seasonality analyzer: {e}")
        
        # Regime detection parameters
        self.adx_threshold = config.get('strategy_configs', {}).get('Combined', {}).get('adx_threshold', 20)
        self.volatility_period = config.get('strategy_configs', {}).get('Combined', {}).get('volatility_period', 20)
        self.regime_lookback = config.get('strategy_configs', {}).get('Combined', {}).get('regime_lookback', 10)
        
        # Market regime filter
        self.use_market_regime_filter = config.get('strategy_configs', {}).get('Combined', {}).get('use_market_regime_filter', False)
        
        # Signal filtering parameters
        self.min_signal_score = config.get('strategy_configs', {}).get('Combined', {}).get('min_signal_score', 0.7)
        
        # Maximum number of signals per day
        self.max_signals_per_day = config.get('general', {}).get('max_signals_per_day', 8)
        
        # Maximum portfolio risk
        self.max_portfolio_risk_pct = config.get('general', {}).get('max_portfolio_risk_pct', 0.015)
        
        # Symbol-specific configurations
        self.symbol_configs = config.get('symbol_configs', {})
        
        # Track performance by regime
        self.regime_performance = {
            'trending': {'trades': 0, 'wins': 0, 'total_return': 0},
            'range_bound': {'trades': 0, 'wins': 0, 'total_return': 0},
            'mixed': {'trades': 0, 'wins': 0, 'total_return': 0}
        }
        
        self.logger.info(f"Initialized Combined Strategy with weights: MR={self.mr_weight}, TF={self.tf_weight}")
    
    def set_symbol_data(self, symbol_data):
        """Store symbol data for later use in the strategy
        
        Args:
            symbol_data (dict): Dictionary of symbol -> dataframe with price data
        """
        self.symbol_data = symbol_data
        
        # Log the number of symbols with data
        valid_symbols = [symbol for symbol, df in symbol_data.items() if df is not None and not df.empty]
        self.logger.info(f"Set symbol data for {len(valid_symbols)} symbols")
        
        # Store sector mapping for symbols if available
        self.symbol_sectors = {}
        if 'sector_mapping' in self.config:
            for sector, symbols in self.config['sector_mapping'].items():
                for symbol in symbols:
                    self.symbol_sectors[symbol] = sector
            
            self.logger.info(f"Loaded sector mappings for {len(self.symbol_sectors)} symbols")
    
    def _load_seasonality_data(self, seasonality_file):
        """Load seasonality data from file
        
        Args:
            seasonality_file (str): Path to seasonality data file
        """
        try:
            with open(seasonality_file, 'r') as f:
                data = yaml.safe_load(f)
                
            # Convert to a more usable format
            self.seasonality_data = {}
            if data and 'opportunities' in data:
                for opportunity in data['opportunities']:
                    symbol = opportunity.get('symbol', '')
                    if not symbol:
                        continue
                        
                    # Convert values to float to handle NumPy scalar values
                    avg_return = float(opportunity.get('avg_return', 0.0))
                    correlation = float(opportunity.get('correlation', 0.0))
                    win_rate = float(opportunity.get('win_rate', 0.0))
                    direction = opportunity.get('direction', 'LONG')
                    season = opportunity.get('season', '')
                    trade_count = int(opportunity.get('trade_count', 0))
                    
                    if symbol not in self.seasonality_data:
                        self.seasonality_data[symbol] = []
                        
                    self.seasonality_data[symbol].append({
                        'avg_return': avg_return,
                        'correlation': correlation,
                        'win_rate': win_rate,
                        'direction': direction,
                        'season': season,
                        'trade_count': trade_count
                    })
                    
            self.logger.info(f"Loaded seasonality data for {len(self.seasonality_data)} symbols")
        except Exception as e:
            self.logger.warning(f"Failed to load seasonality data: {e}")
            self.seasonality_data = {}
            
    def parse_date(self, date_str):
        """Parse date string to datetime object
        
        Args:
            date_str (str): Date string in various formats
            
        Returns:
            datetime: Parsed datetime object or None if parsing fails
        """
        try:
            # Try different date formats
            formats = ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d']
            for fmt in formats:
                try:
                    return dt.datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            
            # If it's a timestamp
            if isinstance(date_str, (int, float)):
                return dt.datetime.fromtimestamp(date_str)
                
            return None
        except Exception as e:
            self.logger.warning(f"Failed to parse date {date_str}: {e}")
            return None
    
    def get_seasonal_score(self, symbol, current_date=None):
        """Get seasonality score for a symbol based on current date
        
        Args:
            symbol (str): Symbol to get seasonality score for
            current_date (datetime, optional): Current date. Defaults to today.
            
        Returns:
            tuple: (seasonality_score, direction) where score is between -1.0 and 1.0
                  and direction is 'LONG' or 'SHORT'
        """
        # Use today's date if not provided
        if current_date is None:
            current_date = dt.datetime.now()
        elif isinstance(current_date, str):
            current_date = self.parse_date(current_date)
            if current_date is None:
                current_date = dt.datetime.now()
        
        # Check if we have the enhanced seasonality analyzer
        if hasattr(self, 'seasonality_analyzer') and self.seasonality_analyzer is not None:
            try:
                # Use the enhanced seasonality analyzer
                score = self.seasonality_analyzer.get_seasonal_score(symbol, current_date)
                
                # Apply additional differentiation to avoid clustering around 0.5
                # If score is close to 0.5 (neutral), add some randomness based on symbol
                if 0.45 <= score <= 0.55:
                    # Use a deterministic "random" value based on symbol and date
                    # This ensures consistent results for the same symbol/date
                    symbol_hash = sum(ord(c) for c in symbol)
                    date_hash = current_date.day + current_date.month * 31
                    combined_hash = (symbol_hash + date_hash) % 100
                    
                    # Add small variation (-0.05 to +0.05) to differentiate similar scores
                    variation = (combined_hash / 100 - 0.5) * 0.1
                    score = max(0.0, min(1.0, score + variation))
                    
                    self.logger.debug(f"Applied differentiation to {symbol} seasonality score: {score}")
                
                # Convert score from 0-1 scale to -1 to 1 scale
                # Scores above 0.5 are considered bullish, below 0.5 are bearish
                normalized_score = (score - 0.5) * 2
                
                # Determine direction based on score
                direction = 'LONG' if score >= 0.5 else 'SHORT'
                
                # Log the score for debugging
                self.logger.debug(f"Seasonality score for {symbol} on {current_date.strftime('%Y-%m-%d')}: {score} ({direction})")
                
                return normalized_score, direction
            except Exception as e:
                self.logger.error(f"Error getting enhanced seasonality score for {symbol}: {e}")
                # Fall back to legacy method
                pass
        else:
            # Fall back to legacy seasonality data if no enhanced analyzer
            if self.use_seasonality and symbol in self.seasonality_data:
                # Get current month name
                current_month = current_date.strftime('%B')
                
                # Find seasonal opportunities for the current month
                seasonal_opportunities = []
                for opportunity in self.seasonality_data.get(symbol, []):
                    if opportunity.get('season', '') == current_month:
                        seasonal_opportunities.append(opportunity)
                
                if not seasonal_opportunities:
                    return 0.0, 'LONG'
                
                # Calculate combined score based on win rate, avg return and correlation
                total_score = 0.0
                total_weight = 0.0
                best_direction = 'LONG'
                
                for opportunity in seasonal_opportunities:
                    win_rate = opportunity.get('win_rate', 0.0)
                    avg_return = opportunity.get('avg_return', 0.0)
                    correlation = opportunity.get('correlation', 0.0)
                    direction = opportunity.get('direction', 'LONG')
                    trade_count = opportunity.get('trade_count', 0)
                    
                    # Skip if we don't have valid data
                    if win_rate == 0.0 and avg_return == 0.0 and correlation == 0.0:
                        continue
                    
                    # Calculate individual score components
                    win_rate_score = win_rate - 0.5  # -0.5 to 0.5
                    avg_return_score = min(max(avg_return * 10, -0.5), 0.5)  # -0.5 to 0.5
                    correlation_score = correlation * 0.5  # -0.5 to 0.5
                    
                    # Weight based on number of trades (more trades = more reliable)
                    weight = min(trade_count / 10, 1.0)
                    
                    # Combined weighted score for this opportunity
                    opportunity_score = (win_rate_score + avg_return_score + correlation_score) / 3
                    
                    # Apply direction (positive for LONG, negative for SHORT)
                    if direction == 'SHORT':
                        opportunity_score = -opportunity_score
                    
                    total_score += opportunity_score * weight
                    total_weight += weight
                    
                    # Track the direction with the highest score
                    if abs(opportunity_score) > abs(total_score / max(total_weight, 1)):
                        best_direction = direction
                
                # Calculate final score (-1.0 to 1.0)
                if total_weight > 0:
                    final_score = total_score / total_weight
                else:
                    final_score = 0.0
                
                return final_score, best_direction
            else:
                # No seasonality data available
                return 0.0, 'LONG'
    
    def detect_market_regime(self, df):
        """
        Detect the current market regime (trending, range-bound, or mixed).
        
        Enhanced version with multiple indicators:
        1. ADX (Average Directional Index) - measures trend strength
        2. Bollinger Band Width - measures volatility
        3. VIX (if available) - measures market fear/volatility
        4. Correlation between sectors - measures sector rotation
        
        Args:
            df (pd.DataFrame): DataFrame with price data (typically for SPY or another index)
            
        Returns:
            MarketRegime: Detected market regime
        """
        if df is None or df.empty or len(df) < 30:
            self.logger.warning("Insufficient data for market regime detection")
            return MarketRegime.MIXED
        
        # Get configuration parameters
        adx_threshold = self.config.get('market_regime', {}).get('adx_threshold', 25)
        bb_width_change_threshold = self.config.get('market_regime', {}).get('bb_width_change_threshold', 0.05)
        lookback_period = self.config.get('market_regime', {}).get('lookback_period', 20)
        vix_threshold = self.config.get('market_regime', {}).get('vix_threshold', 20)
        
        # Ensure we have enough data
        if len(df) < lookback_period + 10:
            self.logger.warning(f"Insufficient data for market regime detection (need {lookback_period + 10} bars)")
            return MarketRegime.MIXED
        
        # Calculate ADX
        try:
            adx = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=lookback_period)
            current_adx = adx[-1]
        except Exception as e:
            self.logger.error(f"Error calculating ADX: {e}")
            current_adx = 0
        
        # Calculate Bollinger Bands Width
        try:
            upper, middle, lower = talib.BBANDS(df['close'].values, timeperiod=lookback_period, nbdevup=2, nbdevdn=2)
            bb_width = (upper - lower) / middle
            current_bb_width = bb_width[-1]
            prev_bb_width = bb_width[-10]
            bb_width_change = (current_bb_width - prev_bb_width) / prev_bb_width
        except Exception as e:
            self.logger.error(f"Error calculating Bollinger Bands: {e}")
            bb_width_change = 0
        
        # Check for VIX data if available (using ^VIX symbol)
        vix_indicator = 0
        if 'VIX' in self.symbol_data:
            try:
                vix_df = self.symbol_data['VIX']
                if not vix_df.empty:
                    current_vix = vix_df['close'].iloc[-1]
                    vix_indicator = 1 if current_vix > vix_threshold else 0
            except Exception as e:
                self.logger.error(f"Error processing VIX data: {e}")
        
        # Check sector correlation if sector data is available
        sector_rotation_indicator = 0
        if hasattr(self, 'sector_data') and self.sector_data:
            try:
                sector_rotation_indicator = self.detect_sector_rotation()
            except Exception as e:
                self.logger.error(f"Error detecting sector rotation: {e}")
        
        # Determine market regime based on indicators
        trending_score = 0
        range_bound_score = 0
        
        # ADX contribution
        if current_adx > adx_threshold:
            trending_score += 1
        else:
            range_bound_score += 1
        
        # Bollinger Band Width contribution
        if abs(bb_width_change) > bb_width_change_threshold:
            trending_score += 1
        else:
            range_bound_score += 1
        
        # VIX contribution (if available)
        if vix_indicator > 0:
            trending_score += 0.5
        
        # Sector rotation contribution (if available)
        if sector_rotation_indicator > 0:
            trending_score += 0.5
        
        # Log the regime detection details
        self.logger.info(f"Market regime indicators - ADX: {current_adx:.2f}, BB Width Change: {bb_width_change:.2%}, " +
                        f"VIX Indicator: {vix_indicator}, Sector Rotation: {sector_rotation_indicator}")
        self.logger.info(f"Market regime scores - Trending: {trending_score}, Range-bound: {range_bound_score}")
        
        # Determine the final regime
        if trending_score > range_bound_score + 0.5:
            self.logger.info("Detected TRENDING market regime")
            return MarketRegime.TRENDING
        elif range_bound_score > trending_score + 0.5:
            self.logger.info("Detected RANGE_BOUND market regime")
            return MarketRegime.RANGE_BOUND
        else:
            self.logger.info("Detected MIXED market regime")
            return MarketRegime.MIXED
    
    def detect_sector_rotation(self):
        """
        Detect sector rotation by analyzing correlation and relative performance of sectors.
        
        Returns:
            int: Sector rotation indicator (1 if rotation detected, 0 otherwise)
        """
        # Check if we have sector ETFs in our data
        sector_etfs = ['XLF', 'XLV', 'XLE', 'XLK', 'XLI', 'XLP', 'XLY', 'XLB', 'XLU', 'XLRE', 'XLC']
        available_sectors = [etf for etf in sector_etfs if etf in self.symbol_data]
        
        if len(available_sectors) < 3:
            self.logger.warning("Insufficient sector data for rotation analysis")
            return 0
        
        # Get configuration parameters
        lookback_period = self.config.get('market_regime', {}).get('sector_rotation_lookback', 30)
        correlation_threshold = self.config.get('market_regime', {}).get('correlation_threshold', 0.7)
        
        # Calculate sector correlations and performance
        sector_returns = {}
        sector_data = {}
        
        for sector in available_sectors:
            df = self.symbol_data[sector]
            if len(df) < lookback_period:
                continue
                
            # Calculate returns
            df_subset = df.tail(lookback_period).copy()
            df_subset['return'] = df_subset['close'].pct_change()
            sector_returns[sector] = df_subset['return'].mean() * 100  # Mean daily return in percent
            sector_data[sector] = df_subset['return']
        
        # Calculate correlation matrix
        if len(sector_data) >= 3:
            returns_df = pd.DataFrame(sector_data)
            correlation_matrix = returns_df.corr()
            
            # Calculate average correlation
            corr_values = correlation_matrix.values
            # Get upper triangle of correlation matrix excluding diagonal
            upper_triangle = np.triu(corr_values, k=1)
            # Calculate average of upper triangle
            avg_correlation = np.mean(upper_triangle[upper_triangle != 0])
            
            # Calculate performance dispersion
            returns_array = np.array(list(sector_returns.values()))
            returns_dispersion = np.std(returns_array)
            
            # Log sector rotation metrics
            self.logger.info(f"Sector rotation metrics - Avg correlation: {avg_correlation:.2f}, " +
                            f"Returns dispersion: {returns_dispersion:.2f}%")
            
            # Determine if sector rotation is occurring
            if avg_correlation < correlation_threshold and returns_dispersion > 0.5:
                self.logger.info("Sector rotation detected")
                return 1
        
        return 0
    
    def identify_leading_sectors(self):
        """
        Identify the top performing sectors based on momentum.
        
        Returns:
            dict: Dictionary of sector -> momentum score
        """
        # Check if we have sector ETFs in our data
        sector_etfs = ['XLF', 'XLV', 'XLE', 'XLK', 'XLI', 'XLP', 'XLY', 'XLB', 'XLU', 'XLRE', 'XLC']
        available_sectors = [etf for etf in sector_etfs if etf in self.symbol_data]
        
        if len(available_sectors) < 3:
            self.logger.warning("Insufficient sector data for leading sector analysis")
            return {}
        
        # Get configuration parameters
        lookback_period = self.config.get('stock_selection', {}).get('sector_rotation', {}).get('lookback_period', 30)
        top_sectors = self.config.get('stock_selection', {}).get('sector_rotation', {}).get('top_sectors', 3)
        
        # Calculate sector momentum
        sector_momentum = {}
        
        for sector in available_sectors:
            df = self.symbol_data[sector]
            if len(df) < lookback_period:
                continue
                
            # Calculate momentum score (combination of short and medium-term performance)
            try:
                # 5-day momentum
                short_momentum = (df['close'].iloc[-1] / df['close'].iloc[-5] - 1) * 100
                # 20-day momentum
                medium_momentum = (df['close'].iloc[-1] / df['close'].iloc[-20] - 1) * 100
                # Combine with more weight on recent performance
                momentum_score = (short_momentum * 0.6) + (medium_momentum * 0.4)
                sector_momentum[sector] = momentum_score
            except Exception as e:
                self.logger.error(f"Error calculating momentum for {sector}: {e}")
        
        # Sort sectors by momentum score
        sorted_sectors = sorted(sector_momentum.items(), key=lambda x: x[1], reverse=True)
        
        # Get top sectors
        top_sectors_dict = dict(sorted_sectors[:top_sectors])
        
        # Log top sectors
        self.logger.info(f"Top performing sectors: {top_sectors_dict}")
        
        return top_sectors_dict
    
    def get_sector_for_symbol(self, symbol):
        """
        Map a symbol to its sector.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            str: Sector name or None if not found
        """
        # Basic sector mapping for common stocks
        # This could be enhanced with a more comprehensive database
        tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'META', 'NVDA', 'AMD', 'INTC', 'CSCO', 'ORCL', 'IBM', 
                      'ADBE', 'CRM', 'PYPL', 'NFLX', 'TSLA', 'AVGO', 'TXN', 'QCOM', 'MU', 'AMAT', 'ADI', 'LRCX', 
                      'KLAC', 'SNPS', 'CDNS', 'INTU', 'NOW', 'WDAY', 'TEAM', 'ZS', 'OKTA', 'CRWD', 'NET', 'DDOG', 'SNOW']
        
        financial_stocks = ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'AXP', 'V', 'MA', 'SCHW', 'PNC', 'USB', 
                           'TFC', 'COF', 'BK', 'STT', 'DFS', 'AIG', 'MET', 'PRU', 'ALL', 'TRV', 'CB', 'PGR', 'MMC', 
                           'AON', 'ICE', 'CME', 'SPGI', 'MCO']
        
        healthcare_stocks = ['JNJ', 'PFE', 'MRK', 'ABBV', 'LLY', 'ABT', 'TMO', 'DHR', 'UNH', 'CVS', 'AMGN', 'GILD', 
                            'ISRG', 'REGN', 'VRTX', 'MRNA', 'BIIB', 'BMY', 'MDT', 'SYK', 'BSX', 'ZTS', 'DXCM', 'ILMN', 
                            'IDXX', 'EW', 'HUM', 'CI', 'ANTM', 'CNC', 'BDX', 'BAX']
        
        consumer_stocks = ['PG', 'KO', 'PEP', 'WMT', 'COST', 'HD', 'LOW', 'MCD', 'SBUX', 'NKE', 'DIS', 'CMCSA', 
                          'TGT', 'BKNG', 'EXPE', 'MAR', 'HLT', 'YUM', 'QSR', 'MO', 'PM', 'KHC', 'GIS', 'K', 'CL', 
                          'CLX', 'CHD', 'EL', 'ULTA', 'VFC', 'TPR', 'RL']
        
        industrial_stocks = ['GE', 'HON', 'MMM', 'CAT', 'DE', 'BA', 'LMT', 'RTX', 'GD', 'NOC', 'UPS', 'FDX', 'UNP', 
                            'CSX', 'NSC', 'EMR', 'ETN', 'PH', 'ITW', 'CMI', 'ROK', 'IR', 'DOV', 'SWK', 'URI', 'PCAR', 'WAB']
        
        energy_stocks = ['XOM', 'CVX', 'COP', 'EOG', 'PXD', 'OXY', 'MPC', 'PSX', 'VLO', 'KMI', 'WMB', 'ET', 'EPD', 
                        'SLB', 'HAL', 'BKR']
        
        materials_stocks = ['LIN', 'APD', 'SHW', 'FCX', 'NEM', 'GOLD', 'NUE', 'STLD', 'DOW', 'DD', 'ECL', 'PPG', 
                           'ALB', 'CF', 'MOS']
        
        utilities_stocks = ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'PCG', 'ED', 'XEL', 'WEC', 'ES', 'PEG', 
                           'DTE', 'FE']
        
        real_estate_stocks = ['AMT', 'CCI', 'EQIX', 'PLD', 'PSA', 'WELL', 'AVB', 'EQR', 'DLR', 'O', 'SPG', 'VICI']
        
        communication_stocks = ['T', 'VZ', 'TMUS', 'CHTR', 'EA', 'ATVI', 'TTWO', 'MTCH', 'PINS', 'SNAP', 'TWTR']
        
        # Map symbol to sector
        if symbol in tech_stocks:
            return 'Technology'
        elif symbol in financial_stocks:
            return 'Financial'
        elif symbol in healthcare_stocks:
            return 'Healthcare'
        elif symbol in consumer_stocks:
            return 'Consumer'
        elif symbol in industrial_stocks:
            return 'Industrial'
        elif symbol in energy_stocks:
            return 'Energy'
        elif symbol in materials_stocks:
            return 'Materials'
        elif symbol in utilities_stocks:
            return 'Utilities'
        elif symbol in real_estate_stocks:
            return 'Real Estate'
        elif symbol in communication_stocks:
            return 'Communication'
        else:
            return None
    
    def adjust_weights_by_regime(self, regime):
        """Adjust strategy weights based on market regime
        
        Args:
            regime (str): Market regime ('trending', 'range_bound', or 'mixed')
            
        Returns:
            tuple: Adjusted weights for mean reversion and trend following
        """
        # Get regime-specific weights from config
        if regime == MarketRegime.TRENDING:
            # In trending markets, favor trend following
            mr_weight = self.config.get('strategy_configs', {}).get('Combined', {}).get('trending_regime_weights', {}).get('mean_reversion', 0.15)
            tf_weight = self.config.get('strategy_configs', {}).get('Combined', {}).get('trending_regime_weights', {}).get('trend_following', 0.85)
        elif regime == MarketRegime.RANGE_BOUND:
            # In range-bound markets, favor mean reversion
            mr_weight = self.config.get('strategy_configs', {}).get('Combined', {}).get('range_bound_regime_weights', {}).get('mean_reversion', 0.7)
            tf_weight = self.config.get('strategy_configs', {}).get('Combined', {}).get('range_bound_regime_weights', {}).get('trend_following', 0.3)
        else:  # Mixed or unknown
            # In mixed markets, use balanced weights
            mr_weight = self.config.get('strategy_configs', {}).get('Combined', {}).get('mixed_regime_weights', {}).get('mean_reversion', 0.4)
            tf_weight = self.config.get('strategy_configs', {}).get('Combined', {}).get('mixed_regime_weights', {}).get('trend_following', 0.6)
        
        self.logger.info(f"Dynamic weight allocation: MR={mr_weight}, TF={tf_weight}")
        
        return mr_weight, tf_weight
    
    def generate_signals(self, df, symbol=None):
        """Generate trading signals by combining mean reversion and trend following strategies
        
        Args:
            df (pd.DataFrame): DataFrame with price data
            symbol (str, optional): Symbol for the data
            
        Returns:
            list: List of signal dictionaries
        """
        logger.info(f"Generating signals for {symbol if symbol else 'unknown symbol'}")
        
        # Generate signals from both strategies
        mr_signals = self.mean_reversion.generate_signals(df, symbol)
        tf_signals = self.trend_following.generate_signals(df, symbol)
        
        # Log signal counts for debugging
        logger.info(f"Generated {len(mr_signals)} mean reversion signals and {len(tf_signals)} trend following signals")
        
        # Combine signals
        all_signals = mr_signals + tf_signals
        
        # Ensure all signals have a symbol
        if symbol:
            for signal in all_signals:
                if 'symbol' not in signal:
                    signal['symbol'] = symbol
        
        # Sort signals by date
        all_signals = sorted(all_signals, key=lambda x: x['date'])
        
        # Apply market regime filter if enabled
        if self.use_market_regime_filter:
            # Calculate market regime
            market_regime = self.calculate_market_regime(df)
            
            # Filter signals based on market regime
            filtered_signals = []
            for signal in all_signals:
                # Get signal date
                signal_date = signal['date']
                
                # Get market regime for this date
                if isinstance(signal_date, str):
                    signal_date = pd.to_datetime(signal_date)
                
                # Find the closest date in the market regime
                closest_date = min(market_regime.keys(), key=lambda x: abs(x - signal_date))
                regime = market_regime.get(closest_date, 'neutral')
                
                # Apply market regime filter
                if (signal['direction'] == 'LONG' and regime in ['bullish', 'neutral']) or \
                   (signal['direction'] == 'SHORT' and regime in ['bearish', 'neutral']):
                    # Adjust signal strength based on market regime
                    if (signal['direction'] == 'LONG' and regime == 'bullish') or \
                       (signal['direction'] == 'SHORT' and regime == 'bearish'):
                        # Boost signal strength in favorable regime
                        signal['strength_value'] = signal.get('strength_value', 1.0) * 1.5
                        signal['weight'] = signal.get('weight', 1.0) * 1.5
                    
                    filtered_signals.append(signal)
            
            all_signals = filtered_signals
        
        # Apply seasonality filter if enabled
        if self.use_seasonality_filter and symbol:
            # Calculate seasonality
            seasonality = self.calculate_seasonality(symbol, df)
            
            # Filter signals based on seasonality
            filtered_signals = []
            for signal in all_signals:
                # Get signal date
                signal_date = signal['date']
                
                # Get seasonality for this date
                if isinstance(signal_date, str):
                    signal_date = pd.to_datetime(signal_date)
                
                # Convert to month-day format
                month_day = f"{signal_date.month:02d}-{signal_date.day:02d}"
                
                # Get seasonality for this month-day
                seasonal_bias = seasonality.get(month_day, 'neutral')
                
                # Apply seasonality filter with more aggressive settings
                if (signal['direction'] == 'LONG' and seasonal_bias != 'bearish') or \
                   (signal['direction'] == 'SHORT' and seasonal_bias != 'bullish'):
                    # Adjust signal strength based on seasonality
                    if (signal['direction'] == 'LONG' and seasonal_bias == 'bullish') or \
                       (signal['direction'] == 'SHORT' and seasonal_bias == 'bearish'):
                        # Boost signal strength in favorable seasonality
                        signal['strength_value'] = signal.get('strength_value', 1.0) * self.seasonal_boost
                        signal['weight'] = signal.get('weight', 1.0) * self.seasonal_boost
                    elif seasonal_bias == 'neutral':
                        # Keep signal as is in neutral seasonality
                        pass
                    else:
                        # Reduce signal strength in unfavorable seasonality but still include it
                        signal['strength_value'] = signal.get('strength_value', 1.0) * self.seasonal_penalty
                        signal['weight'] = signal.get('weight', 1.0) * self.seasonal_penalty
                    
                    filtered_signals.append(signal)
                else:
                    # Include all signals with reduced weight instead of filtering them out
                    signal['strength_value'] = signal.get('strength_value', 1.0) * 0.5
                    signal['weight'] = signal.get('weight', 1.0) * 0.5
                    filtered_signals.append(signal)
            
            all_signals = filtered_signals
        
        # Apply signal score filter
        scored_signals = []
        for signal in all_signals:
            # Calculate signal score
            score = self.calculate_signal_score(signal)
            
            # Add score to signal
            signal['score'] = score
            
            # Filter signals based on score
            if score >= self.min_signal_score:
                scored_signals.append(signal)
        
        # Log final signal count
        logger.info(f"Final signal count after all filters: {len(scored_signals)}")
        
        # Return filtered signals
        return scored_signals
    
    def calculate_position_size(self, signal, capital, current_positions):
        """Calculate position size based on risk and signal strength
        
        Args:
            signal (dict): Signal dictionary
            capital (float): Available capital
            current_positions (int): Number of current open positions
            
        Returns:
            int: Number of shares to trade
        """
        # Get risk parameters
        max_portfolio_risk_pct = self.config.get('general', {}).get('max_portfolio_risk_pct', 0.02)
        max_positions = self.config.get('general', {}).get('max_positions', 15)
        min_capital_per_trade = self.config.get('general', {}).get('min_capital_per_trade', 1000)
        
        # Calculate available risk capital
        available_risk_capital = capital * max_portfolio_risk_pct
        
        # Calculate per-position risk based on max positions
        per_position_risk = available_risk_capital / max_positions
        
        # Adjust risk based on current number of positions (reduce risk as positions increase)
        position_factor = 1 - (current_positions / max_positions * 0.5)  # Scales from 1.0 to 0.5 as positions increase
        adjusted_per_position_risk = per_position_risk * position_factor
        
        # Get price and stop loss from signal
        price = signal['price']
        stop_loss = signal['stop_loss']
        
        # Calculate risk per share
        risk_per_share = abs(price - stop_loss)
        
        if risk_per_share <= 0:
            self.logger.warning(f"Invalid risk per share: {risk_per_share}. Using 1% of price.")
            risk_per_share = price * 0.01  # Default to 1% of price
        
        # Apply risk multiplier from signal
        risk_multiplier = signal.get('risk_multiplier', 1.0)
        
        # Apply strength-based risk adjustment
        strength_value = signal.get('strength_value', 0.7)
        strength_multiplier = 0.7 + (strength_value * 0.6)  # Scales from 0.7 to 1.3 based on strength
        
        # Apply market regime risk adjustment
        regime = signal.get('regime', 'mixed')
        if regime == MarketRegime.TRENDING:
            regime_multiplier = 1.2  # Increase position size in trending markets
        elif regime == MarketRegime.RANGE_BOUND:
            regime_multiplier = 0.9  # Decrease position size in range-bound markets
        else:  # Mixed
            regime_multiplier = 0.8  # More conservative in mixed markets
        
        # Calculate final risk-adjusted position size
        final_risk = adjusted_per_position_risk * risk_multiplier * strength_multiplier * regime_multiplier
        
        # Calculate number of shares based on risk
        shares = int(final_risk / risk_per_share)
        
        # Ensure minimum position size
        min_shares = max(1, int(min_capital_per_trade / price))
        shares = max(shares, min_shares)
        
        # Ensure position doesn't exceed available capital
        max_shares = int(capital / price * 0.95)  # Use 95% of available capital at most
        shares = min(shares, max_shares)
        
        # Log position sizing details
        self.logger.info(f"Position sizing: {shares} shares at ${price:.2f} with risk ${final_risk:.2f} " +
                        f"(strength: {strength_value:.2f}, regime: {regime})")
        
        return shares
    
    def update_regime_performance(self, trade):
        """Update performance tracking by market regime
        
        Args:
            trade (dict): Completed trade information
        """
        regime = trade.get('regime', 'unknown')
        if regime in self.regime_performance:
            self.regime_performance[regime]['trades'] += 1
            if trade['pnl'] > 0:
                self.regime_performance[regime]['wins'] += 1
            self.regime_performance[regime]['total_return'] += trade['pnl']
    
    def get_regime_performance(self):
        """Get performance metrics by regime
        
        Returns:
            dict: Performance metrics by regime
        """
        return self.regime_performance
    
    def calculate_seasonality(self, symbol, df):
        """Calculate seasonality for a symbol
        
        Args:
            symbol (str): Symbol to calculate seasonality for
            df (pd.DataFrame): DataFrame with price data
            
        Returns:
            dict: Dictionary of month-day -> bias ('bullish', 'bearish', or 'neutral')
        """
        # Default to neutral seasonality if no data is available
        seasonality = {}
        
        # If seasonality analyzer is available, use it
        if self.seasonality_analyzer and hasattr(self.seasonality_analyzer, 'get_seasonal_bias'):
            try:
                # Get date range from dataframe
                start_date = df.index[0]
                end_date = df.index[-1]
                
                # Get seasonality for each day in the range
                current_date = start_date
                while current_date <= end_date:
                    month_day = f"{current_date.month:02d}-{current_date.day:02d}"
                    
                    # Get seasonal bias from analyzer
                    bias = self.seasonality_analyzer.get_seasonal_bias(symbol, current_date)
                    
                    # Store bias
                    seasonality[month_day] = bias
                    
                    # Move to next day
                    current_date += pd.Timedelta(days=1)
            except Exception as e:
                logger.error(f"Error calculating seasonality for {symbol}: {e}")
                # Use fallback method
                seasonality = self._calculate_seasonality_fallback(symbol, df)
        else:
            # Use fallback method
            seasonality = self._calculate_seasonality_fallback(symbol, df)
        
        return seasonality
    
    def _calculate_seasonality_fallback(self, symbol, df):
        """Fallback method to calculate seasonality when seasonality analyzer is not available
        
        Args:
            symbol (str): Symbol to calculate seasonality for
            df (pd.DataFrame): DataFrame with price data
            
        Returns:
            dict: Dictionary of month-day -> bias ('bullish', 'bearish', or 'neutral')
        """
        # Default to neutral seasonality
        seasonality = {}
        
        # If seasonality data is available for this symbol, use it
        if symbol in self.seasonality_data:
            # Get seasonality data for this symbol
            symbol_seasonality = self.seasonality_data[symbol]
            
            # Get date range from dataframe
            start_date = df.index[0]
            end_date = df.index[-1]
            
            # Get seasonality for each day in the range
            current_date = start_date
            while current_date <= end_date:
                month_day = f"{current_date.month:02d}-{current_date.day:02d}"
                
                # Default to neutral
                bias = 'neutral'
                
                # Check if month-day is in seasonality data
                if month_day in symbol_seasonality:
                    # Get bias based on historical performance
                    if symbol_seasonality[month_day] > 0.6:
                        bias = 'bullish'
                    elif symbol_seasonality[month_day] < 0.4:
                        bias = 'bearish'
                
                # Store bias
                seasonality[month_day] = bias
                
                # Move to next day
                current_date += pd.Timedelta(days=1)
        else:
            # If no seasonality data is available, use all neutral
            # Get date range from dataframe
            start_date = df.index[0]
            end_date = df.index[-1]
            
            # Get seasonality for each day in the range
            current_date = start_date
            while current_date <= end_date:
                month_day = f"{current_date.month:02d}-{current_date.day:02d}"
                
                # Default to neutral
                seasonality[month_day] = 'neutral'
                
                # Move to next day
                current_date += pd.Timedelta(days=1)
        
        return seasonality
    
    def calculate_technical_score(self, df, symbol, momentum_weight=0.25, trend_weight=0.25, 
                               volatility_weight=0.25, volume_weight=0.25):
        """
        Calculate a technical score for a symbol based on multiple indicators
        
        This method evaluates a stock using several technical indicators:
        1. Momentum (RSI, MACD)
        2. Trend strength (ADX, Moving Average relationships)
        3. Volatility (ATR relative to price, Bollinger Band width)
        4. Volume profile (OBV, Volume relative to average)
        
        Args:
            df (pd.DataFrame): Price data with OHLCV data
            symbol (str): Symbol to calculate score for
            momentum_weight (float, optional): Weight for momentum component
            trend_weight (float, optional): Weight for trend component
            volatility_weight (float, optional): Weight for volume component
            volume_weight (float, optional): Weight for volume component
            
        Returns:
            dict: Dictionary with technical scores and components
        """
        # Use provided weights or fall back to instance variables
        momentum_weight = momentum_weight if momentum_weight is not None else self.momentum_weight
        trend_weight = trend_weight if trend_weight is not None else self.trend_weight
        volatility_weight = volatility_weight if volatility_weight is not None else self.volatility_weight
        volume_weight = volume_weight if volume_weight is not None else self.volume_weight
        
        # Initialize scores dictionary
        scores = {
            'symbol': symbol,
            'momentum_score': 0.5,
            'trend_score': 0.5,
            'volatility_score': 0.5,
            'volume_score': 0.5,
            'total_score': 0.5,
            'direction': 'NEUTRAL',
            'bullish_signals': 0,
            'bearish_signals': 0,
            'combined_score': 0.5
        }
        
        # Check if we have enough data
        if df is None or len(df) < 50:
            self.logger.warning(f"Insufficient data for technical scoring of {symbol}")
            return scores
            
        try:
            # Get the most recent data for analysis
            recent_data = df.iloc[-50:].copy()
            
            # Ensure data types are correct for TA-Lib
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in recent_data.columns:
                    recent_data[col] = recent_data[col].astype(float)
            
            # Calculate indicators if they don't exist
            # 1. RSI
            if 'rsi' not in recent_data.columns:
                if TALIB_AVAILABLE:
                    recent_data['rsi'] = talib.RSI(recent_data['close'].values, timeperiod=14)
                else:
                    recent_data['rsi'] = recent_data['close'].pct_change().rolling(window=14).mean()
                
            # 2. MACD
            if 'macd' not in recent_data.columns:
                if TALIB_AVAILABLE:
                    macd, macd_signal, macd_hist = talib.MACD(
                        recent_data['close'].values, 
                        fastperiod=12, 
                        slowperiod=26, 
                        signalperiod=9
                    )
                    recent_data['macd'] = macd
                    recent_data['macd_signal'] = macd_signal
                    recent_data['macd_hist'] = macd_hist
                else:
                    ema_12 = recent_data['close'].ewm(span=12, adjust=False).mean()
                    ema_26 = recent_data['close'].ewm(span=26, adjust=False).mean()
                    recent_data['macd'] = ema_12 - ema_26
                    recent_data['macd_signal'] = recent_data['macd'].ewm(span=9, adjust=False).mean()
            
            # 3. ADX
            if 'adx' not in recent_data.columns:
                if TALIB_AVAILABLE:
                    recent_data['adx'] = talib.ADX(
                        recent_data['high'].values,
                        recent_data['low'].values,
                        recent_data['close'].values,
                        timeperiod=14
                    )
                else:
                    recent_data['adx'] = recent_data['close'].pct_change().rolling(window=14).std()
            
            # 4. Moving Averages
            if 'sma20' not in recent_data.columns:
                if TALIB_AVAILABLE:
                    recent_data['sma20'] = talib.SMA(recent_data['close'].values, timeperiod=20)
                else:
                    recent_data['sma20'] = recent_data['close'].rolling(window=20).mean()
                
            if 'sma50' not in recent_data.columns:
                if TALIB_AVAILABLE:
                    recent_data['sma50'] = talib.SMA(recent_data['close'].values, timeperiod=50)
                else:
                    recent_data['sma50'] = recent_data['close'].rolling(window=50).mean()
                
            # 5. ATR
            if 'atr' not in recent_data.columns:
                if TALIB_AVAILABLE:
                    recent_data['atr'] = talib.ATR(
                        recent_data['high'].values,
                        recent_data['low'].values,
                        recent_data['close'].values,
                        timeperiod=14
                    )
                else:
                    recent_data['hl'] = recent_data['high'] - recent_data['low']
                    recent_data['hc'] = (recent_data['high'] - recent_data['close'].shift(1)).abs()
                    recent_data['lc'] = (recent_data['low'] - recent_data['close'].shift(1)).abs()
                    recent_data['tr'] = recent_data[['hl', 'hc', 'lc']].max(axis=1)
                    recent_data['atr'] = recent_data['tr'].rolling(window=14).mean()
            
            # 6. Bollinger Bands
            if 'bb_width' not in recent_data.columns:
                if TALIB_AVAILABLE:
                    upper, middle, lower = talib.BBANDS(
                        recent_data['close'].values, 
                        timeperiod=20, 
                        nbdevup=2, 
                        nbdevdn=2
                    )
                    recent_data['bb_upper'] = upper
                    recent_data['bb_middle'] = middle
                    recent_data['bb_lower'] = lower
                    recent_data['bb_width'] = (recent_data['bb_upper'] - recent_data['bb_lower']) / recent_data['bb_middle']
                else:
                    recent_data['bb_middle'] = recent_data['close'].rolling(window=20).mean()
                    recent_data['bb_std'] = recent_data['close'].rolling(window=20).std()
                    recent_data['bb_upper'] = recent_data['bb_middle'] + 2 * recent_data['bb_std']
                    recent_data['bb_lower'] = recent_data['bb_middle'] - 2 * recent_data['bb_std']
                    recent_data['bb_width'] = (recent_data['bb_upper'] - recent_data['bb_lower']) / recent_data['bb_middle']
            
            # 7. OBV (On-Balance Volume)
            if 'obv' not in recent_data.columns:
                if TALIB_AVAILABLE:
                    recent_data['obv'] = talib.OBV(recent_data['close'].values, recent_data['volume'].values)
                else:
                    recent_data['obv'] = recent_data['volume'] * np.where(recent_data['close'] > recent_data['close'].shift(1), 1, -1)
                    recent_data['obv'] = recent_data['obv'].cumsum()
            
            # Get the latest values (handle potential NaN values)
            latest_idx = -1
            while latest_idx >= -len(recent_data):
                latest = recent_data.iloc[latest_idx]
                if not np.isnan(latest['close']):
                    break
                latest_idx -= 1
                
            prev_idx = latest_idx - 1
            if prev_idx < -len(recent_data):
                prev_idx = -len(recent_data)
            prev = recent_data.iloc[prev_idx]
            
            # Calculate momentum score (0-1 scale)
            # RSI: >70 overbought, <30 oversold
            rsi_score = 0.5
            if 'rsi' in latest and not np.isnan(latest['rsi']):
                if latest['rsi'] > 70:
                    rsi_score = 0.2  # Overbought - bearish
                    scores['bearish_signals'] += 1
                elif latest['rsi'] < 30:
                    rsi_score = 0.8  # Oversold - bullish
                    scores['bullish_signals'] += 1
                else:
                    # Linear scale between 30-70
                    rsi_score = 0.8 - ((latest['rsi'] - 30) / 40) * 0.6
                    
            # MACD: Positive and rising is bullish
            macd_score = 0.5
            if 'macd' in latest and 'macd_signal' in latest and not np.isnan(latest['macd']) and not np.isnan(latest['macd_signal']):
                # MACD above signal line is bullish
                if latest['macd'] > latest['macd_signal']:
                    macd_score = 0.7
                    scores['bullish_signals'] += 1
                    # Even more bullish if it just crossed above
                    if prev['macd'] <= prev['macd_signal']:
                        macd_score = 0.9
                        scores['bullish_signals'] += 1
                else:
                    macd_score = 0.3
                    scores['bearish_signals'] += 1
                    # Even more bearish if it just crossed below
                    if prev['macd'] >= prev['macd_signal']:
                        macd_score = 0.1
                        scores['bearish_signals'] += 1
                        
            # Combined momentum score
            scores['momentum_score'] = (rsi_score + macd_score) / 2
            
            # Calculate trend score
            # ADX: >25 indicates strong trend
            adx_score = 0.5
            if 'adx' in latest and not np.isnan(latest['adx']):
                if latest['adx'] > 25:
                    adx_score = 0.8  # Strong trend
                elif latest['adx'] < 15:
                    adx_score = 0.2  # Weak trend
                else:
                    # Linear scale between 15-25
                    adx_score = 0.2 + ((latest['adx'] - 15) / 10) * 0.6
            
            # Moving Average relationships
            ma_score = 0.5
            if 'sma20' in latest and 'sma50' in latest and not np.isnan(latest['sma20']) and not np.isnan(latest['sma50']):
                # Price above both MAs is bullish
                if latest['close'] > latest['sma20'] and latest['close'] > latest['sma50']:
                    ma_score = 0.8
                    scores['bullish_signals'] += 1
                # Price below both MAs is bearish
                elif latest['close'] < latest['sma20'] and latest['close'] < latest['sma50']:
                    ma_score = 0.2
                    scores['bearish_signals'] += 1
                # 20MA > 50MA is bullish
                elif latest['sma20'] > latest['sma50']:
                    ma_score = 0.6
                # 20MA < 50MA is bearish
                else:
                    ma_score = 0.4
            
            # Combined trend score
            scores['trend_score'] = (adx_score + ma_score) / 2
            
            # Calculate volatility score
            # ATR relative to price
            atr_score = 0.5
            if 'atr' in latest and not np.isnan(latest['atr']) and latest['close'] > 0:
                atr_pct = (latest['atr'] / latest['close']) * 100
                if atr_pct > 3:
                    atr_score = 0.8  # High volatility
                elif atr_pct < 1:
                    atr_score = 0.2  # Low volatility
                else:
                    # Linear scale between 1-3%
                    atr_score = 0.2 + ((atr_pct - 1) / 2) * 0.6
            
            # Bollinger Band width
            bb_score = 0.5
            if 'bb_width' in latest and not np.isnan(latest['bb_width']):
                if latest['bb_width'] > 0.1:
                    bb_score = 0.8  # Wide bands - high volatility
                elif latest['bb_width'] < 0.03:
                    bb_score = 0.2  # Narrow bands - low volatility
                else:
                    # Linear scale between 0.03-0.1
                    bb_score = 0.2 + ((latest['bb_width'] - 0.03) / 0.07) * 0.6
            
            # Combined volatility score
            scores['volatility_score'] = (atr_score + bb_score) / 2
            
            # Calculate volume score
            # Volume relative to average
            vol_ratio_score = 0.5
            if 'volume_sma' in recent_data.columns and not np.isnan(latest['volume']) and not np.isnan(latest['volume_sma']) and latest['volume_sma'] > 0:
                vol_ratio = latest['volume'] / latest['volume_sma']
                if vol_ratio > 2:
                    vol_ratio_score = 0.9  # Very high volume
                elif vol_ratio > 1.5:
                    vol_ratio_score = 0.7  # High volume
                elif vol_ratio < 0.5:
                    vol_ratio_score = 0.3  # Low volume
                else:
                    # Linear scale between 0.5-1.5
                    vol_ratio_score = 0.3 + ((vol_ratio - 0.5) / 1.0) * 0.4
            
            # OBV trend
            obv_score = 0.5
            if 'obv' in recent_data.columns:
                # Calculate OBV slope over last 5 days
                if len(recent_data) >= 5:
                    obv_5d = recent_data['obv'].iloc[-5:].values
                    if not np.isnan(obv_5d).any():
                        obv_slope = np.polyfit(range(5), obv_5d, 1)[0]
                        if obv_slope > 0:
                            obv_score = 0.7  # Rising OBV is bullish
                            scores['bullish_signals'] += 1
                        else:
                            obv_score = 0.3  # Falling OBV is bearish
                            scores['bearish_signals'] += 1
            
            # Combined volume score
            scores['volume_score'] = (vol_ratio_score + obv_score) / 2
            
            # Calculate overall technical score
            # Weight the components based on configuration
            scores['total_score'] = (
                scores['momentum_score'] * momentum_weight +
                scores['trend_score'] * trend_weight +
                scores['volatility_score'] * volatility_weight +
                scores['volume_score'] * volume_weight
            ) / (momentum_weight + trend_weight + volatility_weight + volume_weight)
            
            # Determine direction based on signals
            if scores['bullish_signals'] > scores['bearish_signals'] + 1:
                scores['direction'] = 'LONG'
            elif scores['bearish_signals'] > scores['bullish_signals'] + 1:
                scores['direction'] = 'SHORT'
            else:
                scores['direction'] = 'NEUTRAL'
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Error calculating technical score for {symbol}: {e}")
            return scores
            
    def select_stocks_multi_factor(self, symbol_data, current_date=None, top_n=None, direction='ANY', market_regime=None):
        """
        Select stocks based on a multi-factor approach combining technical and seasonality scores.
        
        This method combines:
        1. Seasonality scores (historical performance for the time period)
        2. Technical indicators (momentum, trend, volatility, volume)
        3. Market regime alignment
        4. Sector rotation analysis
        5. Short opportunity identification
        6. Dynamic position sizing
        
        Args:
            symbol_data (dict): Dictionary of symbol -> dataframe with historical price data
            current_date (datetime, optional): Current date for seasonality calculations
            top_n (int, optional): Number of top stocks to select
            direction (str, optional): Direction filter ('LONG', 'SHORT', 'ANY')
            market_regime (MarketRegime, optional): Current market regime for regime-specific adjustments
            
        Returns:
            list: List of dictionaries with score data for selected stocks
        """
        if current_date is None:
            current_date = datetime.now()
        
        # Get configuration parameters
        config = self.config.get('stock_selection', {})
        if top_n is None:
            top_n = config.get('top_n_stocks', 10)
        technical_weight = config.get('technical_weight', 0.7)
        seasonality_weight = config.get('seasonality_weight', 0.3)
        
        # Check if seasonality weight adjustment is enabled
        seasonality_config = self.config.get('seasonality', {})
        weight_adjustment = seasonality_config.get('weight_adjustment', False)
        min_score_threshold = seasonality_config.get('min_score_threshold', 0.6)
        
        # Adjust weights based on market regime if available
        if market_regime:
            technical_weight, seasonality_weight = self.adjust_factor_weights_by_regime(
                market_regime, technical_weight, seasonality_weight
            )
            self.logger.debug(f"Adjusted weights for {market_regime.name}: Technical={technical_weight}, Seasonality={seasonality_weight}")
        
        # Calculate technical and seasonality scores for each symbol
        stock_scores = []
        
        for symbol, df in symbol_data.items():
            if df is None or df.empty:
                continue
                
            try:
                # Calculate technical score
                self.logger.debug(f"Calculating technical score for {symbol}")
                technical_scores = self.calculate_technical_score(df, symbol)
                technical_score = technical_scores.get('total_score', 0.5)
                technical_direction = technical_scores.get('direction', 'NEUTRAL')
                
                # Apply direction bias to technical score
                self.logger.debug(f"Technical score for {symbol}: {technical_score:.4f}, Direction: {technical_direction}")
                
                # Calculate seasonality score if enabled
                seasonal_score = 0.5  # Neutral by default
                seasonality_confidence = 0.0
                
                if self.use_seasonality and self.seasonality_analyzer:
                    try:
                        seasonal_score = self.seasonality_analyzer.get_seasonal_score(symbol, current_date)
                        self.logger.debug(f"Seasonality score for {symbol} on {current_date.strftime('%Y-%m-%d')}: {seasonal_score:.4f}")
                        
                        # Calculate seasonality confidence
                        seasonality_confidence = self.calculate_seasonality_confidence(symbol, current_date)
                        
                        # Apply minimum score threshold if enabled
                        if weight_adjustment and seasonal_score < min_score_threshold:
                            # Reduce seasonality weight for low scores
                            adjusted_seasonality_weight = seasonality_weight * (seasonal_score / min_score_threshold)
                            self.logger.debug(f"Reducing seasonality weight for {symbol} from {seasonality_weight:.2f} to {adjusted_seasonality_weight:.2f}")
                        else:
                            adjusted_seasonality_weight = seasonality_weight
                            
                        # Apply confidence factor to seasonality weight
                        adjusted_seasonality_weight *= seasonality_confidence
                        adjusted_technical_weight = 1.0 - adjusted_seasonality_weight
                    except Exception as e:
                        self.logger.warning(f"Error getting seasonality score for {symbol}: {e}")
                        adjusted_technical_weight = technical_weight
                        adjusted_seasonality_weight = seasonality_weight
                else:
                    # No seasonality, use technical only
                    adjusted_technical_weight = 1.0
                    adjusted_seasonality_weight = 0.0
                
                # Apply sector boost/penalty based on leading sectors
                sector_score_boost = 0
                sector = self.get_sector_for_symbol(symbol)
                
                # Calculate combined score with adjusted weights
                combined_score = (technical_score * adjusted_technical_weight) + (seasonal_score * adjusted_seasonality_weight)
                
                # Apply final adjustments based on market regime
                if market_regime:
                    combined_score = self.adjust_final_score_by_regime(market_regime, combined_score, technical_direction)
                
                # Calculate position size based on score and direction
                position_size = self.calculate_dynamic_position_size(combined_score, technical_direction)
                
                # Add to stock scores
                stock_scores.append({
                    'symbol': symbol,
                    'combined_score': combined_score,
                    'technical_score': technical_score,
                    'seasonal_score': seasonal_score,
                    'seasonality_confidence': seasonality_confidence,
                    'technical_weight': adjusted_technical_weight,
                    'seasonality_weight': adjusted_seasonality_weight,
                    'sector_boost': sector_score_boost,
                    'technical_direction': technical_direction,
                    'direction': technical_direction,  # Add direction field for compatibility
                    'momentum_score': technical_scores.get('momentum_score', 0),
                    'trend_score': technical_scores.get('trend_score', 0),
                    'volatility_score': technical_scores.get('volatility_score', 0),
                    'volume_score': technical_scores.get('volume_score', 0),
                    'position_size': position_size
                })
            except Exception as e:
                self.logger.error(f"Error calculating scores for {symbol}: {e}")
                continue
        
        # Filter by direction if specified
        if direction != 'ANY':
            stock_scores = [s for s in stock_scores if s['direction'] == direction]
        
        # Sort by combined score (descending) and take top N
        sorted_scores = sorted(stock_scores, key=lambda x: x['combined_score'], reverse=True)
        
        # Ensure we have a mix of long and short positions if direction is ANY
        if direction == 'ANY':
            # Get top long and short positions
            long_positions = [s for s in sorted_scores if s['direction'] == 'LONG']
            short_positions = [s for s in sorted_scores if s['direction'] == 'SHORT']
            
            # Ensure we have at least some short positions (if available)
            if short_positions:
                target_short_count = max(1, int(top_n * 0.2))  # Aim for at least 20% shorts
                current_short_count = len([s for s in sorted_scores[:top_n] if s['direction'] == 'SHORT'])
                
                if current_short_count < target_short_count:
                    # Add more shorts to ensure diversity
                    shorts_to_add = target_short_count - current_short_count
                    # Get shorts that aren't already in the top_n
                    additional_shorts = [s for s in short_positions if s not in sorted_scores[:top_n]][:shorts_to_add]
                    
                    if additional_shorts:
                        self.logger.info(f"Adding {len(additional_shorts)} additional short positions for diversification")
                        # Replace lowest-scoring stocks with shorts
                        sorted_scores = sorted_scores[:top_n - len(additional_shorts)] + additional_shorts
        
        # Take top N scores
        top_scores = sorted_scores[:top_n]
        
        self.logger.info(f"Selected {len(top_scores)} stocks using multi-factor approach")
        self.logger.info(f"Sample selected stocks: {list(top_scores)[:3]}")
        self.logger.info(f"Sample combined scores: {[s['combined_score'] for s in top_scores[:3]]}")
        self.logger.info(f"Sample technical scores: {[s['technical_score'] for s in top_scores[:3]]}")
        self.logger.info(f"Sample seasonality scores: {[s['seasonal_score'] for s in top_scores[:3]]}")
        self.logger.info(f"Sample seasonality confidence: {[s['seasonality_confidence'] for s in top_scores[:3]]}")
        self.logger.info(f"Sample technical weights: {[s['technical_weight'] for s in top_scores[:3]]}")
        self.logger.info(f"Sample seasonality weights: {[s['seasonality_weight'] for s in top_scores[:3]]}")
        self.logger.info(f"Sample sector boosts: {[s['sector_boost'] for s in top_scores[:3]]}")
        self.logger.info(f"Sample technical directions: {[s['technical_direction'] for s in top_scores[:3]]}")
        self.logger.info(f"Sample directions: {[s['direction'] for s in top_scores[:3]]}")
        self.logger.info(f"Sample momentum scores: {[s['momentum_score'] for s in top_scores[:3]]}")
        self.logger.info(f"Sample trend scores: {[s['trend_score'] for s in top_scores[:3]]}")
        self.logger.info(f"Sample volatility scores: {[s['volatility_score'] for s in top_scores[:3]]}")
        self.logger.info(f"Sample volume scores: {[s['volume_score'] for s in top_scores[:3]]}")
        self.logger.info(f"Sample position sizes: {[s['position_size'] for s in top_scores[:3]]}")
        
        return top_scores
    
    def calculate_dynamic_position_size(self, score, direction):
        """
        Calculate dynamic position size based on score and direction.
        
        Args:
            score (float): Combined score (0-1)
            direction (str): Trade direction ('LONG', 'SHORT', 'NEUTRAL')
            
        Returns:
            float: Position size as a percentage (0-1)
        """
        if 'position_sizing' not in self.config['stock_selection']:
            # Use default position size if no dynamic sizing config
            return self.config['general'].get('position_size_pct', 0.05)
        
        # Get position sizing parameters
        base_position = self.config['stock_selection']['position_sizing'].get('base_position_pct', 0.05)
        min_position = self.config['stock_selection']['position_sizing'].get('min_position_pct', 0.02)
        max_position = self.config['stock_selection']['position_sizing'].get('max_position_pct', 0.08)
        scaling_factor = self.config['stock_selection']['position_sizing'].get('score_scaling_factor', 0.5)
        
        # Normalize score to 0.5-1.0 range (assuming minimum viable score is 0.5)
        normalized_score = (score - 0.5) * 2 if score > 0.5 else 0
        
        # Calculate position size
        position_size = base_position + (normalized_score * scaling_factor * (max_position - base_position))
        
        # Apply min/max bounds
        position_size = max(min_position, min(max_position, position_size))
        
        # Adjust for direction (optional - can be used to reduce size for shorts)
        if direction == 'SHORT':
            # Optionally reduce short position sizes
            short_size_factor = self.config['stock_selection']['position_sizing'].get('short_size_factor', 0.8)
            position_size *= short_size_factor
        
        return position_size
    
    def adjust_final_score_by_regime(self, regime, score, direction):
        """
        Apply final adjustments to the combined score based on market regime and direction.
        
        Args:
            regime (MarketRegime): Current market regime
            score (float): Combined score
            direction (str): Trade direction ('LONG', 'SHORT', 'NEUTRAL')
            
        Returns:
            float: Adjusted score
        """
        # No adjustment for neutral regime or direction
        if regime == MarketRegime.MIXED or direction == 'NEUTRAL':
            return score
        
        # In trending markets, boost scores for trades aligned with the trend
        if regime == MarketRegime.TRENDING:
            # Determine market trend direction using SPY
            market_trend = 'NEUTRAL'
            if 'SPY' in self.symbol_data:
                spy_df = self.symbol_data['SPY']
                if len(spy_df) >= 20:
                    # Simple trend detection using 20-day EMA
                    ema20 = talib.EMA(spy_df['close'].values, timeperiod=20)
                    if spy_df['close'].iloc[-1] > ema20[-1]:
                        market_trend = 'LONG'
                    else:
                        market_trend = 'SHORT'
            
            # Boost scores for trades aligned with market trend
            if market_trend == direction:
                return min(score * 1.1, 1.0)  # 10% boost, capped at 1.0
        
        # In range-bound markets, boost mean-reversion trades
        elif regime == MarketRegime.RANGE_BOUND:
            # Determine if this is a mean-reversion trade
            is_mean_reversion = False
            
            # For longs, look for oversold conditions
            if direction == 'LONG':
                # Check RSI if available
                if 'rsi' in self.symbol_data[list(self.symbol_data.keys())[0]].columns:
                    symbol_df = self.symbol_data[list(self.symbol_data.keys())[0]]
                    rsi = symbol_df['rsi'].iloc[-1]
                    if rsi < 40:  # Oversold
                        is_mean_reversion = True
            
            # For shorts, look for overbought conditions
            elif direction == 'SHORT':
                # Check RSI if available
                if 'rsi' in self.symbol_data[list(self.symbol_data.keys())[0]].columns:
                    symbol_df = self.symbol_data[list(self.symbol_data.keys())[0]]
                    rsi = symbol_df['rsi'].iloc[-1]
                    if rsi > 60:  # Overbought
                        is_mean_reversion = True
            
            # Boost scores for mean-reversion trades
            if is_mean_reversion:
                return min(score * 1.15, 1.0)  # 15% boost, capped at 1.0
        
        return score
    
    def adjust_factor_weights_by_regime(self, regime, technical_weight, seasonality_weight):
        """
        Adjust the weights of technical and seasonality factors based on market regime.
        
        Args:
            regime (MarketRegime): Current market regime
            technical_weight (float): Base technical weight
            seasonality_weight (float): Base seasonality weight
            
        Returns:
            tuple: Adjusted (technical_weight, seasonality_weight)
        """
        # Get configuration parameters for weight adjustments
        regime_adjustments = self.config.get('stock_selection', {}).get('regime_adjustments', {})
        
        # Default adjustments if not specified in config
        trending_tech_multiplier = regime_adjustments.get('trending', {}).get('technical_weight_multiplier', 1.2)
        trending_season_multiplier = regime_adjustments.get('trending', {}).get('seasonality_weight_multiplier', 0.7)
        
        range_tech_multiplier = regime_adjustments.get('range_bound', {}).get('technical_weight_multiplier', 0.9)
        range_season_multiplier = regime_adjustments.get('range_bound', {}).get('seasonality_weight_multiplier', 1.1)
        
        # Base weights should be higher for technical since it performed better
        base_technical_weight = max(technical_weight, 0.75)  # Ensure technical weight is at least 75%
        base_seasonality_weight = 1.0 - base_technical_weight  # Adjust seasonality accordingly
        
        if regime == MarketRegime.TRENDING:
            # In trending markets, increase the weight of technical factors
            technical_adj = min(base_technical_weight * trending_tech_multiplier, 0.9)
            seasonality_adj = 1.0 - technical_adj
        elif regime == MarketRegime.RANGE_BOUND:
            # In range-bound markets, give a slight boost to seasonality factors
            technical_adj = max(base_technical_weight * range_tech_multiplier, 0.65)
            seasonality_adj = 1.0 - technical_adj
        else:  # MIXED or None
            # Use base weights with slight technical bias
            technical_adj = base_technical_weight
            seasonality_adj = base_seasonality_weight
        
        self.logger.debug(f"Adjusted weights for {regime}: Technical={technical_adj:.2f}, Seasonality={seasonality_adj:.2f}")
        return technical_adj, seasonality_adj
    
    def calculate_signal_score(self, signal):
        """Calculate a score for a signal based on its properties
        
        Args:
            signal (dict): Signal dictionary
            
        Returns:
            float: Signal score between 0 and 1
        """
        # Start with base score from signal strength
        if 'strength_value' in signal:
            score = signal['strength_value']
        elif 'strength' in signal:
            # Map strength to value
            strength_map = {
                'weak': 0.3,
                'moderate': 0.6,
                'strong': 0.9
            }
            score = strength_map.get(signal['strength'], 0.5)
        else:
            # Default score
            score = 0.5
        
        # Adjust score based on strategy
        if 'strategy' in signal:
            if signal['strategy'] == 'mean_reversion':
                # Slightly favor mean reversion in this implementation
                score *= 1.1
            elif signal['strategy'] == 'trend_following':
                # Slightly reduce trend following
                score *= 0.9
        
        # Apply weight if available
        if 'weight' in signal:
            score *= signal['weight']
        
        # Ensure score is between 0 and 1
        score = max(0.0, min(1.0, score))
        
        return score

    def calculate_seasonality_confidence(self, symbol, current_date):
        """
        Calculate a confidence factor for seasonality data based on historical data quality.
        
        Args:
            symbol (str): Symbol to calculate confidence for
            current_date (datetime): Current date for seasonality calculations
            
        Returns:
            float: Confidence factor between 0.0 and 1.0
        """
        if not self.use_seasonality or not self.seasonality_analyzer:
            return 0.0
            
        try:
            # Get seasonality data quality metrics
            data_points = self.seasonality_analyzer.get_data_points_count(symbol, current_date)
            consistency = self.seasonality_analyzer.get_seasonal_consistency(symbol, current_date)
            
            # Calculate confidence based on data points and consistency
            # More data points and higher consistency = higher confidence
            data_points_factor = min(data_points / 5.0, 1.0)  # At least 5 years of data for full confidence
            
            # If we have consistency data, use it
            if consistency is not None:
                confidence = (data_points_factor * 0.6) + (consistency * 0.4)
            else:
                confidence = data_points_factor
                
            # Apply minimum confidence threshold from config
            min_confidence = self.config.get('seasonality', {}).get('min_confidence', 0.3)
            confidence = max(confidence, min_confidence)
            
            self.logger.debug(f"Seasonality confidence for {symbol}: {confidence:.2f} (data points: {data_points}, consistency: {consistency})")
            return confidence
        except Exception as e:
            self.logger.warning(f"Error calculating seasonality confidence for {symbol}: {e}")
            return 0.3  # Default moderate confidence
