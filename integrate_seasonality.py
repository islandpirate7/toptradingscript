#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Seasonality Integration Module
-------------------------------------
This module integrates seasonality analysis with the existing trading strategies,
enhancing signal generation and filtering based on historical seasonal patterns.
"""

import os
import logging
import yaml
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class SeasonalityIntegrator:
    """Class to integrate seasonality analysis with existing strategies"""
    
    def __init__(self, seasonality_file: str, config_file: str = None):
        """Initialize the seasonality integrator
        
        Args:
            seasonality_file (str): Path to seasonality opportunities YAML file
            config_file (str, optional): Path to configuration file. Defaults to None.
        """
        self.seasonality_data = self._load_seasonality_data(seasonality_file)
        self.config = self._load_config(config_file)
        self.seasonal_boost = self.config.get('seasonal_boost', 0.2)
        self.seasonal_penalty = self.config.get('seasonal_penalty', 0.2)
        
        logging.info(f"Initialized SeasonalityIntegrator with {len(self.seasonality_data)} seasonal patterns")
        
    def _load_seasonality_data(self, file_path: str) -> Dict:
        """Load seasonality data from YAML file, handling NumPy scalar values
        
        Args:
            file_path (str): Path to seasonality YAML file
            
        Returns:
            Dict: Dictionary mapping symbols to their seasonal patterns
        """
        seasonality_map = {}
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                logging.warning(f"Seasonality file {file_path} does not exist")
                return {}
                
            # First try to read the file directly as text
            with open(file_path, 'r') as f:
                file_content = f.read()
                
            # Parse the data manually to extract the opportunities
            import re
            
            # Find all opportunities in the file
            opportunity_blocks = re.findall(r'- .*?(?=- |$)', file_content, re.DOTALL)
            
            for block in opportunity_blocks:
                # Extract key information from each opportunity block
                symbol_match = re.search(r'symbol: ([A-Z]+)', block)
                season_match = re.search(r'season: ([A-Za-z]+)', block)
                direction_match = re.search(r'direction: ([A-Z]+)', block)
                win_rate_match = re.search(r'win_rate: ([\d\.]+)', block)
                avg_return_match = re.search(r'avg_return: ([-\d\.]+)', block)
                correlation_match = re.search(r'correlation: ([-\d\.]+)', block)
                
                if symbol_match and season_match:
                    symbol = symbol_match.group(1)
                    season = season_match.group(1)
                    direction = direction_match.group(1) if direction_match else 'LONG'
                    win_rate = float(win_rate_match.group(1)) if win_rate_match else 0.0
                    avg_return = float(avg_return_match.group(1)) if avg_return_match else 0.0
                    correlation = float(correlation_match.group(1)) if correlation_match else 0.0
                    
                    if symbol not in seasonality_map:
                        seasonality_map[symbol] = []
                        
                    seasonality_map[symbol].append({
                        'season': season,
                        'direction': direction,
                        'win_rate': win_rate,
                        'avg_return': avg_return,
                        'correlation': correlation,
                        'score': win_rate * avg_return  # Combined score for ranking
                    })
            
            # If we couldn't extract any data, try an alternative approach
            if not seasonality_map:
                logging.warning("Couldn't parse seasonality data using regex, trying alternative approach")
                
                # Try to convert YAML to JSON and then parse
                try:
                    # Create a temporary file with the YAML content converted to JSON
                    temp_file = file_path + '.json'
                    
                    # Use pandas to read the YAML file as it handles NumPy types better
                    df = pd.read_csv(file_path, sep=':', header=None, names=['key', 'value'])
                    
                    # Extract opportunities
                    current_opp = {}
                    for idx, row in df.iterrows():
                        key = row['key'].strip() if isinstance(row['key'], str) else None
                        value = row['value']
                        
                        if key == 'symbol':
                            if current_opp and 'symbol' in current_opp:
                                # Save previous opportunity
                                symbol = current_opp['symbol']
                                if symbol not in seasonality_map:
                                    seasonality_map[symbol] = []
                                seasonality_map[symbol].append(current_opp)
                                
                            # Start new opportunity
                            current_opp = {'symbol': value}
                        elif key in ['season', 'direction']:
                            current_opp[key] = value
                        elif key in ['win_rate', 'avg_return', 'correlation']:
                            try:
                                current_opp[key] = float(value)
                            except (ValueError, TypeError):
                                current_opp[key] = 0.0
                    
                    # Add the last opportunity
                    if current_opp and 'symbol' in current_opp:
                        symbol = current_opp['symbol']
                        if symbol not in seasonality_map:
                            seasonality_map[symbol] = []
                        seasonality_map[symbol].append(current_opp)
                        
                except Exception as e:
                    logging.error(f"Error in alternative parsing approach: {e}")
            
            logging.info(f"Loaded seasonality data for {len(seasonality_map)} symbols")
            return seasonality_map
            
        except Exception as e:
            logging.error(f"Error loading seasonality data: {e}")
            return {}
    
    def _load_config(self, file_path: str) -> Dict:
        """Load configuration from YAML file
        
        Args:
            file_path (str): Path to configuration file
            
        Returns:
            Dict: Configuration dictionary
        """
        default_config = {
            'seasonal_boost': 0.2,      # Boost factor for signals aligned with seasonal patterns
            'seasonal_penalty': 0.2,    # Penalty factor for signals against seasonal patterns
            'min_win_rate': 0.55,       # Minimum win rate to consider a seasonal pattern
            'min_avg_return': 0.2,      # Minimum average return to consider a seasonal pattern
            'seasonality_weight': 0.3,  # Weight of seasonality in the combined strategy
        }
        
        if not file_path:
            return default_config
            
        try:
            with open(file_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Update default config with loaded values
            default_config.update(config)
            logging.info(f"Loaded configuration from {file_path}")
            
            return default_config
            
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            return default_config
    
    def get_current_season(self, date: Optional[datetime] = None) -> Dict:
        """Get the current season information
        
        Args:
            date (datetime, optional): Date to get season for. Defaults to current date.
            
        Returns:
            Dict: Current season information
        """
        if date is None:
            date = datetime.now()
            
        # Get day of week
        day_of_week = date.strftime('%A')
        
        # Get day of month
        day_of_month = str(date.day)
        
        # Get month
        month = date.strftime('%B')
        
        # Get quarter
        quarter = f"Q{(date.month - 1) // 3 + 1}"
        
        # Get half year
        half_year = f"H{(date.month - 1) // 6 + 1}"
        
        return {
            'day_of_week': day_of_week,
            'day_of_month': day_of_month,
            'month': month,
            'quarter': quarter,
            'half_year': half_year
        }
    
    def check_seasonal_alignment(self, symbol: str, direction: str, 
                               date: Optional[datetime] = None) -> Tuple[bool, float]:
        """Check if a signal aligns with seasonal patterns
        
        Args:
            symbol (str): Stock symbol
            direction (str): Signal direction ('LONG' or 'SHORT')
            date (datetime, optional): Date to check. Defaults to current date.
            
        Returns:
            Tuple[bool, float]: (is_aligned, alignment_score)
        """
        if symbol not in self.seasonality_data:
            return False, 0.0
            
        current_season = self.get_current_season(date)
        seasonal_patterns = self.seasonality_data[symbol]
        
        # Check alignment with each seasonal pattern
        for pattern in seasonal_patterns:
            pattern_season = pattern['season']
            pattern_direction = pattern['direction']
            
            # Check if current date matches the pattern season
            is_matching_season = False
            
            # Check month
            if pattern_season in current_season['month']:
                is_matching_season = True
            # Check day of week
            elif pattern_season in current_season['day_of_week']:
                is_matching_season = True
            # Check day of month
            elif pattern_season == current_season['day_of_month']:
                is_matching_season = True
            # Check quarter
            elif pattern_season == current_season['quarter']:
                is_matching_season = True
            # Check half year
            elif pattern_season == current_season['half_year']:
                is_matching_season = True
                
            if is_matching_season and pattern_direction == direction:
                # Calculate alignment score based on win rate and average return
                alignment_score = pattern['score']
                return True, alignment_score
                
        return False, 0.0
    
    def adjust_signal_score(self, signal: Dict) -> Dict:
        """Adjust signal score based on seasonality
        
        Args:
            signal (Dict): Signal dictionary
            
        Returns:
            Dict: Adjusted signal dictionary
        """
        symbol = signal['symbol']
        direction = signal['direction']
        original_score = signal.get('score', 0.5)
        
        # Check seasonal alignment
        is_aligned, alignment_score = self.check_seasonal_alignment(symbol, direction)
        
        # Adjust score based on alignment
        if is_aligned:
            # Boost score for aligned signals
            adjusted_score = original_score * (1 + self.seasonal_boost * alignment_score)
            seasonal_factor = f"Boosted by {self.seasonal_boost * alignment_score:.2f}"
        else:
            # Penalize score for non-aligned signals
            adjusted_score = original_score * (1 - self.seasonal_penalty)
            seasonal_factor = f"Penalized by {self.seasonal_penalty:.2f}"
        
        # Update signal
        signal['original_score'] = original_score
        signal['score'] = min(adjusted_score, 1.0)  # Cap at 1.0
        signal['seasonal_alignment'] = is_aligned
        signal['seasonal_factor'] = seasonal_factor
        
        return signal
    
    def filter_signals_by_seasonality(self, signals: List[Dict], 
                                    min_score: float = 0.5) -> List[Dict]:
        """Filter signals based on seasonality and adjusted scores
        
        Args:
            signals (List[Dict]): List of signal dictionaries
            min_score (float, optional): Minimum score to keep. Defaults to 0.5.
            
        Returns:
            List[Dict]: Filtered signals
        """
        # Adjust scores for all signals
        adjusted_signals = [self.adjust_signal_score(signal) for signal in signals]
        
        # Filter by minimum score
        filtered_signals = [signal for signal in adjusted_signals if signal['score'] >= min_score]
        
        logging.info(f"Filtered {len(signals)} signals to {len(filtered_signals)} based on seasonality")
        
        return filtered_signals
    
    def create_combined_config(self, base_config_file: str, output_file: str) -> None:
        """Create a combined strategy configuration incorporating seasonality
        
        Args:
            base_config_file (str): Path to base configuration file
            output_file (str): Path to output combined configuration file
        """
        try:
            # Load base configuration
            with open(base_config_file, 'r') as f:
                base_config = yaml.safe_load(f)
                
            # Add seasonality configuration
            if 'strategy_configs' not in base_config:
                base_config['strategy_configs'] = {}
                
            if 'Combined' not in base_config['strategy_configs']:
                base_config['strategy_configs']['Combined'] = {}
                
            # Add seasonality parameters
            base_config['strategy_configs']['Combined']['use_seasonality'] = True
            base_config['strategy_configs']['Combined']['seasonality_weight'] = self.config['seasonality_weight']
            base_config['strategy_configs']['Combined']['seasonal_boost'] = self.seasonal_boost
            base_config['strategy_configs']['Combined']['seasonal_penalty'] = self.seasonal_penalty
            
            # Add symbols with seasonal patterns to universe
            seasonal_symbols = list(self.seasonality_data.keys())
            
            if 'universe' in base_config:
                # Add seasonal symbols to existing universe
                base_config['universe'] = list(set(base_config['universe'] + seasonal_symbols))
            else:
                base_config['universe'] = seasonal_symbols
                
            # Save combined configuration
            with open(output_file, 'w') as f:
                yaml.dump(base_config, f)
                
            logging.info(f"Created combined configuration at {output_file}")
            
        except Exception as e:
            logging.error(f"Error creating combined configuration: {e}")

def main():
    """Main function to integrate seasonality with existing strategies"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Integrate seasonality with existing strategies')
    parser.add_argument('--seasonality', type=str, default='output/seasonal_opportunities.yaml',
                      help='Path to seasonality opportunities file')
    parser.add_argument('--config', type=str, default='configuration_mean_reversion_final.yaml',
                      help='Path to base configuration file')
    parser.add_argument('--output', type=str, default='configuration_combined_strategy_seasonal.yaml',
                      help='Path to output combined configuration file')
    args = parser.parse_args()
    
    # Initialize integrator
    integrator = SeasonalityIntegrator(args.seasonality)
    
    # Create combined configuration
    integrator.create_combined_config(args.config, args.output)
    
    logging.info(f"Seasonality integration completed successfully")

if __name__ == "__main__":
    main()
