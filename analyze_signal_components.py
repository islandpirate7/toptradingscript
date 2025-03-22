#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analyze Signal Components
This script analyzes which technical indicators are contributing to losing SHORT signals
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from final_sp500_strategy import SP500Strategy, load_alpaca_credentials

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"signal_analysis_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SignalAnalyzer:
    def __init__(self, mode='paper'):
        """Initialize the signal analyzer"""
        # Load credentials
        credentials = load_alpaca_credentials(mode)
        self.api_key = credentials['api_key']
        self.api_secret = credentials['api_secret']
        self.base_url = credentials['base_url']
        
        # Initialize API
        self.api = tradeapi.REST(
            self.api_key,
            self.api_secret,
            self.base_url,
            api_version='v2'
        )
        
        # Initialize strategy for technical indicators
        # SP500Strategy takes use_live parameter, not mode
        use_live = (mode == 'live')
        self.strategy = SP500Strategy(use_live=use_live)
        
        # Get current positions
        try:
            positions = self.api.list_positions()
            self.current_positions = {p.symbol: {
                'qty': float(p.qty), 
                'side': 'LONG' if float(p.qty) > 0 else 'SHORT',
                'unrealized_pl': float(p.unrealized_pl),
                'unrealized_plpc': float(p.unrealized_plpc)
            } for p in positions}
            logger.info(f"Loaded {len(self.current_positions)} current positions")
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            self.current_positions = {}
    
    def get_underperforming_shorts(self, threshold=-0.01):
        """Get underperforming SHORT positions"""
        underperforming = []
        for symbol, data in self.current_positions.items():
            if data['side'] == 'SHORT' and data['unrealized_plpc'] < threshold:
                underperforming.append({
                    'symbol': symbol,
                    'unrealized_pl': data['unrealized_pl'],
                    'unrealized_plpc': data['unrealized_plpc']
                })
        
        # Sort by worst performers first
        underperforming = sorted(underperforming, key=lambda x: x['unrealized_plpc'])
        logger.info(f"Found {len(underperforming)} underperforming SHORT positions")
        return underperforming
    
    def analyze_indicator_components(self, symbols):
        """Analyze technical indicator components for the given symbols"""
        results = []
        
        for symbol_data in tqdm(symbols, desc="Analyzing indicators"):
            symbol = symbol_data['symbol']
            try:
                # Get historical data - get more data (60 days instead of default 30)
                bars = self.strategy.get_historical_data(symbol, days=60)
                if bars is None or len(bars) < 20:
                    logger.warning(f"Not enough data for {symbol}")
                    continue
                
                # Calculate indicators
                indicators = self.strategy.calculate_technical_indicators(bars)
                if indicators is None:
                    continue
                
                # Get the latest values
                latest = indicators.iloc[-1]
                
                # Calculate individual scores using the same logic as SP500Strategy
                # RSI component (0-1)
                rsi = latest['rsi']
                if rsi < 30:
                    rsi_score = 1.0  # Strongly oversold - bullish
                elif rsi > 70:
                    rsi_score = 0.0  # Strongly overbought - bearish
                else:
                    rsi_score = 1 - ((rsi - 30) / 40)  # Linear scale between 30-70
                
                # MACD component (0-1)
                macd = latest['macd']
                macd_signal = latest['macd_signal']
                if macd > macd_signal:
                    macd_score = 1.0  # Bullish
                else:
                    macd_score = 0.0  # Bearish
                
                # Bollinger Bands component (0-1)
                close = latest['close']
                lower_band = latest['lower_band']
                upper_band = latest['upper_band']
                if close < lower_band:
                    bb_score = 1.0  # Below lower band - bullish
                elif close > upper_band:
                    bb_score = 0.0  # Above upper band - bearish
                else:
                    bb_score = 1 - ((close - lower_band) / (upper_band - lower_band))
                
                # Combined score (weighted average)
                combined_score = (rsi_score * 0.3) + (macd_score * 0.3) + (bb_score * 0.4)
                
                # Inverted score (for SHORT bias)
                inverted_score = 1 - combined_score
                
                # Get overall score
                scores = {
                    'long_score': combined_score,
                    'short_score': inverted_score
                }
                
                # Determine direction and score
                direction, score = self.strategy.determine_trade_direction(scores)
                
                # Add to results
                result = {
                    'symbol': symbol,
                    'unrealized_pl': symbol_data['unrealized_pl'],
                    'unrealized_plpc': symbol_data['unrealized_plpc'],
                    'rsi': latest['rsi'],
                    'rsi_score': rsi_score,
                    'macd': latest['macd'],
                    'macd_signal': latest['macd_signal'],
                    'macd_score': macd_score,
                    'upper_band': latest['upper_band'],
                    'lower_band': latest['lower_band'],
                    'sma15': latest['sma15'],
                    'close': latest['close'],
                    'bb_score': bb_score,
                    'long_score': combined_score,
                    'short_score': inverted_score,
                    'current_direction': direction,
                    'current_score': score
                }
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {str(e)}")
        
        return pd.DataFrame(results)
    
    def visualize_results(self, df):
        """Visualize the results"""
        if df.empty:
            logger.warning("No data to visualize")
            return
        
        # Create output directory
        os.makedirs('analysis', exist_ok=True)
        
        # Set the style
        sns.set(style="whitegrid")
        
        # 1. Correlation between indicator scores and performance
        plt.figure(figsize=(12, 8))
        correlation = df[['unrealized_plpc', 'rsi_score', 'macd_score', 'bb_score']].corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation between Indicator Scores and Performance')
        plt.tight_layout()
        plt.savefig('analysis/indicator_correlation.png')
        
        # 2. Distribution of indicator scores
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))
        
        sns.histplot(df['rsi_score'], kde=True, ax=axes[0])
        axes[0].set_title('Distribution of RSI Scores')
        axes[0].set_xlabel('RSI Score')
        
        sns.histplot(df['macd_score'], kde=True, ax=axes[1])
        axes[1].set_title('Distribution of MACD Scores')
        axes[1].set_xlabel('MACD Score')
        
        sns.histplot(df['bb_score'], kde=True, ax=axes[2])
        axes[2].set_title('Distribution of Bollinger Band Scores')
        axes[2].set_xlabel('Bollinger Band Score')
        
        plt.tight_layout()
        plt.savefig('analysis/indicator_distributions.png')
        
        # 3. Scatter plots of each indicator score vs performance
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))
        
        sns.scatterplot(x='rsi_score', y='unrealized_plpc', data=df, ax=axes[0])
        axes[0].set_title('RSI Score vs Performance')
        axes[0].set_xlabel('RSI Score')
        axes[0].set_ylabel('Unrealized P&L %')
        
        sns.scatterplot(x='macd_score', y='unrealized_plpc', data=df, ax=axes[1])
        axes[1].set_title('MACD Score vs Performance')
        axes[1].set_xlabel('MACD Score')
        axes[1].set_ylabel('Unrealized P&L %')
        
        sns.scatterplot(x='bb_score', y='unrealized_plpc', data=df, ax=axes[2])
        axes[2].set_title('Bollinger Band Score vs Performance')
        axes[2].set_xlabel('Bollinger Band Score')
        axes[2].set_ylabel('Unrealized P&L %')
        
        plt.tight_layout()
        plt.savefig('analysis/indicator_performance.png')
        
        # 4. Current direction vs original direction
        plt.figure(figsize=(10, 6))
        direction_counts = df['current_direction'].value_counts()
        plt.bar(direction_counts.index, direction_counts.values)
        plt.title('Current Signal Direction for Underperforming SHORT Positions')
        plt.xlabel('Direction')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('analysis/current_directions.png')
        
        # Save the data
        df.to_csv('analysis/indicator_analysis.csv', index=False)
        logger.info(f"Saved analysis results to analysis/indicator_analysis.csv")
        
        # Print summary
        print("\n=== Indicator Analysis Summary ===")
        print(f"Total positions analyzed: {len(df)}")
        print("\nAverage Indicator Scores:")
        print(f"  RSI Score: {df['rsi_score'].mean():.4f}")
        print(f"  MACD Score: {df['macd_score'].mean():.4f}")
        print(f"  Bollinger Band Score: {df['bb_score'].mean():.4f}")
        
        print("\nCorrelation with Performance:")
        print(f"  RSI Score: {correlation.loc['unrealized_plpc', 'rsi_score']:.4f}")
        print(f"  MACD Score: {correlation.loc['unrealized_plpc', 'macd_score']:.4f}")
        print(f"  Bollinger Band Score: {correlation.loc['unrealized_plpc', 'bb_score']:.4f}")
        
        print("\nCurrent Direction Breakdown:")
        for direction, count in direction_counts.items():
            print(f"  {direction}: {count} ({count/len(df)*100:.1f}%)")
        
        # Identify problematic indicators
        problem_indicators = []
        if correlation.loc['unrealized_plpc', 'rsi_score'] < -0.2:
            problem_indicators.append('RSI')
        if correlation.loc['unrealized_plpc', 'macd_score'] < -0.2:
            problem_indicators.append('MACD')
        if correlation.loc['unrealized_plpc', 'bb_score'] < -0.2:
            problem_indicators.append('Bollinger Bands')
        
        if problem_indicators:
            print("\nPotentially Problematic Indicators:")
            for indicator in problem_indicators:
                print(f"  - {indicator}")
        else:
            print("\nNo clearly problematic indicators identified based on correlation.")
        
        return correlation

def main():
    """Main function"""
    analyzer = SignalAnalyzer(mode='paper')
    
    # Get underperforming SHORT positions
    underperforming = analyzer.get_underperforming_shorts()
    
    if not underperforming:
        logger.info("No underperforming SHORT positions found")
        return
    
    # Print the underperforming positions
    print("\n=== Underperforming SHORT Positions ===")
    for pos in underperforming:
        print(f"{pos['symbol']}: {pos['unrealized_pl']:.2f} ({pos['unrealized_plpc']*100:.2f}%)")
    
    # Analyze indicator components
    results = analyzer.analyze_indicator_components(underperforming)
    
    # Visualize results if we have data
    correlation = None
    if not results.empty:
        correlation = analyzer.visualize_results(results)
    else:
        print("\nNo sufficient data available for analysis.")
        print("Possible reasons:")
        print("- Not enough historical data for the symbols")
        print("- Technical indicators could not be calculated")
        print("\n=== Recommendations Based on Current Performance ===")
        print("- Raise the SHORT signal threshold to be more selective")
        print("- Implement market regime detection to reduce SHORT exposure in bullish markets")
        print("- Add stop-loss rules to exit underperforming SHORT positions earlier")
        print("- Consider reducing position sizes for SHORT positions")
        return
    
    # Provide recommendations
    print("\n=== Recommendations ===")
    if correlation is not None:
        if correlation.loc['unrealized_plpc', 'rsi_score'] < -0.2:
            print("- Consider adjusting RSI parameters or reducing its weight for SHORT signals")
        if correlation.loc['unrealized_plpc', 'macd_score'] < -0.2:
            print("- Consider adjusting MACD parameters or reducing its weight for SHORT signals")
        if correlation.loc['unrealized_plpc', 'bb_score'] < -0.2:
            print("- Consider adjusting Bollinger Band parameters or reducing its weight for SHORT signals")
    
    print("- Raise the SHORT signal threshold to be more selective")
    print("- Implement market regime detection to reduce SHORT exposure in bullish markets")
    print("- Add stop-loss rules to exit underperforming SHORT positions earlier")

if __name__ == "__main__":
    main()
