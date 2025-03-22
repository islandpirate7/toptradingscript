#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimized Backtest Runner
-------------------------
This script runs backtests for the optimized multi-strategy trading system
across different time periods and asset classes.
"""

import os
import sys
import yaml
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
from typing import List, Dict, Any, Tuple
import json

# Import our strategy modules
from enhanced_mean_reversion_backtest import EnhancedMeanReversionBacktest
from trend_following_strategy import TrendFollowingStrategy
from mean_reversion_enhanced import CandleData, Signal, MarketState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("backtest_results.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BacktestRunner:
    """Class to run backtests for different time periods and assets"""
    
    def __init__(self, config_path: str, alpaca_key: str = None, alpaca_secret: str = None):
        """Initialize the backtest runner"""
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        # Initialize Alpaca API
        self.alpaca_key = alpaca_key or os.environ.get('ALPACA_API_KEY')
        self.alpaca_secret = alpaca_secret or os.environ.get('ALPACA_API_SECRET')
        
        if not self.alpaca_key or not self.alpaca_secret:
            raise ValueError("Alpaca API credentials not provided")
        
        self.api = tradeapi.REST(
            self.alpaca_key,
            self.alpaca_secret,
            base_url='https://paper-api.alpaca.markets'
        )
        
        # Initialize strategies
        self.mean_reversion = EnhancedMeanReversionBacktest(self.config)
        self.trend_following = TrendFollowingStrategy(self.config)
        
        # Define test periods
        self.test_periods = [
            # 2023 Quarters
            ("2023-01-01", "2023-03-31", "2023_Q1"),
            ("2023-04-01", "2023-06-30", "2023_Q2"),
            ("2023-07-01", "2023-09-30", "2023_Q3"),
            ("2023-10-01", "2023-12-31", "2023_Q4"),
            # 2024 Quarters - only include if data is available
            ("2024-01-01", "2024-03-15", "2024_Q1")
        ]
        
        # Define stock symbols
        self.stock_symbols = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META",
            "NVDA", "TSLA", "JPM", "V", "JNJ",
            "WMT", "PG", "UNH", "HD", "BAC",
            "MA", "XOM", "DIS", "NFLX", "ADBE",
            "CRM", "CSCO", "INTC", "VZ", "KO"
        ]
        
        # Define crypto symbols
        self.crypto_symbols = [
            "BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD",
            "AVAX/USD", "LINK/USD", "MATIC/USD", "XRP/USD", "LTC/USD",
            "DOGE/USD", "UNI/USD", "AAVE/USD", "ATOM/USD", "ALGO/USD",
            "BCH/USD", "XLM/USD", "EOS/USD", "FIL/USD", "NEAR/USD",
            "COMP/USD", "MKR/USD", "YFI/USD", "SNX/USD", "SUSHI/USD"
        ]
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    
    def fetch_historical_data(self, symbols: List[str], start_date: str, end_date: str, 
                             timeframe: str = '1D', is_crypto: bool = False) -> Dict[str, List[Dict]]:
        """Fetch historical data from Alpaca for the given symbols and date range"""
        logger.info(f"Fetching historical data for {len(symbols)} {'crypto' if is_crypto else 'stock'} symbols")
        
        # Convert dates to datetime objects
        start = pd.Timestamp(start_date).tz_localize('America/New_York')
        end = pd.Timestamp(end_date).tz_localize('America/New_York')
        
        data = {}
        
        for symbol in symbols:
            try:
                # Fetch data from Alpaca
                symbol_for_api = symbol if not is_crypto else symbol.replace('/', '')
                
                bars = self.api.get_bars(
                    symbol_for_api,
                    timeframe,
                    start=start.isoformat(),
                    end=end.isoformat(),
                    adjustment='raw'
                ).df
                
                if bars.empty:
                    logger.warning(f"No data found for {symbol}")
                    continue
                
                # Convert to our expected format
                candles = []
                for index, row in bars.iterrows():
                    candle = CandleData(
                        timestamp=index.to_pydatetime(),
                        open=float(row['open']),
                        high=float(row['high']),
                        low=float(row['low']),
                        close=float(row['close']),
                        volume=float(row['volume'])
                    )
                    candles.append(candle)
                
                data[symbol] = candles
                logger.info(f"Fetched {len(candles)} candles for {symbol}")
                
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
        
        return data
    
    def run_backtest(self, symbols: List[str], start_date: str, end_date: str, 
                    period_name: str, is_crypto: bool = False) -> Dict:
        """Run a backtest for the given symbols and date range"""
        logger.info(f"Running backtest for period {period_name} ({start_date} to {end_date})")
        
        # Fetch historical data
        data = self.fetch_historical_data(symbols, start_date, end_date, is_crypto=is_crypto)
        
        if not data:
            logger.error("No data available for backtest")
            return None
        
        # Run the backtest directly using the EnhancedMeanReversionBacktest
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Run mean reversion backtest
        mr_results = self.mean_reversion.run_backtest(
            data=data,
            start_date=start_dt,
            end_date=end_dt,
            initial_capital=self.config.get('global', {}).get('initial_capital', 100000.0)
        )
        
        # Save results
        self._save_results(mr_results, f"{period_name}_mean_reversion", is_crypto)
        
        # For now, we're only running the mean reversion strategy
        # In the future, we can add the trend following strategy and combine results
        
        return mr_results
    
    def _save_results(self, results: Dict, period_name: str, is_crypto: bool) -> None:
        """Save backtest results to file"""
        asset_type = "crypto" if is_crypto else "stocks"
        filename = f"backtest_results_{period_name}_{asset_type}.json"
        
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        # Convert datetime objects to strings for JSON serialization
        results_copy = self._prepare_results_for_json(results)
        
        with open(os.path.join('results', filename), 'w') as f:
            json.dump(results_copy, f, indent=2, default=str)
        
        logger.info(f"Saved results to results/{filename}")
    
    def _prepare_results_for_json(self, results: Dict) -> Dict:
        """Prepare results for JSON serialization by converting datetime objects to strings"""
        if not results:
            return {}
            
        results_copy = {}
        
        for key, value in results.items():
            if isinstance(value, datetime):
                results_copy[key] = value.isoformat()
            elif isinstance(value, list):
                results_copy[key] = []
                for item in value:
                    if isinstance(item, dict):
                        item_copy = {}
                        for k, v in item.items():
                            if isinstance(v, datetime):
                                item_copy[k] = v.isoformat()
                            else:
                                item_copy[k] = v
                        results_copy[key].append(item_copy)
                    else:
                        results_copy[key].append(item)
            else:
                results_copy[key] = value
                
        return results_copy
    
    def run_all_backtests(self) -> Dict:
        """Run backtests for all defined periods and asset types"""
        all_results = {}
        
        # Run stock backtests
        for start_date, end_date, period_name in self.test_periods:
            logger.info(f"Running stock backtest for {period_name}")
            results = self.run_backtest(
                self.stock_symbols[:3],  # Start with a smaller subset for testing
                start_date, 
                end_date, 
                period_name,
                is_crypto=False
            )
            all_results[f"{period_name}_stocks"] = results
        
        # Run crypto backtests
        for start_date, end_date, period_name in self.test_periods:
            logger.info(f"Running crypto backtest for {period_name}")
            results = self.run_backtest(
                self.crypto_symbols[:3],  # Start with a smaller subset for testing
                start_date, 
                end_date, 
                period_name,
                is_crypto=True
            )
            all_results[f"{period_name}_crypto"] = results
        
        # Generate summary report
        self.generate_summary_report(all_results)
        
        return all_results
    
    def generate_summary_report(self, all_results: Dict) -> None:
        """Generate a summary report of all backtest results"""
        summary = []
        
        for key, results in all_results.items():
            if not results:
                continue
                
            period, asset_type = key.split('_', 1)
            
            # Extract key metrics from results
            total_return_pct = results.get('total_return_pct', 0)
            win_rate = results.get('win_rate', 0)
            profit_factor = results.get('profit_factor', 0)
            max_drawdown = results.get('max_drawdown', 0)
            sharpe_ratio = results.get('sharpe_ratio', 0)
            
            summary_row = {
                'period': period,
                'asset_type': asset_type,
                'total_return_pct': total_return_pct,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio
            }
            
            summary.append(summary_row)
        
        # Create DataFrame and save to CSV
        if summary:
            df = pd.DataFrame(summary)
            
            # Create results directory if it doesn't exist
            os.makedirs('results', exist_ok=True)
            
            df.to_csv('results/backtest_summary.csv', index=False)
            
            # Print summary table
            print("\nBacktest Summary:")
            print("=================")
            print(df.to_string(index=False))
            
            # Create performance comparison charts
            self._create_performance_charts(df)
        else:
            logger.warning("No results available for summary report")
    
    def _create_performance_charts(self, df: pd.DataFrame) -> None:
        """Create performance comparison charts"""
        plt.figure(figsize=(15, 10))
        
        # Plot total returns by period and asset type
        plt.subplot(2, 2, 1)
        
        # Group by period and asset_type
        periods = df['period'].unique()
        
        # Prepare data for plotting
        stock_returns = df[df['asset_type'] == 'stocks']['total_return_pct'].values
        crypto_returns = df[df['asset_type'] == 'crypto']['total_return_pct'].values
        
        # Set up bar positions
        x = np.arange(len(periods))
        width = 0.35
        
        # Create bars
        plt.bar(x - width/2, stock_returns, width, label='Stocks')
        plt.bar(x + width/2, crypto_returns, width, label='Crypto')
        
        plt.xlabel('Period')
        plt.ylabel('Total Return (%)')
        plt.title('Total Returns by Period and Asset Type')
        plt.xticks(x, periods)
        plt.legend()
        
        # Plot win rates
        plt.subplot(2, 2, 2)
        stock_winrate = df[df['asset_type'] == 'stocks']['win_rate'].values
        crypto_winrate = df[df['asset_type'] == 'crypto']['win_rate'].values
        
        plt.bar(x - width/2, stock_winrate, width, label='Stocks')
        plt.bar(x + width/2, crypto_winrate, width, label='Crypto')
        plt.xlabel('Period')
        plt.ylabel('Win Rate (%)')
        plt.title('Win Rate by Period and Asset Type')
        plt.xticks(x, periods)
        plt.legend()
        
        # Plot max drawdowns
        plt.subplot(2, 2, 3)
        stock_drawdown = df[df['asset_type'] == 'stocks']['max_drawdown'].values
        crypto_drawdown = df[df['asset_type'] == 'crypto']['max_drawdown'].values
        
        plt.bar(x - width/2, stock_drawdown, width, label='Stocks')
        plt.bar(x + width/2, crypto_drawdown, width, label='Crypto')
        plt.xlabel('Period')
        plt.ylabel('Max Drawdown (%)')
        plt.title('Maximum Drawdown by Period and Asset Type')
        plt.xticks(x, periods)
        plt.legend()
        
        # Plot Sharpe ratios
        plt.subplot(2, 2, 4)
        stock_sharpe = df[df['asset_type'] == 'stocks']['sharpe_ratio'].values
        crypto_sharpe = df[df['asset_type'] == 'crypto']['sharpe_ratio'].values
        
        plt.bar(x - width/2, stock_sharpe, width, label='Stocks')
        plt.bar(x + width/2, crypto_sharpe, width, label='Crypto')
        plt.xlabel('Period')
        plt.ylabel('Sharpe Ratio')
        plt.title('Sharpe Ratio by Period and Asset Type')
        plt.xticks(x, periods)
        plt.legend()
        
        plt.tight_layout()
        
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        plt.savefig('results/backtest_performance_comparison.png')
        plt.close()
        
        logger.info("Performance comparison charts saved to results/backtest_performance_comparison.png")

def main():
    """Main function to run the backtest"""
    # Get Alpaca API credentials from environment variables
    alpaca_key = os.environ.get('ALPACA_API_KEY')
    alpaca_secret = os.environ.get('ALPACA_API_SECRET')
    
    if not alpaca_key or not alpaca_secret:
        print("Please set ALPACA_API_KEY and ALPACA_API_SECRET environment variables")
        alpaca_key = input("Enter your Alpaca API key: ")
        alpaca_secret = input("Enter your Alpaca API secret: ")
    
    # Create backtest runner
    runner = BacktestRunner(
        config_path='multi_strategy_config.yaml',
        alpaca_key=alpaca_key,
        alpaca_secret=alpaca_secret
    )
    
    # Run all backtests
    results = runner.run_all_backtests()
    
    print("\nBacktest completed successfully!")
    print("Check results/backtest_summary.csv for detailed results")
    print("Check results/backtest_performance_comparison.png for performance charts")

if __name__ == "__main__":
    main()
