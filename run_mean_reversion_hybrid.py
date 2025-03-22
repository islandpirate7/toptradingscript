#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hybrid Mean Reversion Strategy Backtest Runner
---------------------------------------------
This script runs a backtest of the enhanced Mean Reversion strategy
with market regime detection and ML signal classification features.
"""

import os
import json
import yaml
import logging
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
import alpaca_trade_api as tradeapi
from enhanced_mean_reversion_backtest import EnhancedMeanReversionBacktest
from market_regime_detector import MarketRegimeDetector
from ml_signal_classifier import MLSignalClassifier
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hybrid_mean_reversion_backtest.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("hybrid_mean_reversion_backtest")

def load_alpaca_credentials(credentials_file="alpaca_credentials.json", env="paper"):
    """Load Alpaca API credentials from JSON file"""
    try:
        with open(credentials_file, 'r') as f:
            credentials = json.load(f)
        
        if env not in credentials:
            logger.error(f"Environment {env} not found in credentials file")
            return None
            
        return credentials[env]
    except Exception as e:
        logger.error(f"Error loading Alpaca credentials: {str(e)}")
        return None

def load_config(config_file="configuration_mean_reversion_final.yaml"):
    """Load configuration from YAML file"""
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return None

def plot_backtest_results(results, output_file="hybrid_mean_reversion_results.png"):
    """Plot backtest results"""
    try:
        # Create figure with multiple subplots
        fig, axs = plt.subplots(4, 1, figsize=(12, 20), gridspec_kw={'height_ratios': [3, 1, 1, 1]})
        
        # Plot equity curve
        equity_curve = results.get('equity_curve', [])
        
        if not equity_curve:
            logger.error("No equity curve data available")
            return False
            
        dates = [dt.datetime.fromisoformat(point[0]) if isinstance(point[0], str) else point[0] 
                for point in equity_curve]
        values = [point[1] for point in equity_curve]
        
        axs[0].plot(dates, values)
        axs[0].set_title('Equity Curve')
        axs[0].set_ylabel('Portfolio Value ($)')
        axs[0].grid(True)
        
        # Plot drawdown
        if 'drawdowns' in results:
            drawdowns = results['drawdowns']
            dd_dates = [dt.datetime.fromisoformat(point[0]) if isinstance(point[0], str) else point[0] 
                      for point in drawdowns]
            dd_values = [point[1] * 100 for point in drawdowns]  # Convert to percentage
            
            axs[1].plot(dd_dates, dd_values)
            axs[1].set_title('Drawdown (%)')
            axs[1].set_ylabel('Drawdown %')
            axs[1].grid(True)
            axs[1].fill_between(dd_dates, dd_values, 0, alpha=0.3, color='red')
        
        # Plot trade outcomes
        if 'trade_history' in results:
            trades = results['trade_history']
            
            # Extract trade data
            trade_dates = []
            trade_returns = []
            trade_types = []
            
            for trade in trades:
                if 'exit_date' in trade and trade['exit_date']:
                    exit_date = dt.datetime.fromisoformat(trade['exit_date']) if isinstance(trade['exit_date'], str) else trade['exit_date']
                    trade_dates.append(exit_date)
                    trade_returns.append(trade['pnl_pct'] * 100)  # Convert to percentage
                    trade_types.append(trade['direction'])
            
            # Plot trade returns
            if trade_dates:
                colors = ['green' if ret > 0 else 'red' for ret in trade_returns]
                axs[2].bar(range(len(trade_returns)), trade_returns, color=colors)
                axs[2].set_title('Trade Returns (%)')
                axs[2].set_ylabel('Return %')
                axs[2].set_xlabel('Trade Number')
                axs[2].grid(True)
                
                # Add average line
                if trade_returns:
                    avg_return = sum(trade_returns) / len(trade_returns)
                    axs[2].axhline(y=avg_return, color='blue', linestyle='--', label=f'Avg: {avg_return:.2f}%')
                    axs[2].legend()
        
        # Plot market regime
        if 'market_regimes' in results:
            regimes = results['market_regimes']
            regime_dates = [dt.datetime.fromisoformat(r[0]) if isinstance(r[0], str) else r[0] for r in regimes]
            regime_values = [r[1] for r in regimes]
            
            # Convert regime strings to numeric values for plotting
            regime_map = {
                'strong_bullish': 1.0,
                'bullish': 0.75,
                'neutral': 0.5,
                'transitional': 0.5,
                'bearish': 0.25,
                'strong_bearish': 0.0
            }
            
            regime_numeric = [regime_map.get(r, 0.5) for r in regime_values]
            
            # Create a colormap
            cmap = plt.cm.get_cmap('RdYlGn')
            colors = [cmap(r) for r in regime_numeric]
            
            # Plot as a colorbar
            axs[3].scatter(regime_dates, [0.5] * len(regime_dates), c=regime_numeric, cmap='RdYlGn', s=100)
            axs[3].set_title('Market Regime')
            axs[3].set_yticks([])
            axs[3].grid(True)
            
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(0, 1))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=axs[3])
            cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
            cbar.set_ticklabels(['Strong Bearish', 'Bearish', 'Neutral', 'Bullish', 'Strong Bullish'])
        
        plt.tight_layout()
        plt.savefig(output_file)
        logger.info(f"Saved backtest results plot to {output_file}")
        
        return True
    except Exception as e:
        logger.error(f"Error plotting backtest results: {str(e)}")
        return False

def save_results_to_json(results, output_file="hybrid_mean_reversion_results.json"):
    """Save backtest results to JSON file"""
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        logger.info(f"Saved backtest results to {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error saving backtest results: {str(e)}")
        return False

def run_backtest_with_ml_training():
    """Run a backtest with ML signal classifier training"""
    # Load configuration
    config = load_config("configuration_mean_reversion_final.yaml")
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("MeanReversionHybrid")
    
    # Load Alpaca credentials
    try:
        with open("alpaca_credentials.json", "r") as f:
            alpaca_credentials = json.load(f)
            
        # Use paper trading credentials for backtesting
        if "paper" in alpaca_credentials:
            paper_credentials = alpaca_credentials["paper"]
            config["alpaca"] = {
                "api_key": paper_credentials["api_key"],
                "api_secret": paper_credentials["api_secret"],
                "base_url": paper_credentials["base_url"]
            }
            logger.info("Loaded Alpaca paper trading credentials")
        else:
            logger.warning("No paper trading credentials found in alpaca_credentials.json")
    except Exception as e:
        logger.error(f"Error loading Alpaca credentials: {str(e)}")
    
    # Initialize the ML signal classifier
    ml_config = {
        'ml_classifier_params': {
            'model_type': 'random_forest',
            'n_estimators': 100,
            'max_depth': 5,
            'min_training_samples': 50,  # Minimum samples before training
            'features': [
                'rsi', 'bb_distance', 'atr_percent', 'volume_change', 
                'price_change', 'trend_strength', 'volatility'
            ]
        }
    }
    
    # Ensure the config has a symbols list
    if 'symbols' not in config:
        # Convert the stocks dictionary to a symbols list format
        symbols_list = []
        for symbol, stock_config in config.get('stocks', {}).items():
            symbols_list.append({
                'symbol': symbol,
                'weight': stock_config.get('weight', 1.0),
                'min_signal_strength': stock_config.get('min_signal_strength', 'weak')
            })
        
        # If no stocks were found, use a default list of symbols
        if not symbols_list:
            symbols_list = [
                {'symbol': 'AAPL', 'weight': 1.0},
                {'symbol': 'MSFT', 'weight': 1.0},
                {'symbol': 'AMZN', 'weight': 1.0},
                {'symbol': 'GOOGL', 'weight': 1.0},
                {'symbol': 'META', 'weight': 1.0}
            ]
        
        config['symbols'] = symbols_list
        logger.info(f"Created symbols list with {len(symbols_list)} symbols")
    
    # Merge ML config with strategy config
    config.update(ml_config)
    
    # Define backtest periods (using 2023 data due to Alpaca free tier limitations)
    backtest_periods = [
        # Q1 2023
        (dt.datetime(2023, 1, 1), dt.datetime(2023, 3, 31)),
        # Q2 2023
        (dt.datetime(2023, 4, 1), dt.datetime(2023, 6, 30)),
        # Q3 2023
        (dt.datetime(2023, 7, 1), dt.datetime(2023, 9, 30)),
        # Q4 2023
        (dt.datetime(2023, 10, 1), dt.datetime(2023, 12, 31))
    ]
    
    # Initialize backtest
    backtest = EnhancedMeanReversionBacktest(config)
    
    # Run backtest for each period
    all_results = []
    
    for start_date, end_date in backtest_periods:
        logger.info(f"Running backtest from {start_date.date()} to {end_date.date()}")
        
        # Run backtest
        results = backtest.run_backtest(start_date, end_date)
        
        if results:
            all_results.append({
                'period': f"{start_date.date()} to {end_date.date()}",
                'results': results
            })
            
            # Log key metrics
            logger.info(f"Backtest results for {start_date.date()} to {end_date.date()}:")
            logger.info(f"  Total trades: {results.get('total_trades', 0)}")
            logger.info(f"  Win rate: {results.get('win_rate', 0):.2f}%")
            logger.info(f"  Profit factor: {results.get('profit_factor', 0):.2f}")
            logger.info(f"  Return: {results.get('return', 0):.2f}%")
            
            # Log ML training status
            if backtest.ml_signal_classifier:
                logger.info(f"  ML model trained: {backtest.ml_signal_classifier.is_trained}")
                logger.info(f"  ML training samples: {len(backtest.ml_training_data)}")
                
                # If model is trained, log feature importances
                if backtest.ml_signal_classifier.is_trained:
                    importances = backtest.ml_signal_classifier.get_feature_importances()
                    if importances:
                        logger.info("  ML feature importances:")
                        for feature, importance in importances.items():
                            logger.info(f"    {feature}: {importance:.4f}")
    
    # Compare results across periods
    if all_results:
        logger.info("\nComparison across all periods:")
        
        # Calculate average metrics
        avg_win_rate = sum(r['results'].get('win_rate', 0) for r in all_results) / len(all_results)
        avg_profit_factor = sum(r['results'].get('profit_factor', 0) for r in all_results) / len(all_results)
        avg_return = sum(r['results'].get('return', 0) for r in all_results) / len(all_results)
        
        logger.info(f"Average win rate: {avg_win_rate:.2f}%")
        logger.info(f"Average profit factor: {avg_profit_factor:.2f}")
        logger.info(f"Average return: {avg_return:.2f}%")
        
        # Save ML model if trained
        if backtest.ml_signal_classifier and backtest.ml_signal_classifier.is_trained:
            model_path = backtest.ml_signal_classifier.save_model("ml_signal_classifier_model.pkl")
            logger.info(f"ML model saved to {model_path}")
    
    return all_results

def main():
    """Main function"""
    # Load configuration
    config_file = "configuration_mean_reversion_final.yaml"
    config = load_config(config_file)
    
    if not config:
        logger.error("Failed to load configuration")
        return
    
    logger.info(f"Loaded configuration from {config_file}")
    
    # Load Alpaca credentials (using paper trading for backtesting)
    credentials = load_alpaca_credentials(env="paper")
    
    if not credentials:
        logger.error("Failed to load Alpaca credentials")
        return
    
    # Initialize Alpaca API
    api = tradeapi.REST(
        key_id=credentials['api_key'],
        secret_key=credentials['api_secret'],
        base_url=credentials['base_url']
    )
    
    # Define backtest parameters
    start_date = dt.datetime(2023, 1, 1)
    end_date = dt.datetime(2023, 12, 31)
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    initial_capital = 100000.0
    
    # Format symbols as required by the backtest class
    formatted_symbols = []
    for symbol in symbols:
        # Use the optimized configuration from our previous work
        # The final configuration in configuration_mean_reversion_final.yaml has been optimized
        # with BB period: 20, BB std dev: 1.9, RSI thresholds: 35/65, etc.
        symbol_config = {
            "symbol": symbol,
            "weight": 1.0,  # Default weight, can be adjusted based on performance
            "is_crypto": False
        }
        formatted_symbols.append(symbol_config)
    
    # Extract Mean Reversion strategy parameters
    mean_reversion_params = config.get('strategies', {}).get('MeanReversion', {})
    
    # Create complete configuration for backtest
    backtest_config = {
        'mean_reversion_params': mean_reversion_params,
        'market_regime_params': config.get('market_regime_params', {}),
        'ml_classifier_params': config.get('ml_classifier_params', {}),
        'initial_capital': initial_capital,
        'start_date': start_date,
        'end_date': end_date,
        'symbols': formatted_symbols,
        'alpaca_api': {
            'key_id': credentials['api_key'],
            'secret_key': credentials['api_secret'],
            'base_url': credentials['base_url']
        }
    }
    
    # Initialize market regime detector
    regime_detector = MarketRegimeDetector(backtest_config)
    
    # Initialize backtest
    backtest = EnhancedMeanReversionBacktest(config=backtest_config)
    
    # Run backtest
    logger.info(f"Starting backtest from {start_date} to {end_date} with symbols {symbols}")
    results = backtest.run_backtest(start_date=start_date, end_date=end_date)
    
    if not results:
        logger.error("Backtest failed")
        return
    
    # Calculate performance metrics
    performance = backtest.calculate_performance()
    
    if performance:
        # Add performance metrics to results
        results.update(performance)
        
        # Log performance metrics
        logger.info("Backtest completed successfully")
        logger.info(f"Total Return: {performance['total_return']:.2%}")
        logger.info(f"Annualized Return: {performance['annualized_return']:.2%}")
        logger.info(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {performance['max_drawdown']:.2%}")
        logger.info(f"Win Rate: {performance['win_rate']:.2%}")
        logger.info(f"Profit Factor: {performance['profit_factor']:.2f}")
        
        # Plot and save results
        plot_backtest_results(results)
        save_results_to_json(results)
        
        # Train ML model with historical data if enough trades
        if 'trade_history' in results and len(results['trade_history']) >= 50:
            logger.info("Training ML model with historical trade data")
            ml_classifier = MLSignalClassifier(backtest_config)
            
            # Prepare training data
            historical_signals = backtest.get_historical_signals()
            historical_trades = results['trade_history']
            candles_by_symbol = backtest.get_historical_candles()
            market_states = backtest.get_market_states()
            
            X_train, y_train = ml_classifier.prepare_training_data(
                historical_signals, 
                historical_trades, 
                candles_by_symbol, 
                market_states
            )
            
            if len(X_train) > 0:
                success = ml_classifier.train_model(X_train, y_train)
                if success:
                    logger.info("ML model trained successfully")
                else:
                    logger.warning("ML model training failed")
            else:
                logger.warning("Not enough valid training samples")
    else:
        logger.error("Failed to calculate performance metrics")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Mean Reversion Hybrid Strategy')
    parser.add_argument('--mode', type=str, choices=['backtest', 'live', 'paper', 'ml_training'], 
                        default='backtest', help='Trading mode')
    parser.add_argument('--config', type=str, default='configuration_mean_reversion_final.yaml', 
                        help='Configuration file path')
    
    args = parser.parse_args()
    
    if args.mode == 'backtest':
        main()
    elif args.mode == 'live':
        # run_live(args.config)
        pass
    elif args.mode == 'paper':
        # run_paper(args.config)
        pass
    elif args.mode == 'ml_training':
        run_backtest_with_ml_training()
