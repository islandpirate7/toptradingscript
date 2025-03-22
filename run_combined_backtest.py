import argparse
import logging
import datetime as dt
import json
import os
import yaml
from typing import Dict, Any, List

from enhanced_mean_reversion_backtest import EnhancedMeanReversionBacktest
from multi_strategy_system import TrendFollowingStrategy, MultiStrategySystem, SystemConfig, StockConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_file}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise

def add_alpaca_credentials(config: Dict[str, Any]) -> Dict[str, Any]:
    """Add Alpaca API credentials to configuration"""
    try:
        with open('alpaca_credentials.json', 'r') as f:
            credentials = json.load(f)
            
        # Use paper trading credentials by default
        paper_creds = credentials.get('paper', {})
        config['alpaca'] = {
            'api_key': paper_creds.get('api_key'),
            'api_secret': paper_creds.get('api_secret'),
            'base_url': paper_creds.get('base_url', 'https://paper-api.alpaca.markets/v2')
        }
        logger.info("Added Alpaca API credentials to configuration")
        return config
    except Exception as e:
        logger.error(f"Error adding Alpaca credentials: {e}")
        raise

def convert_to_system_config(config_dict: Dict[str, Any]) -> SystemConfig:
    """Convert dictionary config to SystemConfig object"""
    # Extract global settings
    global_config = config_dict.get('global', {})
    
    # Create stock configs
    stocks = []
    for symbol, stock_config in config_dict.get('stocks', {}).items():
        stock = StockConfig(
            symbol=symbol,
            max_position_size=stock_config.get('max_position_size', 100),
            min_position_size=stock_config.get('min_position_size', 10),
            max_risk_per_trade_pct=stock_config.get('max_risk_per_trade_pct', 0.5),
            min_volume=stock_config.get('min_volume', 100000)
        )
        stocks.append(stock)
    
    # Extract strategy weights
    strategy_weights = {}
    for strategy_name, strategy_config in config_dict.get('strategies', {}).items():
        strategy_weights[strategy_name] = strategy_config.get('base_weight', 0.25)
    
    # Create SystemConfig
    system_config = SystemConfig(
        stocks=stocks,
        initial_capital=global_config.get('initial_capital', 100000),
        max_open_positions=global_config.get('max_open_positions', 5),
        max_positions_per_symbol=global_config.get('max_positions_per_symbol', 2),
        max_correlated_positions=global_config.get('max_correlated_positions', 3),
        max_sector_exposure_pct=global_config.get('max_sector_exposure_pct', 30.0),
        max_portfolio_risk_daily_pct=global_config.get('max_portfolio_risk_daily_pct', 2.0),
        strategy_weights=strategy_weights,
        rebalance_interval=dt.timedelta(days=1),  # Default to daily rebalancing
        data_lookback_days=global_config.get('data_lookback_days', 30),
        market_hours_start=dt.time(9, 30),  # Default market open time
        market_hours_end=dt.time(16, 0),    # Default market close time
        enable_auto_trading=False,          # Default to no auto trading
        backtesting_mode=True,              # Set to backtesting mode
        data_source="alpaca"                # Use Alpaca as data source
    )
    
    return system_config

def run_combined_backtest(config: Dict[str, Any], start_date: dt.datetime, end_date: dt.datetime):
    """Run backtest with both Mean Reversion and Trend Following strategies"""
    
    # Convert dictionary config to SystemConfig
    system_config = convert_to_system_config(config)
    
    # Create a MultiStrategySystem instance
    multi_strategy = MultiStrategySystem(system_config)
    
    # Clear existing strategies and add our own
    multi_strategy.strategies = {}
    
    # Add strategies
    if 'MeanReversion' in config.get('strategies', {}):
        logger.info("Adding Mean Reversion strategy")
        mean_reversion_config = config.get('strategies', {}).get('MeanReversion', {})
        mean_reversion_backtest = EnhancedMeanReversionBacktest({'strategies': {'MeanReversion': mean_reversion_config}})
        multi_strategy.add_strategy(mean_reversion_backtest.strategy)
    
    if 'TrendFollowing' in config.get('strategies', {}):
        logger.info("Adding Trend Following strategy")
        trend_following_config = config.get('strategies', {}).get('TrendFollowing', {})
        trend_following = TrendFollowingStrategy(trend_following_config)
        multi_strategy.add_strategy(trend_following)
    
    # Run the backtest
    logger.info(f"Running combined backtest from {start_date} to {end_date}")
    results = multi_strategy.run_backtest(start_date, end_date)
    
    # Print results
    print("\n=== Combined Strategy Backtest Results ===")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Capital: ${results.get('initial_capital', 0):.2f}")
    print(f"Final Equity: ${results.get('final_equity', 0):.2f}")
    print(f"Total Return: {results.get('total_return_pct', 0):.2f}%")
    print(f"Annualized Return: {results.get('annualized_return_pct', 0):.2f}%")
    print(f"Maximum Drawdown: {results.get('max_drawdown_pct', 0):.2f}%")
    print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
    print(f"Win Rate: {results.get('win_rate', 0):.2f}%")
    print(f"Profit Factor: {results.get('profit_factor', 0):.2f}")
    print(f"Total Trades: {results.get('total_trades', 0)}")
    
    # Save results
    results_dir = os.path.join(os.getcwd(), 'results')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"combined_backtest_results_{timestamp}.json")
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Saved backtest results to {results_file}")
    print(f"\nResults saved to: {results_file}")
    
    if 'equity_curve_file' in results:
        print(f"Equity curve saved to: {results.get('equity_curve_file')}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Run combined Mean Reversion and Trend Following backtest')
    parser.add_argument('--config', type=str, default='configuration_mean_reversion_trend_combo.yaml', 
                        help='Path to configuration file')
    parser.add_argument('--start_date', type=str, default='2023-01-01', 
                        help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2023-12-31', 
                        help='End date for backtest (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Parse dates
    start_date = dt.datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = dt.datetime.strptime(args.end_date, '%Y-%m-%d')
    
    # Load and prepare configuration
    config = load_config(args.config)
    config = add_alpaca_credentials(config)
    
    # Run backtest
    run_combined_backtest(config, start_date, end_date)

if __name__ == "__main__":
    main()
