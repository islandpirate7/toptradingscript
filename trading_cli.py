#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trading CLI

A comprehensive command-line interface for the S&P 500 Trading Strategy.
This script provides commands for backtesting, paper trading, and live trading,
all using the original strategy implementation in final_sp500_strategy.py.
"""

import os
import sys
import json
import yaml
import argparse
import datetime
from pathlib import Path
import importlib.util
import traceback
from error_handler import get_error_handler, error_context, ConfigurationError, BacktestError, APIError

# Import our custom trading logger
from trading_logger import get_logger

# Initialize the logger
logger = get_logger("trading_cli")

def load_strategy_module():
    """Load the strategy module from final_sp500_strategy.py"""
    strategy_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'final_sp500_strategy.py')
    
    try:
        spec = importlib.util.spec_from_file_location("strategy_module", strategy_file)
        strategy_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(strategy_module)
        logger.debug(f"Loaded strategy module from {strategy_file}")
        return strategy_module
    except Exception as e:
        error = ConfigurationError(
            f"Error loading strategy module from {strategy_file}: {str(e)}",
            severity="CRITICAL",
            details={"file": strategy_file}
        )
        get_error_handler().handle_error(error)
        logger.critical(f"Error loading strategy module: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

def load_config():
    """Load the configuration from the YAML file"""
    config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sp500_config.yaml')
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        logger.debug(f"Loaded configuration from {config_file}")
        return config
    except Exception as e:
        error = ConfigurationError(
            f"Error loading configuration from {config_file}: {str(e)}",
            severity="ERROR",
            details={"file": config_file}
        )
        get_error_handler().handle_error(error)
        logger.error(f"Error loading configuration: {str(e)}")
        # Return a default configuration
        return {
            'initial_capital': 300,
            'backtest': {
                'max_signals_per_day': 40,
                'tier1_threshold': 0.8,
                'tier2_threshold': 0.7,
                'tier3_threshold': 0.6
            },
            'paper_trading': {
                'max_signals_per_day': 40,
                'tier1_threshold': 0.8,
                'tier2_threshold': 0.7,
                'tier3_threshold': 0.6
            },
            'live_trading': {
                'max_signals_per_day': 40,
                'tier1_threshold': 0.8,
                'tier2_threshold': 0.7,
                'tier3_threshold': 0.6
            }
        }

def ensure_directories():
    """Ensure all required directories exist"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    required_dirs = [
        'backtest_results',
        'data',
        'logs',
        'models',
        'plots',
        'results',
        'trades',
        os.path.join('performance', 'SP500Strategy')
    ]
    
    for directory in required_dirs:
        dir_path = os.path.join(script_dir, directory)
        os.makedirs(dir_path, exist_ok=True)
        logger.debug(f"Ensured directory exists: {dir_path}")

def run_backtest(args):
    with error_context({"operation": "backtest", "args": vars(args)}) as handler:
        """Run a backtest using the original strategy"""
        logger.info("Starting backtest")
        
        # Load strategy module and configuration
        strategy_module = load_strategy_module()
        config = load_config()
        
        # Ensure all required directories exist
        ensure_directories()
        
        # Get parameters from config if not specified in args
        if args.initial_capital is None and 'initial_capital' in config:
            args.initial_capital = config['initial_capital']
        
        if args.max_signals is None and 'backtest' in config and 'max_signals_per_day' in config['backtest']:
            args.max_signals = config['backtest']['max_signals_per_day']
        
        if args.tier1_threshold is None and 'backtest' in config and 'tier1_threshold' in config['backtest']:
            args.tier1_threshold = config['backtest']['tier1_threshold']
        
        if args.tier2_threshold is None and 'backtest' in config and 'tier2_threshold' in config['backtest']:
            args.tier2_threshold = config['backtest']['tier2_threshold']
        
        if args.tier3_threshold is None and 'backtest' in config and 'tier3_threshold' in config['backtest']:
            args.tier3_threshold = config['backtest']['tier3_threshold']
        
        # Set default values if not in config
        if args.initial_capital is None:
            args.initial_capital = 300
        
        if args.max_signals is None:
            args.max_signals = 40
        
        if args.tier1_threshold is None:
            args.tier1_threshold = 0.8
        
        if args.tier2_threshold is None:
            args.tier2_threshold = 0.7
        
        if args.tier3_threshold is None:
            args.tier3_threshold = 0.6
        
        # Log backtest parameters
        logger.info(f"Running backtest with parameters:")
        logger.info(f"  - Start date: {args.start_date}")
        logger.info(f"  - End date: {args.end_date}")
        logger.info(f"  - Initial capital: {args.initial_capital}")
        logger.info(f"  - Max signals: {args.max_signals}")
        logger.info(f"  - Random seed: {args.random_seed}")
        logger.info(f"  - Weekly selection: {args.weekly_selection}")
        logger.info(f"  - Continuous capital: {args.continuous_capital}")
        logger.info(f"  - Tier 1 threshold: {args.tier1_threshold}")
        logger.info(f"  - Tier 2 threshold: {args.tier2_threshold}")
        logger.info(f"  - Tier 3 threshold: {args.tier3_threshold}")
        
        try:
            # Verify Alpaca credentials before running backtest
            from verify_credentials import verify_credentials
            if not verify_credentials('paper', verbose=False):
                error = APIError(
                    "Alpaca API credentials verification failed",
                    severity="ERROR",
                    details={"mode": "paper"}
                )
                handler.handle_error(error)
                logger.error("Backtest aborted due to invalid Alpaca credentials")
                return
            
            # Call the run_backtest function from the strategy module
            result, signals = strategy_module.run_backtest(
                start_date=args.start_date,
                end_date=args.end_date,
                mode='backtest',
                max_signals=args.max_signals,
                initial_capital=args.initial_capital,
                random_seed=args.random_seed,
                weekly_selection=args.weekly_selection,
                continuous_capital=args.continuous_capital,
                tier1_threshold=args.tier1_threshold,
                tier2_threshold=args.tier2_threshold,
                tier3_threshold=args.tier3_threshold
            )
            
            logger.info("Backtest completed successfully")
            
            # Check if we got signals
            if not signals or len(signals) == 0:
                error = BacktestError(
                    "Backtest did not generate any signals",
                    severity="WARNING",
                    details={
                        "start_date": args.start_date,
                        "end_date": args.end_date,
                        "tier1_threshold": args.tier1_threshold,
                        "tier2_threshold": args.tier2_threshold
                    }
                )
                handler.handle_error(error)
                logger.warning("No signals were generated during the backtest. Check strategy parameters.")
            
            # Log the backtest summary
            if isinstance(result, dict):
                # Create a backtest summary for logging
                backtest_summary = {
                    'start_date': args.start_date,
                    'end_date': args.end_date,
                    'initial_capital': args.initial_capital,
                    'final_equity': result.get('final_capital', args.initial_capital),
                    'return': result.get('total_return', 0),
                    'trade_history': result.get('trade_history', [])
                }
                
                # Log the summary using our specialized logger
                logger.log_backtest_summary(backtest_summary)
                
                # Check if we have trades
                trade_history = result.get('trade_history', [])
                trade_count = len(trade_history)
                
                # Store trade count in the handler context for error recovery
                if handler:
                    handler.context["trade_count"] = trade_count
                    handler.context["backtest_result"] = result
                    # Also store these values to prevent duplicate logging
                    handler.context["start_date"] = args.start_date
                    handler.context["end_date"] = args.end_date
                    handler.context["initial_capital"] = args.initial_capital
                    handler.context["final_capital"] = result.get('final_capital', args.initial_capital)
                    handler.context["total_return"] = result.get('total_return', 0)
                
                # Log the backtest summary regardless of trades
                logger.info(f"Backtest completed: {args.start_date} to {args.end_date}")
                logger.info(f"Initial capital: ${args.initial_capital:.2f}")
                logger.info(f"Final equity: ${result.get('final_capital', args.initial_capital):.2f}")
                logger.info(f"Total return: {result.get('total_return', 0):.2f}%")
                logger.info(f"Total trades: {trade_count}")
                
                # Only generate the "no trades" error if there are actually no trades
                # This prevents the error handler from generating misleading logs
                if trade_history and trade_count > 0:
                    # Log individual trades for detailed analysis
                    for trade in trade_history:
                        logger.log_trade(trade)
                    
                    # IMPORTANT: Add a flag to the context to indicate that we have trades
                    # This will be used by the error handler to suppress the "no trades" error
                    if handler:
                        handler.context["has_trades"] = True
                else:
                    # Only generate the error if there are actually no trades
                    error = BacktestError(
                        "Backtest did not generate any trades",
                        severity="WARNING",
                        details={
                            "start_date": args.start_date,
                            "end_date": args.end_date,
                            "signals_count": len(signals) if signals else 0,
                            "trade_count": 0  # Explicitly set trade count to 0 for the error
                        }
                    )
                    if handler:
                        handler.handle_error(error)
                    logger.warning("No trades were executed during the backtest. Check strategy parameters.")
                    
                # Save the result to a file
                result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
                os.makedirs(result_dir, exist_ok=True)
                
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                result_file = os.path.join(result_dir, f"backtest_results_{timestamp}.json")
                
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
                logger.info(f"Saved backtest results to {result_file}")
                
                # Save with date range in filename
                date_range_file = os.path.join(result_dir, f"backtest_{args.start_date}_to_{args.end_date}.json")
                with open(date_range_file, 'w') as f:
                    json.dump(result, f, indent=2)
                logger.info(f"Saved backtest results to {date_range_file}")
            else:
                error = BacktestError(
                    "Backtest did not return a valid result dictionary",
                    severity="ERROR",
                    details={
                        "start_date": args.start_date,
                        "end_date": args.end_date,
                        "result_type": type(result).__name__
                    }
                )
                handler.handle_error(error)
                logger.warning("Backtest did not return a valid result dictionary")
        
        except Exception as e:
            error = BacktestError(
                f"Error during backtest execution: {str(e)}",
                severity="CRITICAL",
                details={
                    "start_date": args.start_date,
                    "end_date": args.end_date,
                    "exception": str(e),
                    "traceback": traceback.format_exc()
                }
            )
            handler.handle_error(error)
            logger.critical(f"Error during backtest execution: {str(e)}")
            traceback.print_exc()

def run_paper_trading(args):
    """Run paper trading using the original strategy"""
    with error_context({"operation": "paper_trading", "args": vars(args)}) as handler:
        logger.info("Starting paper trading")
        
        # Load strategy module and configuration
        strategy_module = load_strategy_module()
        config = load_config()
        
        # Ensure all required directories exist
        ensure_directories()
        
        # Get parameters from config if not specified in args
        if args.initial_capital is None and 'initial_capital' in config:
            args.initial_capital = config['initial_capital']
        
        if args.max_signals is None and 'paper_trading' in config and 'max_signals_per_day' in config['paper_trading']:
            args.max_signals = config['paper_trading']['max_signals_per_day']
        
        if args.tier1_threshold is None and 'paper_trading' in config and 'tier1_threshold' in config['paper_trading']:
            args.tier1_threshold = config['paper_trading']['tier1_threshold']
        
        if args.tier2_threshold is None and 'paper_trading' in config and 'tier2_threshold' in config['paper_trading']:
            args.tier2_threshold = config['paper_trading']['tier2_threshold']
        
        if args.tier3_threshold is None and 'paper_trading' in config and 'tier3_threshold' in config['paper_trading']:
            args.tier3_threshold = config['paper_trading']['tier3_threshold']
        
        # Set default values if not in config
        if args.initial_capital is None:
            args.initial_capital = 300
        
        if args.max_signals is None:
            args.max_signals = 40
        
        if args.tier1_threshold is None:
            args.tier1_threshold = 0.8
        
        if args.tier2_threshold is None:
            args.tier2_threshold = 0.7
        
        if args.tier3_threshold is None:
            args.tier3_threshold = 0.6
        
        # Log paper trading parameters
        logger.info(f"Running paper trading with parameters:")
        logger.info(f"  - Initial capital: {args.initial_capital}")
        logger.info(f"  - Max signals: {args.max_signals}")
        logger.info(f"  - Tier 1 threshold: {args.tier1_threshold}")
        logger.info(f"  - Tier 2 threshold: {args.tier2_threshold}")
        logger.info(f"  - Tier 3 threshold: {args.tier3_threshold}")
        
        try:
            # Verify Alpaca credentials before running paper trading
            from verify_credentials import verify_credentials
            if not verify_credentials('paper', verbose=True):
                error = APIError(
                    "Alpaca API credentials verification failed",
                    severity="ERROR",
                    details={"mode": "paper"}
                )
                handler.handle_error(error)
                logger.error("Paper trading aborted due to invalid Alpaca credentials")
                return
            
            # Call the run_backtest function from the strategy module with mode='paper'
            result, signals = strategy_module.run_backtest(
                start_date=None,  # Not used in paper trading mode
                end_date=None,    # Not used in paper trading mode
                mode='paper',
                max_signals=args.max_signals,
                initial_capital=args.initial_capital,
                tier1_threshold=args.tier1_threshold,
                tier2_threshold=args.tier2_threshold,
                tier3_threshold=args.tier3_threshold
            )
            
            logger.info("Paper trading completed successfully")
            
            # Check if we got signals
            if not signals or len(signals) == 0:
                error = APIError(
                    "Paper trading did not generate any signals",
                    severity="WARNING",
                    details={
                        "tier1_threshold": args.tier1_threshold,
                        "tier2_threshold": args.tier2_threshold
                    }
                )
                handler.handle_error(error)
                logger.warning("No signals were generated during paper trading. Check strategy parameters.")
            
            # Save the result to a file
            result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
            os.makedirs(result_dir, exist_ok=True)
            
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            result_file = os.path.join(result_dir, f"paper_trading_results_{timestamp}.json")
            
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"Saved paper trading results to {result_file}")
            
            # Log trades if available
            if isinstance(result, dict) and 'trade_history' in result and result['trade_history']:
                logger.info(f"Paper trading executed {len(result['trade_history'])} trades")
                
                # Log individual trades for detailed analysis
                for trade in result['trade_history']:
                    logger.log_trade(trade)
            else:
                logger.warning("No trades were executed during paper trading")
        
        except Exception as e:
            error = APIError(
                f"Error during paper trading execution: {str(e)}",
                severity="CRITICAL",
                details={
                    "exception": str(e),
                    "traceback": traceback.format_exc()
                }
            )
            handler.handle_error(error)
            logger.critical(f"Error during paper trading execution: {str(e)}")
            traceback.print_exc()

def run_live_trading(args):
    """Run live trading using the original strategy"""
    with error_context({"operation": "live_trading", "args": vars(args)}) as handler:
        logger.info("Starting live trading")
        
        # Load strategy module and configuration
        strategy_module = load_strategy_module()
        config = load_config()
        
        # Ensure all required directories exist
        ensure_directories()
        
        # Get parameters from config if not specified in args
        if args.initial_capital is None and 'initial_capital' in config:
            args.initial_capital = config['initial_capital']
        
        if args.max_signals is None and 'live_trading' in config and 'max_signals_per_day' in config['live_trading']:
            args.max_signals = config['live_trading']['max_signals_per_day']
        
        if args.tier1_threshold is None and 'live_trading' in config and 'tier1_threshold' in config['live_trading']:
            args.tier1_threshold = config['live_trading']['tier1_threshold']
        
        if args.tier2_threshold is None and 'live_trading' in config and 'tier2_threshold' in config['live_trading']:
            args.tier2_threshold = config['live_trading']['tier2_threshold']
        
        if args.tier3_threshold is None and 'live_trading' in config and 'tier3_threshold' in config['live_trading']:
            args.tier3_threshold = config['live_trading']['tier3_threshold']
        
        # Set default values if not in config
        if args.initial_capital is None:
            args.initial_capital = 300
        
        if args.max_signals is None:
            args.max_signals = 40
        
        if args.tier1_threshold is None:
            args.tier1_threshold = 0.8
        
        if args.tier2_threshold is None:
            args.tier2_threshold = 0.7
        
        if args.tier3_threshold is None:
            args.tier3_threshold = 0.6
        
        # Log live trading parameters
        logger.info(f"Running live trading with parameters:")
        logger.info(f"  - Initial capital: {args.initial_capital}")
        logger.info(f"  - Max signals: {args.max_signals}")
        logger.info(f"  - Tier 1 threshold: {args.tier1_threshold}")
        logger.info(f"  - Tier 2 threshold: {args.tier2_threshold}")
        logger.info(f"  - Tier 3 threshold: {args.tier3_threshold}")
        
        try:
            # Verify Alpaca credentials before running live trading
            from verify_credentials import verify_credentials
            if not verify_credentials('live', verbose=True):
                error = APIError(
                    "Alpaca API credentials verification failed",
                    severity="ERROR",
                    details={"mode": "live"}
                )
                handler.handle_error(error)
                logger.error("Live trading aborted due to invalid Alpaca credentials")
                return
            
            # Get user confirmation before starting live trading
            print("\n⚠️ WARNING: You are about to start LIVE TRADING with REAL MONEY ⚠️")
            print(f"Initial capital: ${args.initial_capital}")
            print(f"Max signals: {args.max_signals}")
            confirmation = input("\nAre you sure you want to proceed? (yes/no): ")
            
            if confirmation.lower() != "yes":
                logger.info("Live trading aborted by user")
                return
            
            # Call the run_backtest function from the strategy module with mode='live'
            result, signals = strategy_module.run_backtest(
                start_date=None,  # Not used in live trading mode
                end_date=None,    # Not used in live trading mode
                mode='live',
                max_signals=args.max_signals,
                initial_capital=args.initial_capital,
                tier1_threshold=args.tier1_threshold,
                tier2_threshold=args.tier2_threshold,
                tier3_threshold=args.tier3_threshold
            )
            
            logger.info("Live trading completed successfully")
            
            # Check if we got signals
            if not signals or len(signals) == 0:
                error = APIError(
                    "Live trading did not generate any signals",
                    severity="WARNING",
                    details={
                        "tier1_threshold": args.tier1_threshold,
                        "tier2_threshold": args.tier2_threshold
                    }
                )
                handler.handle_error(error)
                logger.warning("No signals were generated during live trading. Check strategy parameters.")
            
            # Save the result to a file
            result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
            os.makedirs(result_dir, exist_ok=True)
            
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            result_file = os.path.join(result_dir, f"live_trading_results_{timestamp}.json")
            
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"Saved live trading results to {result_file}")
            
            # Log trades if available
            if isinstance(result, dict) and 'trade_history' in result and result['trade_history']:
                logger.info(f"Live trading executed {len(result['trade_history'])} trades")
                
                # Log individual trades for detailed analysis
                for trade in result['trade_history']:
                    logger.log_trade(trade)
            else:
                logger.warning("No trades were executed during live trading")
        
        except Exception as e:
            error = APIError(
                f"Error during live trading execution: {str(e)}",
                severity="CRITICAL",
                details={
                    "exception": str(e),
                    "traceback": traceback.format_exc()
                }
            )
            handler.handle_error(error)
            logger.critical(f"Error during live trading execution: {str(e)}")
            traceback.print_exc()

def view_results(args):
    """View backtest, paper trading, or live trading results"""
    logger.info("Viewing results")
    
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    
    if not os.path.exists(result_dir):
        error_handler.handle_error(ConfigurationError(f"Results directory not found: {result_dir}"))
        return
    
    # List all result files
    if args.list:
        result_files = sorted([f for f in os.listdir(result_dir) if f.endswith('.json')], 
                             key=lambda x: os.path.getmtime(os.path.join(result_dir, x)),
                             reverse=True)
        
        if not result_files:
            logger.info("No result files found")
            return
        
        logger.info("Available result files:")
        for i, file in enumerate(result_files):
            file_time = datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(result_dir, file)))
            logger.info(f"{i+1}. {file} - {file_time}")
        
        return
    
    # View the latest result
    if args.latest:
        result_files = sorted([f for f in os.listdir(result_dir) if f.endswith('.json')], 
                             key=lambda x: os.path.getmtime(os.path.join(result_dir, x)),
                             reverse=True)
        
        if not result_files:
            error_handler.handle_error(ConfigurationError("No result files found"))
            return
        
        latest_file = os.path.join(result_dir, result_files[0])
        logger.info(f"Viewing latest result file: {latest_file}")
        view_result_file(latest_file)
        return
    
    # View a specific result file
    if args.file:
        file_path = os.path.join(result_dir, args.file)
        
        if not os.path.exists(file_path):
            # Try to find a file that contains the specified name
            matching_files = [f for f in os.listdir(result_dir) 
                             if f.endswith('.json') and args.file in f]
            
            if not matching_files:
                error_handler.handle_error(ConfigurationError(f"Result file not found: {args.file}"))
                return
            
            file_path = os.path.join(result_dir, matching_files[0])
        
        logger.info(f"Viewing result file: {file_path}")
        view_result_file(file_path)
        return
    
    # If no specific option was chosen, show the list of files
    result_files = sorted([f for f in os.listdir(result_dir) if f.endswith('.json')], 
                         key=lambda x: os.path.getmtime(os.path.join(result_dir, x)),
                         reverse=True)
    
    if not result_files:
        logger.info("No result files found")
        return
    
    logger.info("Available result files:")
    for i, file in enumerate(result_files):
        file_time = datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(result_dir, file)))
        logger.info(f"{i+1}. {file} - {file_time}")

def view_result_file(file_path):
    """View a result file"""
    try:
        with open(file_path, 'r') as f:
            result = json.load(f)
        
        # Extract basic information
        initial_capital = result.get('initial_capital', 0)
        final_equity = result.get('final_equity', 0)
        total_return = result.get('return', 0) * 100  # Convert to percentage
        trade_history = result.get('trade_history', [])
        
        # Calculate additional metrics
        winning_trades = [t for t in trade_history if t.get('profit', 0) > 0]
        win_rate = len(winning_trades) / len(trade_history) * 100 if trade_history else 0
        
        total_profit = sum(t.get('profit', 0) for t in trade_history if t.get('profit', 0) > 0)
        total_loss = abs(sum(t.get('profit', 0) for t in trade_history if t.get('profit', 0) < 0))
        profit_factor = total_profit / total_loss if total_loss else float('inf')
        
        avg_trade = sum(t.get('profit', 0) for t in trade_history) / len(trade_history) if trade_history else 0
        
        # Extract date range if available
        equity_curve = result.get('equity_curve', {})
        date_range = ""
        if equity_curve:
            dates = list(equity_curve.keys())
            if dates:
                start_date = min(dates)
                end_date = max(dates)
                date_range = f"{start_date} to {end_date}"
        
        # Calculate max drawdown
        max_drawdown = 0
        if equity_curve:
            values = list(equity_curve.values())
            peak = values[0]
            
            for value in values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak * 100
                max_drawdown = max(max_drawdown, drawdown)
        
        # Print summary
        logger.info("\nRESULT SUMMARY")
        logger.info("==============")
        if date_range:
            logger.info(f"Period: {date_range}")
        logger.info(f"Initial Capital: ${initial_capital:.2f}")
        logger.info(f"Final Equity: ${final_equity:.2f}")
        logger.info(f"Total Return: {total_return:.2f}%")
        logger.info(f"Total Trades: {len(trade_history)}")
        logger.info(f"Winning Trades: {len(winning_trades)}")
        logger.info(f"Win Rate: {win_rate:.2f}%")
        logger.info(f"Total Profit: ${total_profit:.2f}")
        logger.info(f"Total Loss: ${total_loss:.2f}")
        logger.info(f"Profit Factor: {profit_factor:.2f}")
        logger.info(f"Average Trade: ${avg_trade:.2f}")
        logger.info(f"Max Drawdown: {max_drawdown:.2f}%")
        
        # Print trade history
        if trade_history and len(trade_history) > 0:
            logger.info("\nTRADE HISTORY")
            logger.info("=============")
            
            for i, trade in enumerate(trade_history[:10]):  # Show only first 10 trades
                symbol = trade.get('symbol', 'Unknown')
                entry_date = trade.get('entry_date', 'Unknown')
                exit_date = trade.get('exit_date', 'Unknown')
                entry_price = trade.get('entry_price', 0)
                exit_price = trade.get('exit_price', 0)
                profit = trade.get('profit', 0)
                profit_pct = trade.get('profit_pct', 0) * 100
                
                logger.info(f"Trade {i+1}: {symbol} - Entry: {entry_date} at ${entry_price:.2f}, Exit: {exit_date} at ${exit_price:.2f}, Profit: ${profit:.2f} ({profit_pct:.2f}%)")
            
            if len(trade_history) > 10:
                logger.info(f"... and {len(trade_history) - 10} more trades")
    
    except Exception as e:
        error_handler.handle_error(ConfigurationError(f"Error viewing result file: {str(e)}"))

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Trading CLI for S&P 500 Strategy')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Initialize error handler
    error_handler = get_error_handler("error_handler_config.yaml")
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run a backtest')
    backtest_parser.add_argument('--start-date', dest='start_date', help='Start date (YYYY-MM-DD)')
    backtest_parser.add_argument('--end-date', dest='end_date', help='End date (YYYY-MM-DD)')
    backtest_parser.add_argument('--initial-capital', dest='initial_capital', type=float, help='Initial capital')
    backtest_parser.add_argument('--max-signals', dest='max_signals', type=int, help='Maximum number of signals per day')
    backtest_parser.add_argument('--random-seed', dest='random_seed', type=int, default=42, help='Random seed')
    backtest_parser.add_argument('--weekly-selection', dest='weekly_selection', action='store_true', help='Use weekly symbol selection')
    backtest_parser.add_argument('--continuous-capital', dest='continuous_capital', action='store_true', help='Use continuous capital')
    backtest_parser.add_argument('--tier1-threshold', dest='tier1_threshold', type=float, help='Tier 1 threshold')
    backtest_parser.add_argument('--tier2-threshold', dest='tier2_threshold', type=float, help='Tier 2 threshold')
    backtest_parser.add_argument('--tier3-threshold', dest='tier3_threshold', type=float, help='Tier 3 threshold')
    
    # Paper trading command
    paper_parser = subparsers.add_parser('paper', help='Run paper trading')
    paper_parser.add_argument('--initial-capital', dest='initial_capital', type=float, help='Initial capital')
    paper_parser.add_argument('--max-signals', dest='max_signals', type=int, help='Maximum number of signals per day')
    paper_parser.add_argument('--tier1-threshold', dest='tier1_threshold', type=float, help='Tier 1 threshold')
    paper_parser.add_argument('--tier2-threshold', dest='tier2_threshold', type=float, help='Tier 2 threshold')
    paper_parser.add_argument('--tier3-threshold', dest='tier3_threshold', type=float, help='Tier 3 threshold')
    
    # Live trading command
    live_parser = subparsers.add_parser('live', help='Run live trading')
    live_parser.add_argument('--initial-capital', dest='initial_capital', type=float, help='Initial capital')
    live_parser.add_argument('--max-signals', dest='max_signals', type=int, help='Maximum number of signals per day')
    live_parser.add_argument('--tier1-threshold', dest='tier1_threshold', type=float, help='Tier 1 threshold')
    live_parser.add_argument('--tier2-threshold', dest='tier2_threshold', type=float, help='Tier 2 threshold')
    live_parser.add_argument('--tier3-threshold', dest='tier3_threshold', type=float, help='Tier 3 threshold')
    
    # Results command
    results_parser = subparsers.add_parser('results', help='View results')
    results_parser.add_argument('--list', action='store_true', help='List all result files')
    results_parser.add_argument('--latest', action='store_true', help='View the latest result')
    results_parser.add_argument('--file', help='View a specific result file')
    
    args = parser.parse_args()
    
    # Run the appropriate command
    try:
        if args.command == 'backtest':
            run_backtest(args)
        elif args.command == 'paper':
            run_paper_trading(args)
        elif args.command == 'live':
            run_live_trading(args)
        elif args.command == 'results':
            view_results(args)
        else:
            parser.print_help()
    except Exception as e:
        logger.critical(f"Unhandled exception in main: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
