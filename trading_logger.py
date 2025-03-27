#!/usr/bin/env python
"""
Trading System Logger
--------------------
A comprehensive logging system for the S&P 500 Multi-Strategy Trading System.
This module provides structured logging capabilities for backtesting, paper trading,
and live trading operations, with configurable verbosity levels and output formats.
"""

import os
import sys
import logging
import json
import yaml
from datetime import datetime
import traceback
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

# Default log directory
DEFAULT_LOG_DIR = "logs"

# Log levels
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

class TradingLogger:
    """
    Trading system logger that provides structured logging for all trading operations.
    Features:
    - Multiple output formats (console, file, JSON)
    - Rotating log files to prevent excessive disk usage
    - Different log levels for different components
    - Trade-specific logging with performance metrics
    - Error tracking with stack traces
    """
    
    def __init__(self, name="trading_system", config_file=None):
        """
        Initialize the logger with the specified name and configuration.
        
        Args:
            name (str): Logger name, used as a prefix for log files
            config_file (str): Path to a YAML configuration file for logging settings
        """
        self.name = name
        self.loggers = {}
        self.config = self._load_config(config_file)
        self._setup_log_directory()
        
        # Create the main logger
        self.main_logger = self._create_logger("main")
        
        # Create specialized loggers
        self.trade_logger = self._create_logger("trade")
        self.performance_logger = self._create_logger("performance")
        self.error_logger = self._create_logger("error")
        
        self.main_logger.info(f"Trading logger initialized: {name}")
    
    def _load_config(self, config_file):
        """Load logger configuration from a YAML file."""
        default_config = {
            "log_dir": DEFAULT_LOG_DIR,
            "console_level": "INFO",
            "file_level": "DEBUG",
            "max_file_size_mb": 10,
            "backup_count": 5,
            "enable_json_logs": True,
            "enable_trade_logs": True,
            "enable_performance_logs": True,
            "enable_error_logs": True,
            "log_rotation": "size"  # 'size' or 'time'
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = yaml.safe_load(f)
                    if user_config and isinstance(user_config, dict):
                        # Merge with default config
                        default_config.update(user_config)
            except Exception as e:
                print(f"Error loading logger config: {e}")
        
        # Also check if there's a logging section in the main config
        try:
            with open('sp500_config.yaml', 'r') as f:
                main_config = yaml.safe_load(f)
                if main_config and isinstance(main_config, dict) and 'logging' in main_config:
                    default_config.update(main_config['logging'])
        except Exception:
            pass
            
        return default_config
    
    def _setup_log_directory(self):
        """Create the log directory if it doesn't exist."""
        log_dir = self.config["log_dir"]
        if not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir)
                print(f"Created log directory: {log_dir}")
            except Exception as e:
                print(f"Error creating log directory: {e}")
                # Fall back to current directory
                self.config["log_dir"] = "."
    
    def _create_logger(self, logger_type):
        """
        Create a logger with the specified type.
        
        Args:
            logger_type (str): Type of logger to create (main, trade, performance, error)
            
        Returns:
            logging.Logger: Configured logger instance
        """
        logger_name = f"{self.name}.{logger_type}"
        logger = logging.getLogger(logger_name)
        
        # Avoid adding handlers multiple times
        if logger_name in self.loggers:
            return self.loggers[logger_name]
        
        # Set the level to the lowest level we'll use
        logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        if logger.handlers:
            logger.handlers = []
        
        # Console handler
        console_level = LOG_LEVELS.get(self.config["console_level"].upper(), logging.INFO)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler
        if self.config.get(f"enable_{logger_type}_logs", True):
            file_level = LOG_LEVELS.get(self.config["file_level"].upper(), logging.DEBUG)
            log_file = os.path.join(self.config["log_dir"], f"{logger_type}_{datetime.now().strftime('%Y%m%d')}.log")
            
            # Choose rotation method
            if self.config["log_rotation"] == "time":
                file_handler = TimedRotatingFileHandler(
                    log_file,
                    when="midnight",
                    backupCount=self.config["backup_count"]
                )
            else:  # Default to size-based rotation
                file_handler = RotatingFileHandler(
                    log_file,
                    maxBytes=self.config["max_file_size_mb"] * 1024 * 1024,
                    backupCount=self.config["backup_count"]
                )
            
            file_handler.setLevel(file_level)
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
            # JSON handler for structured logging
            if self.config["enable_json_logs"]:
                json_log_file = os.path.join(self.config["log_dir"], f"{logger_type}_{datetime.now().strftime('%Y%m%d')}.json")
                json_handler = RotatingFileHandler(
                    json_log_file,
                    maxBytes=self.config["max_file_size_mb"] * 1024 * 1024,
                    backupCount=self.config["backup_count"]
                )
                json_handler.setLevel(file_level)
                
                class JsonFormatter(logging.Formatter):
                    def format(self, record):
                        log_data = {
                            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                            "level": record.levelname,
                            "logger": record.name,
                            "message": record.getMessage(),
                            "module": record.module,
                            "function": record.funcName,
                            "line": record.lineno
                        }
                        if hasattr(record, 'trade_data'):
                            log_data['trade_data'] = record.trade_data
                        if hasattr(record, 'performance_data'):
                            log_data['performance_data'] = record.performance_data
                        if hasattr(record, 'error_data'):
                            log_data['error_data'] = record.error_data
                        return json.dumps(log_data)
                
                json_handler.setFormatter(JsonFormatter())
                logger.addHandler(json_handler)
        
        self.loggers[logger_name] = logger
        return logger
    
    def debug(self, message, *args, **kwargs):
        """Log a debug message."""
        self.main_logger.debug(message, *args, **kwargs)
    
    def info(self, message, *args, **kwargs):
        """Log an info message."""
        self.main_logger.info(message, *args, **kwargs)
    
    def warning(self, message, *args, **kwargs):
        """Log a warning message."""
        self.main_logger.warning(message, *args, **kwargs)
    
    def error(self, message, *args, **kwargs):
        """Log an error message."""
        self.main_logger.error(message, *args, **kwargs)
        
        # Also log to the error logger with stack trace
        error_record = logging.LogRecord(
            name=self.error_logger.name,
            level=logging.ERROR,
            pathname=__file__,
            lineno=0,
            msg=message,
            args=args,
            exc_info=sys.exc_info()
        )
        error_record.error_data = {
            "traceback": traceback.format_exc() if sys.exc_info()[0] else "No traceback available",
            "args": kwargs.get("extra", {})
        }
        for handler in self.error_logger.handlers:
            handler.handle(error_record)
    
    def critical(self, message, *args, **kwargs):
        """Log a critical message."""
        self.main_logger.critical(message, *args, **kwargs)
        
        # Also log to the error logger with stack trace
        error_record = logging.LogRecord(
            name=self.error_logger.name,
            level=logging.CRITICAL,
            pathname=__file__,
            lineno=0,
            msg=message,
            args=args,
            exc_info=sys.exc_info()
        )
        error_record.error_data = {
            "traceback": traceback.format_exc() if sys.exc_info()[0] else "No traceback available",
            "args": kwargs.get("extra", {})
        }
        for handler in self.error_logger.handlers:
            handler.handle(error_record)
    
    def log_trade(self, trade_data):
        """
        Log trade information.
        
        Args:
            trade_data (dict): Dictionary containing trade information
        """
        if not self.config.get("enable_trade_logs", True):
            return
            
        # Create a custom record with trade data
        trade_record = logging.LogRecord(
            name=self.trade_logger.name,
            level=logging.INFO,
            pathname=__file__,
            lineno=0,
            msg=f"Trade: {trade_data.get('symbol')} {trade_data.get('action')} at {trade_data.get('price')}",
            args=(),
            exc_info=None
        )
        trade_record.trade_data = trade_data
        
        # Process through all handlers
        for handler in self.trade_logger.handlers:
            handler.handle(trade_record)
    
    def log_performance(self, performance_data):
        """
        Log performance metrics.
        
        Args:
            performance_data (dict): Dictionary containing performance metrics
        """
        if not self.config.get("enable_performance_logs", True):
            return
            
        # Create a custom record with performance data
        perf_record = logging.LogRecord(
            name=self.performance_logger.name,
            level=logging.INFO,
            pathname=__file__,
            lineno=0,
            msg=f"Performance update: equity={performance_data.get('equity')}, return={performance_data.get('return')}%",
            args=(),
            exc_info=None
        )
        perf_record.performance_data = performance_data
        
        # Process through all handlers
        for handler in self.performance_logger.handlers:
            handler.handle(perf_record)
    
    def log_backtest_summary(self, backtest_data):
        """
        Log a summary of backtest results.
        
        Args:
            backtest_data (dict): Dictionary containing backtest results
        """
        self.info(f"Backtest completed: {backtest_data.get('start_date')} to {backtest_data.get('end_date')}")
        self.info(f"Initial capital: ${backtest_data.get('initial_capital'):.2f}")
        self.info(f"Final equity: ${backtest_data.get('final_equity'):.2f}")
        self.info(f"Total return: {backtest_data.get('return', 0):.2f}%")
        self.info(f"Total trades: {len(backtest_data.get('trade_history', []))}")
        
        # Also log detailed performance data
        self.log_performance({
            "type": "backtest_summary",
            "start_date": backtest_data.get('start_date'),
            "end_date": backtest_data.get('end_date'),
            "initial_capital": backtest_data.get('initial_capital'),
            "final_equity": backtest_data.get('final_equity'),
            "return": backtest_data.get('return', 0),
            "trade_count": len(backtest_data.get('trade_history', [])),
            "timestamp": datetime.now().isoformat()
        })

# Global logger instance
_logger = None

def get_logger(name="trading_system", config_file=None):
    """
    Get or create a logger instance.
    
    Args:
        name (str): Logger name
        config_file (str): Path to logger configuration file
        
    Returns:
        TradingLogger: Logger instance
    """
    global _logger
    if _logger is None:
        _logger = TradingLogger(name, config_file)
    return _logger

# Example usage
if __name__ == "__main__":
    # Example of how to use the logger
    logger = get_logger("example_trading_system")
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    
    try:
        # Simulate an error
        result = 1 / 0
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    
    # Log a trade
    logger.log_trade({
        "symbol": "AAPL",
        "action": "BUY",
        "quantity": 10,
        "price": 150.25,
        "timestamp": datetime.now().isoformat(),
        "order_id": "12345",
        "strategy": "momentum"
    })
    
    # Log performance
    logger.log_performance({
        "equity": 10500.75,
        "cash": 5000.25,
        "positions_value": 5500.50,
        "return": 5.01,
        "drawdown": 0.5,
        "sharpe": 1.2,
        "timestamp": datetime.now().isoformat()
    })
    
    # Log backtest summary
    logger.log_backtest_summary({
        "start_date": "2023-01-01",
        "end_date": "2023-03-31",
        "initial_capital": 10000,
        "final_equity": 10501.23,
        "return": 0.05,
        "trade_history": [{"id": 1}, {"id": 2}, {"id": 3}]
    })
