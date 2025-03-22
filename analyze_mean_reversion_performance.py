#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analyze the performance of the optimized MeanReversion strategy using historical data from 2023.
"""

import os
import sys
import json
import datetime as dt
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional
import alpaca_trade_api as tradeapi
from dataclasses import dataclass
import yaml

# Define necessary classes and enums from multi_strategy_system
class TradeDirection:
    LONG = "LONG"
    SHORT = "SHORT"

class SignalStrength:
    STRONG_BUY = "STRONG_BUY"
    MODERATE_BUY = "MODERATE_BUY"
    WEAK_BUY = "WEAK_BUY"
    WEAK_SELL = "WEAK_SELL"
    MODERATE_SELL = "MODERATE_SELL"
    STRONG_SELL = "STRONG_SELL"

@dataclass
class CandleData:
    """Class to represent candle data."""
    timestamp: dt.datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    symbol: str = None

@dataclass
class Signal:
    """Class to represent a trading signal."""
    symbol: str
    direction: str
    strategy: str
    entry_price: float
    stop_loss: float
    take_profit: float
    timestamp: dt.datetime
    strength: str
    expiration: dt.datetime

@dataclass
class Trade:
    """Class to represent a trade with entry and exit information."""
    symbol: str
    direction: str
    entry_price: float
    entry_time: dt.datetime
    exit_price: float = None
    exit_time: dt.datetime = None
    stop_loss: float = None
    take_profit: float = None
    pnl: float = None
    pnl_percent: float = None
    status: str = "OPEN"  # OPEN, CLOSED, STOPPED, TARGET_HIT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MeanReversionAnalysis')

# Create a directory for plots if it doesn't exist
os.makedirs('performance_analysis', exist_ok=True)

def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_file}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}

def calculate_bollinger_bands(candles: List[CandleData], period: int = 20, std_dev: float = 2.0) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Calculate Bollinger Bands for the given candles"""
    if len(candles) < period:
        return None, None, None
    
    # Calculate SMA
    closes = [candle.close for candle in candles[-period:]]
    sma = sum(closes) / period
    
    # Calculate standard deviation
    variance = sum((price - sma) ** 2 for price in closes) / period
    std = variance ** 0.5
    
    # Calculate upper and lower bands
    upper_band = sma + (std_dev * std)
    lower_band = sma - (std_dev * std)
    
    return sma, upper_band, lower_band

def calculate_rsi(candles: List[CandleData], period: int = 14) -> Optional[float]:
    """Calculate RSI for the given candles"""
    if len(candles) < period + 1:
        return None
    
    # Calculate price changes
    deltas = [candles[i].close - candles[i-1].close for i in range(1, len(candles))]
    
    # Calculate gains and losses
    gains = [delta if delta > 0 else 0 for delta in deltas]
    losses = [-delta if delta < 0 else 0 for delta in deltas]
    
    # Calculate average gains and losses
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    
    if avg_loss == 0:
        return 100
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_atr(candles: List[CandleData], period: int = 14) -> Optional[float]:
    """Calculate ATR for the given candles"""
    if len(candles) < period + 1:
        return None
    
    # Calculate true ranges
    true_ranges = []
    for i in range(1, len(candles)):
        high = candles[i].high
        low = candles[i].low
        prev_close = candles[i-1].close
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = max(tr1, tr2, tr3)
        true_ranges.append(true_range)
    
    # Calculate ATR
    atr = sum(true_ranges[-period:]) / period
    
    return atr

def fetch_historical_data(symbol: str, start_date: dt.datetime, end_date: dt.datetime) -> List[CandleData]:
    """Fetch historical data from Alpaca API"""
    try:
        # Load Alpaca credentials
        with open('alpaca_credentials.json', 'r') as f:
            credentials = json.load(f)
        
        # Use paper trading credentials
        paper_creds = credentials.get('paper', {})
        api_key = paper_creds.get('api_key')
        api_secret = paper_creds.get('api_secret')
        base_url = paper_creds.get('base_url', 'https://paper-api.alpaca.markets/v2')
        
        if not api_key or not api_secret:
            logger.error("Missing Alpaca API credentials")
            return []
        
        # Initialize Alpaca API
        api = tradeapi.REST(api_key, api_secret, base_url)
        
        # Format dates
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Fetch historical data
        logger.info(f"Fetching historical data for {symbol} from {start_str} to {end_str}")
        bars = api.get_bars(symbol, '1D', start=start_str, end=end_str).df
        
        # Convert to CandleData objects
        candles = []
        for index, row in bars.iterrows():
            candle = CandleData(
                timestamp=index.to_pydatetime(),
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=int(row['volume'])
            )
            candle.symbol = symbol
            candles.append(candle)
        
        logger.info(f"Fetched {len(candles)} candles for {symbol}")
        return candles
    
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        return []

def generate_mean_reversion_signals(candles: List[CandleData], config: Dict[str, Any], symbol: str) -> List[Signal]:
    """Generate signals using the MeanReversion strategy with the provided configuration"""
    signals = []
    
    # Extract parameters from config
    bb_period = config.get('bb_period', 20)
    bb_std_dev = config.get('bb_std_dev', 2.0)
    rsi_period = config.get('rsi_period', 14)
    rsi_overbought = config.get('rsi_overbought', 70)
    rsi_oversold = config.get('rsi_oversold', 30)
    min_reversal_candles = config.get('min_reversal_candles', 2)
    require_reversal = config.get('require_reversal', True)
    stop_loss_atr = config.get('stop_loss_atr', 2.0)
    take_profit_atr = config.get('take_profit_atr', 3.0)
    
    # Need enough candles for calculations
    if len(candles) < max(bb_period, rsi_period) + 1:
        logger.warning(f"Not enough candles for calculations. Need at least {max(bb_period, rsi_period) + 1}, got {len(candles)}")
        return signals
    
    # Generate signals for each candle (except the first few needed for indicators)
    all_signals = []
    for i in range(max(bb_period, rsi_period) + 1, len(candles)):
        # Use candles up to current point for calculations
        current_candles = candles[:i+1]
        current_candle = current_candles[-1]
        
        # Calculate indicators
        sma, upper_band, lower_band = calculate_bollinger_bands(current_candles, bb_period, bb_std_dev)
        rsi = calculate_rsi(current_candles, rsi_period)
        atr = calculate_atr(current_candles, 14)
        
        if sma is None or rsi is None or atr is None:
            continue
        
        current_price = current_candle.close
        
        # Check for oversold condition (buy signal)
        if current_price < lower_band * 1.05 and rsi < rsi_oversold * 1.2:
            # Check for price reversal if required
            reversal = True
            if require_reversal:
                reversal = current_candle.close > current_candles[-2].close
            
            if reversal:
                # Calculate stop loss and take profit using ATR
                stop_loss = current_price - (atr * stop_loss_atr)
                take_profit = current_price + (atr * take_profit_atr)
                
                # Determine signal strength
                rsi_strength = (rsi_oversold * 1.2 - rsi) / (rsi_oversold * 1.2)
                price_strength = (lower_band * 1.05 - current_price) / (lower_band * 1.05)
                
                if rsi_strength > 0.2 and price_strength > 0.05:
                    strength = SignalStrength.STRONG_BUY
                elif rsi_strength > 0.1 or price_strength > 0.02:
                    strength = SignalStrength.MODERATE_BUY
                else:
                    strength = SignalStrength.WEAK_BUY
                
                signal = Signal(
                    symbol=symbol,
                    direction=TradeDirection.LONG,
                    strategy="MeanReversion",
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    timestamp=current_candle.timestamp,
                    strength=strength,
                    expiration=current_candle.timestamp + dt.timedelta(days=3)
                )
                all_signals.append(signal)
        
        # Check for overbought condition (sell signal)
        elif current_price > upper_band / 1.05 and rsi > rsi_overbought / 1.2:
            # Check for price reversal if required
            reversal = True
            if require_reversal:
                reversal = current_candle.close < current_candles[-2].close
            
            if reversal:
                # Calculate stop loss and take profit using ATR
                stop_loss = current_price + (atr * stop_loss_atr)
                take_profit = current_price - (atr * take_profit_atr)
                
                # Determine signal strength
                rsi_strength = (rsi - (rsi_overbought / 1.2)) / (100 - (rsi_overbought / 1.2))
                price_strength = (current_price - (upper_band / 1.05)) / (upper_band / 1.05)
                
                if rsi_strength > 0.2 and price_strength > 0.05:
                    strength = SignalStrength.STRONG_SELL
                elif rsi_strength > 0.1 or price_strength > 0.02:
                    strength = SignalStrength.MODERATE_SELL
                else:
                    strength = SignalStrength.WEAK_SELL
                
                signal = Signal(
                    symbol=symbol,
                    direction=TradeDirection.SHORT,
                    strategy="MeanReversion",
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    timestamp=current_candle.timestamp,
                    strength=strength,
                    expiration=current_candle.timestamp + dt.timedelta(days=3)
                )
                all_signals.append(signal)
    
    return all_signals

def simulate_trades(candles: List[CandleData], signals: List[Signal]) -> List[Trade]:
    """Simulate trades based on signals and candle data"""
    if not signals or not candles:
        return []
    
    # Sort candles and signals by timestamp
    candles = sorted(candles, key=lambda x: x.timestamp)
    signals = sorted(signals, key=lambda x: x.timestamp)
    
    # Create a dictionary mapping timestamps to candles for easy lookup
    candle_dict = {candle.timestamp: candle for candle in candles}
    
    # Create a list to store trades
    trades = []
    
    # Simulate trades
    for signal in signals:
        # Find candles after the signal
        future_candles = [c for c in candles if c.timestamp > signal.timestamp]
        
        if not future_candles:
            continue
        
        # Create a new trade
        trade = Trade(
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=signal.entry_price,
            entry_time=signal.timestamp,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            status="OPEN"
        )
        
        # Simulate the trade
        for candle in future_candles:
            # Check if stop loss was hit
            if signal.direction == TradeDirection.LONG and candle.low <= signal.stop_loss:
                trade.exit_price = signal.stop_loss
                trade.exit_time = candle.timestamp
                trade.status = "STOPPED"
                break
            elif signal.direction == TradeDirection.SHORT and candle.high >= signal.stop_loss:
                trade.exit_price = signal.stop_loss
                trade.exit_time = candle.timestamp
                trade.status = "STOPPED"
                break
            
            # Check if take profit was hit
            if signal.direction == TradeDirection.LONG and candle.high >= signal.take_profit:
                trade.exit_price = signal.take_profit
                trade.exit_time = candle.timestamp
                trade.status = "TARGET_HIT"
                break
            elif signal.direction == TradeDirection.SHORT and candle.low <= signal.take_profit:
                trade.exit_price = signal.take_profit
                trade.exit_time = candle.timestamp
                trade.status = "TARGET_HIT"
                break
            
            # Check if signal expired (3 days)
            if candle.timestamp >= signal.timestamp + dt.timedelta(days=3):
                trade.exit_price = candle.close
                trade.exit_time = candle.timestamp
                trade.status = "CLOSED"
                break
        
        # If the trade is still open at the end of the data, close it at the last price
        if trade.status == "OPEN":
            trade.exit_price = future_candles[-1].close
            trade.exit_time = future_candles[-1].timestamp
            trade.status = "CLOSED"
        
        # Calculate P&L
        if signal.direction == TradeDirection.LONG:
            trade.pnl = trade.exit_price - trade.entry_price
            trade.pnl_percent = (trade.exit_price / trade.entry_price - 1) * 100
        else:
            trade.pnl = trade.entry_price - trade.exit_price
            trade.pnl_percent = (trade.entry_price / trade.exit_price - 1) * 100
        
        trades.append(trade)
    
    return trades

def analyze_performance(trades: List[Trade]) -> Dict[str, Any]:
    """Analyze the performance of trades"""
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0,
            "avg_profit": 0,
            "avg_loss": 0,
            "profit_factor": 0,
            "avg_pnl_percent": 0,
            "max_drawdown": 0,
            "sharpe_ratio": 0
        }
    
    # Calculate basic statistics
    total_trades = len(trades)
    winning_trades = [t for t in trades if t.pnl > 0]
    losing_trades = [t for t in trades if t.pnl <= 0]
    
    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
    
    avg_profit = sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0
    avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0
    
    total_profit = sum(t.pnl for t in winning_trades)
    total_loss = abs(sum(t.pnl for t in losing_trades))
    
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    
    avg_pnl_percent = sum(t.pnl_percent for t in trades) / total_trades
    
    # Calculate equity curve
    equity_curve = []
    initial_equity = 10000  # Starting with $10,000
    current_equity = initial_equity
    
    # Sort trades by exit time
    sorted_trades = sorted(trades, key=lambda x: x.exit_time)
    
    for trade in sorted_trades:
        # Assume each trade uses 10% of the equity
        position_size = current_equity * 0.1
        trade_pnl = position_size * (trade.pnl_percent / 100)
        current_equity += trade_pnl
        equity_curve.append((trade.exit_time, current_equity))
    
    # Calculate drawdown
    max_equity = initial_equity
    max_drawdown = 0
    
    for _, equity in equity_curve:
        max_equity = max(max_equity, equity)
        drawdown = (max_equity - equity) / max_equity
        max_drawdown = max(max_drawdown, drawdown)
    
    # Calculate Sharpe ratio (assuming risk-free rate of 0)
    if len(equity_curve) > 1:
        equity_values = [e[1] for e in equity_curve]
        returns = [(equity_values[i] / equity_values[i-1] - 1) for i in range(1, len(equity_values))]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
    else:
        sharpe_ratio = 0
    
    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "avg_profit": avg_profit,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "avg_pnl_percent": avg_pnl_percent,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "equity_curve": equity_curve,
        "final_equity": current_equity,
        "return_percent": (current_equity / initial_equity - 1) * 100
    }

def plot_equity_curve(equity_curve: List[Tuple[dt.datetime, float]], symbol: str, config_name: str):
    """Plot the equity curve"""
    if not equity_curve:
        return
    
    dates = [e[0] for e in equity_curve]
    equity = [e[1] for e in equity_curve]
    
    plt.figure(figsize=(12, 6))
    plt.plot(dates, equity, label=f'{symbol} Equity Curve')
    plt.title(f'Equity Curve for {symbol} - {config_name}')
    plt.xlabel('Date')
    plt.ylabel('Equity ($)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'performance_analysis/{symbol}_{config_name}_equity_curve.png')
    plt.close()

def plot_trade_distribution(trades: List[Trade], symbol: str, config_name: str):
    """Plot the distribution of trade P&L"""
    if not trades:
        return
    
    pnl_values = [t.pnl_percent for t in trades]
    
    plt.figure(figsize=(12, 6))
    plt.hist(pnl_values, bins=20, alpha=0.7)
    plt.axvline(0, color='r', linestyle='--')
    plt.title(f'Trade P&L Distribution for {symbol} - {config_name}')
    plt.xlabel('P&L (%)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'performance_analysis/{symbol}_{config_name}_pnl_distribution.png')
    plt.close()

def compare_configurations(symbols: List[str], configs: Dict[str, Dict[str, Any]], start_date: dt.datetime, end_date: dt.datetime):
    """Compare the performance of different configurations"""
    results = {}
    
    for symbol in symbols:
        symbol_results = {}
        
        # Fetch historical data once for each symbol
        candles = fetch_historical_data(symbol, start_date, end_date)
        
        if not candles:
            logger.warning(f"No historical data available for {symbol}")
            continue
        
        for config_name, config in configs.items():
            logger.info(f"Testing {config_name} configuration for {symbol}")
            
            # Generate signals
            signals = generate_mean_reversion_signals(candles, config, symbol)
            
            if not signals:
                logger.warning(f"No signals generated for {symbol} with {config_name}")
                continue
            
            logger.info(f"Generated {len(signals)} signals for {symbol} with {config_name}")
            
            # Simulate trades
            trades = simulate_trades(candles, signals)
            
            if not trades:
                logger.warning(f"No trades executed for {symbol} with {config_name}")
                continue
            
            logger.info(f"Simulated {len(trades)} trades for {symbol} with {config_name}")
            
            # Analyze performance
            performance = analyze_performance(trades)
            
            logger.info(f"Performance for {symbol} with {config_name}:")
            logger.info(f"  Total trades: {performance['total_trades']}")
            logger.info(f"  Win rate: {performance['win_rate']:.2%}")
            logger.info(f"  Profit factor: {performance['profit_factor']:.2f}")
            logger.info(f"  Average P&L: {performance['avg_pnl_percent']:.2f}%")
            logger.info(f"  Max drawdown: {performance['max_drawdown']:.2%}")
            logger.info(f"  Sharpe ratio: {performance['sharpe_ratio']:.2f}")
            logger.info(f"  Return: {performance['return_percent']:.2f}%")
            
            # Plot equity curve
            plot_equity_curve(performance['equity_curve'], symbol, config_name)
            
            # Plot trade distribution
            plot_trade_distribution(trades, symbol, config_name)
            
            # Store results
            symbol_results[config_name] = {
                "signals": len(signals),
                "trades": len(trades),
                "win_rate": performance['win_rate'],
                "profit_factor": performance['profit_factor'],
                "avg_pnl_percent": performance['avg_pnl_percent'],
                "max_drawdown": performance['max_drawdown'],
                "sharpe_ratio": performance['sharpe_ratio'],
                "return_percent": performance['return_percent']
            }
        
        results[symbol] = symbol_results
    
    return results

def plot_comparison_chart(results: Dict[str, Dict[str, Dict[str, Any]]]):
    """Plot a comparison chart of different configurations"""
    if not results:
        return
    
    # Prepare data for plotting
    symbols = list(results.keys())
    config_names = list(results[symbols[0]].keys()) if symbols else []
    
    metrics = ['win_rate', 'profit_factor', 'sharpe_ratio', 'return_percent']
    metric_labels = ['Win Rate', 'Profit Factor', 'Sharpe Ratio', 'Return (%)']
    
    for metric, metric_label in zip(metrics, metric_labels):
        plt.figure(figsize=(14, 8))
        
        x = np.arange(len(symbols))
        width = 0.8 / len(config_names)
        
        for i, config_name in enumerate(config_names):
            values = [results[symbol][config_name][metric] if symbol in results and config_name in results[symbol] else 0 for symbol in symbols]
            plt.bar(x + i * width - 0.4 + width / 2, values, width, label=config_name)
        
        plt.xlabel('Symbol')
        plt.ylabel(metric_label)
        plt.title(f'Comparison of {metric_label} Across Configurations')
        plt.xticks(x, symbols)
        plt.legend()
        plt.grid(True, axis='y')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(f'performance_analysis/comparison_{metric}.png')
        plt.close()
    
    # Create a summary table
    summary_data = []
    
    for config_name in config_names:
        row = [config_name]
        
        for metric in metrics:
            values = [results[symbol][config_name][metric] for symbol in symbols if symbol in results and config_name in results[symbol]]
            avg_value = sum(values) / len(values) if values else 0
            row.append(avg_value)
        
        summary_data.append(row)
    
    # Create a DataFrame for the summary
    summary_df = pd.DataFrame(summary_data, columns=['Configuration'] + metric_labels)
    
    # Save the summary to a CSV file
    summary_df.to_csv('performance_analysis/configuration_comparison_summary.csv', index=False)
    
    logger.info("Comparison charts and summary created")
    logger.info("\nConfiguration Comparison Summary:")
    logger.info(summary_df.to_string(index=False))

def main():
    """Main function to analyze the performance of the MeanReversion strategy"""
    # Define symbols to test
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    
    # Define date range for 2023 (use multiple periods for more robust testing)
    periods = [
        ("Q1 2023", dt.datetime(2023, 1, 1), dt.datetime(2023, 3, 31)),
        ("Q2 2023", dt.datetime(2023, 4, 1), dt.datetime(2023, 6, 30)),
        ("Q3 2023", dt.datetime(2023, 7, 1), dt.datetime(2023, 9, 30)),
        ("Q4 2023", dt.datetime(2023, 10, 1), dt.datetime(2023, 12, 31))
    ]
    
    # Load configurations
    default_config = load_config('configuration_12.yaml')
    optimized_config = load_config('configuration_mean_reversion_optimized.yaml')
    
    # Extract MeanReversion strategy configurations
    mr_configs = {
        "Default": default_config.get('strategies', {}).get('MeanReversion', {}),
        "Optimized": optimized_config.get('strategies', {}).get('MeanReversion', {})
    }
    
    # Create a variant with different parameters for comparison
    variant_config = mr_configs["Optimized"].copy()
    variant_config['bb_std_dev'] = 2.0  # Wider Bollinger Bands
    variant_config['require_reversal'] = True  # Require reversal pattern
    
    mr_configs["Variant"] = variant_config
    
    # Analyze each period
    all_results = {}
    
    for period_name, start_date, end_date in periods:
        logger.info(f"\nAnalyzing period: {period_name} ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})")
        
        # Compare configurations
        results = compare_configurations(symbols, mr_configs, start_date, end_date)
        
        # Plot comparison chart
        plot_comparison_chart(results)
        
        # Store results
        all_results[period_name] = results
    
    # Create overall summary
    logger.info("\nOverall Performance Summary:")
    
    for period_name, results in all_results.items():
        logger.info(f"\n{period_name} Summary:")
        
        for config_name in mr_configs.keys():
            avg_win_rate = sum(results[symbol][config_name]['win_rate'] for symbol in symbols if symbol in results and config_name in results[symbol]) / len(symbols) if symbols else 0
            avg_return = sum(results[symbol][config_name]['return_percent'] for symbol in symbols if symbol in results and config_name in results[symbol]) / len(symbols) if symbols else 0
            
            logger.info(f"  {config_name} Configuration:")
            logger.info(f"    Average Win Rate: {avg_win_rate:.2%}")
            logger.info(f"    Average Return: {avg_return:.2f}%")
    
    logger.info("\nAnalysis complete. Results saved to 'performance_analysis' directory.")

if __name__ == "__main__":
    main()
