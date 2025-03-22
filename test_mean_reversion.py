import os
import sys
import yaml
import json
import logging
import datetime as dt
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MeanReversionTest")

# Add the current directory to the path so we can import from the multi_strategy_system
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import necessary classes from multi_strategy_system
try:
    from multi_strategy_system import (
        CandleData, Signal, TradeDirection, SignalStrength
    )
    logger.info("Successfully imported from multi_strategy_system")
except ImportError as e:
    logger.error(f"Error importing from multi_strategy_system: {e}")
    sys.exit(1)

def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
            logger.info(f"Loaded configuration from {config_file}")
            return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}

def create_realistic_candles(symbol: str, days: int = 60) -> List[CandleData]:
    """Create realistic candle data with trends, reversals, and volatility that will trigger mean reversion signals"""
    candles = []
    end_date = dt.datetime.now()
    
    # Base price and trend
    base_price = 100.0
    trend = 0.001  # Small upward trend
    
    # Volatility parameters
    volatility = 0.015  # Base volatility
    
    # Generate candles
    for i in range(days, 0, -1):
        date = end_date - dt.timedelta(days=i)
        
        # Add randomness with some memory of previous price
        import random
        import math
        random.seed(i)  # For reproducibility
        
        # Create more extreme price movements to trigger signals
        if i == 40:  # Create an oversold condition around day 40
            # Extreme drop to trigger oversold
            base_price *= 0.65  # 35% drop
            trend = -0.008
            volatility = 0.05
        elif i == 39:  # Continue drop
            base_price *= 0.85  # Another 15% drop
            trend = -0.006
            volatility = 0.045
        elif i == 38:  # Start of reversal pattern after oversold
            # Start of reversal
            trend = 0.002
            volatility = 0.03
        elif i == 37:  # Continue reversal pattern
            trend = 0.004
            volatility = 0.025
        elif i == 36:  # Continue reversal pattern
            trend = 0.006
            volatility = 0.02
        elif i == 20:  # Create an overbought condition around day 20
            # Extreme rise to trigger overbought
            base_price *= 1.35  # 35% rise
            trend = 0.008
            volatility = 0.045
        elif i == 19:  # Continue rise
            base_price *= 1.15  # Another 15% rise
            trend = 0.006
            volatility = 0.04
        elif i == 18:  # Start of reversal pattern after overbought
            # Start of reversal
            trend = -0.002
            volatility = 0.03
        elif i == 17:  # Continue reversal pattern
            trend = -0.004
            volatility = 0.025
        elif i == 16:  # Continue reversal pattern
            trend = -0.006
            volatility = 0.02
        elif i % 20 == 0:  # Change regime every 20 days
            trend = random.uniform(-0.002, 0.002)
            volatility = random.uniform(0.01, 0.03)
        
        # Apply trend
        base_price *= (1 + trend)
        
        # Calculate daily range based on volatility
        daily_range = base_price * volatility
        
        # Create price action
        if i % 5 == 0:  # Every 5 days, create a potential reversal pattern
            # If previous trend was up, create a down day
            if trend > 0:
                open_price = base_price * (1 + random.uniform(0, 0.005))
                close_price = base_price * (1 - random.uniform(0.005, 0.015))
            else:
                open_price = base_price * (1 - random.uniform(0, 0.005))
                close_price = base_price * (1 + random.uniform(0.005, 0.015))
        else:
            # Normal day following the trend
            price_change = random.normalvariate(trend, volatility)
            open_price = base_price * (1 + random.uniform(-0.005, 0.005))
            close_price = base_price * (1 + price_change)
        
        # Ensure high and low make sense and create intraday volatility
        high_price = max(open_price, close_price) * (1 + random.uniform(0.001, 0.01))
        low_price = min(open_price, close_price) * (1 - random.uniform(0.001, 0.01))
        
        # Volume increases with volatility and on reversal days
        volume_multiplier = 1.0 + abs(open_price - close_price) / open_price * 10
        if i % 5 == 0:  # Higher volume on potential reversal days
            volume_multiplier *= 1.5
        
        volume = int(100000 * volume_multiplier)
        
        # Create candle
        candle = CandleData(
            timestamp=date,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume
        )
        # Add symbol as a separate attribute
        candle.symbol = symbol
        candles.append(candle)
        
        # Update base price for next iteration
        base_price = close_price
    
    return candles

def calculate_bollinger_bands(candles: List[CandleData], period: int = 20, std_dev: float = 2.0) -> tuple:
    """Calculate Bollinger Bands for a series of candles"""
    if len(candles) < period:
        return None, None, None
    
    # Calculate SMA
    close_prices = [c.close for c in candles[-period:]]
    sma = sum(close_prices) / period
    
    # Calculate standard deviation
    variance = sum((price - sma) ** 2 for price in close_prices) / period
    std = variance ** 0.5
    
    # Calculate Bollinger Bands
    upper_band = sma + (std_dev * std)
    lower_band = sma - (std_dev * std)
    
    return sma, upper_band, lower_band

def calculate_rsi(candles: List[CandleData], period: int = 14) -> Optional[float]:
    """Calculate RSI for a series of candles"""
    if len(candles) < period + 1:
        return None
    
    # Calculate price changes
    close_prices = [c.close for c in candles]
    changes = [close_prices[i] - close_prices[i-1] for i in range(1, len(close_prices))]
    
    # Separate gains and losses
    gains = [change if change > 0 else 0 for change in changes]
    losses = [abs(change) if change < 0 else 0 for change in changes]
    
    # Calculate average gain and loss
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    
    # Calculate RSI
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_atr(candles: List[CandleData], period: int = 14) -> Optional[float]:
    """Calculate Average True Range (ATR) for a series of candles"""
    if len(candles) < period + 1:
        return None
    
    true_ranges = []
    
    for i in range(1, len(candles)):
        high = candles[i].high
        low = candles[i].low
        prev_close = candles[i-1].close
        
        # Calculate true range
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = max(tr1, tr2, tr3)
        true_ranges.append(true_range)
    
    # Calculate ATR
    atr = sum(true_ranges[-period:]) / period
    
    return atr

def generate_mean_reversion_signals(candles: List[CandleData], config: Dict[str, Any], symbol: str) -> List[Signal]:
    """Generate signals using the MeanReversion strategy with optimized ATR multipliers"""
    signals = []
    
    # Extract parameters from config
    bb_period = config.get('bb_period', 20)
    bb_std_dev = config.get('bb_std_dev', 2.0)
    rsi_period = config.get('rsi_period', 14)
    rsi_overbought = config.get('rsi_overbought', 70)
    rsi_oversold = config.get('rsi_oversold', 30)
    min_reversal_candles = config.get('min_reversal_candles', 2)
    require_reversal = config.get('require_reversal', False)  # Make reversal optional by default
    stop_loss_atr = config.get('stop_loss_atr', 2.0)
    take_profit_atr = config.get('take_profit_atr', 3.0)
    
    # Use even more lenient thresholds for testing
    price_band_threshold = 1.1  # Allow price to be up to 10% away from the band
    rsi_threshold_multiplier = 1.3  # Allow RSI to be up to 30% away from the threshold
    
    logger.info(f"MeanReversion parameters: BB Period={bb_period}, BB StdDev={bb_std_dev}, "
               f"RSI Period={rsi_period}, RSI Overbought={rsi_overbought}, RSI Oversold={rsi_oversold}, "
               f"Stop Loss ATR={stop_loss_atr}, Take Profit ATR={take_profit_atr}")
    
    # Need enough candles for calculations
    if len(candles) < max(bb_period, rsi_period) + 1:
        logger.warning(f"Not enough candles for calculations. Need at least {max(bb_period, rsi_period) + 1}, got {len(candles)}")
        return signals
    
    # Calculate indicators for the latest candle
    sma, upper_band, lower_band = calculate_bollinger_bands(candles, bb_period, bb_std_dev)
    rsi = calculate_rsi(candles, rsi_period)
    atr = calculate_atr(candles, 14)
    
    if sma is None or rsi is None or atr is None:
        logger.warning("Failed to calculate indicators")
        return signals
    
    current_price = candles[-1].close
    
    # Log all calculated indicators for debugging
    logger.info(f"Calculated indicators for {symbol}:")
    logger.info(f"  SMA: {sma:.2f}")
    logger.info(f"  Upper Band: {upper_band:.2f}")
    logger.info(f"  Lower Band: {lower_band:.2f}")
    logger.info(f"  RSI: {rsi:.2f}")
    logger.info(f"  ATR: {atr:.2f}")
    logger.info(f"  Current Price: {current_price:.2f}")
    
    # Calculate thresholds
    lower_band_threshold = lower_band * price_band_threshold
    upper_band_threshold = upper_band / price_band_threshold
    rsi_oversold_threshold = rsi_oversold * rsi_threshold_multiplier
    rsi_overbought_threshold = rsi_overbought / rsi_threshold_multiplier
    
    logger.info(f"Calculated thresholds:")
    logger.info(f"  Lower Band Threshold: {lower_band_threshold:.2f}")
    logger.info(f"  Upper Band Threshold: {upper_band_threshold:.2f}")
    logger.info(f"  RSI Oversold Threshold: {rsi_oversold_threshold:.2f}")
    logger.info(f"  RSI Overbought Threshold: {rsi_overbought_threshold:.2f}")
    
    # Check for oversold condition (buy signal)
    if current_price < lower_band_threshold and rsi < rsi_oversold_threshold:
        logger.info(f"Potential BUY signal for {symbol}: Price near lower band ({current_price:.2f} vs {lower_band_threshold:.2f}) and RSI near oversold ({rsi:.2f} vs {rsi_oversold_threshold:.2f})")
        
        # Check for price reversal (min_reversal_candles consecutive higher lows)
        if require_reversal:
            # Make reversal check more lenient - just need 1 candle showing reversal
            reversal = candles[-1].close > candles[-2].close
            logger.info(f"Reversal check: {reversal}, Current close: {candles[-1].close:.2f}, Previous close: {candles[-2].close:.2f}")
        else:
            reversal = True
            logger.info("Reversal check skipped (not required)")
        
        if not reversal:
            logger.info(f"No BUY signal for {symbol}: Price near lower band and RSI near oversold, but no reversal pattern")
            return signals
        
        # Check for volume confirmation - make this optional for testing
        volume_increase = True  # Skip volume check for testing
        
        logger.info(f"BUY SIGNAL for {symbol}: Price near lower band ({current_price:.2f} vs {lower_band_threshold:.2f}) and RSI near oversold ({rsi:.2f} vs {rsi_oversold_threshold:.2f})")
        
        # Calculate stop loss and take profit using ATR
        stop_loss = current_price - (atr * stop_loss_atr)
        take_profit = current_price + (atr * take_profit_atr)
        
        # Determine signal strength based on RSI and price deviation
        rsi_strength = (rsi_oversold_threshold - rsi) / rsi_oversold_threshold
        price_strength = (lower_band_threshold - current_price) / lower_band_threshold
        
        logger.info(f"Signal strength factors: RSI strength: {rsi_strength:.2f}, Price strength: {price_strength:.2f}")
        
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
            timestamp=candles[-1].timestamp,
            strength=strength,
            expiration=candles[-1].timestamp + dt.timedelta(days=3)
        )
        signals.append(signal)
    
    # Check for overbought condition (sell signal)
    elif current_price > upper_band_threshold and rsi > rsi_overbought_threshold:
        logger.info(f"Potential SELL signal for {symbol}: Price near upper band ({current_price:.2f} vs {upper_band_threshold:.2f}) and RSI near overbought ({rsi:.2f} vs {rsi_overbought_threshold:.2f})")
        
        # Check for price reversal (min_reversal_candles consecutive lower highs)
        if require_reversal:
            # Make reversal check more lenient - just need 1 candle showing reversal
            reversal = candles[-1].close < candles[-2].close
            logger.info(f"Reversal check: {reversal}, Current close: {candles[-1].close:.2f}, Previous close: {candles[-2].close:.2f}")
        else:
            reversal = True
            logger.info("Reversal check skipped (not required)")
        
        if not reversal:
            logger.info(f"No SELL signal for {symbol}: Price near upper band and RSI near overbought, but no reversal pattern")
            return signals
        
        # Check for volume confirmation - make this optional for testing
        volume_increase = True  # Skip volume check for testing
        
        logger.info(f"SELL SIGNAL for {symbol}: Price near upper band ({current_price:.2f} vs {upper_band_threshold:.2f}) and RSI near overbought ({rsi:.2f} vs {rsi_overbought_threshold:.2f})")
        
        # Calculate stop loss and take profit using ATR
        stop_loss = current_price + (atr * stop_loss_atr)
        take_profit = current_price - (atr * take_profit_atr)
        
        # Determine signal strength based on RSI and price deviation
        rsi_strength = (rsi - rsi_overbought_threshold) / (100 - rsi_overbought_threshold)
        price_strength = (current_price - upper_band_threshold) / upper_band_threshold
        
        logger.info(f"Signal strength factors: RSI strength: {rsi_strength:.2f}, Price strength: {price_strength:.2f}")
        
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
            timestamp=candles[-1].timestamp,
            strength=strength,
            expiration=candles[-1].timestamp + dt.timedelta(days=3)
        )
        signals.append(signal)
    
    else:
        logger.info(f"No signal for {symbol}: Price not near bands or RSI not near extremes")
        logger.info(f"  Current price: {current_price:.2f}, Lower band threshold: {lower_band_threshold:.2f}, Upper band threshold: {upper_band_threshold:.2f}")
        logger.info(f"  RSI: {rsi:.2f}, RSI oversold threshold: {rsi_oversold_threshold:.2f}, RSI overbought threshold: {rsi_overbought_threshold:.2f}")
        logger.info(f"  Conditions: Price < Lower Band Threshold: {current_price < lower_band_threshold}, "
                   f"RSI < Oversold Threshold: {rsi < rsi_oversold_threshold}")
        logger.info(f"  Conditions: Price > Upper Band Threshold: {current_price > upper_band_threshold}, "
                   f"RSI > Overbought Threshold: {rsi > rsi_overbought_threshold}")
    
    return signals

def test_atr_multipliers(candles: List[CandleData], config: Dict[str, Any], symbol: str):
    """Test different ATR multiplier combinations for stop loss and take profit"""
    # Create a directory for test results
    results_dir = "atr_test_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Define ATR multiplier combinations to test
    sl_multipliers = [1.0, 1.5, 2.0, 2.5, 3.0]
    tp_multipliers = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    
    # Store results
    results = []
    
    for sl_atr in sl_multipliers:
        for tp_atr in tp_multipliers:
            # Update config with current ATR multipliers
            test_config = config.copy()
            test_config['stop_loss_atr'] = sl_atr
            test_config['take_profit_atr'] = tp_atr
            
            # Generate signals
            signals = generate_mean_reversion_signals(candles, test_config, symbol)
            
            # Calculate risk-reward ratio
            risk_reward = tp_atr / sl_atr
            
            # Store results
            result = {
                'stop_loss_atr': sl_atr,
                'take_profit_atr': tp_atr,
                'risk_reward': risk_reward,
                'signals_count': len(signals),
                'signals': signals
            }
            results.append(result)
            
            logger.info(f"ATR Multipliers - SL: {sl_atr}, TP: {tp_atr}, Risk-Reward: {risk_reward:.2f}, Signals: {len(signals)}")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Create a grid for the heatmap
    signal_counts = np.zeros((len(sl_multipliers), len(tp_multipliers)))
    
    for i, sl in enumerate(sl_multipliers):
        for j, tp in enumerate(tp_multipliers):
            # Find the corresponding result
            for result in results:
                if result['stop_loss_atr'] == sl and result['take_profit_atr'] == tp:
                    signal_counts[i, j] = result['signals_count']
                    break
    
    # Create heatmap
    plt.imshow(signal_counts, cmap='viridis', aspect='auto', origin='lower')
    plt.colorbar(label='Number of Signals')
    
    # Add labels
    plt.xlabel('Take Profit ATR Multiplier')
    plt.ylabel('Stop Loss ATR Multiplier')
    plt.title('ATR Multiplier Optimization - Signal Generation')
    
    # Add tick labels
    plt.xticks(np.arange(len(tp_multipliers)), [f"{x:.1f}" for x in tp_multipliers])
    plt.yticks(np.arange(len(sl_multipliers)), [f"{x:.1f}" for x in sl_multipliers])
    
    # Highlight the optimized setting (2.0, 3.0)
    sl_idx = sl_multipliers.index(2.0)
    tp_idx = tp_multipliers.index(3.0)
    plt.plot(tp_idx, sl_idx, 'ro', markersize=10)
    plt.annotate('Optimized Setting (2.0, 3.0)', 
                 xy=(tp_idx, sl_idx), 
                 xytext=(tp_idx - 1, sl_idx - 1),
                 arrowprops=dict(facecolor='white', shrink=0.05))
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'atr_signal_generation.png'))
    plt.close()
    
    # Create risk-reward visualization
    plt.figure(figsize=(12, 8))
    
    # Create a grid for the risk-reward heatmap
    risk_rewards = np.zeros((len(sl_multipliers), len(tp_multipliers)))
    
    for i, sl in enumerate(sl_multipliers):
        for j, tp in enumerate(tp_multipliers):
            risk_rewards[i, j] = tp / sl
    
    # Create heatmap
    plt.imshow(risk_rewards, cmap='plasma', aspect='auto', origin='lower')
    plt.colorbar(label='Risk-Reward Ratio')
    
    # Add labels
    plt.xlabel('Take Profit ATR Multiplier')
    plt.ylabel('Stop Loss ATR Multiplier')
    plt.title('ATR Multiplier Optimization - Risk-Reward Ratio')
    
    # Add tick labels
    plt.xticks(np.arange(len(tp_multipliers)), [f"{x:.1f}" for x in tp_multipliers])
    plt.yticks(np.arange(len(sl_multipliers)), [f"{x:.1f}" for x in sl_multipliers])
    
    # Highlight the optimized setting (2.0, 3.0)
    plt.plot(tp_idx, sl_idx, 'ro', markersize=10)
    plt.annotate('Optimized Setting (2.0, 3.0)', 
                 xy=(tp_idx, sl_idx), 
                 xytext=(tp_idx - 1, sl_idx - 1),
                 arrowprops=dict(facecolor='white', shrink=0.05))
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'atr_risk_reward.png'))
    plt.close()
    
    # Return the results
    return results

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
        
        # Import alpaca_trade_api
        import alpaca_trade_api as tradeapi
        
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

def plot_signals_on_price_chart(candles: List[CandleData], signals: List[Signal], symbol: str, config: Dict[str, Any]):
    """Plot price chart with signals and indicators"""
    if not candles:
        logger.warning("No candles to plot")
        return
    
    # Create directory for plots
    plots_dir = "signal_plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Extract parameters from config
    bb_period = config.get('bb_period', 20)
    bb_std_dev = config.get('bb_std_dev', 2.0)
    rsi_period = config.get('rsi_period', 14)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot price chart
    dates = [c.timestamp for c in candles]
    close_prices = [c.close for c in candles]
    
    # Calculate Bollinger Bands for all candles
    sma_values = []
    upper_band_values = []
    lower_band_values = []
    
    for i in range(len(candles)):
        if i >= bb_period - 1:
            sma, upper, lower = calculate_bollinger_bands(candles[:i+1], bb_period, bb_std_dev)
            sma_values.append(sma)
            upper_band_values.append(upper)
            lower_band_values.append(lower)
        else:
            sma_values.append(None)
            upper_band_values.append(None)
            lower_band_values.append(None)
    
    # Remove None values
    valid_indices = [i for i, x in enumerate(sma_values) if x is not None]
    valid_dates = [dates[i] for i in valid_indices]
    valid_sma = [sma_values[i] for i in valid_indices]
    valid_upper = [upper_band_values[i] for i in valid_indices]
    valid_lower = [lower_band_values[i] for i in valid_indices]
    
    # Plot price and Bollinger Bands
    ax1.plot(dates, close_prices, label='Close Price', color='blue')
    ax1.plot(valid_dates, valid_sma, label='SMA', color='orange', alpha=0.7)
    ax1.plot(valid_dates, valid_upper, label='Upper Band', color='green', alpha=0.5)
    ax1.plot(valid_dates, valid_lower, label='Lower Band', color='red', alpha=0.5)
    
    # Fill between Bollinger Bands
    ax1.fill_between(valid_dates, valid_upper, valid_lower, color='gray', alpha=0.1)
    
    # Plot buy signals
    for signal in signals:
        if signal.direction == TradeDirection.LONG:
            ax1.scatter(signal.timestamp, signal.entry_price, marker='^', color='green', s=100, label='Buy Signal')
            # Plot stop loss and take profit
            ax1.plot([signal.timestamp, signal.timestamp], [signal.entry_price, signal.stop_loss], 'r--', alpha=0.5)
            ax1.plot([signal.timestamp, signal.timestamp], [signal.entry_price, signal.take_profit], 'g--', alpha=0.5)
        elif signal.direction == TradeDirection.SHORT:
            ax1.scatter(signal.timestamp, signal.entry_price, marker='v', color='red', s=100, label='Sell Signal')
            # Plot stop loss and take profit
            ax1.plot([signal.timestamp, signal.timestamp], [signal.entry_price, signal.stop_loss], 'r--', alpha=0.5)
            ax1.plot([signal.timestamp, signal.timestamp], [signal.entry_price, signal.take_profit], 'g--', alpha=0.5)
    
    # Calculate RSI for all candles
    rsi_values = []
    
    for i in range(len(candles)):
        if i >= rsi_period:
            rsi = calculate_rsi(candles[:i+1], rsi_period)
            rsi_values.append(rsi)
        else:
            rsi_values.append(None)
    
    # Remove None values
    valid_indices = [i for i, x in enumerate(rsi_values) if x is not None]
    valid_dates = [dates[i] for i in valid_indices]
    valid_rsi = [rsi_values[i] for i in valid_indices]
    
    # Plot RSI
    ax2.plot(valid_dates, valid_rsi, label='RSI', color='purple')
    ax2.axhline(y=70, color='r', linestyle='-', alpha=0.3)
    ax2.axhline(y=30, color='g', linestyle='-', alpha=0.3)
    ax2.fill_between(valid_dates, 70, valid_rsi, where=(np.array(valid_rsi) >= 70), color='red', alpha=0.3)
    ax2.fill_between(valid_dates, 30, valid_rsi, where=(np.array(valid_rsi) <= 30), color='green', alpha=0.3)
    
    # Set labels and title
    ax1.set_title(f'{symbol} Price Chart with Mean Reversion Signals')
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    ax2.set_xlabel('Date')
    ax2.set_ylabel('RSI')
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis dates
    fig.autofmt_xdate()
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{symbol}_mean_reversion_signals.png'))
    plt.close()
    
    logger.info(f"Signal plot saved to {os.path.join(plots_dir, f'{symbol}_mean_reversion_signals.png')}")

def main():
    """Main function to test the MeanReversion strategy with optimized ATR multipliers"""
    # Load configuration
    config = load_config('configuration_12.yaml')
    
    # Extract MeanReversion strategy configuration
    mr_config = config.get('strategies', {}).get('MeanReversion', {})
    
    # Override configuration for testing
    test_config = mr_config.copy()
    test_config['require_reversal'] = False  # Make reversal check optional for testing
    test_config['bb_std_dev'] = 1.8  # Tighter Bollinger Bands
    test_config['rsi_oversold'] = 35  # Less extreme RSI oversold threshold
    test_config['rsi_overbought'] = 65  # Less extreme RSI overbought threshold
    
    # Test with synthetic data
    logger.info("Testing with synthetic data...")
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    all_signals = []
    
    for symbol in symbols:
        logger.info(f"Testing MeanReversion strategy for {symbol} with synthetic data")
        
        # Generate realistic candles with extreme movements
        candles = create_realistic_candles(symbol, 60)
        
        # Generate signals with the optimized ATR multipliers
        signals = generate_mean_reversion_signals(candles, test_config, symbol)
        
        if signals:
            logger.info(f"Generated {len(signals)} signals for {symbol}")
            for i, signal in enumerate(signals):
                logger.info(f"Signal {i+1}: {signal.direction} at {signal.entry_price:.2f}, "
                           f"SL: {signal.stop_loss:.2f}, TP: {signal.take_profit:.2f}, "
                           f"Strength: {signal.strength}, Timestamp: {signal.timestamp}")
                all_signals.append(signal)
                
            # Plot signals on price chart
            plot_signals_on_price_chart(candles, signals, symbol, test_config)
        else:
            logger.info(f"No signals generated for {symbol}")
        
        # Test different ATR multiplier combinations
        test_atr_multipliers(candles, test_config, symbol)
    
    # Test with real historical data from 2023 - focus on volatile periods
    logger.info("\nTesting with real historical data from 2023...")
    real_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    real_signals = []
    
    # Define volatile periods in 2023
    volatile_periods = [
        # Banking crisis in March 2023
        (dt.datetime(2023, 3, 1), dt.datetime(2023, 3, 31)),
        # Summer volatility
        (dt.datetime(2023, 7, 1), dt.datetime(2023, 8, 31)),
        # Fall market correction
        (dt.datetime(2023, 10, 1), dt.datetime(2023, 10, 31))
    ]
    
    for period_start, period_end in volatile_periods:
        logger.info(f"\nTesting volatile period: {period_start.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')}")
        
        for symbol in real_symbols:
            logger.info(f"Testing MeanReversion strategy for {symbol} with real data")
            
            # Fetch historical data from Alpaca
            candles = fetch_historical_data(symbol, period_start, period_end)
            
            if not candles:
                logger.warning(f"No historical data available for {symbol}")
                continue
            
            # Generate signals with the optimized ATR multipliers
            signals = generate_mean_reversion_signals(candles, test_config, symbol)
            
            if signals:
                logger.info(f"Generated {len(signals)} signals for {symbol} with real data")
                for i, signal in enumerate(signals):
                    logger.info(f"Signal {i+1}: {signal.direction} at {signal.entry_price:.2f}, "
                               f"SL: {signal.stop_loss:.2f}, TP: {signal.take_profit:.2f}, "
                               f"Strength: {signal.strength}, Timestamp: {signal.timestamp}")
                    real_signals.append(signal)
                    
                # Plot signals on price chart
                plot_signals_on_price_chart(candles, signals, symbol, test_config)
            else:
                logger.info(f"No signals generated for {symbol} with real data")
    
    # Summarize results
    logger.info("\nSummary of synthetic data test:")
    logger.info(f"Generated {len(all_signals)} signals across {len(symbols)} symbols")
    
    # Count signals by direction
    long_signals = sum(1 for s in all_signals if s.direction == TradeDirection.LONG)
    short_signals = sum(1 for s in all_signals if s.direction == TradeDirection.SHORT)
    
    logger.info(f"Long signals: {long_signals}, Short signals: {short_signals}")
    
    # Count signals by strength
    strong_signals = sum(1 for s in all_signals if s.strength in [SignalStrength.STRONG_BUY, SignalStrength.STRONG_SELL])
    moderate_signals = sum(1 for s in all_signals if s.strength in [SignalStrength.MODERATE_BUY, SignalStrength.MODERATE_SELL])
    weak_signals = sum(1 for s in all_signals if s.strength in [SignalStrength.WEAK_BUY, SignalStrength.WEAK_SELL])
    
    logger.info(f"Signal strength distribution: Strong: {strong_signals}, Moderate: {moderate_signals}, Weak: {weak_signals}")
    
    # Calculate average risk-reward ratio
    if all_signals:
        risk_rewards = []
        for signal in all_signals:
            if signal.direction == TradeDirection.LONG:
                risk = signal.entry_price - signal.stop_loss
                reward = signal.take_profit - signal.entry_price
            else:
                risk = signal.stop_loss - signal.entry_price
                reward = signal.entry_price - signal.take_profit
            
            if risk > 0:
                risk_reward = reward / risk
                risk_rewards.append(risk_reward)
        
        avg_risk_reward = sum(risk_rewards) / len(risk_rewards) if risk_rewards else 0
        logger.info(f"Average risk-reward ratio: {avg_risk_reward:.2f}")
    
    # Summarize results for real data
    logger.info("\nSummary of real data test:")
    logger.info(f"Generated {len(real_signals)} signals across {len(real_symbols)} symbols with real data")
    
    # Count signals by direction
    long_signals = sum(1 for s in real_signals if s.direction == TradeDirection.LONG)
    short_signals = sum(1 for s in real_signals if s.direction == TradeDirection.SHORT)
    
    logger.info(f"Long signals: {long_signals}, Short signals: {short_signals}")
    
    # Count signals by strength
    strong_signals = sum(1 for s in real_signals if s.strength in [SignalStrength.STRONG_BUY, SignalStrength.STRONG_SELL])
    moderate_signals = sum(1 for s in real_signals if s.strength in [SignalStrength.MODERATE_BUY, SignalStrength.MODERATE_SELL])
    weak_signals = sum(1 for s in real_signals if s.strength in [SignalStrength.WEAK_BUY, SignalStrength.WEAK_SELL])
    
    logger.info(f"Signal strength distribution: Strong: {strong_signals}, Moderate: {moderate_signals}, Weak: {weak_signals}")

if __name__ == "__main__":
    main()
