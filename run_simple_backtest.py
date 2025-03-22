#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple Backtest Runner for Enhanced Mean Reversion Strategy
----------------------------------------------------------
This script runs a simplified backtest for the Enhanced Mean Reversion strategy
using either real Alpaca data or generated mock data.
"""

import os
import sys
import random
import logging
import datetime
import numpy as np
import pandas as pd
import alpaca_trade_api
from alpaca_trade_api.rest import REST, TimeFrame

# Add parent directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import strategy modules
from enhanced_mean_reversion_backtest import EnhancedMeanReversionBacktest, CandleData

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CandleData:
    """Class to store candle data"""
    
    def __init__(self, timestamp, open, high, low, close, volume):
        self.timestamp = timestamp
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
    
    def __str__(self):
        return f"CandleData(timestamp={self.timestamp}, open={self.open}, high={self.high}, low={self.low}, close={self.close}, volume={self.volume})"

class Position:
    """Class to represent a trading position"""
    
    def __init__(self, symbol, entry_price, entry_time, position_size, direction, stop_loss=None, take_profit=None):
        self.symbol = symbol
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.position_size = position_size
        self.direction = direction  # 'long' or 'short'
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.exit_price = None
        self.exit_time = None
        self.profit_loss = None
        self.status = "open"
        self.exit_reason = None
    
    def close_position(self, exit_price, exit_time, reason="manual"):
        """Close the position and calculate profit/loss"""
        self.exit_price = exit_price
        self.exit_time = exit_time
        
        if self.direction == "long":
            self.profit_loss = (exit_price - self.entry_price) * self.position_size
        else:  # short
            self.profit_loss = (self.entry_price - exit_price) * self.position_size
        
        self.status = "closed"
        self.exit_reason = reason
        
        return self.profit_loss
    
    def calculate_current_profit_loss(self, current_price):
        """Calculate current profit/loss without closing the position"""
        if self.direction == "long":
            return (current_price - self.entry_price) * self.position_size
        else:  # short
            return (self.entry_price - current_price) * self.position_size
    
    def __str__(self):
        status_str = f"{self.symbol} {self.direction} position, size: {self.position_size}, entry: {self.entry_price:.2f}"
        
        if self.status == "closed":
            status_str += f", exit: {self.exit_price:.2f}, P/L: {self.profit_loss:.2f}, reason: {self.exit_reason}"
        
        return status_str

class Portfolio:
    """Class to manage trading portfolio and positions"""
    
    def __init__(self, initial_capital):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.open_positions = {}  # symbol -> Position
        self.closed_positions = []
        self.equity_curve = []
        self.logger = logging.getLogger(__name__)
    
    def reset(self, initial_capital=None):
        """Reset the portfolio to initial state"""
        if initial_capital is not None:
            self.initial_capital = initial_capital
        
        self.cash = self.initial_capital
        self.open_positions = {}
        self.closed_positions = []
        self.equity_curve = []
    
    def open_position(self, symbol, entry_price, entry_time, position_size, direction, stop_loss=None, take_profit=None):
        """Open a new position"""
        # Check if we already have a position for this symbol
        if symbol in self.open_positions:
            self.logger.warning(f"Already have an open position for {symbol}, cannot open another")
            return False
        
        # Calculate cost to open position
        position_cost = entry_price * position_size
        
        # Check if we have enough cash
        if position_cost > self.cash:
            self.logger.warning(f"Not enough cash to open position for {symbol}. Need {position_cost:.2f}, have {self.cash:.2f}")
            return False
        
        # Create position
        position = Position(
            symbol=symbol,
            entry_price=entry_price,
            entry_time=entry_time,
            position_size=position_size,
            direction=direction,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        # Update cash
        self.cash -= position_cost
        
        # Add to open positions
        self.open_positions[symbol] = position
        
        self.logger.info(f"Opened {direction} position for {symbol}: {position_size} shares at {entry_price:.2f}")
        
        return True
    
    def close_position(self, symbol, exit_price, exit_time, reason="manual"):
        """Close an open position"""
        if symbol not in self.open_positions:
            self.logger.warning(f"No open position for {symbol} to close")
            return False
        
        position = self.open_positions[symbol]
        profit_loss = position.close_position(exit_price, exit_time, reason)
        
        # Update cash
        self.cash += (exit_price * position.position_size)
        
        # Move from open to closed positions
        self.closed_positions.append(position)
        del self.open_positions[symbol]
        
        self.logger.info(f"Closed {position.direction} position for {symbol}: {position.position_size} shares at {exit_price:.2f}, P/L: {profit_loss:.2f}")
        
        return True
    
    def update_equity_curve(self, timestamp):
        """Update the equity curve with current portfolio value"""
        equity = self.get_equity()
        self.equity_curve.append((timestamp, equity))
    
    def get_equity(self):
        """Calculate total portfolio value (cash + open positions)"""
        equity = self.cash
        
        for symbol, position in self.open_positions.items():
            # In a real implementation, we would use current market prices
            # For simplicity, we'll use the entry price here
            position_value = position.entry_price * position.position_size
            equity += position_value
        
        return equity
    
    def get_win_rate(self):
        """Calculate win rate from closed positions"""
        if not self.closed_positions:
            return 0.0
        
        winners = sum(1 for pos in self.closed_positions if pos.profit_loss > 0)
        return winners / len(self.closed_positions)
    
    def get_profit_factor(self):
        """Calculate profit factor (gross profits / gross losses)"""
        gross_profit = sum(pos.profit_loss for pos in self.closed_positions if pos.profit_loss > 0)
        gross_loss = sum(abs(pos.profit_loss) for pos in self.closed_positions if pos.profit_loss < 0)
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def get_max_drawdown(self):
        """Calculate maximum drawdown from equity curve"""
        if not self.equity_curve:
            return 0.0
        
        # Extract equity values
        equity_values = [equity for _, equity in self.equity_curve]
        
        # Calculate drawdown
        max_dd = 0
        peak = equity_values[0]
        
        for equity in equity_values:
            if equity > peak:
                peak = equity
            
            dd = (peak - equity) / peak
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def get_sharpe_ratio(self, risk_free_rate=0.0):
        """Calculate Sharpe ratio from equity curve"""
        if len(self.equity_curve) < 2:
            return 0.0
        
        # Extract equity values
        equity_values = [equity for _, equity in self.equity_curve]
        
        # Calculate returns
        returns = [(equity_values[i] / equity_values[i-1]) - 1 for i in range(1, len(equity_values))]
        
        # Calculate Sharpe ratio
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualize (assuming daily returns)
        sharpe = (mean_return - risk_free_rate) / std_return * np.sqrt(252)
        
        return sharpe

class MeanReversionStrategy:
    """Mean reversion trading strategy using Bollinger Bands and RSI"""
    
    def __init__(self, bb_period=20, bb_std=2.0, rsi_period=14, rsi_overbought=70, rsi_oversold=30, 
                 require_reversal=False, stop_loss_atr=2.0, take_profit_atr=3.0, atr_period=14):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.require_reversal = require_reversal
        self.stop_loss_atr = stop_loss_atr
        self.take_profit_atr = take_profit_atr
        self.atr_period = atr_period
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Initialized with parameters: BB period={bb_period}, BB std={bb_std}, "
                        f"RSI period={rsi_period}, RSI thresholds={rsi_oversold}/{rsi_overbought}, "
                        f"Require reversal={require_reversal}")
    
    def calculate_indicators(self, candles):
        """Calculate Bollinger Bands, RSI, and ATR for the given candles"""
        if len(candles) < max(self.bb_period, self.rsi_period, self.atr_period) + 10:
            return None, None, None, None, None
        
        # Extract close prices
        closes = np.array([candle.close for candle in candles])
        highs = np.array([candle.high for candle in candles])
        lows = np.array([candle.low for candle in candles])
        
        # Calculate Bollinger Bands
        bb_middle = self.calculate_sma(closes, self.bb_period)
        bb_std = self.calculate_std(closes, self.bb_period)
        bb_upper = bb_middle + self.bb_std * bb_std
        bb_lower = bb_middle - self.bb_std * bb_std
        
        # Calculate RSI
        rsi = self.calculate_rsi(closes, self.rsi_period)
        
        # Calculate ATR
        atr = self.calculate_atr(highs, lows, closes, self.atr_period)
        
        return bb_upper, bb_middle, bb_lower, rsi, atr
    
    def calculate_sma(self, data, period):
        """Calculate Simple Moving Average"""
        sma = np.zeros_like(data)
        for i in range(period - 1, len(data)):
            sma[i] = np.mean(data[i - period + 1:i + 1])
        return sma
    
    def calculate_std(self, data, period):
        """Calculate Standard Deviation"""
        std = np.zeros_like(data)
        for i in range(period - 1, len(data)):
            std[i] = np.std(data[i - period + 1:i + 1])
        return std
    
    def calculate_rsi(self, prices, period):
        """Calculate Relative Strength Index"""
        rsi = np.zeros_like(prices)
        
        # Calculate price changes
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        
        if down != 0:
            rs = up / down
        else:
            rs = 1.0
        
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))
        
        # Calculate RSI
        for i in range(period + 1, len(prices)):
            delta = deltas[i - 1]
            
            if delta > 0:
                up_val = delta
                down_val = 0.0
            else:
                up_val = 0.0
                down_val = -delta
            
            up = (up * (period - 1) + up_val) / period
            down = (down * (period - 1) + down_val) / period
            
            if down != 0:
                rs = up / down
            else:
                rs = 1.0
            
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        
        return rsi
    
    def calculate_atr(self, high, low, close, period):
        """Calculate Average True Range"""
        atr = np.zeros_like(high)
        
        # Calculate True Range
        tr = np.zeros_like(high)
        tr[0] = high[0] - low[0]
        
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Calculate ATR
        atr[period-1] = np.mean(tr[:period])
        
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        
        return atr
    
    def generate_signals(self, candles):
        """Generate trading signals based on the strategy"""
        if len(candles) < max(self.bb_period, self.rsi_period, self.atr_period) + 10:
            return []
        
        # Calculate indicators
        bb_upper, bb_middle, bb_lower, rsi, atr = self.calculate_indicators(candles)
        
        if bb_upper is None or bb_lower is None or rsi is None or atr is None:
            return []
        
        # Generate signals
        signals = []
        
        for i in range(max(self.bb_period, self.rsi_period, self.atr_period) + 1, len(candles)):
            current_candle = candles[i]
            prev_candle = candles[i-1]
            
            # Check for buy signal (price below lower BB and RSI oversold)
            if current_candle.close < bb_lower[i] and rsi[i] < self.rsi_oversold:
                # If we require price reversal, check if price is starting to move up
                if not self.require_reversal or current_candle.close > prev_candle.close:
                    # Calculate stop loss and take profit levels
                    stop_loss = current_candle.close - (atr[i] * self.stop_loss_atr)
                    take_profit = current_candle.close + (atr[i] * self.take_profit_atr)
                    
                    signals.append({
                        'timestamp': current_candle.timestamp,
                        'signal': 'buy',
                        'price': current_candle.close,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'strength': min(1.0, (self.rsi_oversold - rsi[i]) / 10)  # Signal strength based on RSI distance
                    })
            
            # Check for sell signal (price above upper BB and RSI overbought)
            elif current_candle.close > bb_upper[i] and rsi[i] > self.rsi_overbought:
                # If we require price reversal, check if price is starting to move down
                if not self.require_reversal or current_candle.close < prev_candle.close:
                    # Calculate stop loss and take profit levels
                    stop_loss = current_candle.close + (atr[i] * self.stop_loss_atr)
                    take_profit = current_candle.close - (atr[i] * self.take_profit_atr)
                    
                    signals.append({
                        'timestamp': current_candle.timestamp,
                        'signal': 'sell',
                        'price': current_candle.close,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'strength': min(1.0, (rsi[i] - self.rsi_overbought) / 10)  # Signal strength based on RSI distance
                    })
        
        return signals

class MarketRegimeDetector:
    """Detect market regime (trending, mean-reverting, volatile)"""
    
    def __init__(self, lookback_period=50):
        self.lookback_period = lookback_period
        self.market_state = "unknown"
        self.logger = logging.getLogger(__name__)
        self.logger.info("Market regime detector initialized")
    
    def update_market_state(self, spy_data):
        """Update market state based on SPY data"""
        if len(spy_data) < self.lookback_period:
            self.logger.warning(f"Not enough data to determine market state. Need {self.lookback_period} candles, got {len(spy_data)}")
            return
        
        # Get recent SPY data
        recent_data = spy_data[-self.lookback_period:]
        
        # Extract close prices
        closes = [candle.close for candle in recent_data]
        
        # Calculate metrics
        volatility = np.std(np.diff(closes) / closes[:-1])
        trend_strength = self.calculate_trend_strength(closes)
        mean_reversion_strength = self.calculate_mean_reversion_strength(closes)
        
        # Determine market state
        if volatility > 0.015:  # High volatility threshold
            self.market_state = "volatile"
        elif trend_strength > 0.7:  # Strong trend threshold
            self.market_state = "trending"
        elif mean_reversion_strength > 0.7:  # Strong mean-reversion threshold
            self.market_state = "mean_reverting"
        else:
            self.market_state = "neutral"
        
        self.logger.info(f"Market state updated to: {self.market_state}")
        self.logger.info(f"Metrics: volatility={volatility:.4f}, trend_strength={trend_strength:.4f}, mean_reversion_strength={mean_reversion_strength:.4f}")
        
        return self.market_state
    
    def calculate_trend_strength(self, prices):
        """Calculate trend strength using linear regression R-squared"""
        x = np.arange(len(prices))
        y = np.array(prices)
        
        # Calculate linear regression
        slope, intercept = np.polyfit(x, y, 1)
        
        # Calculate R-squared
        y_pred = slope * x + intercept
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        
        r_squared = 1 - (ss_residual / ss_total)
        
        return r_squared
    
    def calculate_mean_reversion_strength(self, prices):
        """Calculate mean reversion strength using Hurst exponent"""
        # Simplified Hurst exponent calculation
        lags = range(2, min(20, len(prices) // 4))
        tau = [np.sqrt(np.std(np.subtract(prices[lag:], prices[:-lag]))) for lag in lags]
        
        # Estimate Hurst exponent from the power law
        m = np.polyfit(np.log(lags), np.log(tau), 1)
        hurst = m[0] / 2.0
        
        # Convert to mean reversion strength (H < 0.5 indicates mean reversion)
        mean_reversion_strength = max(0, 1 - 2 * hurst)
        
        return mean_reversion_strength
    
    def get_market_state(self):
        """Get current market state"""
        return self.market_state

class MLSignalClassifier:
    """Machine learning classifier for trading signals"""
    
    def __init__(self, model_path=None):
        self.model = None
        self.model_path = model_path
        self.logger = logging.getLogger(__name__)
        
        # Try to load existing model
        if model_path:
            try:
                import pickle
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                self.logger.info(f"Loaded ML model from {model_path}")
            except Exception as e:
                self.logger.warning(f"Could not load ML model: {str(e)}")
        
        if self.model is None:
            self.logger.info("No existing model found or error loading model. Will train new model when sufficient data is available.")
        
        self.logger.info("ML signal classifier initialized successfully")
    
    def train_model(self, signals, outcomes):
        """Train the ML model on historical signals and outcomes"""
        if len(signals) < 50:
            self.logger.warning(f"Not enough data to train model. Need at least 50 signals, got {len(signals)}")
            return False
        
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            import numpy as np
            
            # Prepare features and labels
            X = np.array([self.extract_features(signal) for signal in signals])
            y = np.array(outcomes)
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train model
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_scaled, y)
            
            # Save model if path is provided
            if self.model_path:
                import pickle
                with open(self.model_path, 'wb') as f:
                    pickle.dump(self.model, f)
                self.logger.info(f"Saved ML model to {self.model_path}")
            
            self.logger.info("ML model trained successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error training ML model: {str(e)}")
            return False
    
    def extract_features(self, signal):
        """Extract features from a signal for ML model"""
        # Example features (expand as needed)
        features = [
            signal.get('strength', 0),
            signal.get('price', 0),
            signal.get('volume', 0) if 'volume' in signal else 0,
            1 if signal.get('signal') == 'buy' else 0,  # One-hot encoding for signal type
            1 if signal.get('signal') == 'sell' else 0,
        ]
        
        return features
    
    def classify_signal(self, signal, market_state):
        """Classify a trading signal using the ML model"""
        if self.model is None:
            # If no model is trained, use simple rules
            if market_state == "trending" and signal.get('signal') == 'buy':
                return 0.8  # Higher probability for buy signals in trending markets
            elif market_state == "mean_reverting" and (signal.get('signal') == 'buy' or signal.get('signal') == 'sell'):
                return 0.9  # Higher probability for mean reversion signals in mean-reverting markets
            elif market_state == "volatile":
                return 0.4  # Lower probability in volatile markets
            else:
                return 0.6  # Default probability
        
        try:
            from sklearn.preprocessing import StandardScaler
            import numpy as np
            
            # Extract features
            features = np.array([self.extract_features(signal)])
            
            # Scale features (in real implementation, use the same scaler as training)
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Predict probability
            proba = self.model.predict_proba(features_scaled)[0]
            
            # Return probability of positive outcome
            return proba[1]
            
        except Exception as e:
            self.logger.error(f"Error classifying signal: {str(e)}")
            return 0.5  # Default probability on error

class RealAlpacaBacktest(EnhancedMeanReversionBacktest):
    """Version of the EnhancedMeanReversionBacktest that uses real Alpaca data"""
    
    def __init__(self, config):
        """Initialize with the given configuration"""
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # Set up configuration
        self.config = config
        self.initial_capital = config.get('initial_capital', 100000)
        self.symbols = config.get('symbols', [])
        
        # Initialize portfolio
        self.portfolio = Portfolio(self.initial_capital)
        
        # Initialize strategies
        self.strategies = {}
        strategy_configs = config.get('strategies', {})
        
        if 'mean_reversion' in strategy_configs:
            params = strategy_configs['mean_reversion']['params']
            self.strategies['mean_reversion'] = MeanReversionStrategy(
                bb_period=params.get('bb_period', 20),
                bb_std=params.get('bb_std', 2.0),
                rsi_period=params.get('rsi_period', 14),
                rsi_overbought=params.get('rsi_overbought', 70),
                rsi_oversold=params.get('rsi_oversold', 30),
                require_reversal=params.get('require_reversal', False),
                stop_loss_atr=params.get('stop_loss_atr', 2.0),
                take_profit_atr=params.get('take_profit_atr', 3.0),
                atr_period=params.get('atr_period', 14)
            )
            self.logger.info(f"Initialized with parameters: BB period={params.get('bb_period', 20)}, BB std={params.get('bb_std', 2.0)}, RSI period={params.get('rsi_period', 14)}, RSI thresholds={params.get('rsi_oversold', 30)}/{params.get('rsi_overbought', 70)}, Require reversal={params.get('require_reversal', False)}")
        
        if 'trend_following' in strategy_configs:
            params = strategy_configs['trend_following']['params']
            self.strategies['trend_following'] = MeanReversionStrategy(
                bb_period=params.get('bb_period', 20),
                bb_std=params.get('bb_std', 2.0),
                rsi_period=params.get('rsi_period', 14),
                rsi_overbought=params.get('rsi_overbought', 70),
                rsi_oversold=params.get('rsi_oversold', 30),
                require_reversal=params.get('require_reversal', False),
                stop_loss_atr=params.get('stop_loss_atr', 2.0),
                take_profit_atr=params.get('take_profit_atr', 3.0),
                atr_period=params.get('atr_period', 14)
            )
            self.logger.info(f"Initialized with parameters: BB period={params.get('bb_period', 20)}, BB std={params.get('bb_std', 2.0)}, RSI period={params.get('rsi_period', 14)}, RSI thresholds={params.get('rsi_oversold', 30)}/{params.get('rsi_overbought', 70)}, Require reversal={params.get('require_reversal', False)}")
        
        # Initialize market state detector
        self.market_state_detector = MarketRegimeDetector()
        self.logger.info("Market regime detector initialized")
        
        # Initialize ML signal classifier
        self.ml_signal_classifier = MLSignalClassifier()
        
        # Initialize Alpaca API
        self.api = None
        
        # Separate symbols by type
        self.stock_symbols = []
        self.crypto_symbols = []
        
        for symbol in self.symbols:
            if symbol.endswith('USD'):
                self.crypto_symbols.append(symbol)
            else:
                self.stock_symbols.append(symbol)
        
        self.logger.info(f"Separated symbols: {len(self.stock_symbols)} stocks, {len(self.crypto_symbols)} cryptos")
    
    def initialize_alpaca_api(self):
        """Initialize the Alpaca API client with credentials from alpaca_credentials.json"""
        try:
            # Load credentials from JSON file
            import json
            import os
            
            credentials_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'alpaca_credentials.json')
            
            if os.path.exists(credentials_path):
                with open(credentials_path, 'r') as f:
                    credentials = json.load(f)
                
                # Use paper trading credentials by default
                paper_creds = credentials.get('paper', {})
                api_key = paper_creds.get('api_key')
                api_secret = paper_creds.get('api_secret')
                base_url = paper_creds.get('base_url', 'https://paper-api.alpaca.markets')
                
                self.logger.info(f"Using paper trading credentials from file")
            else:
                # Fallback to config
                api_key = self.config.get('alpaca', {}).get('api_key')
                api_secret = self.config.get('alpaca', {}).get('api_secret')
                base_url = self.config.get('alpaca', {}).get('base_url', 'https://paper-api.alpaca.markets')
                
                self.logger.info(f"Using credentials from config")
            
            # Initialize Alpaca API client
            self.api = REST(
                key_id=api_key,
                secret_key=api_secret,
                base_url=base_url
            )
            
            # Test the connection
            try:
                account = self.api.get_account()
                self.logger.info(f"Connected to Alpaca API. Account status: {account.status}")
                self.logger.info(f"Account equity: ${float(account.equity):.2f}")
            except Exception as e:
                self.logger.warning(f"Could not get account info: {e}")
            
            self.logger.info("Alpaca API initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error initializing Alpaca API: {str(e)}")
            return False
    
    def fetch_historical_data(self, symbol, start_date, end_date):
        """Fetch historical price data from Alpaca"""
        self.logger.info(f"Fetching historical data for {symbol} from {start_date.date()} to {end_date.date()}")
        
        try:
            # Make sure we have an Alpaca API instance
            if not self.api:
                self.initialize_alpaca_api()
            
            # Ensure dates are in the correct format
            if isinstance(start_date, str):
                start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
            if isinstance(end_date, str):
                end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
            
            # Format dates for Alpaca API
            start_str = pd.Timestamp(start_date, tz='America/New_York').isoformat()
            end_str = pd.Timestamp(end_date, tz='America/New_York').isoformat()
            
            # Try with direct HTTP request to Alpaca API
            import requests
            import json
            import os
            from datetime import datetime, timedelta
            
            # Load credentials from JSON file
            credentials_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'alpaca_credentials.json')
            
            if os.path.exists(credentials_path):
                with open(credentials_path, 'r') as f:
                    credentials = json.load(f)
                
                # Use paper trading credentials by default
                paper_creds = credentials.get('paper', {})
                api_key = paper_creds.get('api_key')
                api_secret = paper_creds.get('api_secret')
                
                self.logger.info(f"Using paper trading credentials from file for data API")
            else:
                # Fallback to config
                api_key = self.config.get('alpaca', {}).get('api_key')
                api_secret = self.config.get('alpaca', {}).get('api_secret')
            
            # Set up the headers with API key authentication
            headers = {
                'APCA-API-KEY-ID': api_key,
                'APCA-API-SECRET-KEY': api_secret
            }
            
            # Format the URL for the bars endpoint
            url = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
            params = {
                'timeframe': '1Day',
                'adjustment': 'raw',
                'start': start_str,
                'end': end_str
            }
            
            self.logger.info(f"Making direct API request to: {url}")
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                self.logger.info(f"Successfully fetched data for {symbol}")
                
                # Convert to CandleData objects
                candles = []
                for bar in data.get('bars', []):
                    timestamp = datetime.fromisoformat(bar['t'].replace('Z', '+00:00'))
                    candle = CandleData(
                        timestamp=timestamp,
                        open=float(bar['o']),
                        high=float(bar['h']),
                        low=float(bar['l']),
                        close=float(bar['c']),
                        volume=int(bar['v'])
                    )
                    candles.append(candle)
                
                self.logger.info(f"Fetched {len(candles)} candles for {symbol}")
                return candles
            else:
                self.logger.error(f"API request failed with status {response.status_code}: {response.text}")
                
                # Try with mock data as fallback
                self.logger.warning(f"Falling back to mock data for {symbol}")
                return self.generate_mock_data(symbol, start_date, end_date)
                
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            # Fallback to mock data
            self.logger.warning(f"Falling back to mock data for {symbol}")
            return self.generate_mock_data(symbol, start_date, end_date)
    
    def generate_mock_data(self, symbol, start_date, end_date, base_price=None, volatility=0.015, seed=None):
        """Generate mock price data for testing"""
        self.logger.info(f"Generating mock data for {symbol} from {start_date.date()} to {end_date.date()}")
        
        # Set random seed for reproducibility if provided
        if seed is not None:
            # Ensure seed is a valid integer for numpy
            seed_int = abs(hash(str(seed))) % (2**32 - 1)
            np.random.seed(seed_int)
        
        # Generate dates
        dates = []
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5:  # Monday to Friday
                dates.append(current_date)
            current_date += datetime.timedelta(days=1)
        
        # Set base price if not provided
        if base_price is None:
            if symbol == "SPY":
                base_price = 400.0
            elif symbol == "AAPL":
                base_price = 150.0
            elif symbol == "MSFT":
                base_price = 250.0
            elif symbol == "GOOGL":
                base_price = 100.0
            elif symbol == "AMZN":
                base_price = 120.0
            elif symbol == "META":
                base_price = 200.0
            elif symbol == "TSLA":
                base_price = 180.0
            elif symbol == "NVDA":
                base_price = 220.0
            elif symbol == "JPM":
                base_price = 140.0
            elif symbol == "V":
                base_price = 230.0
            elif symbol == "JNJ":
                base_price = 160.0
            elif symbol == "WMT":
                base_price = 140.0
            elif symbol == "PG":
                base_price = 150.0
            else:
                base_price = 100.0
        
        # Generate price data
        num_days = len(dates)
        returns = np.random.normal(0.0005, volatility, num_days)  # Mean slightly positive for upward bias
        prices = [base_price]
        
        for i in range(1, num_days):
            prices.append(prices[i-1] * (1 + returns[i]))
        
        # Generate OHLCV data
        candles = []
        for i, date in enumerate(dates):
            close_price = prices[i]
            daily_volatility = volatility * close_price
            
            # Generate open, high, low with some randomness
            open_price = close_price * (1 + np.random.normal(0, 0.005))
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.008)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.008)))
            
            # Ensure high >= open, close and low <= open, close
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            # Generate volume
            volume = int(np.random.lognormal(15, 0.5))
            
            # Store data
            candles.append({
                'timestamp': date,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        
        # Create DataFrame
        df = pd.DataFrame(candles)
        df.set_index('timestamp', inplace=True)
        
        self.logger.info(f"Generated {len(df)} candles for {symbol}")
        return df
    
    def run_backtest(self, start_date, end_date):
        """Run the backtest from start_date to end_date"""
        self.logger.info(f"Starting backtest from {start_date.date()} to {end_date.date()}")
        
        # Initialize Alpaca API
        self.initialize_alpaca_api()
        
        # Get market data for SPY to determine market state
        try:
            spy_data = self.fetch_historical_data("SPY", start_date, end_date)
            if spy_data:
                self.market_state_detector.update_market_state(spy_data)
            else:
                self.logger.error("Failed to fetch SPY data, cannot determine market state")
        except Exception as e:
            self.logger.error(f"Error updating market state: {str(e)}")
        
        # Fetch data for each symbol
        all_symbol_data = {}
        for symbol in self.symbols:
            candles = self.fetch_historical_data(symbol, start_date, end_date)
            if candles:
                all_symbol_data[symbol] = candles
        
        # Generate trading dates from the data
        trading_dates = []
        if all_symbol_data:
            # Use the first symbol's data to determine trading dates
            first_symbol = list(all_symbol_data.keys())[0]
            trading_dates = [candle.timestamp for candle in all_symbol_data[first_symbol]]
        else:
            # If no data was fetched, generate dates manually
            current_date = start_date
            while current_date <= end_date:
                if current_date.weekday() < 5:  # Monday to Friday
                    trading_dates.append(current_date)
                current_date += datetime.timedelta(days=1)
        
        # Sort trading dates
        trading_dates.sort()
        
        # Initialize portfolio
        self.portfolio.reset(self.initial_capital)
        
        # Run the backtest day by day
        for date in trading_dates:
            self.logger.info(f"Processing date: {date.strftime('%Y-%m-%d')}")
            self.process_trading_day(date, all_symbol_data)
        
        # Calculate final results
        final_equity = self.portfolio.get_equity()
        total_return = (final_equity - self.initial_capital) / self.initial_capital if self.initial_capital > 0 else 0
        
        self.logger.info("Backtest completed with {} trades".format(len(self.portfolio.closed_positions)))
        self.logger.info(f"Final equity: ${final_equity:.2f}")
        self.logger.info(f"Total return: {total_return:.2%}")
        
        # Compile results
        results = {
            'final_equity': final_equity,
            'initial_capital': self.initial_capital,
            'total_return': total_return,
            'return': total_return,  
            'trades': len(self.portfolio.closed_positions),
            'win_rate': self.portfolio.get_win_rate(),
            'profit_factor': self.portfolio.get_profit_factor(),
            'max_drawdown': self.portfolio.get_max_drawdown(),
            'sharpe_ratio': self.portfolio.get_sharpe_ratio(),
            'closed_positions': self.portfolio.closed_positions
        }
        
        return results

def run_backtest():
    """Run a backtest with real Alpaca data"""
    # Define configuration
    config = {
        'initial_capital': 100000,
        'symbols': [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 
            'TSLA', 'NVDA', 'JPM', 'V', 'JNJ',
            'WMT', 'PG'
        ],
        'start_date': '2023-01-01',
        'end_date': '2023-03-31',
        'strategies': {
            'mean_reversion': {
                'weight': 1.0,
                'params': {
                    'bb_period': 20,
                    'bb_std': 1.9,
                    'rsi_period': 14,
                    'rsi_overbought': 65,
                    'rsi_oversold': 35,
                    'require_reversal': True,
                    'stop_loss_atr': 1.8,
                    'take_profit_atr': 3.0,
                    'atr_period': 14
                }
            },
            'trend_following': {
                'weight': 1.0,
                'params': {
                    'bb_period': 20,
                    'bb_std': 2.0,
                    'rsi_period': 14,
                    'rsi_overbought': 70,
                    'rsi_oversold': 30,
                    'require_reversal': True,
                    'stop_loss_atr': 2.0,
                    'take_profit_atr': 4.0,
                    'atr_period': 14
                }
            }
        },
        'alpaca': {
            'api_key': 'PK3MIMOSIMVY8A9IYXE5',
            'api_secret': 'GMXfCCDGQYSPyGrJZPwrIUUgmMO5XIOhXKWJnL3f',
            'base_url': 'https://api.alpaca.markets'
        }
    }
    
    # Initialize backtest
    backtest = RealAlpacaBacktest(config)
    
    # Parse dates
    start_date = datetime.datetime.strptime(config['start_date'], '%Y-%m-%d')
    end_date = datetime.datetime.strptime(config['end_date'], '%Y-%m-%d')
    
    # Run backtest
    results = backtest.run_backtest(start_date, end_date)
    
    # Print results
    logger.info("Backtest Results:")
    logger.info(f"Final Equity: ${results['final_equity']:.2f}")
    logger.info(f"Initial Capital: ${results['initial_capital']:.2f}")
    logger.info(f"Return: {results['total_return'] * 100:.2f}%")
    logger.info(f"Number of Trades: {results['trades']}")
    logger.info(f"Win Rate: {results['win_rate'] * 100:.2f}%")
    logger.info(f"Profit Factor: {results['profit_factor']:.2f}")
    logger.info(f"Max Drawdown: {results['max_drawdown'] * 100:.2f}%")
    logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    
    return results

if __name__ == "__main__":
    run_backtest()
