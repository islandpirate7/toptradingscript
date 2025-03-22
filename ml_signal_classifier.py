#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Machine Learning Signal Classifier
Trains and uses ML models to classify trading signals by expected quality/outcome
"""

import numpy as np
import pandas as pd
import datetime
import pickle
import os
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
import joblib

# Import data classes
from mean_reversion_enhanced import CandleData, Signal, Trade, MarketState

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLSignalClassifier:
    """
    Machine Learning Signal Classifier for predicting signal quality
    """
    
    def __init__(self, config: Dict = None):
        """Initialize the ML signal classifier with configuration"""
        self.config = config or {}
        self.logger = logging.getLogger("MLSignalClassifier")
        
        # Configuration parameters
        self.params = self.config.get('ml_classifier_params', {})
        
        # Model parameters
        self.model_type = self.params.get('model_type', 'random_forest')
        self.feature_lookback = self.params.get('feature_lookback', 20)
        self.min_training_samples = self.params.get('min_training_samples', 100)
        self.model_path = self.params.get('model_path', 'models/signal_classifier.pkl')
        self.retrain_frequency = self.params.get('retrain_frequency', 30)  # Days
        self.last_training_date = None
        
        # Ensure model directory exists
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Initialize model
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.is_trained = False
        
        # Load existing model if available
        self._load_model()
    
    def _load_model(self):
        """Load the trained model if it exists"""
        try:
            if os.path.exists(self.model_path):
                self.logger.info(f"Loading existing model from {self.model_path}")
                model_data = joblib.load(self.model_path)
                self.model = model_data.get('model')
                self.scaler = model_data.get('scaler')
                self.feature_names = model_data.get('feature_names', [])
                self.last_training_date = model_data.get('training_date')
                
                self.logger.info(f"Model loaded successfully. Last trained: {self.last_training_date}")
                return True
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
        
        self.logger.info("No existing model found or error loading model. Will train new model when sufficient data is available.")
        return False
    
    def _save_model(self):
        """Save the trained model"""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'training_date': datetime.datetime.now()
            }
            joblib.dump(model_data, self.model_path)
            self.logger.info(f"Model saved to {self.model_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            return False
    
    def _extract_features(self, candles: List[CandleData], signal: Signal, market_state: Optional[MarketState] = None) -> List[float]:
        """Extract features from candles for ML model"""
        if len(candles) < self.feature_lookback:
            self.logger.warning(f"Not enough candles for feature extraction: {len(candles)} < {self.feature_lookback}")
            return []
        
        # Extract price data
        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        volumes = np.array([c.volume for c in candles])
        
        # Get recent data for feature calculation
        recent_closes = closes[-self.feature_lookback:]
        recent_highs = highs[-self.feature_lookback:]
        recent_lows = lows[-self.feature_lookback:]
        recent_volumes = volumes[-self.feature_lookback:]
        
        features = []
        
        try:
            # Price features
            current_price = recent_closes[-1]
            
            # Moving averages
            ma5 = np.mean(recent_closes[-5:]) if len(recent_closes) >= 5 else current_price
            ma10 = np.mean(recent_closes[-10:]) if len(recent_closes) >= 10 else current_price
            ma20 = np.mean(recent_closes[-20:]) if len(recent_closes) >= 20 else current_price
            
            # Price relative to moving averages
            price_to_ma5 = current_price / ma5 - 1
            price_to_ma10 = current_price / ma10 - 1
            price_to_ma20 = current_price / ma20 - 1
            
            features.extend([price_to_ma5, price_to_ma10, price_to_ma20])
            
            # Volatility features
            atr = self._calculate_atr(recent_highs, recent_lows, recent_closes, 14)
            atr_pct = atr / current_price
            
            std_dev_5 = np.std(recent_closes[-5:]) / ma5 if len(recent_closes) >= 5 else 0
            std_dev_10 = np.std(recent_closes[-10:]) / ma10 if len(recent_closes) >= 10 else 0
            std_dev_20 = np.std(recent_closes[-20:]) / ma20 if len(recent_closes) >= 20 else 0
            
            features.extend([atr_pct, std_dev_5, std_dev_10, std_dev_20])
            
            # Momentum features
            roc_1 = (current_price / recent_closes[-2] - 1) if len(recent_closes) >= 2 else 0
            roc_5 = (current_price / recent_closes[-6] - 1) if len(recent_closes) >= 6 else 0
            roc_10 = (current_price / recent_closes[-11] - 1) if len(recent_closes) >= 11 else 0
            
            features.extend([roc_1, roc_5, roc_10])
            
            # RSI
            rsi_14 = self._calculate_rsi(recent_closes, 14)
            features.append(rsi_14)
            
            # Bollinger Bands
            bb_period = 20
            if len(recent_closes) >= bb_period:
                bb_ma = np.mean(recent_closes[-bb_period:])
                bb_std = np.std(recent_closes[-bb_period:])
                bb_upper = bb_ma + 2 * bb_std
                bb_lower = bb_ma - 2 * bb_std
                
                # Position within Bollinger Bands (0 = lower band, 1 = upper band)
                if bb_upper > bb_lower:
                    bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
                else:
                    bb_position = 0.5
                
                features.append(bb_position)
            else:
                features.append(0.5)  # Default middle position
            
            # Volume features
            vol_avg_5 = np.mean(recent_volumes[-5:]) if len(recent_volumes) >= 5 else recent_volumes[-1]
            vol_avg_10 = np.mean(recent_volumes[-10:]) if len(recent_volumes) >= 10 else recent_volumes[-1]
            
            vol_ratio_5 = recent_volumes[-1] / vol_avg_5 if vol_avg_5 > 0 else 1
            vol_ratio_10 = recent_volumes[-1] / vol_avg_10 if vol_avg_10 > 0 else 1
            
            features.extend([vol_ratio_5, vol_ratio_10])
            
            # Signal features
            features.append(signal.strength)
            features.append(1 if signal.direction == "long" else 0)
            
            # Risk-reward ratio
            if signal.stop_loss and signal.take_profit:
                if signal.direction == "long":
                    risk = (signal.entry_price - signal.stop_loss) / signal.entry_price
                    reward = (signal.take_profit - signal.entry_price) / signal.entry_price
                else:
                    risk = (signal.stop_loss - signal.entry_price) / signal.entry_price
                    reward = (signal.entry_price - signal.take_profit) / signal.entry_price
                
                risk_reward = reward / risk if risk > 0 else 0
                features.append(risk_reward)
            else:
                features.append(0)
            
            # Market state features
            if market_state:
                # Encode regime as one-hot
                regime_map = {
                    "strong_bullish": [1, 0, 0, 0, 0],
                    "bullish": [0, 1, 0, 0, 0],
                    "neutral": [0, 0, 1, 0, 0],
                    "bearish": [0, 0, 0, 1, 0],
                    "strong_bearish": [0, 0, 0, 0, 1],
                    "transitional": [0, 0, 1, 0, 0]  # Map transitional to neutral
                }
                
                regime_features = regime_map.get(market_state.regime, [0, 0, 1, 0, 0])  # Default to neutral
                features.extend(regime_features)
                
                # Add volatility and trend strength
                features.append(market_state.volatility)
                features.append(market_state.trend_strength)
                features.append(1 if market_state.is_range_bound else 0)
            else:
                # Default market state features if not available
                features.extend([0, 0, 1, 0, 0])  # Neutral regime
                features.append(0.01)  # Default volatility
                features.append(0)     # Default trend strength
                features.append(0)     # Not range-bound
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {str(e)}")
            return []
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI without using TA-Lib"""
        if len(prices) < period + 1:
            return 50.0  # Default to neutral if not enough data
        
        # Calculate price changes
        deltas = np.diff(prices)
        
        # Get gains and losses
        gains = np.copy(deltas)
        losses = np.copy(deltas)
        
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        # Calculate average gains and losses
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        """Calculate Average True Range without using TA-Lib"""
        if len(high) < period + 1:
            return high[-1] - low[-1]  # Default to current range if not enough data
        
        tr = np.zeros(len(high))
        tr[0] = high[0] - low[0]
        
        for i in range(1, len(high)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        
        atr = np.mean(tr[-period:])
        return atr
    
    def train_model(self, training_data: List[Dict], force_retrain: bool = False) -> bool:
        """
        Train the ML model using historical data
        
        Args:
            training_data: List of dictionaries with features and labels
            force_retrain: Force retraining even if model exists
            
        Returns:
            bool: True if training was successful
        """
        if len(training_data) < self.min_training_samples and not force_retrain:
            self.logger.warning(f"Not enough training samples ({len(training_data)}/{self.min_training_samples}). Skipping training.")
            return False
            
        try:
            # Extract features and labels
            X = []
            y = []
            
            for data in training_data:
                X.append(data['features'])
                y.append(data['label'])
                
            X = np.array(X)
            y = np.array(y)
            
            # Split into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train the model
            self.logger.info(f"Training ML model with {len(X_train)} samples...")
            
            # Initialize and train the model
            if self.model_type == 'random_forest':
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
            elif self.model_type == 'gradient_boosting':
                self.model = GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                )
            else:  # Default to random forest
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
                
            # Train the model
            self.model.fit(X_train, y_train)
            
            # Evaluate on validation set
            y_pred = self.model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, average='binary', zero_division=0)
            recall = recall_score(y_val, y_pred, average='binary', zero_division=0)
            f1 = f1_score(y_val, y_pred, average='binary', zero_division=0)
            
            self.logger.info(f"Model trained successfully. Validation metrics:")
            self.logger.info(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            
            # Save the model
            self.save_model()
            
            self.is_trained = True
            return True
            
        except Exception as e:
            self.logger.error(f"Error training ML model: {str(e)}")
            return False
            
    def collect_training_data(self, candles: List[CandleData], signal: Signal, market_state: MarketState, outcome: float) -> Dict:
        """
        Collect training data for the ML model
        
        Args:
            candles: List of candle data
            signal: Signal generated by the strategy
            market_state: Current market state
            outcome: Outcome of the trade (profit/loss percentage)
            
        Returns:
            Dict: Dictionary with features and label
        """
        # Extract features
        features = self._extract_features(candles, signal, market_state)
        
        # Determine label (1 for profitable trade, 0 for losing trade)
        label = 1 if outcome > 0 else 0
        
        return {
            'features': features,
            'label': label,
            'signal_type': signal.signal_type,
            'timestamp': signal.timestamp,
            'outcome': outcome
        }
    
    def predict_signal_quality(self, candles: List[CandleData], signal: Signal, market_state: MarketState) -> float:
        """
        Predict the quality of a trading signal
        
        Args:
            candles: Historical candle data
            signal: Trading signal to evaluate
            market_state: Current market state
            
        Returns:
            Quality score between 0 and 1, where 1 is highest quality
        """
        if not self.model or not self.is_trained:
            self.logger.warning("Model not trained yet, returning default quality score")
            return 0.5
        
        try:
            # Extract features
            features = self._extract_features(candles, signal, market_state)
            
            if not features:
                return 0.5
            
            # Create DataFrame with the same columns as training data
            feature_df = pd.DataFrame([features])
            
            # Ensure all expected features are present
            for feature in self.feature_names:
                if feature not in feature_df.columns:
                    feature_df[feature] = 0.0
            
            # Keep only the features used during training
            feature_df = feature_df[self.feature_names]
            
            # Scale features
            X_scaled = self.scaler.transform(feature_df)
            
            # Get prediction probability
            quality_score = self.model.predict_proba(X_scaled)[0][1]
            
            self.logger.info(f"Predicted quality score for {signal.symbol} {signal.direction} signal: {quality_score:.4f}")
            
            return quality_score
            
        except Exception as e:
            self.logger.error(f"Error predicting signal quality: {str(e)}")
            return 0.5
    
    def should_retrain(self, current_date: datetime.datetime) -> bool:
        """Check if model should be retrained based on last training date"""
        if not self.last_training_date:
            return True
            
        days_since_training = (current_date - self.last_training_date).days
        return days_since_training >= self.retrain_frequency
