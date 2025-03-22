#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ML Strategy Selector
-------------------
This module implements a machine learning-based strategy selector
that predicts strategy performance in different market regimes.
"""

import datetime as dt
import numpy as np
import pandas as pd
import pickle
import os
import logging
from typing import Dict, Any, List, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

class MLStrategySelector:
    """
    Machine learning-based strategy selector that predicts strategy performance
    in different market regimes.
    """
    
    def __init__(self, config: Dict[str, Any], logger=None):
        """
        Initialize the ML strategy selector.
        
        Args:
            config: Configuration dictionary
            logger: Logger object
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize models and scalers
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.last_training_date = None
        self.performance_history = {}
        
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Load existing models if available
        self._load_models()
        
    def _load_models(self):
        """Load existing models from disk"""
        try:
            strategy_names = self.config.get("strategies", [])
            
            for strategy_name in strategy_names:
                model_path = f"models/{strategy_name}_model.pkl"
                scaler_path = f"models/{strategy_name}_scaler.pkl"
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    with open(model_path, 'rb') as f:
                        self.models[strategy_name] = pickle.load(f)
                    
                    with open(scaler_path, 'rb') as f:
                        self.scalers[strategy_name] = pickle.load(f)
                    
                    self.logger.info(f"Loaded existing model for strategy {strategy_name}")
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
    
    def _save_models(self):
        """Save models to disk"""
        try:
            for strategy_name, model in self.models.items():
                model_path = f"models/{strategy_name}_model.pkl"
                
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                
                if strategy_name in self.scalers:
                    scaler_path = f"models/{strategy_name}_scaler.pkl"
                    
                    with open(scaler_path, 'wb') as f:
                        pickle.dump(self.scalers[strategy_name], f)
                
                self.logger.info(f"Saved model for strategy {strategy_name}")
        except Exception as e:
            self.logger.error(f"Error saving models: {str(e)}")
    
    def _extract_features(self, market_state):
        """
        Extract features from market state for model input.
        
        Args:
            market_state: Market state object
            
        Returns:
            np.ndarray: Feature vector
        """
        features = []
        
        # Extract regime
        features.append(market_state.regime.value)
        
        # Extract market indicators
        if hasattr(market_state, 'market_indicators'):
            for key, value in market_state.market_indicators.items():
                features.append(value)
        
        # Extract breadth indicators
        if hasattr(market_state, 'breadth_indicators'):
            for key, value in market_state.breadth_indicators.items():
                features.append(value)
        
        # Extract intermarket indicators
        if hasattr(market_state, 'intermarket_indicators'):
            for key, value in market_state.intermarket_indicators.items():
                features.append(value)
        
        # Extract sector performance
        if hasattr(market_state, 'sector_performance'):
            for key, value in market_state.sector_performance.items():
                features.append(value)
        
        # Extract sentiment indicators
        if hasattr(market_state, 'sentiment_indicators'):
            for key, value in market_state.sentiment_indicators.items():
                features.append(value)
        
        # Convert to numpy array
        return np.array(features).reshape(1, -1)
    
    def _record_performance(self, strategy_name, market_state, performance):
        """
        Record strategy performance for training data.
        
        Args:
            strategy_name: Name of the strategy
            market_state: Market state object
            performance: Performance metric (e.g., return)
        """
        if strategy_name not in self.performance_history:
            self.performance_history[strategy_name] = []
        
        # Extract features
        features = self._extract_features(market_state)[0]
        
        # Record features and performance
        self.performance_history[strategy_name].append({
            'features': features,
            'performance': performance,
            'timestamp': dt.datetime.now()
        })
    
    def record_trade_result(self, strategy_name, market_state, profit_pct):
        """
        Record trade result for training data.
        
        Args:
            strategy_name: Name of the strategy
            market_state: Market state object
            profit_pct: Profit percentage
        """
        self._record_performance(strategy_name, market_state, profit_pct)
    
    def train_models(self, current_date=None):
        """
        Train ML models to predict strategy performance.
        
        Args:
            current_date: Current date (for retraining frequency check)
        """
        # Check if we need to retrain
        if self.last_training_date is not None:
            if current_date is None:
                current_date = dt.datetime.now()
            
            days_since_last_training = (current_date - self.last_training_date).days
            
            if days_since_last_training < self.config.get("retraining_interval_days", 7):
                self.logger.debug(f"Skipping training: Last trained {days_since_last_training} days ago")
                return
        
        # Train models for each strategy
        for strategy_name, history in self.performance_history.items():
            try:
                # Skip if not enough data
                if len(history) < self.config.get("min_training_samples", 30):
                    self.logger.info(f"Skipping training for {strategy_name}: Not enough data ({len(history)} samples)")
                    continue
                
                # Prepare training data
                X = np.array([record['features'] for record in history])
                y = np.array([record['performance'] for record in history])
                
                # Save feature names for the first time
                if not self.feature_names:
                    self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                
                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Train model
                model = RandomForestRegressor(
                    n_estimators=self.config.get("n_estimators", 100),
                    max_depth=self.config.get("max_depth", 10),
                    random_state=42
                )
                
                model.fit(X_scaled, y)
                
                # Save model and scaler
                self.models[strategy_name] = model
                self.scalers[strategy_name] = scaler
                
                self.logger.info(f"Trained model for strategy {strategy_name} with {len(history)} samples")
                
                # Calculate feature importance
                importances = model.feature_importances_
                
                # Log feature importance
                feature_importance = dict(zip(self.feature_names, importances))
                top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                
                self.logger.info(f"Top features for {strategy_name}: {top_features}")
                
            except Exception as e:
                self.logger.error(f"Error training model for {strategy_name}: {str(e)}")
        
        # Save models to disk
        self._save_models()
        
        # Update last training date
        self.last_training_date = current_date or dt.datetime.now()
    
    def predict_strategy_performance(self, strategy_name, market_state):
        """
        Predict strategy performance in the current market state.
        
        Args:
            strategy_name: Name of the strategy
            market_state: Market state object
            
        Returns:
            float: Predicted performance
        """
        try:
            # Check if model exists
            if strategy_name not in self.models:
                self.logger.warning(f"No model available for strategy {strategy_name}")
                return 0.0
            
            # Extract features
            features = self._extract_features(market_state)
            
            # Scale features
            scaler = self.scalers.get(strategy_name)
            if scaler:
                features_scaled = scaler.transform(features)
            else:
                features_scaled = features
            
            # Predict performance
            model = self.models[strategy_name]
            predicted_performance = model.predict(features_scaled)[0]
            
            self.logger.debug(f"Predicted performance for {strategy_name}: {predicted_performance:.2%}")
            
            return predicted_performance
        except Exception as e:
            self.logger.error(f"Error predicting performance for {strategy_name}: {str(e)}")
            return 0.0
    
    def get_optimal_strategy_weights(self, market_state, strategies):
        """
        Get optimal strategy weights based on predicted performance.
        
        Args:
            market_state: Market state object
            strategies: List of strategy names
            
        Returns:
            Dict[str, float]: Strategy weights
        """
        try:
            # Predict performance for each strategy
            performances = {}
            for strategy_name in strategies:
                performance = self.predict_strategy_performance(strategy_name, market_state)
                performances[strategy_name] = max(0.0, performance)  # Ensure non-negative
            
            # Calculate weights based on relative performance
            total_performance = sum(performances.values())
            
            if total_performance > 0:
                weights = {strategy: perf / total_performance for strategy, perf in performances.items()}
            else:
                # Equal weights if all performances are zero or negative
                weights = {strategy: 1.0 / len(strategies) for strategy in strategies}
            
            self.logger.info(f"Optimal strategy weights: {weights}")
            
            return weights
        except Exception as e:
            self.logger.error(f"Error calculating optimal strategy weights: {str(e)}")
            # Return equal weights as fallback
            return {strategy: 1.0 / len(strategies) for strategy in strategies}
