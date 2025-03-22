import datetime as dt
import numpy as np
import logging
import copy
from typing import List, Dict, Any
from enum import Enum

class SignalStrength(Enum):
    STRONG_BUY = 3
    MODERATE_BUY = 2
    WEAK_BUY = 1
    NEUTRAL = 0
    WEAK_SELL = -1
    MODERATE_SELL = -2
    STRONG_SELL = -3

class MarketRegime(Enum):
    UNKNOWN = 0
    TRENDING_BULLISH = 1
    TRENDING_BEARISH = 2
    RANGE_BOUND = 3
    HIGH_VOLATILITY = 4
    LOW_VOLATILITY = 5
    BEARISH_BREAKDOWN = 6
    BULLISH_BREAKOUT = 7
    CONSOLIDATION = 8

def calculate_adaptive_position_size(signal, market_state, candle_data, 
                                    current_equity, position_sizing_config, logger):
    """
    Calculate adaptive position size based on signal strength, volatility, and market conditions
    
    Args:
        signal: The trading signal object
        market_state: Current market state/regime
        candle_data: Historical price data for the symbol
        current_equity: Current portfolio equity
        position_sizing_config: Configuration for position sizing
        logger: Logger object
        
    Returns:
        float: Position size in dollars
    """
    try:
        # Base position size as percentage of portfolio
        base_risk = position_sizing_config["base_risk_per_trade"]
        
        # Adjust for signal strength (score)
        signal_adjustment = 1.0
        if position_sizing_config["signal_strength_adjustment"] and hasattr(signal, 'score'):
            # Scale from 0.7 to 1.3 based on signal score
            signal_adjustment = 0.7 + (min(1.0, max(0.0, signal.score)) * 0.6)
            logger.debug(f"Signal strength adjustment for {signal.symbol}: {signal_adjustment:.2f}")
        
        # Adjust for volatility
        volatility_adjustment = 1.0
        if position_sizing_config["volatility_adjustment"] and signal.symbol in candle_data:
            # Calculate historical volatility
            close_prices = [candle.close for candle in candle_data[signal.symbol][-20:]]
            if len(close_prices) > 1:
                returns = [close_prices[i] / close_prices[i-1] - 1 for i in range(1, len(close_prices))]
                volatility = np.std(returns) * np.sqrt(252)  # Annualized
                
                # Inverse relationship with volatility - lower position size for higher volatility
                if volatility > 0:
                    volatility_adjustment = min(1.5, max(0.5, 1.0 / (volatility * 10)))
                    logger.debug(f"Volatility adjustment for {signal.symbol}: {volatility_adjustment:.2f} (volatility: {volatility:.2%})")
        
        # Adjust for market regime
        regime_adjustment = 1.0
        if market_state:
            if market_state.regime == MarketRegime.HIGH_VOLATILITY:
                regime_adjustment = 0.7  # Reduce position size in high volatility
            elif market_state.regime == MarketRegime.TRENDING_BULLISH:
                regime_adjustment = 1.2  # Increase in strong uptrend
            elif market_state.regime == MarketRegime.BEARISH_BREAKDOWN:
                regime_adjustment = 0.5  # Reduce significantly in breakdown
            elif market_state.regime == MarketRegime.BULLISH_BREAKOUT:
                regime_adjustment = 1.3  # Increase in breakout
            elif market_state.regime == MarketRegime.CONSOLIDATION:
                regime_adjustment = 0.8  # Slightly reduce in consolidation
            
            logger.debug(f"Regime adjustment for {signal.symbol}: {regime_adjustment:.2f} (regime: {market_state.regime})")
        
        # Adjust for strategy predicted performance if available
        strategy_adjustment = 1.0
        if hasattr(signal, 'metadata') and 'predicted_performance' in signal.metadata:
            pred_performance = signal.metadata['predicted_performance']
            # Scale from 0.8 to 1.2 based on predicted performance
            strategy_adjustment = 0.8 + min(0.4, max(0.0, pred_performance * 2))
            logger.debug(f"Strategy performance adjustment for {signal.symbol}: {strategy_adjustment:.2f}")
        
        # Calculate adjusted risk
        adjusted_risk = base_risk * signal_adjustment * volatility_adjustment * regime_adjustment * strategy_adjustment
        
        # Apply min/max constraints
        adjusted_risk = max(
            position_sizing_config["min_position_size"],
            min(position_sizing_config["max_position_size"], adjusted_risk)
        )
        
        # Calculate dollar amount
        position_dollars = current_equity * adjusted_risk
        
        logger.info(f"Calculated position size for {signal.symbol}: {adjusted_risk:.2%} of portfolio (${position_dollars:.2f})")
        
        return position_dollars
    except Exception as e:
        logger.error(f"Error calculating position size for {signal.symbol}: {str(e)}")
        # Fall back to minimum position size
        return current_equity * position_sizing_config["min_position_size"]

def filter_signals(signals, candle_data, config, signal_quality_filters, logger):
    """
    Apply enhanced quality filters to signals
    
    Args:
        signals: List of trading signals
        candle_data: Historical price data for symbols
        config: System configuration
        signal_quality_filters: Configuration for signal filtering
        logger: Logger object
        
    Returns:
        List: Filtered signals that passed quality checks
    """
    if not signals:
        return []
        
    filtered_signals = []
    signals_by_sector = {}
    signals_today = 0
    
    # Sort signals by score (descending)
    sorted_signals = sorted(signals, key=lambda s: getattr(s, 'score', 0.5), reverse=True)
    
    for signal in sorted_signals:
        try:
            # Skip if we've reached the daily signal limit
            if signals_today >= signal_quality_filters["max_signals_per_day"]:
                logger.info(f"Daily signal limit reached, skipping {signal.symbol}")
                continue
            
            # Score threshold filter
            if hasattr(signal, 'score') and signal.score < signal_quality_filters["min_score_threshold"]:
                logger.info(f"Signal for {signal.symbol} rejected: score {signal.score:.2f} below threshold")
                continue
            
            # Basic price filter
            if signal.entry_price < signal_quality_filters["min_price"]:
                logger.info(f"Signal for {signal.symbol} rejected: price {signal.entry_price} below minimum")
                continue
            
            # Check volume if data is available
            if signal.symbol in candle_data and len(candle_data[signal.symbol]) > 0:
                volume_data = [candle.volume for candle in candle_data[signal.symbol][-20:]]
                if volume_data:
                    current_volume = volume_data[-1]
                    avg_volume = sum(volume_data) / len(volume_data)
                    
                    if avg_volume > 0 and current_volume < avg_volume * 0.5:
                        logger.info(f"Signal for {signal.symbol} rejected: volume {current_volume} below threshold")
                        continue
            
            # Check sector exposure
            sector = "Unknown"
            for stock in config.stocks:
                if stock.symbol == signal.symbol:
                    sector = stock.sector
                    break
            
            if sector not in signals_by_sector:
                signals_by_sector[sector] = 0
            
            max_sector_signals = int(signal_quality_filters["max_sector_exposure"] * signal_quality_filters["max_signals_per_day"])
            if signals_by_sector[sector] >= max_sector_signals:
                logger.info(f"Signal for {signal.symbol} rejected: sector {sector} exposure limit reached")
                continue
            
            # Check correlation with existing signals
            if signal_quality_filters["max_correlation_threshold"] < 1.0:
                correlated = False
                for existing_signal in filtered_signals:
                    # Skip same symbol
                    if existing_signal.symbol == signal.symbol:
                        continue
                    
                    # Check if symbols are in same sector
                    existing_sector = "Unknown"
                    for stock in config.stocks:
                        if stock.symbol == existing_signal.symbol:
                            existing_sector = stock.sector
                            break
                    
                    # If same sector, consider them correlated
                    if sector == existing_sector and sector != "Unknown":
                        correlated = True
                        logger.info(f"Signal for {signal.symbol} rejected: correlated with {existing_signal.symbol} (same sector)")
                        break
                
                if correlated:
                    continue
            
            # Signal passed all filters
            filtered_signals.append(signal)
            signals_by_sector[sector] = signals_by_sector.get(sector, 0) + 1
            signals_today += 1
            
            logger.info(f"Signal for {signal.symbol} accepted")
            
        except Exception as e:
            logger.error(f"Error filtering signal for {signal.symbol}: {str(e)}")
    
    return filtered_signals

def generate_ml_signals(stocks, strategies, candle_data, market_state, ml_strategy_selector, logger):
    """
    Generate trading signals using ML-based strategy selection
    
    Args:
        stocks: List of stock configurations
        strategies: Dictionary of strategy objects
        candle_data: Historical price data for symbols
        market_state: Current market state/regime
        ml_strategy_selector: ML strategy selector object
        logger: Logger object
        
    Returns:
        List: Generated trading signals
    """
    if not market_state:
        logger.warning("Cannot generate signals: Market state not available")
        return []
        
    logger.info(f"Generating signals for market regime: {market_state.regime}")
    
    all_signals = []
    
    # Train ML models if needed
    ml_strategy_selector.train_models(dt.datetime.now())
    
    # Process each stock
    for stock_config in stocks:
        symbol = stock_config.symbol
        
        if symbol not in candle_data or len(candle_data[symbol]) < 20:
            logger.debug(f"Skipping {symbol}: Insufficient data")
            continue
            
        # Get candle data for the stock
        candles = candle_data[symbol]
        
        # Process each strategy
        for name, strategy in strategies.items():
            try:
                # Use ML to predict strategy performance in current market state
                predicted_performance = ml_strategy_selector.predict_strategy_performance(
                    name, market_state
                )
                
                # Skip strategies with negative expected performance
                if predicted_performance < 0:
                    logger.debug(f"Skipping {name} for {symbol}: Negative expected performance ({predicted_performance:.2%})")
                    continue
                
                # Calculate strategy weight based on market regime
                regime_weight = strategy.calculate_regime_weight(market_state)
                
                # Skip strategies with low regime weight
                if regime_weight < 0.3:
                    logger.debug(f"Skipping {name} for {symbol}: Low regime weight ({regime_weight:.2f})")
                    continue
                
                # Apply stock-specific strategy parameters if available
                if name == "MeanReversion" and hasattr(stock_config, "mean_reversion_params"):
                    strategy_copy = copy.deepcopy(strategy)
                    strategy_copy.config.update(stock_config.mean_reversion_params)
                    strategy_to_use = strategy_copy
                elif name == "TrendFollowing" and hasattr(stock_config, "trend_following_params"):
                    strategy_copy = copy.deepcopy(strategy)
                    strategy_copy.config.update(stock_config.trend_following_params)
                    strategy_to_use = strategy_copy
                elif name == "VolatilityBreakout" and hasattr(stock_config, "volatility_breakout_params"):
                    strategy_copy = copy.deepcopy(strategy)
                    strategy_copy.config.update(stock_config.volatility_breakout_params)
                    strategy_to_use = strategy_copy
                elif name == "GapTrading" and hasattr(stock_config, "gap_trading_params"):
                    strategy_copy = copy.deepcopy(strategy)
                    strategy_copy.config.update(stock_config.gap_trading_params)
                    strategy_to_use = strategy_copy
                else:
                    strategy_to_use = strategy
                
                # Generate signals
                new_signals = strategy_to_use.generate_signals(
                    symbol=symbol,
                    candles=candles,
                    stock_config=stock_config,
                    market_state=market_state
                )
                
                # Log signal generation results
                logger.info(f"Strategy {name} for {symbol} generated {len(new_signals)} signals")
                
                if new_signals:
                    # Enhance signals with additional metadata
                    for signal in new_signals:
                        # Add market regime info
                        signal.metadata["market_regime"] = market_state.regime.value
                        signal.metadata["regime_weight"] = regime_weight
                        signal.metadata["predicted_performance"] = predicted_performance
                        signal.metadata["strategy_name"] = name
                        
                        # Calculate signal score based on strategy performance and regime weight
                        signal.score = (regime_weight * 0.5) + (predicted_performance * 5)
                        
                        # Add timestamp if not present
                        if not hasattr(signal, 'timestamp') or not signal.timestamp:
                            signal.timestamp = dt.datetime.now()
                    
                    # Add to all signals list
                    all_signals.extend(new_signals)
            except Exception as e:
                logger.error(f"Error generating signals for {symbol} with strategy {name}: {str(e)}")
    
    return all_signals
