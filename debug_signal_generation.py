import os
import sys
import yaml
import json
import logging
import datetime as dt
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SignalDebugger")

# Add the current directory to the path so we can import from the multi_strategy_system
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import necessary classes from multi_strategy_system
try:
    from multi_strategy_system import (
        MultiStrategySystem, CandleData, Signal, TradeDirection, 
        MeanReversionStrategy, VolatilityBreakoutStrategy,
        TrendFollowingStrategy, GapTradingStrategy,
        MarketState, StockConfig, SignalStrength, Strategy
    )
    logger.info("Successfully imported from multi_strategy_system")
except ImportError as e:
    logger.error(f"Error importing from multi_strategy_system: {e}")
    sys.exit(1)

# Create concrete implementations of the abstract strategy classes
class ConcreteMeanReversionStrategy(MeanReversionStrategy):
    def calculate_regime_weight(self, market_state: MarketState) -> float:
        # Simple implementation for testing
        return 0.8
    
    def calculate_stop_loss(self, signal: Signal, candles: List[CandleData]) -> float:
        # Use the ATR-based stop loss calculation
        if len(candles) < 14:
            return signal.stop_loss
        
        atr = self._calculate_atr(candles, 14)
        stop_loss_atr_multiplier = self.get_param('stop_loss_atr', 2.0)
        
        if signal.direction == TradeDirection.LONG:
            return signal.entry_price - (atr * stop_loss_atr_multiplier)
        else:
            return signal.entry_price + (atr * stop_loss_atr_multiplier)
    
    def calculate_take_profit(self, signal: Signal, candles: List[CandleData]) -> float:
        # Use the ATR-based take profit calculation
        if len(candles) < 14:
            return signal.take_profit
        
        atr = self._calculate_atr(candles, 14)
        take_profit_atr_multiplier = self.get_param('take_profit_atr', 3.0)
        
        if signal.direction == TradeDirection.LONG:
            return signal.entry_price + (atr * take_profit_atr_multiplier)
        else:
            return signal.entry_price - (atr * take_profit_atr_multiplier)
    
    def should_exit_position(self, signal: Signal, candles: List[CandleData], current_price: float) -> bool:
        # Simple implementation for testing
        # Exit if the signal has expired
        if dt.datetime.now() > signal.expiration:
            return True
        
        # Exit if price hits stop loss or take profit
        if signal.direction == TradeDirection.LONG:
            if current_price <= signal.stop_loss or current_price >= signal.take_profit:
                return True
        else:
            if current_price >= signal.stop_loss or current_price <= signal.take_profit:
                return True
        
        return False

class ConcreteVolatilityBreakoutStrategy(VolatilityBreakoutStrategy):
    def calculate_regime_weight(self, market_state: MarketState) -> float:
        # Simple implementation for testing
        return 0.7
    
    def calculate_stop_loss(self, signal: Signal, candles: List[CandleData]) -> float:
        # Simple implementation for testing
        if len(candles) < 14:
            return signal.stop_loss
        
        atr = self._calculate_atr(candles, 14)
        stop_loss_atr_multiplier = self.get_param('stop_loss_atr', 1.5)
        
        if signal.direction == TradeDirection.LONG:
            return signal.entry_price - (atr * stop_loss_atr_multiplier)
        else:
            return signal.entry_price + (atr * stop_loss_atr_multiplier)
    
    def calculate_take_profit(self, signal: Signal, candles: List[CandleData]) -> float:
        # Simple implementation for testing
        if len(candles) < 14:
            return signal.take_profit
        
        atr = self._calculate_atr(candles, 14)
        take_profit_atr_multiplier = self.get_param('take_profit_atr', 2.5)
        
        if signal.direction == TradeDirection.LONG:
            return signal.entry_price + (atr * take_profit_atr_multiplier)
        else:
            return signal.entry_price - (atr * take_profit_atr_multiplier)
    
    def should_exit_position(self, signal: Signal, candles: List[CandleData], current_price: float) -> bool:
        # Simple implementation for testing
        if dt.datetime.now() > signal.expiration:
            return True
        
        if signal.direction == TradeDirection.LONG:
            if current_price <= signal.stop_loss or current_price >= signal.take_profit:
                return True
        else:
            if current_price >= signal.stop_loss or current_price <= signal.take_profit:
                return True
        
        return False

class ConcreteTrendFollowingStrategy(TrendFollowingStrategy):
    def calculate_regime_weight(self, market_state: MarketState) -> float:
        # Simple implementation for testing
        return 0.7
    
    def calculate_stop_loss(self, signal: Signal, candles: List[CandleData]) -> float:
        # Simple implementation for testing
        return signal.stop_loss
    
    def calculate_take_profit(self, signal: Signal, candles: List[CandleData]) -> float:
        # Simple implementation for testing
        return signal.take_profit
    
    def should_exit_position(self, signal: Signal, candles: List[CandleData], current_price: float) -> bool:
        # Simple implementation for testing
        return False

class ConcreteGapTradingStrategy(GapTradingStrategy):
    def calculate_regime_weight(self, market_state: MarketState) -> float:
        # Simple implementation for testing
        return 0.6
    
    def calculate_stop_loss(self, signal: Signal, candles: List[CandleData]) -> float:
        # Simple implementation for testing
        return signal.stop_loss
    
    def calculate_take_profit(self, signal: Signal, candles: List[CandleData]) -> float:
        # Simple implementation for testing
        return signal.take_profit
    
    def should_exit_position(self, signal: Signal, candles: List[CandleData], current_price: float) -> bool:
        # Simple implementation for testing
        return False

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

def create_sample_candles(symbol: str, days: int = 30) -> List[CandleData]:
    """Create sample candle data for testing"""
    candles = []
    end_date = dt.datetime.now()
    
    # Generate candles for the specified number of days
    for i in range(days, 0, -1):
        date = end_date - dt.timedelta(days=i)
        
        # Create a basic candle with some volatility
        base_price = 100 + (i % 10)  # Oscillate between 100-110
        
        # Add some randomness
        import random
        random.seed(i)  # For reproducibility
        
        # Create more realistic price action
        open_price = base_price * (1 + random.uniform(-0.02, 0.02))
        close_price = base_price * (1 + random.uniform(-0.02, 0.02))
        
        # Ensure high and low make sense
        high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.01))
        low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.01))
        
        # Volume varies but trends upward on bigger price moves
        volume = 100000 * (1 + random.uniform(0, 0.5) + abs(open_price - close_price) / open_price * 10)
        
        # Create candle
        candle = CandleData(
            symbol=symbol,
            timestamp=date,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume
        )
        candles.append(candle)
    
    return candles

def create_realistic_candles(symbol: str, days: int = 60) -> List[CandleData]:
    """Create more realistic candle data with trends, reversals, and volatility"""
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
        
        # Adjust trend and volatility periodically to create regimes
        if i % 20 == 0:  # Change regime every 20 days
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
        
        volume = 100000 * volume_multiplier
        
        # Create candle
        candle = CandleData(
            symbol=symbol,
            timestamp=date,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume
        )
        candles.append(candle)
        
        # Update base price for next iteration
        base_price = close_price
    
    return candles

def test_strategy_signal_generation(strategy_name: str, config: Dict[str, Any]):
    """Test signal generation for a specific strategy"""
    logger.info(f"Testing signal generation for {strategy_name}")
    
    # Create strategy instance based on name
    strategy = None
    if strategy_name == "MeanReversion":
        strategy = ConcreteMeanReversionStrategy(config.get('strategies', {}).get('MeanReversion', {}))
    elif strategy_name == "VolatilityBreakout":
        strategy = ConcreteVolatilityBreakoutStrategy(config.get('strategies', {}).get('VolatilityBreakout', {}))
    elif strategy_name == "TrendFollowing":
        strategy = ConcreteTrendFollowingStrategy(config.get('strategies', {}).get('TrendFollowing', {}))
    elif strategy_name == "GapTrading":
        strategy = ConcreteGapTradingStrategy(config.get('strategies', {}).get('GapTrading', {}))
    else:
        logger.error(f"Unknown strategy: {strategy_name}")
        return
    
    # Create sample stock config
    stock_config = StockConfig(
        symbol="AAPL",
        max_position_size=100,
        min_position_size=10,
        max_risk_per_trade_pct=0.5,
        min_volume=10000,
        beta=1.2,
        sector="Technology"
    )
    
    # Create sample market state
    market_state = MarketState()
    
    # Test with both sample and realistic candles
    for candle_type, candle_generator in [
        ("Sample", create_sample_candles),
        ("Realistic", create_realistic_candles)
    ]:
        logger.info(f"Testing with {candle_type} candles")
        
        # Generate candles for testing
        candles = candle_generator("AAPL", 60)  # 60 days of data
        
        # Generate signals
        signals = strategy.generate_signals("AAPL", candles, stock_config, market_state)
        
        # Log results
        if signals:
            logger.info(f"Generated {len(signals)} signals with {candle_type} candles")
            for i, signal in enumerate(signals):
                logger.info(f"Signal {i+1}: {signal.direction} at {signal.entry_price:.2f}, "
                           f"SL: {signal.stop_loss:.2f}, TP: {signal.take_profit:.2f}, "
                           f"Strength: {signal.strength}, Timestamp: {signal.timestamp}")
        else:
            logger.info(f"No signals generated with {candle_type} candles")
            
            # Debug why no signals were generated
            if strategy_name == "MeanReversion":
                # Check if any candles meet the basic conditions
                bb_period = strategy.get_param('bb_period', 20)
                bb_std_dev = strategy.get_param('bb_std_dev', 2.0)
                rsi_period = strategy.get_param('rsi_period', 14)
                rsi_overbought = strategy.get_param('rsi_overbought', 70)
                rsi_oversold = strategy.get_param('rsi_oversold', 30)
                
                logger.info(f"MeanReversion parameters: BB Period={bb_period}, BB StdDev={bb_std_dev}, "
                           f"RSI Period={rsi_period}, RSI Overbought={rsi_overbought}, RSI Oversold={rsi_oversold}")
                
                # Check the last 10 candles
                for i in range(min(10, len(candles))):
                    idx = len(candles) - 1 - i
                    candle = candles[idx]
                    
                    # Calculate SMA
                    if idx >= bb_period:
                        close_prices = [c.close for c in candles[idx-bb_period+1:idx+1]]
                        sma = sum(close_prices) / bb_period
                        
                        # Calculate standard deviation
                        variance = sum((price - sma) ** 2 for price in close_prices) / bb_period
                        import math
                        std_dev = math.sqrt(variance)
                        
                        # Calculate Bollinger Bands
                        upper_band = sma + (bb_std_dev * std_dev)
                        lower_band = sma - (bb_std_dev * std_dev)
                        
                        # Calculate RSI
                        if idx >= rsi_period:
                            changes = [close_prices[j] - close_prices[j-1] for j in range(1, len(close_prices))]
                            gains = [change if change > 0 else 0 for change in changes]
                            losses = [abs(change) if change < 0 else 0 for change in changes]
                            
                            avg_gain = sum(gains[-rsi_period:]) / rsi_period
                            avg_loss = sum(losses[-rsi_period:]) / rsi_period
                            
                            if avg_loss == 0:
                                rsi = 100
                            else:
                                rs = avg_gain / avg_loss
                                rsi = 100 - (100 / (1 + rs))
                            
                            logger.info(f"Candle {idx} ({candle.timestamp}): Close={candle.close:.2f}, "
                                       f"Upper Band={upper_band:.2f}, Lower Band={lower_band:.2f}, RSI={rsi:.2f}")
                            
                            # Check buy conditions
                            buy_condition = candle.close < lower_band * 1.02 and rsi < rsi_oversold * 1.1
                            
                            # Check sell conditions
                            sell_condition = candle.close > upper_band * 0.98 and rsi > rsi_overbought * 0.9
                            
                            logger.info(f"Buy condition: {buy_condition}, Sell condition: {sell_condition}")
                        else:
                            logger.info(f"Not enough data for RSI calculation at index {idx}")
                    else:
                        logger.info(f"Not enough data for Bollinger Bands calculation at index {idx}")

def create_configuration_12(config_11: Dict[str, Any]) -> Dict[str, Any]:
    """Create configuration_12 based on configuration_11 with optimized settings"""
    # Create a deep copy of configuration_11
    config_12 = {k: v.copy() if isinstance(v, dict) else v for k, v in config_11.items()}
    
    # Update strategy weights based on our analysis
    if 'strategy_weights' in config_12:
        config_12['strategy_weights'] = {
            'MeanReversion': 0.35,  # Increased weight as it's our most optimized strategy
            'TrendFollowing': 0.30,
            'VolatilityBreakout': 0.25,
            'GapTrading': 0.10      # Reduced weight as it's less reliable
        }
    
    # Ensure all strategies have configurations
    if 'strategies' not in config_12:
        config_12['strategies'] = {}
    
    # MeanReversion strategy is already configured in configuration_11
    
    # Add or update TrendFollowing strategy configuration
    config_12['strategies']['TrendFollowing'] = {
        'ma_short_period': 20,
        'ma_long_period': 50,
        'atr_period': 14,
        'atr_multiplier': 2.5,
        'min_trend_strength': 0.5,
        'volume_confirmation': True,
        'stop_loss_atr': 2.0,
        'take_profit_atr': 3.0
    }
    
    # Add or update GapTrading strategy configuration
    config_12['strategies']['GapTrading'] = {
        'min_gap_percent': 1.5,
        'max_gap_percent': 7.0,
        'volume_threshold': 1.5,
        'stop_loss_atr': 1.5,
        'take_profit_atr': 2.5,
        'max_risk_percent': 1.0
    }
    
    # Update risk management parameters
    config_12['position_sizing_config'] = {
        'base_risk_per_trade': 0.008,  # Conservative risk per trade
        'max_position_size': 0.08,     # Maximum position size as percentage of portfolio
        'min_position_size': 0.005,    # Minimum position size as percentage of portfolio
        'volatility_adjustment': True,  # Adjust position size based on volatility
        'signal_strength_adjustment': True,  # Adjust position size based on signal strength
        'atr_multiplier': 2.0,         # ATR multiplier for stop loss calculation
        'max_risk_per_trade_pct': 1.0   # Maximum risk per trade percentage
    }
    
    # Update signal quality filters
    config_12['signal_quality_filters'] = {
        'min_score_threshold': 0.7,
        'max_correlation_threshold': 0.6,
        'min_volume_percentile': 60,
        'min_price': 10.0,
        'max_spread_percent': 0.8,
        'min_volatility_percentile': 30,
        'max_volatility_percentile': 80,
        'min_regime_weight': 0.4,
        'max_signals_per_regime': 3,
        'max_sector_exposure': 0.25,
        'max_signals_per_day': 5
    }
    
    return config_12

def main():
    """Main function to debug signal generation and create configuration_12"""
    # Load configuration_11
    config_11 = load_config('configuration_11.yaml')
    
    # Test signal generation for each strategy
    strategies = ["MeanReversion", "VolatilityBreakout", "TrendFollowing", "GapTrading"]
    for strategy in strategies:
        test_strategy_signal_generation(strategy, config_11)
    
    # Create configuration_12
    config_12 = create_configuration_12(config_11)
    
    # Save configuration_12
    try:
        with open('configuration_12.yaml', 'w') as file:
            yaml.dump(config_12, file, default_flow_style=False)
            logger.info("Saved configuration_12.yaml")
    except Exception as e:
        logger.error(f"Error saving configuration_12: {e}")

if __name__ == "__main__":
    main()
