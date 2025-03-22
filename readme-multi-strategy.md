# Multi-Strategy Trading System

A comprehensive trading system that combines multiple strategies and adapts to changing market conditions.

## Features

### Core Features
- Multiple trading strategies with dynamic weights
- Automated signal generation and position management
- Risk management and position sizing
- Performance tracking and analysis
- Backtesting capabilities

### Enhanced Features
- **Advanced Market Regime Detection**: Identifies market conditions using multiple indicators and classifies them into detailed regimes
- **ML-Based Strategy Selection**: Uses machine learning to dynamically optimize strategy weights based on market conditions
- **Signal Quality Filtering**: Applies sophisticated filters to trading signals based on various criteria
- **Adaptive Position Sizing**: Dynamically adjusts position sizes based on signal strength, volatility, and market conditions
- **Broader Stock Universe**: Monitors a wider range of stocks to increase opportunities

## Getting Started

### Prerequisites
- Python 3.7+
- Required packages: pandas, numpy, scikit-learn, matplotlib, etc.

### Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure the system in `multi_strategy_config.yaml`

### Configuration
The system is configured through the `multi_strategy_config.yaml` file, which includes:

- Overall system parameters
- Strategy weights
- Stock configurations
- Signal quality filters
- Adaptive position sizing parameters
- ML-based strategy selection parameters

## Usage

### Basic Usage
```python
from multi_strategy_system import MultiStrategySystem, SystemConfig
import yaml

# Load configuration
with open('multi_strategy_config.yaml', 'r') as file:
    config_dict = yaml.safe_load(file)
    
# Create system config
config = SystemConfig(**config_dict)

# Initialize system
system = MultiStrategySystem(config)

# Start the system
system.start()
```

### Enhanced Usage
```python
# Import enhanced trading functions
from enhanced_trading_functions import (
    calculate_adaptive_position_size,
    filter_signals,
    generate_ml_signals
)

# Generate signals using ML-based strategy selection
all_signals = generate_ml_signals(
    stocks=system.config.stocks,
    strategies=system.strategies,
    candle_data=system.candle_data,
    market_state=system.market_state,
    ml_strategy_selector=system.ml_strategy_selector,
    logger=system.logger
)

# Apply enhanced quality filters
filtered_signals = filter_signals(
    signals=all_signals,
    candle_data=system.candle_data,
    config=system.config,
    signal_quality_filters=system.signal_quality_filters,
    logger=system.logger
)

# Calculate position size using adaptive sizing
position_size = calculate_adaptive_position_size(
    signal=signal,
    market_state=system.market_state,
    candle_data=system.candle_data,
    current_equity=system.current_equity,
    position_sizing_config=system.position_sizing_config,
    logger=system.logger
)
```

## Strategies

### Mean Reversion
Trades reversals from overbought/oversold conditions using Bollinger Bands and RSI.

### Trend Following
Follows established trends using moving average crossovers and ADX.

### Volatility Breakout
Trades breakouts from low volatility periods (Bollinger Band squeeze).

### Gap Trading
Trades significant gaps at market open with follow-through potential.

## Market Regime Detection

The system detects various market regimes:
- **Trending Bullish**: Strong uptrend with high ADX and positive directional movement
- **Trending Bearish**: Strong downtrend with high ADX and negative directional movement
- **Range Bound**: Low ADX indicating sideways market
- **High Volatility**: High VIX and wide price swings
- **Low Volatility**: Low VIX and compressed price action
- **Bearish Breakdown**: Market breaking below key support levels
- **Bullish Breakout**: Market breaking above key resistance levels
- **Consolidation**: Market in a tight trading range before a potential breakout

## ML-Based Strategy Selection

The system uses machine learning to:
- Predict strategy performance in different market conditions
- Dynamically adjust strategy weights
- Optimize signal generation based on historical performance
- Features used include market indicators, regime characteristics, and past performance

## Signal Quality Filtering

Signals are filtered based on:
- Signal score and strength
- Price and volume thresholds
- Sector exposure limits
- Correlation with existing positions
- Volatility constraints
- Market regime compatibility

## Adaptive Position Sizing

Position sizes are dynamically calculated based on:
- Signal strength and quality score
- Historical and current volatility
- Market regime
- Strategy predicted performance
- Portfolio risk parameters

## License
[MIT License](LICENSE)
