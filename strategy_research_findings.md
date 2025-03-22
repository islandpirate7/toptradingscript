# Research Findings: Optimizing Mean Reversion and Combining with Trend-Following Strategies

## Optimal Parameters for Mean Reversion Strategy

### Bollinger Bands Parameters
- **Period**: 20 days is the standard setting, but research suggests 15-25 days works well for most markets
- **Standard Deviation Multiplier**: 
  - Traditional setting: 2.0 standard deviations
  - For more signals: 1.5-1.8 standard deviations
  - For higher quality signals: 2.0-2.5 standard deviations
- **Moving Average Type**: Simple Moving Average (SMA) is most common, but Exponential Moving Average (EMA) can be more responsive to recent price changes

### RSI Parameters
- **Period**: 14 days is standard, but 9-21 days can be used depending on the desired sensitivity
- **Overbought/Oversold Thresholds**:
  - Traditional: 70/30
  - More conservative: 80/20
  - More signals: 65/35

### Stop Loss and Take Profit Parameters
- **Stop Loss**: 1.5-2.0x ATR from entry point is generally effective
- **Take Profit**: 2.5-3.5x ATR from entry point for a favorable risk-reward ratio

### Signal Confirmation
- **Price Reversal**: Requiring a price reversal after crossing a Bollinger Band increases signal quality but reduces frequency
- **Volume Confirmation**: Higher than average volume on signal days increases reliability

## Combining Mean Reversion with Trend-Following Strategies

### Complementary Characteristics
- **Mean Reversion**: Works best in range-bound, sideways markets
- **Trend-Following**: Works best in strong directional markets
- **Combined Approach**: Provides more consistent returns across different market conditions

### Integration Methods

#### 1. Regime-Based Allocation
- Detect market regime (trending vs. range-bound)
- Allocate more capital to trend-following during trending markets
- Allocate more capital to mean reversion during range-bound markets

#### 2. Signal Confirmation
- Use trend indicators to filter mean reversion signals
- Only take mean reversion signals that don't contradict the primary trend
- Example: Only take "buy" mean reversion signals when the longer-term trend is upward

#### 3. Time Frame Separation
- Apply trend-following on longer timeframes (daily/weekly)
- Apply mean reversion on shorter timeframes (hourly/4-hour)
- This captures both macro trends and short-term price oscillations

#### 4. Equal Weight Allocation
- Allocate capital equally between both strategies (e.g., 50/50)
- Rebalance periodically to maintain the allocation
- Research shows this simple approach often outperforms more complex allocations

### Specific Implementation Techniques

#### AlphaTrend + Bollinger Bands Hybrid
- **AlphaTrend**: Trend-following indicator that adapts to volatility
- **Entry Conditions**:
  - Long: Price breaks above upper Bollinger Band AND AlphaTrend is upward
  - Short: Price breaks below lower Bollinger Band AND AlphaTrend is downward
- **Exit Conditions**: 
  - Close position when price crosses below/above the AlphaTrend indicator
- **Advantages**: Captures both trend and mean reversion opportunities

#### Multi-Indicator Fusion Approach
- **Trend Indicators**: Moving Averages (50-day, 200-day), MACD
- **Mean Reversion Indicators**: RSI, Bollinger Bands
- **Volatility Filter**: ATR to adjust position sizing
- **Entry Logic**: 
  - Primary signal from one system (e.g., mean reversion)
  - Confirmation from the other system (e.g., trend alignment)
- **Position Sizing**: Larger positions when both systems align, smaller when they diverge

## Risk Management for Combined Strategies

### Position Sizing
- Reduce position size during high volatility periods
- Increase position size when both strategies generate aligned signals
- Maximum risk per trade: 1-2% of total capital

### Correlation Management
- Monitor correlation between strategy returns
- Adjust allocation when correlation increases to maintain diversification benefits

### Drawdown Control
- Implement strategy-specific stop losses
- Consider portfolio-level circuit breakers (e.g., reduce exposure after X% drawdown)
- Use time-based exits for mean reversion trades that don't reach targets

## Optimization Directions

### Parameter Optimization
- Perform walk-forward optimization to avoid overfitting
- Test different parameter sets across various market regimes
- Consider symbol-specific parameters for better performance

### Signal Quality Improvement
- Add additional filters to reduce false signals:
  - Volume confirmation
  - Multiple timeframe alignment
  - Volatility-based filters

### Dynamic Adaptation
- Implement adaptive parameters based on recent market conditions
- Adjust standard deviation multipliers based on recent volatility
- Modify RSI thresholds based on trending/ranging market detection

## Performance Metrics to Monitor

- **Win Rate**: Target 40-50% for combined strategy
- **Profit Factor**: Target above 1.5
- **Maximum Drawdown**: Target below 20%
- **Sharpe Ratio**: Target above 1.0
- **Strategy Correlation**: Target below 0.5 between mean reversion and trend-following components

## Implementation Roadmap

1. Implement both strategies separately with basic parameters
2. Evaluate individual performance across different market regimes
3. Implement one of the integration methods (start with equal weight)
4. Gradually introduce dynamic allocation based on market regime
5. Fine-tune parameters and risk management rules
6. Implement portfolio-level safeguards

## References
- "The Bollinger Bands Mean Reversion Strategy" - FMZQuant
- "AlphaTrend and Bollinger Bands Combined Mean Reversion + Trend Following Strategy" - FMZQuant
- "Combining Trend-Following and Mean-Reversion" - Price Action Lab
- "Multi-Technical Indicator Based Mean Reversion and Trend Following Strategy" - Medium
