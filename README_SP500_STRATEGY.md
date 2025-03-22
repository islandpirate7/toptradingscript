# S&P 500 Multi-Factor Stock Selection Strategy

This project implements a multi-factor stock selection strategy for S&P 500 stocks, combining technical analysis and seasonality factors to identify the most promising investment opportunities.

## Strategy Overview

The strategy selects the top 25 stocks from the S&P 500 index based on a combined score that incorporates:

1. **Technical Analysis (92.5% weight)**: Evaluates stocks using multiple technical indicators:
   - RSI (Relative Strength Index)
   - MACD (Moving Average Convergence Divergence)
   - Bollinger Bands
   - Stochastic Oscillator
   - ADX (Average Directional Index)

2. **Seasonality Analysis (7.5% weight)**: Analyzes historical seasonal patterns to identify stocks that tend to perform well during specific calendar periods.

## Backtest Results

The strategy has been backtested across multiple time periods in 2023:

### Q1 2023 (Jan-Mar)
- **Win Rate**: 64.19%
- **Average Return**: 1.51%
- **Total Return**: 2418.63%
- **Profit Factor**: 2.28
- **Sharpe Ratio**: 0.29

### Q3 2023 (Jul-Sep)
- **Win Rate**: 54.44%
- **Average Return**: 0.13%
- **Total Return**: 208.76%
- **Profit Factor**: 1.09
- **Sharpe Ratio**: 0.03

### Q4 2023 (Oct-Dec)
- **Win Rate**: 53.87%
- **Average Return**: 0.21%
- **Total Return**: 333.61%
- **Profit Factor**: 1.14
- **Sharpe Ratio**: 0.05

## Key Findings

1. **Direction Performance**:
   - LONG positions generally performed better in Q1 and Q4 2023
   - SHORT positions performed better in Q3 2023
   - This suggests adapting the strategy to market conditions

2. **Score Range Performance**:
   - Stocks with scores in the 0.6-0.7 range consistently showed good performance
   - Higher scores (0.7-0.8) showed mixed results across different periods

3. **Top Performing Stocks**:
   - Q1 2023: NVDA, TSLA, META, AMD, SCHW
   - Q3 2023: RTX, TGT, AMT, UPS, NFLX
   - Q4 2023: AMD, INTC, AXP, RTX, JPM

## Implementation

The implementation consists of several Python scripts:

1. `test_sp500_selection.py`: Runs backtests for the strategy
2. `analyze_results.py`: Analyzes backtest results with detailed statistics
3. `compare_backtest_periods.py`: Compares performance across different time periods
4. `sp500_live_trading.py`: Implements live trading using the Alpaca API

## Configuration

The strategy can be configured using the `configuration_enhanced_multi_factor_500.json` file, which allows you to adjust:

- Lookback period for historical data
- Holding period for positions
- Number of top stocks to select
- Position sizing
- Weights for technical and seasonality factors
- Parameters for individual technical indicators

## Usage

### Running a Backtest

```bash
python test_sp500_selection.py --start_date 2023-01-01 --end_date 2023-03-31 --holding_period 5 --top_n 25
```

### Analyzing Results

```bash
python analyze_results.py --results_file backtest_results_20230101_20230331.csv
```

### Comparing Multiple Periods

```bash
python compare_backtest_periods.py --results_files backtest_results_20230101_20230331.csv backtest_results_20230701_20230930.csv backtest_results_20231001_20231231.csv
```

### Live Trading

```bash
# Paper trading
python sp500_live_trading.py --mode paper --top_n 25 --position_size 0.04

# Live trading (use with caution)
python sp500_live_trading.py --mode live --top_n 25 --position_size 0.04
```

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- requests
- bs4 (BeautifulSoup)
- alpaca-trade-api
- pytz

## Disclaimer

This strategy is provided for informational purposes only and does not constitute investment advice. Past performance is not indicative of future results. Always conduct your own research and consider your financial situation before making investment decisions.
