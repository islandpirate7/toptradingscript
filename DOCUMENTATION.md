# Multi-Strategy Trading System Documentation

## Core Files

### 1. `final_sp500_strategy.py`
- **Purpose**: Main strategy implementation file containing the core trading logic
- **Key Components**:
  - `SP500Strategy` class: Implements the trading strategy
  - `_get_seasonality_score_from_data()`: Calculates seasonality scores for stocks
  - `run_backtest()`: Function to run backtests with specified parameters
- **Recent Fixes**:
  - Fixed seasonality score calculation by converting month and day to integers
  - Added weekly stock selection refresh functionality

### 2. `portfolio.py`
- **Purpose**: Handles position sizing, trade execution, and performance tracking
- **Key Components**:
  - `get_equity()`: Calculates portfolio equity value
  - `update_equity_curve()`: Updates the equity curve with new values

### 3. `fixed_backtest_v2.py`
- **Purpose**: Enhanced backtest functionality with improved logging
- **Key Components**:
  - Implements backtesting logic with detailed performance metrics

### 4. `run_comprehensive_backtest_fixed.py`
- **Purpose**: Runs comprehensive backtests for specified quarters with detailed analysis
- **Key Components**:
  - `run_quarter_backtest()`: Runs backtest for a specific quarter
  - `run_multiple_backtests()`: Runs multiple backtests and averages results
  - `display_performance_metrics()`: Displays performance metrics from backtest results
- **Features**:
  - Supports weekly stock selection refresh with `--weekly_selection` flag
  - Handles continuous capital across quarters with `--continuous_capital` flag

## Utility Scripts

### 1. `fix_seasonality.py`
- **Purpose**: Fixed the seasonality score calculation error
- **Key Fix**: Changed `date_key = f"{month:02d}-{day:02d}"` to `date_key = f"{int(month):02d}-{int(day):02d}"`

### 2. `enhance_weekly_selection.py`
- **Purpose**: Enhanced the weekly stock selection refresh functionality
- **Key Fix**: Fixed indentation issues in the `get_symbols()` method

### 3. `fix_run_backtest_function.py`
- **Purpose**: Updated the `run_backtest()` function to accept the weekly selection parameter
- **Key Change**: Added `weekly_selection` parameter to control symbol reselection interval

### 4. `fix_performance_metrics.py`
- **Purpose**: Fixed the performance metrics display function to handle tuple return values
- **Key Fix**: Added handling for tuple return values from `run_backtest()`

## Configuration

### `sp500_config.yaml`
- **Purpose**: Stores configuration for the trading system
- **Key Components**:
  - API credentials for Alpaca
  - Strategy parameters
  - Path configurations

## Running Backtests

### Basic Usage
```bash
python run_comprehensive_backtest_fixed.py Q1_2023 --max_signals 50 --initial_capital 10000
```

### With Weekly Selection
```bash
python run_comprehensive_backtest_fixed.py Q1_2023 --max_signals 50 --initial_capital 10000 --weekly_selection
```

### Multiple Quarters
```bash
python run_comprehensive_backtest_fixed.py Q1_2023 Q2_2023 --max_signals 50 --initial_capital 10000
```

### All Quarters
```bash
python run_comprehensive_backtest_fixed.py all --max_signals 50 --initial_capital 10000
```

## Key Parameters

- `--max_signals`: Maximum number of signals to use (default: 100)
- `--initial_capital`: Initial capital for the backtest (default: 300)
- `--multiple_runs`: Run multiple backtests and average results
- `--num_runs`: Number of backtest runs to perform (default: 5)
- `--continuous_capital`: Use continuous capital across quarters
- `--weekly_selection`: Enable weekly stock selection refresh

## Tiered Signal Approach

The system uses a tiered approach to position sizing based on signal strength:
- **Tier 1 (≥0.9)**: Strongest signals, largest position sizes
- **Tier 2 (0.8-0.9)**: Medium-strength signals, medium position sizes
- **Tier 3 (0.7-0.8)**: Weaker signals, smaller position sizes
