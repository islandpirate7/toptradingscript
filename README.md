# S&P 500 Multi-Strategy Trading System

This repository contains a comprehensive trading system for the S&P 500 index, focusing on technical analysis and optimized position sizing. The strategy has been refined through extensive backtesting to maximize returns.

## Strategy Overview

The strategy uses a combination of technical indicators to select stocks from the S&P 500 and mid-cap universe to determine optimal entry and exit points. Key features include:

- **Technical Indicator Scoring**: Combines RSI, MACD, and Bollinger Bands to generate trade signals
- **LONG-Only Approach**: Focuses exclusively on LONG positions based on improved backtest performance
- **Dynamic Position Sizing**: Adjusts position sizes based on technical score and historical performance
- **Optimized Holding Period**: Configurable holding period for maximum returns
- **Adaptive Stop Loss**: Implements volatility and signal-quality based stop loss mechanisms
- **Mid-Cap Integration**: Dynamically balances large-cap and mid-cap stocks based on configuration

## Backtest Results

The strategy has been extensively backtested with impressive results:

- **Win Rate**: 96.67%
- **LONG Win Rate**: 96.67%
- **Profit Factor**: 308.47
- **Total Trades**: 120
- **Winning Trades**: 116
- **Losing Trades**: 4

## Files in this Repository

- `final_sp500_strategy.py`: The main strategy implementation with live trading capabilities
- `sp500_config.yaml`: Configuration file with all strategy parameters
- `trading_cli.py`: Command-line interface for running backtests, paper trading, and live trading
- `vercel_deploy.py`: API endpoints for deploying the strategy to Vercel

## Command-Line Usage

The system now provides a comprehensive command-line interface for all trading operations, eliminating the need for the web interface.

### Running Backtests

```bash
# Run a backtest for Q1 2023
python trading_cli.py backtest --quarter Q1_2023

# Run a backtest with custom date range
python trading_cli.py backtest --start-date 2023-01-01 --end-date 2023-03-31

# Run a backtest with custom parameters
python trading_cli.py backtest --quarter Q1_2023 --initial-capital 500 --max-signals 30 --tier1-threshold 0.85
```

### Running Paper Trading

```bash
# Run paper trading with default parameters
python trading_cli.py paper

# Run paper trading with custom parameters
python trading_cli.py paper --initial-capital 1000 --max-signals 25 --weekly-selection
```

### Running Live Trading

```bash
# Run live trading with default parameters
python trading_cli.py live

# Run live trading with custom parameters
python trading_cli.py live --initial-capital 5000 --max-signals 20 --tier1-threshold 0.85
```

### Viewing Results

```bash
# List all available result files
python trading_cli.py results --list

# View the latest result
python trading_cli.py results --latest

# View a specific result file
python trading_cli.py results --file backtest_Q1_2023.json

# View results for a specific quarter
python trading_cli.py results --quarter Q1_2023
```

## Vercel Deployment

The system can be deployed to Vercel for uninterrupted operation. The `vercel_deploy.py` file provides API endpoints for running backtests, paper trading, and live trading from a server.

### API Endpoints

- `GET /`: Root endpoint that returns the API status
- `POST /api/backtest`: Run a backtest with specified parameters
- `POST /api/paper`: Run paper trading with specified parameters
- `POST /api/live`: Run live trading with specified parameters
- `GET /api/results`: Get a list of all result files or view a specific result

### Deployment Steps

1. Create a Vercel account and install the Vercel CLI
2. Initialize a new Vercel project in the repository:
   ```bash
   vercel init
   ```
3. Configure the project to use the `vercel_deploy.py` file as the entry point
4. Deploy the project to Vercel:
   ```bash
   vercel --prod
   ```

### Example API Usage

```bash
# Run a backtest
curl -X POST https://your-vercel-app.vercel.app/api/backtest \
  -H "Content-Type: application/json" \
  -d '{"quarter": "Q1_2023", "initial_capital": 500}'

# View the latest result
curl https://your-vercel-app.vercel.app/api/results?latest=true
```

## Configuration

All strategy parameters are stored in the `sp500_config.yaml` file. This includes:

- Initial capital
- Position sizing parameters
- Tier thresholds for signal quality
- API credentials for Alpaca
- Backtest parameters

The command-line tools and API endpoints will use these parameters as defaults if not explicitly specified.

## Requirements

- Python 3.8+
- alpaca-trade-api
- pandas
- numpy
- tqdm
- flask (for web interface)
- pyyaml

## Setup

1. Create an `alpaca_credentials.json` file with your API keys:

```json
{
    "paper": {
        "api_key": "YOUR_PAPER_API_KEY",
        "api_secret": "YOUR_PAPER_API_SECRET",
        "base_url": "https://paper-api.alpaca.markets/v2"
    },
    "live": {
        "api_key": "YOUR_LIVE_API_KEY",
        "api_secret": "YOUR_LIVE_API_SECRET",
        "base_url": "https://api.alpaca.markets"
    }
}
```

2. Install the required packages:

```bash
pip install alpaca-trade-api pandas numpy tqdm flask pyyaml
```

## Strategy Optimization

The strategy has been optimized based on extensive backtesting. Key optimizations include:

1. **LONG-Only Approach**: Switched to LONG-only strategy for improved stability and performance
2. **Multiple Backtests**: Runs multiple backtests with different random seeds for more reliable results
3. **Adaptive Stop Loss**: Implements stop loss mechanisms that adapt to market conditions
4. **Mid-Cap Integration**: Balances between large-cap and mid-cap stocks for improved diversification
5. **Signal Quality Assessment**: Prioritizes signals based on technical score and market conditions

## Emergency Procedures

The system includes safety mechanisms for handling unexpected shutdowns:

1. All open positions are logged to CSV files
2. The `handle_shutdown.py` script can be run to safely close all positions in case of system failure
