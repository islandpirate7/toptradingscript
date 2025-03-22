#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Market Simulation for S&P 500 Strategy
This script simulates market behavior to test the strategy without requiring actual market connectivity
"""

import os
import json
import yaml
import time
import random
import logging
import argparse
import traceback
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from final_sp500_strategy import SP500Strategy, run_backtest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"market_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_file='sp500_config.yaml'):
    """Load configuration from YAML file"""
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Successfully loaded configuration from {config_file}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise

class MarketSimulator:
    """Simulates market behavior for testing trading strategies"""
    
    def __init__(self, config, initial_capital=100000, volatility=0.015):
        """
        Initialize the market simulator
        
        Args:
            config (dict): Strategy configuration
            initial_capital (float): Initial capital for the simulation
            volatility (float): Daily price volatility factor
        """
        self.config = config
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.volatility = volatility
        self.positions = {}  # symbol -> {qty, entry_price, current_price}
        self.trades = []
        self.market_data = {}
        self.simulation_date = datetime.now()
        
        # Create output directories
        for path_key in ['simulation_results', 'plots', 'trades']:
            path = config.get('paths', {}).get(path_key, f"./{path_key}")
            os.makedirs(path, exist_ok=True)
        
        # Initialize market data
        self._initialize_market_data()
    
    def _initialize_market_data(self):
        """Initialize market data for simulation"""
        try:
            # Get S&P 500 symbols
            from final_sp500_strategy import get_sp500_symbols, get_midcap_symbols
            
            sp500_symbols = get_sp500_symbols()
            midcap_symbols = get_midcap_symbols()
            all_symbols = list(set(sp500_symbols + midcap_symbols))
            
            logger.info(f"Initializing market data for {len(all_symbols)} symbols")
            
            # Generate random initial prices between $10 and $500
            for symbol in all_symbols:
                initial_price = random.uniform(10, 500)
                self.market_data[symbol] = {
                    'price': initial_price,
                    'volume': random.randint(100000, 10000000),
                    'high': initial_price * 1.02,
                    'low': initial_price * 0.98,
                    'open': initial_price * 0.99,
                    'history': []  # Will store price history
                }
            
            # Add SPY for market regime detection
            self.market_data['SPY'] = {
                'price': 450.0,
                'volume': 50000000,
                'high': 455.0,
                'low': 445.0,
                'open': 448.0,
                'history': []
            }
            
            # Add sector ETFs
            sector_etfs = {
                'XLK': 'Technology',
                'XLF': 'Financials',
                'XLV': 'Healthcare',
                'XLE': 'Energy',
                'XLI': 'Industrials',
                'XLY': 'Consumer Discretionary',
                'XLP': 'Consumer Staples',
                'XLB': 'Materials',
                'XLU': 'Utilities',
                'XLRE': 'Real Estate',
                'XLC': 'Communication Services'
            }
            
            for etf in sector_etfs:
                if etf not in self.market_data:
                    self.market_data[etf] = {
                        'price': random.uniform(50, 150),
                        'volume': random.randint(5000000, 20000000),
                        'high': 0,
                        'low': 0,
                        'open': 0,
                        'history': []
                    }
            
            logger.info("Market data initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing market data: {str(e)}")
            traceback.print_exc()
    
    def update_market_prices(self):
        """Update market prices for all symbols"""
        try:
            # Update date
            self.simulation_date += timedelta(days=1)
            
            # First update SPY (market index)
            spy_change = np.random.normal(0.0005, self.volatility)  # Slight upward bias
            spy_data = self.market_data['SPY']
            spy_data['open'] = spy_data['price']
            spy_data['price'] *= (1 + spy_change)
            spy_data['high'] = max(spy_data['price'] * (1 + random.uniform(0, 0.01)), spy_data['price'])
            spy_data['low'] = min(spy_data['price'] * (1 - random.uniform(0, 0.01)), spy_data['price'])
            spy_data['volume'] = int(spy_data['volume'] * random.uniform(0.8, 1.2))
            spy_data['history'].append(spy_data['price'])
            
            # Now update all other symbols with some correlation to SPY
            for symbol, data in self.market_data.items():
                if symbol == 'SPY':
                    continue
                
                # Correlation with SPY plus random noise
                correlation = random.uniform(0.3, 0.8)
                symbol_specific = np.random.normal(0, self.volatility)
                price_change = (correlation * spy_change) + symbol_specific
                
                # Update price and related data
                data['open'] = data['price']
                data['price'] *= (1 + price_change)
                data['high'] = max(data['price'] * (1 + random.uniform(0, 0.02)), data['price'])
                data['low'] = min(data['price'] * (1 - random.uniform(0, 0.02)), data['price'])
                data['volume'] = int(data['volume'] * random.uniform(0.7, 1.3))
                data['history'].append(data['price'])
                
                # Update current price for any open positions
                if symbol in self.positions:
                    self.positions[symbol]['current_price'] = data['price']
            
            logger.info(f"Updated market prices for simulation date: {self.simulation_date.strftime('%Y-%m-%d')}")
            logger.info(f"SPY: ${self.market_data['SPY']['price']:.2f} ({spy_change*100:.2f}%)")
            
        except Exception as e:
            logger.error(f"Error updating market prices: {str(e)}")
            traceback.print_exc()
    
    def execute_trade(self, symbol, direction, quantity, score):
        """
        Execute a simulated trade
        
        Args:
            symbol (str): Stock symbol
            direction (str): 'LONG' or 'SHORT'
            quantity (int): Number of shares
            score (float): Signal score
            
        Returns:
            dict: Trade details
        """
        try:
            if symbol not in self.market_data:
                logger.warning(f"Symbol {symbol} not found in market data")
                return None
            
            price = self.market_data[symbol]['price']
            trade_value = price * quantity
            
            # Check if we have enough capital
            if direction == 'LONG' and trade_value > self.current_capital:
                logger.warning(f"Insufficient capital for trade: {symbol} {direction} {quantity} shares")
                # Adjust quantity based on available capital
                quantity = int(self.current_capital / price)
                if quantity <= 0:
                    return None
                trade_value = price * quantity
            
            # Execute the trade
            trade = {
                'symbol': symbol,
                'direction': direction,
                'quantity': quantity,
                'price': price,
                'value': trade_value,
                'score': score,
                'timestamp': self.simulation_date,
                'status': 'FILLED'
            }
            
            # Update positions and capital
            if direction == 'LONG':
                self.current_capital -= trade_value
                
                if symbol in self.positions:
                    # Average down/up
                    pos = self.positions[symbol]
                    total_shares = pos['qty'] + quantity
                    total_value = (pos['qty'] * pos['entry_price']) + trade_value
                    new_avg_price = total_value / total_shares
                    
                    self.positions[symbol] = {
                        'qty': total_shares,
                        'entry_price': new_avg_price,
                        'current_price': price,
                        'direction': direction
                    }
                else:
                    self.positions[symbol] = {
                        'qty': quantity,
                        'entry_price': price,
                        'current_price': price,
                        'direction': direction
                    }
            
            # Add to trades list
            self.trades.append(trade)
            
            logger.info(f"Executed trade: {symbol} {direction} {quantity} shares at ${price:.2f}")
            return trade
            
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            traceback.print_exc()
            return None
    
    def close_position(self, symbol, reason=""):
        """
        Close a position
        
        Args:
            symbol (str): Stock symbol
            reason (str): Reason for closing
            
        Returns:
            dict: Trade details
        """
        try:
            if symbol not in self.positions:
                logger.warning(f"No position found for {symbol}")
                return None
            
            position = self.positions[symbol]
            current_price = self.market_data[symbol]['price']
            trade_value = current_price * position['qty']
            
            # Calculate P/L
            if position['direction'] == 'LONG':
                pl = (current_price - position['entry_price']) * position['qty']
            else:
                pl = (position['entry_price'] - current_price) * position['qty']
            
            # Execute the closing trade
            trade = {
                'symbol': symbol,
                'direction': 'SELL' if position['direction'] == 'LONG' else 'BUY_TO_COVER',
                'quantity': position['qty'],
                'price': current_price,
                'value': trade_value,
                'pl': pl,
                'pl_percent': (pl / (position['entry_price'] * position['qty'])) * 100,
                'timestamp': self.simulation_date,
                'status': 'FILLED',
                'reason': reason
            }
            
            # Update capital and remove position
            self.current_capital += trade_value
            del self.positions[symbol]
            
            # Add to trades list
            self.trades.append(trade)
            
            logger.info(f"Closed position: {symbol} {trade['direction']} {trade['quantity']} shares " +
                       f"at ${current_price:.2f}, P/L: ${pl:.2f} ({trade['pl_percent']:.2f}%)")
            return trade
            
        except Exception as e:
            logger.error(f"Error closing position: {str(e)}")
            traceback.print_exc()
            return None
    
    def check_stop_losses(self):
        """Check stop loss conditions for all positions"""
        try:
            positions_to_close = []
            
            for symbol, position in self.positions.items():
                current_price = self.market_data[symbol]['price']
                entry_price = position['entry_price']
                
                # Simple stop loss at 5%
                if position['direction'] == 'LONG' and current_price < entry_price * 0.95:
                    positions_to_close.append((symbol, "Stop Loss"))
                elif position['direction'] == 'SHORT' and current_price > entry_price * 1.05:
                    positions_to_close.append((symbol, "Stop Loss"))
            
            # Close positions
            for symbol, reason in positions_to_close:
                self.close_position(symbol, reason)
            
            return len(positions_to_close)
            
        except Exception as e:
            logger.error(f"Error checking stop losses: {str(e)}")
            traceback.print_exc()
            return 0
    
    def get_account_summary(self):
        """Get account summary"""
        try:
            # Calculate portfolio value
            portfolio_value = 0
            for symbol, position in self.positions.items():
                current_price = self.market_data[symbol]['price']
                portfolio_value += current_price * position['qty']
            
            total_value = self.current_capital + portfolio_value
            
            # Calculate P/L
            total_pl = total_value - self.initial_capital
            total_pl_percent = (total_pl / self.initial_capital) * 100
            
            summary = {
                'date': self.simulation_date,
                'cash': self.current_capital,
                'portfolio_value': portfolio_value,
                'total_value': total_value,
                'total_pl': total_pl,
                'total_pl_percent': total_pl_percent,
                'open_positions': len(self.positions),
                'total_trades': len(self.trades)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting account summary: {str(e)}")
            traceback.print_exc()
            return {}
    
    def get_historical_data(self, symbol, days=20):
        """Get historical data for a symbol"""
        try:
            if symbol not in self.market_data:
                logger.warning(f"Symbol {symbol} not found in market data")
                return None
            
            data = self.market_data[symbol]
            history = data['history']
            
            # If we don't have enough history, pad with generated data
            if len(history) < days:
                # Generate some fake history
                current_price = data['price']
                fake_history = []
                for i in range(days - len(history)):
                    change = np.random.normal(0, self.volatility)
                    current_price /= (1 + change)  # Work backwards
                    fake_history.append(current_price)
                
                # Combine fake history with real history
                history = fake_history[::-1] + history
            
            # Take the last 'days' elements
            recent_history = history[-days:]
            
            # Create a dataframe
            dates = [(self.simulation_date - timedelta(days=days-i)) for i in range(len(recent_history))]
            df = pd.DataFrame({
                'date': dates,
                'close': recent_history,
                'volume': [random.randint(100000, 10000000) for _ in range(len(recent_history))]
            })
            
            # Add open, high, low
            df['open'] = df['close'].shift(1).fillna(df['close'] * 0.99)
            df['high'] = df.apply(lambda row: max(row['open'], row['close']) * (1 + random.uniform(0, 0.02)), axis=1)
            df['low'] = df.apply(lambda row: min(row['open'], row['close']) * (1 - random.uniform(0, 0.02)), axis=1)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting historical data: {str(e)}")
            traceback.print_exc()
            return None
    
    def save_results(self):
        """Save simulation results"""
        try:
            # Save account summary
            summary = self.get_account_summary()
            summary_df = pd.DataFrame([summary])
            summary_file = os.path.join(
                self.config['paths']['simulation_results'], 
                f"sim_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            summary_df.to_csv(summary_file, index=False)
            
            # Save trades
            if self.trades:
                trades_df = pd.DataFrame(self.trades)
                trades_file = os.path.join(
                    self.config['paths']['trades'], 
                    f"sim_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                )
                trades_df.to_csv(trades_file, index=False)
            
            # Save positions
            if self.positions:
                positions_data = []
                for symbol, pos in self.positions.items():
                    current_price = self.market_data[symbol]['price']
                    entry_price = pos['entry_price']
                    pl = (current_price - entry_price) * pos['qty'] if pos['direction'] == 'LONG' else (entry_price - current_price) * pos['qty']
                    pl_percent = (pl / (entry_price * pos['qty'])) * 100
                    
                    positions_data.append({
                        'symbol': symbol,
                        'direction': pos['direction'],
                        'quantity': pos['qty'],
                        'entry_price': entry_price,
                        'current_price': current_price,
                        'market_value': current_price * pos['qty'],
                        'unrealized_pl': pl,
                        'unrealized_pl_percent': pl_percent
                    })
                
                positions_df = pd.DataFrame(positions_data)
                positions_file = os.path.join(
                    self.config['paths']['trades'], 
                    f"sim_positions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                )
                positions_df.to_csv(positions_file, index=False)
            
            logger.info(f"Simulation results saved")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            traceback.print_exc()

def run_simulation(days=30, initial_capital=100000, max_signals=20, check_interval=5):
    """
    Run a market simulation
    
    Args:
        days (int): Number of days to simulate
        initial_capital (float): Initial capital
        max_signals (int): Maximum number of signals to act on
        check_interval (int): How often to check for new signals (in days)
    """
    try:
        # Load configuration
        config = load_config()
        
        # Initialize simulator
        simulator = MarketSimulator(config, initial_capital)
        
        logger.info(f"Starting market simulation for {days} days")
        logger.info(f"Initial capital: ${initial_capital}")
        
        # Run simulation
        day = 0
        while day < days:
            # Update market prices
            simulator.update_market_prices()
            day += 1
            
            # Check if we should generate signals
            if day % check_interval == 0 or day == days:
                logger.info(f"Day {day}/{days}: Generating signals")
                
                # Generate signals using the strategy logic
                # This is a simplified version that generates random signals
                all_symbols = list(simulator.market_data.keys())
                random.shuffle(all_symbols)
                
                # Filter out ETFs and select a subset
                trading_symbols = [s for s in all_symbols if not s.startswith('X') and s != 'SPY'][:100]
                
                # Generate random signals
                signals = []
                for symbol in trading_symbols[:max_signals*2]:  # Generate more than we need
                    if random.random() < 0.7:  # 70% chance of LONG signal
                        score = random.uniform(0.5, 1.0)
                        signals.append({
                            'symbol': symbol,
                            'direction': 'LONG',
                            'score': score,
                            'is_midcap': random.random() < 0.3  # 30% chance of being mid-cap
                        })
                
                # Sort by score and limit
                signals = sorted(signals, key=lambda x: x['score'], reverse=True)[:max_signals]
                
                if signals:
                    logger.info(f"Generated {len(signals)} signals")
                    
                    # Execute trades
                    for signal in signals:
                        # Calculate position size (1-5% of capital per trade)
                        position_pct = 0.01 + (signal['score'] - 0.5) * 0.08  # 1-5% based on score
                        position_value = simulator.current_capital * position_pct
                        price = simulator.market_data[signal['symbol']]['price']
                        quantity = int(position_value / price)
                        
                        if quantity > 0:
                            simulator.execute_trade(
                                signal['symbol'], 
                                signal['direction'], 
                                quantity, 
                                signal['score']
                            )
                
                # Check stop losses
                closed_count = simulator.check_stop_losses()
                if closed_count > 0:
                    logger.info(f"Closed {closed_count} positions due to stop loss")
                
                # Log account summary
                summary = simulator.get_account_summary()
                logger.info(f"Account value: ${summary['total_value']:.2f}")
                logger.info(f"P/L: ${summary['total_pl']:.2f} ({summary['total_pl_percent']:.2f}%)")
                logger.info(f"Open positions: {summary['open_positions']}")
            
            # Sleep to simulate passage of time (only in interactive mode)
            if days <= 10:  # Only sleep for short simulations
                time.sleep(1)
        
        # End of simulation
        logger.info("Simulation completed")
        
        # Final account summary
        summary = simulator.get_account_summary()
        logger.info("=== FINAL ACCOUNT SUMMARY ===")
        logger.info(f"Initial capital: ${initial_capital:.2f}")
        logger.info(f"Final account value: ${summary['total_value']:.2f}")
        logger.info(f"Total P/L: ${summary['total_pl']:.2f} ({summary['total_pl_percent']:.2f}%)")
        logger.info(f"Cash: ${summary['cash']:.2f}")
        logger.info(f"Portfolio value: ${summary['portfolio_value']:.2f}")
        logger.info(f"Open positions: {summary['open_positions']}")
        logger.info(f"Total trades: {summary['total_trades']}")
        
        # Save results
        simulator.save_results()
        
        return summary
        
    except Exception as e:
        logger.error(f"Error in simulation: {str(e)}")
        traceback.print_exc()
        return None

def main():
    """Main function to run market simulation"""
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Run market simulation for S&P 500 strategy')
        parser.add_argument('--days', type=int, default=30, 
                           help='Number of days to simulate')
        parser.add_argument('--capital', type=float, default=100000, 
                           help='Initial capital')
        parser.add_argument('--max_signals', type=int, default=20, 
                           help='Maximum number of signals to act on')
        parser.add_argument('--interval', type=int, default=5, 
                           help='Check interval in days')
        args = parser.parse_args()
        
        # Run simulation
        run_simulation(
            days=args.days,
            initial_capital=args.capital,
            max_signals=args.max_signals,
            check_interval=args.interval
        )
        
    except Exception as e:
        logger.error(f"Error running simulation: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
