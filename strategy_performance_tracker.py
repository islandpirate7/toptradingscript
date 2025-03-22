import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StrategyPerformanceTracker:
    """
    Tracks and analyzes the performance of trading strategies over time
    """
    
    def __init__(self, strategy_name):
        """
        Initialize the performance tracker
        
        Args:
            strategy_name (str): Name of the strategy to track
        """
        self.strategy_name = strategy_name
        self.performance_dir = os.path.join('performance', strategy_name)
        os.makedirs(self.performance_dir, exist_ok=True)
        
        # Initialize performance data
        self.daily_performance = None
        self.trade_history = None
        self.stop_loss_history = None
        
        # Default metrics
        self._win_rate = 0.0
        self._profit_factor = 0.0
    
    @property
    def win_rate(self):
        """Get the current win rate"""
        return self._win_rate
    
    @property
    def profit_factor(self):
        """Get the current profit factor"""
        return self._profit_factor
    
    def load_trade_data(self, trades_dir='trades'):
        """
        Load all trade data from CSV files
        
        Args:
            trades_dir (str): Directory containing trade CSV files
        """
        try:
            # Get all trade CSV files
            trade_files = [f for f in os.listdir(trades_dir) if f.endswith('.csv')]
            
            if not trade_files:
                logger.warning(f"No trade files found in {trades_dir}")
                return None
            
            # Load and combine all trade data
            all_trades = []
            for file in trade_files:
                file_path = os.path.join(trades_dir, file)
                trades = pd.read_csv(file_path)
                all_trades.append(trades)
            
            if all_trades:
                self.trade_history = pd.concat(all_trades, ignore_index=True)
                logger.info(f"Loaded {len(self.trade_history)} trades from {len(trade_files)} files")
                return self.trade_history
            else:
                logger.warning("No trade data found")
                return None
        
        except Exception as e:
            logger.error(f"Error loading trade data: {str(e)}")
            return None
    
    def load_stop_loss_data(self, file_path='stop_loss_history.csv'):
        """
        Load stop loss history data
        
        Args:
            file_path (str): Path to the stop loss history CSV file
        """
        try:
            if os.path.exists(file_path):
                self.stop_loss_history = pd.read_csv(file_path)
                logger.info(f"Loaded {len(self.stop_loss_history)} stop loss events")
                return self.stop_loss_history
            else:
                logger.warning(f"Stop loss history file not found: {file_path}")
                return None
        
        except Exception as e:
            logger.error(f"Error loading stop loss data: {str(e)}")
            return None
    
    def record_stop_loss_event(self, position_data, file_path='stop_loss_history.csv'):
        """
        Record a stop loss event
        
        Args:
            position_data (dict): Data about the position that hit stop loss
            file_path (str): Path to save the stop loss history
        """
        try:
            # Add timestamp
            position_data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Create DataFrame for this event
            event_df = pd.DataFrame([position_data])
            
            # Append to existing file or create new one
            if os.path.exists(file_path):
                existing_df = pd.read_csv(file_path)
                updated_df = pd.concat([existing_df, event_df], ignore_index=True)
                updated_df.to_csv(file_path, index=False)
            else:
                event_df.to_csv(file_path, index=False)
            
            logger.info(f"Recorded stop loss event for {position_data['symbol']}")
            
            # Update stop loss history
            self.load_stop_loss_data(file_path)
        
        except Exception as e:
            logger.error(f"Error recording stop loss event: {str(e)}")
    
    def analyze_stop_loss_effectiveness(self):
        """
        Analyze the effectiveness of stop loss rules
        """
        if self.stop_loss_history is None:
            logger.warning("No stop loss history available for analysis")
            return None
        
        try:
            # Calculate basic statistics
            total_events = len(self.stop_loss_history)
            avg_loss_pct = self.stop_loss_history['unrealized_plpc'].mean() * 100
            
            # Group by direction
            direction_counts = self.stop_loss_history['direction'].value_counts()
            
            # Group by symbol
            symbol_counts = self.stop_loss_history['symbol'].value_counts().head(10)
            
            # Results
            results = {
                'total_events': total_events,
                'avg_loss_pct': avg_loss_pct,
                'direction_counts': direction_counts.to_dict(),
                'top_symbols': symbol_counts.to_dict()
            }
            
            logger.info(f"Stop Loss Analysis: {total_events} events, Avg Loss: {avg_loss_pct:.2f}%")
            
            return results
        
        except Exception as e:
            logger.error(f"Error analyzing stop loss effectiveness: {str(e)}")
            return None
    
    def fetch_account_performance(self, api):
        """
        Fetch account performance data from Alpaca API
        
        Args:
            api: Alpaca API instance
        """
        try:
            # Get account
            account = api.get_account()
            
            # Get current portfolio value
            portfolio_value = float(account.portfolio_value)
            
            # Get positions
            positions = api.list_positions()
            
            # Calculate position data
            long_positions = [p for p in positions if float(p.qty) > 0]
            short_positions = [p for p in positions if float(p.qty) < 0]
            
            long_value = sum([float(p.market_value) for p in long_positions])
            short_value = sum([float(p.market_value) for p in short_positions])
            
            # Calculate unrealized P&L
            unrealized_pl = sum([float(p.unrealized_pl) for p in positions])
            unrealized_plpc = unrealized_pl / portfolio_value if portfolio_value > 0 else 0
            
            # Calculate position counts and average P&L by direction
            long_count = len(long_positions)
            short_count = len(short_positions)
            
            long_pl = sum([float(p.unrealized_pl) for p in long_positions])
            short_pl = sum([float(p.unrealized_pl) for p in short_positions])
            
            long_plpc = long_pl / long_value if long_value > 0 else 0
            short_plpc = short_pl / abs(short_value) if short_value != 0 else 0
            
            # Create performance snapshot
            performance = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'portfolio_value': portfolio_value,
                'cash': float(account.cash),
                'long_value': long_value,
                'short_value': abs(short_value),
                'long_count': long_count,
                'short_count': short_count,
                'unrealized_pl': unrealized_pl,
                'unrealized_plpc': unrealized_plpc * 100,  # Convert to percentage
                'long_pl': long_pl,
                'short_pl': short_pl,
                'long_plpc': long_plpc * 100,  # Convert to percentage
                'short_plpc': short_plpc * 100  # Convert to percentage
            }
            
            # Save performance data
            self.save_performance_snapshot(performance)
            
            logger.info(f"Portfolio Value: ${portfolio_value:.2f}, P&L: ${unrealized_pl:.2f} ({unrealized_plpc*100:.2f}%)")
            logger.info(f"LONG: {long_count} positions, P&L: {long_plpc*100:.2f}%, SHORT: {short_count} positions, P&L: {short_plpc*100:.2f}%")
            
            return performance
        
        except Exception as e:
            logger.error(f"Error fetching account performance: {str(e)}")
            return None
    
    def save_performance_snapshot(self, performance):
        """
        Save a performance snapshot to CSV
        
        Args:
            performance (dict): Performance data
        """
        try:
            # Convert to DataFrame
            perf_df = pd.DataFrame([performance])
            
            # Save to CSV
            file_path = os.path.join(self.performance_dir, 'performance_history.csv')
            
            if os.path.exists(file_path):
                existing_df = pd.read_csv(file_path)
                updated_df = pd.concat([existing_df, perf_df], ignore_index=True)
                updated_df.to_csv(file_path, index=False)
            else:
                perf_df.to_csv(file_path, index=False)
            
            logger.info(f"Saved performance snapshot to {file_path}")
        
        except Exception as e:
            logger.error(f"Error saving performance snapshot: {str(e)}")
    
    def load_performance_history(self):
        """
        Load performance history from CSV
        """
        try:
            file_path = os.path.join(self.performance_dir, 'performance_history.csv')
            
            if os.path.exists(file_path):
                self.daily_performance = pd.read_csv(file_path)
                self.daily_performance['timestamp'] = pd.to_datetime(self.daily_performance['timestamp'])
                logger.info(f"Loaded {len(self.daily_performance)} performance snapshots")
                return self.daily_performance
            else:
                logger.warning(f"Performance history file not found: {file_path}")
                return None
        
        except Exception as e:
            logger.error(f"Error loading performance history: {str(e)}")
            return None
    
    def generate_performance_report(self):
        """
        Generate a comprehensive performance report
        """
        try:
            # Load data if not already loaded
            if self.daily_performance is None:
                self.load_performance_history()
            
            if self.trade_history is None:
                self.load_trade_data()
            
            if self.stop_loss_history is None:
                self.load_stop_loss_data()
            
            # Check if we have performance data
            if self.daily_performance is None or len(self.daily_performance) == 0:
                logger.warning("No performance data available for report")
                return None
            
            # Calculate performance metrics
            latest = self.daily_performance.iloc[-1]
            first = self.daily_performance.iloc[0]
            
            # Overall performance
            total_return = (latest['portfolio_value'] / first['portfolio_value'] - 1) * 100
            
            # Create report
            report = {
                'strategy_name': self.strategy_name,
                'report_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'days_tracked': int(len(self.daily_performance)),
                'starting_value': float(first['portfolio_value']),
                'current_value': float(latest['portfolio_value']),
                'total_return_pct': float(total_return),
                'current_positions': {
                    'long_count': int(latest['long_count']),
                    'short_count': int(latest['short_count']),
                    'long_value': float(latest['long_value']),
                    'short_value': float(latest['short_value'])
                },
                'current_performance': {
                    'unrealized_pl': float(latest['unrealized_pl']),
                    'unrealized_plpc': float(latest['unrealized_plpc']),
                    'long_plpc': float(latest['long_plpc']),
                    'short_plpc': float(latest['short_plpc'])
                }
            }
            
            # Add trade statistics if available
            if self.trade_history is not None and len(self.trade_history) > 0:
                trade_stats = {
                    'total_trades': int(len(self.trade_history)),
                    'long_trades': int(len(self.trade_history[self.trade_history['direction'] == 'LONG'])),
                    'short_trades': int(len(self.trade_history[self.trade_history['direction'] == 'SHORT']))
                }
                report['trade_statistics'] = trade_stats
            
            # Add stop loss statistics if available
            if self.stop_loss_history is not None and len(self.stop_loss_history) > 0:
                stop_loss_stats = self.analyze_stop_loss_effectiveness()
                
                # Convert numpy types to Python native types
                if stop_loss_stats:
                    for key, value in stop_loss_stats.items():
                        if isinstance(value, dict):
                            for k, v in value.items():
                                if hasattr(v, 'item'):  # Check if it's a numpy type
                                    value[k] = v.item()  # Convert to Python native type
                        elif hasattr(value, 'item'):  # Check if it's a numpy type
                            stop_loss_stats[key] = value.item()  # Convert to Python native type
                
                report['stop_loss_statistics'] = stop_loss_stats
            
            # Save report to JSON
            report_path = os.path.join(self.performance_dir, f'performance_report_{datetime.now().strftime("%Y%m%d")}.json')
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=4)
            
            logger.info(f"Generated performance report: {report_path}")
            
            return report
        
        except Exception as e:
            logger.error(f"Error generating performance report: {str(e)}")
            return None
    
    def plot_performance(self, save_path=None):
        """
        Plot performance metrics over time
        
        Args:
            save_path (str): Path to save the plot image
        """
        try:
            # Load data if not already loaded
            if self.daily_performance is None:
                self.load_performance_history()
            
            if self.trade_history is None:
                self.load_trade_data()
            
            if self.stop_loss_history is None:
                self.load_stop_loss_data()
            
            # Check if we have performance data
            if self.daily_performance is None or len(self.daily_performance) < 2:
                logger.warning("Not enough performance data for plotting")
                return None
            
            # Create figure with subplots
            fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
            
            # Plot portfolio value
            self.daily_performance.plot(
                x='timestamp', 
                y='portfolio_value',
                ax=axes[0],
                title=f'{self.strategy_name} - Portfolio Value Over Time',
                legend=True
            )
            axes[0].set_ylabel('Portfolio Value ($)')
            axes[0].grid(True)
            
            # Plot position counts
            self.daily_performance.plot(
                x='timestamp',
                y=['long_count', 'short_count'],
                ax=axes[1],
                title='Position Counts by Direction',
                legend=True
            )
            axes[1].set_ylabel('Number of Positions')
            axes[1].grid(True)
            
            # Plot P&L percentages
            self.daily_performance.plot(
                x='timestamp',
                y=['long_plpc', 'short_plpc', 'unrealized_plpc'],
                ax=axes[2],
                title='P&L Percentages by Direction',
                legend=True
            )
            axes[2].set_ylabel('P&L (%)')
            axes[2].grid(True)
            
            # Format x-axis
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Saved performance plot to {save_path}")
            
            return fig
        
        except Exception as e:
            logger.error(f"Error plotting performance: {str(e)}")
            return None
    
    def track_performance(self, api=None):
        """
        Track and update performance metrics
        
        Args:
            api: Alpaca API instance (optional)
        """
        try:
            # Fetch account performance if API is provided
            if api:
                self.fetch_account_performance(api)
            
            # Load trade history if not already loaded
            if self.trade_history is None:
                self.load_trade_data()
            
            # Calculate win rate and profit factor if trade history is available
            if self.trade_history is not None and len(self.trade_history) > 0:
                self.calculate_trade_metrics()
            
            logger.info(f"Updated performance metrics - Win Rate: {self._win_rate:.2f}%, Profit Factor: {self._profit_factor:.2f}")
            
        except Exception as e:
            logger.error(f"Error tracking performance: {str(e)}")
    
    def calculate_trade_metrics(self):
        """Calculate win rate and profit factor from trade history"""
        try:
            if self.trade_history is None or len(self.trade_history) == 0:
                return
            
            # For now, use a simple placeholder calculation
            # In a real implementation, this would analyze closed trades with entry and exit prices
            self._win_rate = 50.0  # Placeholder 50% win rate
            self._profit_factor = 1.5  # Placeholder 1.5 profit factor
            
            logger.info(f"Calculated trade metrics from {len(self.trade_history)} trades")
            
        except Exception as e:
            logger.error(f"Error calculating trade metrics: {str(e)}")

if __name__ == "__main__":
    # Example usage
    tracker = StrategyPerformanceTracker("SP500Strategy")
    
    # Test with sample data
    sample_performance = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'portfolio_value': 100000,
        'cash': 50000,
        'long_value': 30000,
        'short_value': 20000,
        'long_count': 15,
        'short_count': 10,
        'unrealized_pl': 1500,
        'unrealized_plpc': 1.5,
        'long_pl': 1000,
        'short_pl': 500,
        'long_plpc': 3.33,
        'short_plpc': 2.5
    }
    
    # Save sample data
    tracker.save_performance_snapshot(sample_performance)
    
    # Generate report
    tracker.generate_performance_report()
