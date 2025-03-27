#!/usr/bin/env python
"""
Log Analyzer for Trading System
-------------------------------
This tool analyzes log files from the trading system to provide insights,
identify patterns, and help debug issues.
"""

import os
import re
import json
import argparse
import datetime
from collections import Counter, defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

def find_log_files(log_dir="logs", pattern=None, days=7):
    """Find log files in the specified directory."""
    if not os.path.exists(log_dir):
        print(f"Log directory '{log_dir}' does not exist.")
        return []
    
    # Get all log files
    log_files = []
    for root, _, files in os.walk(log_dir):
        for file in files:
            if file.endswith('.log') or file.endswith('.json'):
                if pattern and pattern not in file:
                    continue
                
                file_path = os.path.join(root, file)
                file_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
                
                # Filter by date if specified
                if days:
                    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
                    if file_time < cutoff_date:
                        continue
                
                log_files.append((file_path, file_time))
    
    # Sort by modification time (newest first)
    log_files.sort(key=lambda x: x[1], reverse=True)
    return log_files

def parse_log_file(file_path):
    """Parse a log file and extract entries."""
    entries = []
    
    try:
        with open(file_path, 'r') as f:
            if file_path.endswith('.json'):
                # Parse JSON log file
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        entries.append(entry)
                    except json.JSONDecodeError:
                        continue
            else:
                # Parse regular log file
                for line in f:
                    # Parse log line with regex
                    match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - ([^-]+) - ([^-]+) - (.+)', line.strip())
                    if match:
                        timestamp, logger, level, message = match.groups()
                        entries.append({
                            'timestamp': timestamp,
                            'logger': logger.strip(),
                            'level': level.strip(),
                            'message': message.strip()
                        })
    except Exception as e:
        print(f"Error parsing log file {file_path}: {e}")
    
    return entries

def analyze_log_entries(entries):
    """Analyze log entries and extract insights."""
    # Count log levels
    level_counts = Counter(entry['level'] for entry in entries)
    
    # Count loggers
    logger_counts = Counter(entry['logger'] for entry in entries)
    
    # Extract errors and warnings
    errors = [entry for entry in entries if entry['level'] in ('ERROR', 'CRITICAL')]
    warnings = [entry for entry in entries if entry['level'] == 'WARNING']
    
    # Extract trade information
    trades = []
    for entry in entries:
        if 'trade_data' in entry:
            trades.append(entry['trade_data'])
        elif 'message' in entry and 'Trade' in entry['message']:
            # Try to extract trade info from message
            match = re.search(r'Trade (\d+): ([A-Z]+) - Entry: ([^at]+) at \$([0-9.]+), Exit: ([^at]+) at \$([0-9.]+), Profit: \$([0-9.-]+) \(([0-9.-]+)%\)', entry['message'])
            if match:
                trade_num, symbol, entry_date, entry_price, exit_date, exit_price, profit, profit_pct = match.groups()
                trades.append({
                    'symbol': symbol,
                    'entry_date': entry_date.strip(),
                    'exit_date': exit_date.strip(),
                    'entry_price': float(entry_price),
                    'exit_price': float(exit_price),
                    'profit': float(profit),
                    'profit_pct': float(profit_pct)
                })
    
    # Extract performance metrics
    performance = []
    for entry in entries:
        if 'performance_data' in entry:
            performance.append(entry['performance_data'])
        elif 'message' in entry and 'Backtest Summary' in entry['message']:
            # Try to extract summary from message
            try:
                # Extract the JSON part of the message
                json_str = entry['message'].split('Backtest Summary:')[1].strip()
                summary = eval(json_str)  # Using eval since the string contains numpy objects
                performance.append(summary)
            except:
                pass
    
    # Extract system events (startup, shutdown, etc.)
    system_events = []
    for entry in entries:
        if 'message' in entry:
            message = entry['message']
            if any(keyword in message for keyword in ['initialized', 'started', 'completed', 'shutdown', 'error']):
                system_events.append({
                    'timestamp': entry.get('timestamp', ''),
                    'level': entry.get('level', ''),
                    'message': message
                })
    
    return {
        'level_counts': level_counts,
        'logger_counts': logger_counts,
        'errors': errors,
        'warnings': warnings,
        'trades': trades,
        'performance': performance,
        'system_events': system_events
    }

def print_summary(analysis):
    """Print a summary of the log analysis."""
    print("\nLOG ANALYSIS SUMMARY")
    print("====================")
    
    # Print log level counts
    print("\nLog Levels:")
    level_table = [[level, count] for level, count in analysis['level_counts'].items()]
    print(tabulate(level_table, headers=["Level", "Count"], tablefmt="grid"))
    
    # Print logger counts
    print("\nLoggers:")
    logger_table = [[logger, count] for logger, count in analysis['logger_counts'].items()]
    print(tabulate(logger_table, headers=["Logger", "Count"], tablefmt="grid"))
    
    # Print errors and warnings
    if analysis['errors']:
        print(f"\nErrors ({len(analysis['errors'])}):")
        for i, error in enumerate(analysis['errors'][:5]):  # Show only first 5
            print(f"{i+1}. {error.get('timestamp', '')}: {error.get('message', '')}")
        if len(analysis['errors']) > 5:
            print(f"... and {len(analysis['errors']) - 5} more errors")
    else:
        print("\nNo errors found.")
    
    if analysis['warnings']:
        print(f"\nWarnings ({len(analysis['warnings'])}):")
        for i, warning in enumerate(analysis['warnings'][:5]):  # Show only first 5
            print(f"{i+1}. {warning.get('timestamp', '')}: {warning.get('message', '')}")
        if len(analysis['warnings']) > 5:
            print(f"... and {len(analysis['warnings']) - 5} more warnings")
    else:
        print("\nNo warnings found.")
    
    # Print trade summary
    if analysis['trades']:
        trades = analysis['trades']
        winning_trades = [t for t in trades if t.get('profit', 0) > 0]
        
        print("\nTrade Summary:")
        print(f"  - Total Trades: {len(trades)}")
        print(f"  - Winning Trades: {len(winning_trades)} ({len(winning_trades)/len(trades)*100:.2f}%)")
        print(f"  - Losing Trades: {len(trades) - len(winning_trades)} ({(len(trades) - len(winning_trades))/len(trades)*100:.2f}%)")
        
        if trades:
            total_profit = sum(t.get('profit', 0) for t in trades)
            print(f"  - Total Profit: ${total_profit:.2f}")
            
            # Show top 3 winning and losing trades
            if winning_trades:
                top_winners = sorted(winning_trades, key=lambda x: x.get('profit', 0), reverse=True)[:3]
                print("\nTop Winners:")
                for i, trade in enumerate(top_winners):
                    print(f"  {i+1}. {trade.get('symbol', 'Unknown')}: ${trade.get('profit', 0):.2f} ({trade.get('profit_pct', 0):.2f}%)")
            
            losing_trades = [t for t in trades if t.get('profit', 0) < 0]
            if losing_trades:
                top_losers = sorted(losing_trades, key=lambda x: x.get('profit', 0))[:3]
                print("\nTop Losers:")
                for i, trade in enumerate(top_losers):
                    print(f"  {i+1}. {trade.get('symbol', 'Unknown')}: ${trade.get('profit', 0):.2f} ({trade.get('profit_pct', 0):.2f}%)")
    else:
        print("\nNo trade data found.")
    
    # Print performance summary
    if analysis['performance']:
        print("\nPerformance Summary:")
        for i, perf in enumerate(analysis['performance'][:3]):  # Show only first 3
            if 'type' in perf:
                print(f"\n{perf.get('type', 'Unknown').title()}:")
            else:
                print(f"\nBacktest {i+1}:")
            
            # Print key metrics
            metrics = [
                ('Initial Capital', f"${perf.get('initial_capital', 0):.2f}"),
                ('Final Equity', f"${perf.get('final_equity', perf.get('final_capital', 0)):.2f}"),
                ('Return', f"{perf.get('return', perf.get('total_return', 0)):.2f}%"),
                ('Win Rate', f"{perf.get('win_rate', 0):.2f}%"),
                ('Profit Factor', f"{perf.get('profit_factor', 0)}")
            ]
            
            for metric, value in metrics:
                print(f"  - {metric}: {value}")
        
        if len(analysis['performance']) > 3:
            print(f"... and {len(analysis['performance']) - 3} more performance records")
    else:
        print("\nNo performance data found.")
    
    # Print system events
    if analysis['system_events']:
        print("\nSystem Events:")
        for i, event in enumerate(analysis['system_events'][:10]):  # Show only first 10
            print(f"{i+1}. {event.get('timestamp', '')}: {event.get('message', '')}")
        if len(analysis['system_events']) > 10:
            print(f"... and {len(analysis['system_events']) - 10} more events")
    else:
        print("\nNo system events found.")

def plot_log_activity(entries, output_file=None):
    """Plot log activity over time."""
    if not entries:
        print("No log entries to plot.")
        return
    
    # Extract timestamps
    timestamps = []
    for entry in entries:
        if 'timestamp' in entry:
            try:
                ts = datetime.datetime.strptime(entry['timestamp'], '%Y-%m-%d %H:%M:%S,%f')
                timestamps.append(ts)
            except:
                continue
    
    if not timestamps:
        print("No valid timestamps found in log entries.")
        return
    
    # Create DataFrame with timestamps
    df = pd.DataFrame({'timestamp': timestamps})
    df['hour'] = df['timestamp'].dt.floor('H')
    
    # Count logs per hour
    hourly_counts = df.groupby('hour').size()
    
    # Plot
    plt.figure(figsize=(12, 6))
    hourly_counts.plot(kind='bar')
    plt.title('Log Activity by Hour')
    plt.xlabel('Hour')
    plt.ylabel('Number of Log Entries')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        print(f"Log activity plot saved to {output_file}")
    else:
        plt.show()

def plot_error_distribution(entries, output_file=None):
    """Plot distribution of errors and warnings."""
    if not entries:
        print("No log entries to plot.")
        return
    
    # Extract errors and warnings
    errors = [entry for entry in entries if entry.get('level') in ('ERROR', 'CRITICAL')]
    warnings = [entry for entry in entries if entry.get('level') == 'WARNING']
    
    if not errors and not warnings:
        print("No errors or warnings found in log entries.")
        return
    
    # Group errors by message pattern
    error_types = defaultdict(int)
    for error in errors:
        message = error.get('message', '')
        # Extract the first part of the error message (before any variable data)
        pattern = re.sub(r'[0-9]', '#', message.split(':')[0])
        error_types[pattern] += 1
    
    # Group warnings by message pattern
    warning_types = defaultdict(int)
    for warning in warnings:
        message = warning.get('message', '')
        # Extract the first part of the warning message (before any variable data)
        pattern = re.sub(r'[0-9]', '#', message.split(':')[0])
        warning_types[pattern] += 1
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Error types
    if error_types:
        error_df = pd.DataFrame({'count': list(error_types.values())}, index=list(error_types.keys()))
        error_df = error_df.sort_values('count', ascending=False)
        error_df = error_df.head(10)  # Show only top 10
        error_df.plot(kind='barh', ax=ax1, color='red')
        ax1.set_title('Top Error Types')
        ax1.set_xlabel('Count')
        ax1.set_ylabel('Error Type')
    else:
        ax1.text(0.5, 0.5, 'No errors found', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
    
    # Warning types
    if warning_types:
        warning_df = pd.DataFrame({'count': list(warning_types.values())}, index=list(warning_types.keys()))
        warning_df = warning_df.sort_values('count', ascending=False)
        warning_df = warning_df.head(10)  # Show only top 10
        warning_df.plot(kind='barh', ax=ax2, color='orange')
        ax2.set_title('Top Warning Types')
        ax2.set_xlabel('Count')
        ax2.set_ylabel('Warning Type')
    else:
        ax2.text(0.5, 0.5, 'No warnings found', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        print(f"Error distribution plot saved to {output_file}")
    else:
        plt.show()

def plot_trade_performance(trades, output_file=None):
    """Plot trade performance."""
    if not trades:
        print("No trade data to plot.")
        return
    
    # Create DataFrame
    trade_data = []
    for trade in trades:
        if 'symbol' in trade and 'profit' in trade:
            trade_data.append({
                'symbol': trade['symbol'],
                'profit': trade['profit'],
                'profit_pct': trade.get('profit_pct', 0),
                'entry_date': trade.get('entry_date', ''),
                'exit_date': trade.get('exit_date', '')
            })
    
    if not trade_data:
        print("No valid trade data found.")
        return
    
    df = pd.DataFrame(trade_data)
    
    # Convert dates if possible
    try:
        df['entry_date'] = pd.to_datetime(df['entry_date'])
        df['exit_date'] = pd.to_datetime(df['exit_date'])
        df = df.sort_values('entry_date')
    except:
        # If date conversion fails, just use the order in the log
        pass
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Trade P&L
    df['profit'].cumsum().plot(ax=ax1)
    ax1.set_title('Cumulative Profit/Loss')
    ax1.set_xlabel('Trade Number')
    ax1.set_ylabel('Cumulative P&L ($)')
    ax1.grid(True)
    
    # Trade P&L by symbol
    symbol_pnl = df.groupby('symbol')['profit'].sum().sort_values(ascending=False)
    symbol_pnl.plot(kind='bar', ax=ax2)
    ax2.set_title('Profit/Loss by Symbol')
    ax2.set_xlabel('Symbol')
    ax2.set_ylabel('P&L ($)')
    ax2.grid(True)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        print(f"Trade performance plot saved to {output_file}")
    else:
        plt.show()

def find_issues(analysis):
    """Find potential issues in the logs."""
    issues = []
    
    # Check for errors
    if analysis['errors']:
        issues.append(f"Found {len(analysis['errors'])} errors in the logs.")
        
        # Group errors by type
        error_types = defaultdict(int)
        for error in analysis['errors']:
            message = error.get('message', '')
            # Extract the first part of the error message (before any variable data)
            pattern = re.sub(r'[0-9]', '#', message.split(':')[0])
            error_types[pattern] += 1
        
        # Add most common error types
        for pattern, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:3]:
            issues.append(f"  - {pattern}: {count} occurrences")
    
    # Check for warnings
    if analysis['warnings']:
        issues.append(f"Found {len(analysis['warnings'])} warnings in the logs.")
        
        # Group warnings by type
        warning_types = defaultdict(int)
        for warning in analysis['warnings']:
            message = warning.get('message', '')
            # Extract the first part of the warning message (before any variable data)
            pattern = re.sub(r'[0-9]', '#', message.split(':')[0])
            warning_types[pattern] += 1
        
        # Add most common warning types
        for pattern, count in sorted(warning_types.items(), key=lambda x: x[1], reverse=True)[:3]:
            issues.append(f"  - {pattern}: {count} occurrences")
    
    # Check for performance issues
    if analysis['performance']:
        # Look for negative returns
        negative_returns = [p for p in analysis['performance'] if p.get('return', p.get('total_return', 0)) < 0]
        if negative_returns:
            issues.append(f"Found {len(negative_returns)} instances of negative returns.")
    
    # Check for trade issues
    if analysis['trades']:
        # Calculate win rate
        winning_trades = [t for t in analysis['trades'] if t.get('profit', 0) > 0]
        win_rate = len(winning_trades) / len(analysis['trades']) * 100 if analysis['trades'] else 0
        
        if win_rate < 50:
            issues.append(f"Low win rate: {win_rate:.2f}%")
        
        # Check for consecutive losses
        consecutive_losses = 0
        max_consecutive_losses = 0
        
        for trade in analysis['trades']:
            if trade.get('profit', 0) < 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0
        
        if max_consecutive_losses >= 3:
            issues.append(f"Found {max_consecutive_losses} consecutive losing trades.")
    
    # Check for system issues
    startup_count = 0
    shutdown_count = 0
    
    for event in analysis['system_events']:
        message = event.get('message', '')
        if 'initialized' in message or 'started' in message:
            startup_count += 1
        elif 'shutdown' in message:
            shutdown_count += 1
    
    if startup_count > shutdown_count + 1:
        issues.append(f"Found {startup_count} startups but only {shutdown_count} shutdowns. Possible abnormal terminations.")
    
    return issues

def main():
    parser = argparse.ArgumentParser(description='Analyze trading system log files')
    parser.add_argument('--log-dir', dest='log_dir', default='logs', help='Directory containing log files')
    parser.add_argument('--pattern', help='Pattern to filter log files')
    parser.add_argument('--days', type=int, default=7, help='Number of days to look back')
    parser.add_argument('--output', help='Output directory for plots')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--issues', action='store_true', help='Find potential issues')
    
    args = parser.parse_args()
    
    # Find log files
    log_files = find_log_files(args.log_dir, args.pattern, args.days)
    
    if not log_files:
        print("No log files found.")
        return
    
    print(f"Found {len(log_files)} log files:")
    for i, (file_path, file_time) in enumerate(log_files[:10]):  # Show only first 10
        print(f"{i+1}. {os.path.basename(file_path)} - {file_time}")
    
    if len(log_files) > 10:
        print(f"... and {len(log_files) - 10} more files")
    
    # Parse and analyze log files
    all_entries = []
    for file_path, _ in log_files:
        entries = parse_log_file(file_path)
        all_entries.extend(entries)
    
    if not all_entries:
        print("No log entries found.")
        return
    
    print(f"Parsed {len(all_entries)} log entries.")
    
    # Analyze log entries
    analysis = analyze_log_entries(all_entries)
    
    # Print summary
    print_summary(analysis)
    
    # Find issues if requested
    if args.issues:
        issues = find_issues(analysis)
        
        if issues:
            print("\nPOTENTIAL ISSUES")
            print("================")
            for i, issue in enumerate(issues):
                print(f"{i+1}. {issue}")
        else:
            print("\nNo potential issues found.")
    
    # Generate plots if requested
    if args.plot:
        output_dir = args.output or 'plots'
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot log activity
        plot_log_activity(all_entries, os.path.join(output_dir, 'log_activity.png'))
        
        # Plot error distribution
        plot_error_distribution(all_entries, os.path.join(output_dir, 'error_distribution.png'))
        
        # Plot trade performance
        plot_trade_performance(analysis['trades'], os.path.join(output_dir, 'trade_performance.png'))

if __name__ == "__main__":
    main()
