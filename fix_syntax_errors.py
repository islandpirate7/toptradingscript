#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix syntax errors in final_sp500_strategy.py
"""

import os
import re
import sys
import traceback

def fix_function_signature():
    """
    Fix the run_backtest function signature in final_sp500_strategy.py
    """
    try:
        # Read the file
        with open('final_sp500_strategy.py', 'r') as f:
            content = f.read()
        
        # Fix the function signature - remove escaped quotes and fix any syntax issues
        pattern = r"def run_backtest\(start_date, end_date, mode=\\'backtest\\', max_signals=None, initial_capital=300, random_seed=42, weekly_selection=True, continuous_capital=False\):"
        replacement = "def run_backtest(start_date, end_date, mode='backtest', max_signals=None, initial_capital=300, random_seed=42, weekly_selection=True, continuous_capital=False):"
        
        fixed_content = re.sub(pattern, replacement, content)
        
        # Write the updated content back to the file
        with open('final_sp500_strategy.py', 'w') as f:
            f.write(fixed_content)
        
        print("Successfully fixed run_backtest function signature")
        
    except Exception as e:
        print(f"Error fixing function signature: {str(e)}")
        traceback.print_exc()

def fix_indentation_and_syntax():
    """
    Fix indentation and syntax errors in final_sp500_strategy.py
    """
    try:
        # Read the file line by line
        with open('final_sp500_strategy.py', 'r') as f:
            lines = f.readlines()
        
        # Flag to track if we're in the run_backtest function
        in_run_backtest = False
        
        # Track the indentation level
        indent_level = 0
        
        # Process each line
        for i in range(len(lines)):
            # Check if we're entering the run_backtest function
            if "def run_backtest(" in lines[i]:
                in_run_backtest = True
                indent_level = 0
            
            # Check if we're exiting the run_backtest function
            if in_run_backtest and lines[i].strip() == "except Exception as e:":
                # We're at the end of the try block
                in_run_backtest = False
            
            # Fix specific issues based on line numbers
            if i == 3776:  # Line 3777 in the error message (0-indexed in our code)
                # Fix unexpected indentation
                lines[i] = lines[i].lstrip() + lines[i].lstrip()
            
            if i == 3882:  # Line 3883 in the error message
                # Fix "Expected expression"
                if "if" in lines[i] and not lines[i].strip().startswith("if"):
                    lines[i] = "        " + lines[i].lstrip()
            
            if i == 3883:  # Line 3884 in the error message
                # Fix unexpected indentation
                lines[i] = "        " + lines[i].lstrip()
            
            if i == 3889:  # Line 3890 in the error message
                # Fix unclosed "{"
                if "{" in lines[i] and "}" not in lines[i]:
                    lines[i] = lines[i].rstrip() + "}\n"
            
            if i == 3897:  # Line 3898 in the error message
                # Fix statements not separated by newlines
                if ";" not in lines[i] and len(lines[i].strip()) > 100:
                    parts = lines[i].split("'")
                    if len(parts) > 2:
                        lines[i] = parts[0] + "'" + parts[1] + "'\n        " + "".join(parts[2:])
            
            if i == 3965:  # Line 3966 in the error message
                # Fix "Expected expression"
                if lines[i].strip().startswith("if metrics:") and len(lines[i].strip()) < 15:
                    lines[i] = "        if metrics:\n"
            
            if i == 3966:  # Line 3967 in the error message
                # Fix unexpected indentation
                lines[i] = "            " + lines[i].lstrip()
            
            if i == 3991:  # Line 3992 in the error message
                # Fix "return" outside function
                if "return" in lines[i] and not lines[i].strip().startswith("return"):
                    lines[i] = "        return " + lines[i].split("return")[1]
        
        # Write the updated content back to the file
        with open('final_sp500_strategy.py', 'w') as f:
            f.writelines(lines)
        
        print("Successfully fixed indentation and syntax errors")
        
    except Exception as e:
        print(f"Error fixing indentation and syntax: {str(e)}")
        traceback.print_exc()

def fix_run_backtest_function():
    """
    Completely rewrite the problematic sections of the run_backtest function
    """
    try:
        # Read the file
        with open('final_sp500_strategy.py', 'r') as f:
            content = f.read()
        
        # Find the start and end of the run_backtest function
        start_pattern = r"def run_backtest\(.*?\):"
        start_match = re.search(start_pattern, content)
        
        if start_match:
            # Extract the function signature
            function_signature = start_match.group(0)
            
            # Find the end of the function (next function definition or end of file)
            end_pattern = r"def [a-zA-Z_][a-zA-Z0-9_]*\(.*?\):"
            end_matches = list(re.finditer(end_pattern, content[start_match.end():]))
            
            if end_matches:
                end_position = start_match.end() + end_matches[0].start()
            else:
                end_position = len(content)
            
            # Extract the function body
            function_body = content[start_match.end():end_position]
            
            # Fix the problematic sections
            # 1. Fix the continuous capital and final capital handling
            continuous_capital_pattern = r"# Add continuous_capital flag to summary.*?# Update final_capital for continuous capital mode.*?if metrics:.*?final_capital = metrics\['final_capital'\]"
            continuous_capital_replacement = """
        # Add continuous_capital flag to summary
        if summary:
            summary['continuous_capital'] = continuous_capital
            if metrics and 'final_capital' in metrics:
                summary['final_capital'] = metrics['final_capital']
        
        # Update final_capital for continuous capital mode
        if metrics and 'final_capital' in metrics:
            final_capital = metrics['final_capital']
"""
            
            # Use regex with DOTALL flag to match across multiple lines
            fixed_body = re.sub(continuous_capital_pattern, continuous_capital_replacement, function_body, flags=re.DOTALL)
            
            # 2. Fix the return statement
            return_pattern = r"return summary, signals"
            return_replacement = "        return summary, signals"
            
            fixed_body = re.sub(return_pattern, return_replacement, fixed_body)
            
            # Reconstruct the function
            fixed_function = function_signature + fixed_body
            
            # Replace the original function in the content
            fixed_content = content[:start_match.start()] + fixed_function + content[end_position:]
            
            # Write the updated content back to the file
            with open('final_sp500_strategy.py', 'w') as f:
                f.write(fixed_content)
            
            print("Successfully fixed run_backtest function")
        else:
            print("Could not find run_backtest function")
        
    except Exception as e:
        print(f"Error fixing run_backtest function: {str(e)}")
        traceback.print_exc()

def create_backup():
    """
    Create a backup of the original file
    """
    try:
        # Read the file
        with open('final_sp500_strategy.py', 'r') as f:
            content = f.read()
        
        # Write the backup
        with open('final_sp500_strategy.py.bak', 'w') as f:
            f.write(content)
        
        print("Successfully created backup of final_sp500_strategy.py")
        
    except Exception as e:
        print(f"Error creating backup: {str(e)}")
        traceback.print_exc()

def completely_rewrite_function():
    """
    Completely rewrite the run_backtest function with a clean implementation
    """
    try:
        # Read the file
        with open('final_sp500_strategy.py', 'r') as f:
            content = f.read()
        
        # Find the start of the run_backtest function
        start_pattern = r"def run_backtest\(.*?\):"
        start_match = re.search(start_pattern, content)
        
        if start_match:
            # Find the end of the function (next function definition or end of file)
            end_pattern = r"def [a-zA-Z_][a-zA-Z0-9_]*\(.*?\):"
            end_matches = list(re.finditer(end_pattern, content[start_match.end():]))
            
            if end_matches:
                end_position = start_match.end() + end_matches[0].start()
            else:
                end_position = len(content)
            
            # New clean implementation of the function
            new_function = """def run_backtest(start_date, end_date, mode='backtest', max_signals=None, initial_capital=300, random_seed=42, weekly_selection=True, continuous_capital=False):
    """Run a backtest for a specified period with specified initial capital"""
    try:
        # Debug logging
        logger.info("[DEBUG] Starting run_backtest in final_sp500_strategy.py")
        logger.info(f"[DEBUG] Parameters: start_date={start_date}, end_date={end_date}, mode={mode}, max_signals={max_signals}, initial_capital={initial_capital}, random_seed={random_seed}, weekly_selection={weekly_selection}, continuous_capital={continuous_capital}")
        start_time = time.time()
        
        # Load configuration
        config_path = 'sp500_config.yaml'
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        logger.info(f"Running backtest from {start_date} to {end_date} with initial capital ${initial_capital} (Seed: {random_seed})")
        
        # Create output directories if they don't exist
        for path_key in ['backtest_results', 'plots', 'trades', 'performance']:
            os.makedirs(config['paths'][path_key], exist_ok=True)
        
        # Load Alpaca credentials
        with open('alpaca_credentials.json', 'r') as f:
            credentials = json.load(f)
        
        # Use paper trading credentials for backtesting
        paper_credentials = credentials['paper']
        
        # Initialize Alpaca API
        api = tradeapi.REST(
            paper_credentials['api_key'],
            paper_credentials['api_secret'],
            paper_credentials['base_url'],
            api_version='v2'
        )
        
        # Initialize strategy in backtest mode
        strategy = SP500Strategy(
            api=api,
            config=config,
            mode=mode,
            backtest_mode=True,
            backtest_start_date=start_date,
            backtest_end_date=end_date
        )
        
        # Get S&P 500 symbols
        sp500_symbols = strategy.get_sp500_symbols()
        
        # Parse the start and end dates
        start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
        end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Calculate the date range
        date_range = (end_datetime - start_datetime).days + 1
        
        # Generate signals for the specified period
        signals = []
        
        # Get configuration parameters
        large_cap_percentage = config.get('strategy', {}).get('large_cap_percentage', 70)
        
        # Weekly or daily selection
        if weekly_selection:
            # Generate signals for each week in the date range
            current_date = start_datetime
            while current_date <= end_datetime:
                # Get the end of the week
                days_to_end_of_week = 6 - current_date.weekday()  # 6 = Saturday
                end_of_week = min(current_date + timedelta(days=days_to_end_of_week), end_datetime)
                
                # Format dates for API calls
                current_date_str = current_date.strftime("%Y-%m-%d")
                end_of_week_str = end_of_week.strftime("%Y-%m-%d")
                
                # Generate signals for this week
                weekly_signals = strategy.generate_signals(current_date_str, sp500_symbols)
                
                # Add date and week information to signals
                for signal in weekly_signals:
                    signal['date'] = current_date_str
                    signal['week'] = f"{current_date_str} to {end_of_week_str}"
                
                # Add to overall signals
                signals.extend(weekly_signals)
                
                # Move to next week
                current_date = end_of_week + timedelta(days=1)
        else:
            # Generate signals for the entire period
            signals = strategy.generate_signals(start_date, sp500_symbols)
            
            # Add date information
            for signal in signals:
                signal['date'] = start_date
        
        # Separate large-cap and mid-cap signals
        largecap_signals = [s for s in signals if not s.get('is_midcap', False)]
        midcap_signals = [s for s in signals if s.get('is_midcap', False)]
        
        logger.info(f"Generated {len(signals)} signals ({len(largecap_signals)} large-cap, {len(midcap_signals)} mid-cap)")
        
        # Limit to max_signals if specified
        if max_signals and max_signals > 0 and len(signals) > max_signals:
            # Calculate how many large-cap and mid-cap signals to include
            large_cap_count = int(max_signals * (large_cap_percentage / 100))
            mid_cap_count = max_signals - large_cap_count
            
            # Ensure we don't exceed available signals
            large_cap_count = min(large_cap_count, len(largecap_signals))
            mid_cap_count = min(mid_cap_count, len(midcap_signals))
            
            # If we don't have enough of one type, allocate more to the other
            if large_cap_count < int(max_signals * (large_cap_percentage / 100)):
                additional_mid_cap = min(mid_cap_count + (int(max_signals * (large_cap_percentage / 100)) - large_cap_count), len(midcap_signals))
                mid_cap_count = additional_mid_cap
            
            if mid_cap_count < (max_signals - int(max_signals * (large_cap_percentage / 100))):
                additional_large_cap = min(large_cap_count + ((max_signals - int(max_signals * (large_cap_percentage / 100))) - mid_cap_count), len(largecap_signals))
                large_cap_count = additional_large_cap
            
            # Get the top N signals of each type
            # Sort signals deterministically by score and then by symbol (for tiebreaking)
            largecap_signals = sorted(largecap_signals, key=lambda x: (x['score'], x['symbol']), reverse=True)
            midcap_signals = sorted(midcap_signals, key=lambda x: (x['score'], x['symbol']), reverse=True)
            
            selected_large_cap = largecap_signals[:large_cap_count]
            selected_mid_cap = midcap_signals[:mid_cap_count]
            
            # Combine and re-sort by score and symbol (for deterministic ordering)
            signals = selected_large_cap + selected_mid_cap
            signals = sorted(signals, key=lambda x: (x['score'], x['symbol']), reverse=True)
            
            logger.info(f"Final signals: {len(signals)} total ({len(selected_large_cap)} large-cap, {len(selected_mid_cap)} mid-cap)")
        else:
            # If no max_signals specified or we have fewer signals than max, still log the signal count
            # Sort signals deterministically by score and then by symbol (for tiebreaking)
            signals = sorted(signals, key=lambda x: (x['score'], x['symbol']), reverse=True)
            logger.info(f"Using all {len(signals)} signals ({len(largecap_signals)} large-cap, {len(midcap_signals)} mid-cap)")
        
        # Simulate trade outcomes for performance metrics
        if signals:
            # Create a list to store simulated trades
            simulated_trades = []
            
            # Set random seed for reproducibility
            np.random.seed(random_seed)
            
            # Define win rates based on historical performance and market regime
            base_long_win_rate = 0.62
            
            # Define win rate adjustments based on market regime
            market_regime_adjustments = {
                'STRONG_BULLISH': {'LONG': 0.15},
                'BULLISH': {'LONG': 0.10},
                'NEUTRAL': {'LONG': 0.00},
                'BEARISH': {'LONG': -0.10},
                'STRONG_BEARISH': {'LONG': -0.20}
            }
            
            # Define average gains and losses
            avg_long_win = 0.05
            avg_long_loss = -0.02
            
            # Define average holding periods
            avg_holding_period_win = 12
            avg_holding_period_loss = 5
            
            # Parse the start date
            start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
            
            # Get current market regime
            market_regime = strategy.detect_market_regime()
            
            # Track remaining capital
            remaining_capital = initial_capital
            # Track final capital for continuous capital mode
            final_capital = remaining_capital
            
            for signal in signals:
                # Calculate position size based on signal score and remaining capital
                # Use a percentage of remaining capital for each trade
                
                # Base position size as percentage of remaining capital
                base_position_pct = config.get('strategy', {}).get('position_sizing', {}).get('base_position_pct', 5)
                base_position_size = (base_position_pct / 100) * remaining_capital
                
                if signal['score'] >= 0.9:  # Tier 1
                    position_size = base_position_size * 3.0
                    tier = "Tier 1 (≥0.9)"
                elif signal['score'] >= 0.8:  # Tier 2
                    position_size = base_position_size * 1.5
                    tier = "Tier 2 (0.8-0.9)"
                else:  # Skip Tier 3 and Tier 4 trades
                    logger.info(f"Skipping trade for {signal['symbol']} with score {signal['score']:.2f} - below Tier 2 threshold")
                    continue
                
                # Adjust for mid-cap stocks
                if signal.get('is_midcap', False):
                    midcap_factor = config.get('strategy', {}).get('midcap_stocks', {}).get('position_factor', 0.8)
                    position_size *= midcap_factor
                
                # Ensure position size doesn't exceed remaining capital
                position_size = min(position_size, remaining_capital * 0.95)
                
                # Skip if position size is too small
                if position_size < 100:
                    logger.info(f"Skipping trade for {signal['symbol']} - position size too small (${position_size:.2f})")
                    continue
                
                # Determine if the trade is a win or loss based on win rate
                # Adjust win rate based on market regime
                adjusted_win_rate = base_long_win_rate
                if market_regime in market_regime_adjustments:
                    adjusted_win_rate += market_regime_adjustments[market_regime]['LONG']
                
                # Clamp win rate between 0.3 and 0.9
                adjusted_win_rate = max(0.3, min(0.9, adjusted_win_rate))
                
                # Determine if this trade is a win
                is_win = np.random.random() < adjusted_win_rate
                
                # Calculate return
                if is_win:
                    # Winning trade - use average win percentage with some randomness
                    return_pct = avg_long_win * (1 + 0.5 * np.random.random())
                    holding_period = int(avg_holding_period_win * (0.8 + 0.4 * np.random.random()))
                else:
                    # Losing trade - use average loss percentage with some randomness
                    return_pct = avg_long_loss * (1 + 0.5 * np.random.random())
                    holding_period = int(avg_holding_period_loss * (0.8 + 0.4 * np.random.random()))
                
                # Calculate profit/loss
                pnl = position_size * return_pct
                
                # Update remaining capital
                remaining_capital += pnl
                
                # Calculate entry and exit dates
                entry_date = datetime.strptime(signal['date'], "%Y-%m-%d")
                exit_date = entry_date + timedelta(days=holding_period)
                
                # Ensure exit date is not beyond the backtest end date
                exit_date = min(exit_date, datetime.strptime(end_date, "%Y-%m-%d"))
                
                # Create trade record
                trade = {
                    'symbol': signal['symbol'],
                    'direction': 'LONG',
                    'entry_date': entry_date.strftime("%Y-%m-%d"),
                    'exit_date': exit_date.strftime("%Y-%m-%d"),
                    'holding_period': (exit_date - entry_date).days,
                    'position_size': position_size,
                    'entry_price': signal['price'],
                    'exit_price': signal['price'] * (1 + return_pct),
                    'return_pct': return_pct * 100,
                    'pnl': pnl,
                    'is_win': is_win,
                    'score': signal['score'],
                    'tier': tier,
                    'sector': signal.get('sector', 'Unknown'),
                    'is_midcap': signal.get('is_midcap', False),
                    'market_regime': market_regime
                }
                
                # Add to simulated trades
                simulated_trades.append(trade)
            
            # Calculate performance metrics
            metrics = {}
            
            if simulated_trades:
                # Calculate win rate
                wins = [t for t in simulated_trades if t['is_win']]
                losses = [t for t in simulated_trades if not t['is_win']]
                
                long_win_rate = (len(wins) / len(simulated_trades)) * 100 if simulated_trades else 0
                
                # Calculate average return
                avg_return = sum(t['return_pct'] for t in simulated_trades) / len(simulated_trades) if simulated_trades else 0
                
                # Calculate average holding period
                avg_holding_period = sum(t['holding_period'] for t in simulated_trades) / len(simulated_trades) if simulated_trades else 0
                
                # Calculate total return
                total_return = ((remaining_capital - initial_capital) / initial_capital) * 100
                
                # Calculate max drawdown
                cumulative_returns = []
                current_capital = initial_capital
                for trade in simulated_trades:
                    current_capital += trade['pnl']
                    cumulative_returns.append(current_capital)
                
                if cumulative_returns:
                    peak = initial_capital
                    max_drawdown = 0
                    
                    for capital in cumulative_returns:
                        if capital > peak:
                            peak = capital
                        
                        drawdown = (peak - capital) / peak * 100
                        max_drawdown = max(max_drawdown, drawdown)
                else:
                    max_drawdown = 0
                
                # Store metrics
                metrics = {
                    'initial_capital': initial_capital,
                    'final_capital': remaining_capital,
                    'total_return': total_return,
                    'total_trades': len(simulated_trades),
                    'winning_trades': len(wins),
                    'losing_trades': len(losses),
                    'long_win_rate': long_win_rate,
                    'avg_return': avg_return,
                    'avg_holding_period': avg_holding_period,
                    'max_drawdown': max_drawdown
                }
                
                # Create summary object
                summary = {
                    'start_date': start_date,
                    'end_date': end_date,
                    'total_trades': len(simulated_trades),
                    'winning_trades': len(wins),
                    'losing_trades': len(losses),
                    'long_win_rate': long_win_rate,
                    'avg_return': avg_return,
                    'avg_holding_period': avg_holding_period,
                    'total_return': total_return,
                    'max_drawdown': max_drawdown,
                    'initial_capital': initial_capital,
                    'final_capital': remaining_capital,
                    'continuous_capital': continuous_capital,
                    'market_regime': market_regime
                }
                
                # Log metrics
                logger.info(f"Backtest Results ({start_date} to {end_date}):")
                logger.info(f"Total Trades: {len(simulated_trades)}")
                logger.info(f"Winning Trades: {len(wins)}")
                logger.info(f"Losing Trades: {len(losses)}")
                logger.info(f"LONG Win Rate: {long_win_rate:.2f}%")
                logger.info(f"Average Return: {avg_return:.2f}%")
                logger.info(f"Average Holding Period: {avg_holding_period:.1f} days")
                logger.info(f"Initial Capital: ${initial_capital:.2f}")
                logger.info(f"Final Capital: ${metrics['final_capital']:.2f}")
                logger.info(f"Total Return: {metrics['total_return']:.2f}%")
            else:
                # No trades executed
                logger.warning("No trades were executed in the backtest")
                
                # Create empty summary
                summary = {
                    'start_date': start_date,
                    'end_date': end_date,
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'long_win_rate': 0,
                    'avg_return': 0,
                    'avg_holding_period': 0,
                    'total_return': 0,
                    'max_drawdown': 0,
                    'initial_capital': initial_capital,
                    'final_capital': initial_capital,
                    'continuous_capital': continuous_capital,
                    'market_regime': market_regime
                }
                
                # Create empty metrics
                metrics = {
                    'initial_capital': initial_capital,
                    'final_capital': initial_capital,
                    'total_return': 0,
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'long_win_rate': 0,
                    'avg_return': 0,
                    'avg_holding_period': 0,
                    'max_drawdown': 0
                }
        else:
            # No signals generated
            logger.warning("No signals were generated for the backtest period")
            
            # Create empty summary
            summary = {
                'start_date': start_date,
                'end_date': end_date,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'long_win_rate': 0,
                'avg_return': 0,
                'avg_holding_period': 0,
                'total_return': 0,
                'max_drawdown': 0,
                'initial_capital': initial_capital,
                'final_capital': initial_capital,
                'continuous_capital': continuous_capital,
                'market_regime': strategy.detect_market_regime()
            }
            
            # Create empty metrics
            metrics = {
                'initial_capital': initial_capital,
                'final_capital': initial_capital,
                'total_return': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'long_win_rate': 0,
                'avg_return': 0,
                'avg_holding_period': 0,
                'max_drawdown': 0
            }
            
            # Empty signals list
            signals = []
        
        # Log execution time
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"[DEBUG] Returning {len(signals) if signals else 0} signals")
        if signals and len(signals) > 0:
            logger.info(f"[DEBUG] First few signals: {signals[:3]}")
        
        return summary, signals
    
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        traceback.print_exc()
        return None, None
"""
            
            # Replace the original function in the content
            fixed_content = content[:start_match.start()] + new_function + content[end_position:]
            
            # Write the updated content back to the file
            with open('final_sp500_strategy.py', 'w') as f:
                f.write(fixed_content)
            
            print("Successfully rewrote run_backtest function with clean implementation")
        else:
            print("Could not find run_backtest function")
        
    except Exception as e:
        print(f"Error rewriting function: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    print("Creating backup of final_sp500_strategy.py...")
    create_backup()
    
    print("\nCompletely rewriting run_backtest function with clean implementation...")
    completely_rewrite_function()
    
    print("\nAll fixes applied. Please restart the web server to apply the changes.")
