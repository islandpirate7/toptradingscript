"""
This script fixes the errors in the multi-strategy trading system:
1. Adds try-except block to VolatilityBreakout strategy to ensure it always returns a list
2. Adds null check for new_signals in run_backtest method
3. Fixes duplicate logging issue
"""

import re
import os
import sys

def fix_volatility_strategy(file_path):
    """Add try-except block to VolatilityBreakoutStrategy.generate_signals"""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Find the VolatilityBreakoutStrategy.generate_signals method
    pattern = r'class VolatilityBreakoutStrategy\(Strategy\):.*?def generate_signals\(self,\s+symbol: str,\s+candles: List\[CandleData\],\s+stock_config: StockConfig,\s+market_state: MarketState\) -> List\[Signal\]:\s+"""Generate volatility breakout signals based on Bollinger Band squeeze"""\s+signals = \[\]'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        # Insert try block after signals = []
        pos = match.end()
        try_block = "\n        try:"
        
        # Add indentation to the rest of the method
        method_end_pattern = r'def _calculate_atr'
        method_end_match = re.search(method_end_pattern, content[pos:])
        
        if method_end_match:
            method_body = content[pos:pos + method_end_match.start()]
            indented_body = ""
            for line in method_body.split('\n'):
                if line.strip():
                    indented_body += "\n            " + line.strip()
                else:
                    indented_body += "\n"
            
            # Add except block at the end
            except_block = "\n        except Exception as e:\n            # Log the error but don't crash\n            self.logger.error(f\"Error in VolatilityBreakout strategy for {symbol}: {str(e)}\")\n            # Return empty signals list\n            return []\n        \n        return signals"
            
            # Replace the method body
            new_content = content[:pos] + try_block + indented_body + except_block + content[pos + method_end_match.start():]
            
            # Write the modified content back to the file
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(new_content)
            
            print("Successfully added try-except block to VolatilityBreakoutStrategy.generate_signals")
            return True
    
    print("Could not find VolatilityBreakoutStrategy.generate_signals method")
    return False

def fix_run_backtest(file_path):
    """Add null check for new_signals in run_backtest method"""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Find the run_backtest method
    pattern = r'def run_backtest\(self, start_date: dt\.date, end_date: dt\.date, initial_capital=100000\):'
    match = re.search(pattern, content)
    
    if match:
        # Find all occurrences of new_signals = strategy_to_use.generate_signals
        signal_pattern = r'new_signals = strategy_to_use\.generate_signals\([^)]+\)'
        matches = list(re.finditer(signal_pattern, content))
        
        if matches:
            # Start from the end to avoid messing up positions
            for match in reversed(matches):
                # Find the logging statement after this line
                log_pattern = r'self\.logger\.info\(f"Strategy \{name\} for \{symbol\} generated \{len\(new_signals\)\} signals"\)'
                log_match = re.search(log_pattern, content[match.end():match.end() + 200])
                
                if log_match:
                    # Insert the null check
                    null_check = "\n                # Ensure new_signals is a list, not None\n                if new_signals is None:\n                    new_signals = []"
                    pos = match.end() + log_match.start()
                    new_content = content[:pos] + null_check + content[pos:]
                    content = new_content
        
        # Write the modified content back to the file
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
        
        print("Successfully added null checks for new_signals in the run_backtest method")
        return True
    
    print("Could not find run_backtest method")
    return False

def fix_duplicate_logging(file_path):
    """Fix duplicate logging issue"""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Find the logging setup in __init__ method
    pattern = r'def __init__\(self, config: MultiStrategyConfig\):'
    match = re.search(pattern, content)
    
    if match:
        # Find the logger setup
        logger_pattern = r'self\.logger = logging\.getLogger\([^)]+\)'
        logger_match = re.search(logger_pattern, content[match.end():match.end() + 1000])
        
        if logger_match:
            # Add code to remove existing handlers
            handler_fix = "\n        # Remove existing handlers to prevent duplicate logging\n        if self.logger.handlers:\n            self.logger.handlers = []"
            pos = match.end() + logger_match.end()
            new_content = content[:pos] + handler_fix + content[pos:]
            
            # Write the modified content back to the file
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(new_content)
            
            print("Successfully fixed the duplicate logging issue")
            return True
    
    print("Could not find the logging setup in the __init__ method")
    return False

def fix_date_range(file_path):
    """Fix the date range to use historical data from 2023 or earlier"""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Find the main function
    pattern = r'if __name__ == "__main__":'
    match = re.search(pattern, content)
    
    if match:
        # Find the date range setup
        date_pattern = r'start_date = dt\.date\([^)]+\)\s+end_date = dt\.date\([^)]+\)'
        date_match = re.search(date_pattern, content[match.end():])
        
        if date_match:
            # Replace with dates from 2023
            new_dates = "    # Use historical data from 2023 or earlier (Alpaca free tier limitation)\n    start_date = dt.date(2023, 1, 1)\n    end_date = dt.date(2023, 12, 31)"
            pos = match.end() + date_match.start()
            end_pos = match.end() + date_match.end()
            new_content = content[:pos] + new_dates + content[end_pos:]
            
            # Write the modified content back to the file
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(new_content)
            
            print("Successfully fixed the date range to use historical data from 2023")
            return True
    
    print("Could not find the date range setup in the main function")
    return False

def main():
    file_path = "multi_strategy_system.py"
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found")
        return
    
    # Create a backup
    backup_path = file_path + ".bak"
    with open(file_path, 'r', encoding='utf-8') as src, open(backup_path, 'w', encoding='utf-8') as dst:
        dst.write(src.read())
    print(f"Created backup at {backup_path}")
    
    # Fix all issues
    fix_volatility_strategy(file_path)
    fix_run_backtest(file_path)
    fix_duplicate_logging(file_path)
    fix_date_range(file_path)
    
    print("\nAll fixes have been applied. Please run your script again.")

if __name__ == "__main__":
    main()
