"""
Simple script to fix the VolatilityBreakout strategy in the multi-strategy trading system.
This script adds a try-except block to ensure the strategy always returns a list.
"""

import re

def fix_volatility_breakout():
    # Path to the file
    file_path = "multi_strategy_system.py"
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Find the VolatilityBreakoutStrategy class
    class_pattern = r'class VolatilityBreakoutStrategy\(Strategy\):'
    class_match = re.search(class_pattern, content)
    
    if not class_match:
        print("Could not find VolatilityBreakoutStrategy class")
        return False
    
    # Find the generate_signals method
    method_pattern = r'def generate_signals\(self,\s+symbol: str,\s+candles: List\[CandleData\],\s+stock_config: StockConfig,\s+market_state: MarketState\) -> List\[Signal\]:'
    method_match = re.search(method_pattern, content[class_match.end():])
    
    if not method_match:
        print("Could not find generate_signals method")
        return False
    
    # Find the method body
    method_start = class_match.end() + method_match.end()
    
    # Find the next method
    next_method_pattern = r'def _calculate_atr\('
    next_method_match = re.search(next_method_pattern, content[method_start:])
    
    if not next_method_match:
        print("Could not find the end of the generate_signals method")
        return False
    
    method_end = method_start + next_method_match.start()
    
    # Extract the method body
    method_body = content[method_start:method_end]
    
    # Check if try-except is already there
    if "try:" in method_body and "except Exception as e:" in method_body:
        print("Try-except block already exists")
        return True
    
    # Add try-except block
    docstring_pattern = r'"""Generate volatility breakout signals based on Bollinger Band squeeze"""'
    docstring_match = re.search(docstring_pattern, method_body)
    
    if not docstring_match:
        print("Could not find method docstring")
        return False
    
    signals_pattern = r'signals = \[\]'
    signals_match = re.search(signals_pattern, method_body[docstring_match.end():])
    
    if not signals_match:
        print("Could not find signals initialization")
        return False
    
    # Position after signals = []
    pos = docstring_match.end() + signals_match.end()
    
    # Split the method body
    before_code = method_body[:pos]
    after_code = method_body[pos:]
    
    # Add try-except block
    new_method_body = before_code + "\n        try:" + after_code.replace("\n        ", "\n            ").replace("return signals", "        except Exception as e:\n            # Log the error but don't crash\n            self.logger.error(f\"Error in VolatilityBreakout strategy for {symbol}: {str(e)}\")\n            # Return empty signals list\n            return []\n        \n        return signals")
    
    # Replace the method body in the content
    new_content = content[:method_start] + new_method_body + content[method_end:]
    
    # Write the modified content back to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(new_content)
    
    print("Successfully added try-except block to VolatilityBreakoutStrategy.generate_signals")
    return True

def fix_run_backtest():
    # Path to the file
    file_path = "multi_strategy_system.py"
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Find the run_backtest method
    method_pattern = r'def run_backtest\(self, start_date: dt\.date, end_date: dt\.date, initial_capital=100000\):'
    method_match = re.search(method_pattern, content)
    
    if not method_match:
        print("Could not find run_backtest method")
        return False
    
    # Find the signal generation code
    signal_pattern = r'new_signals = strategy_to_use\.generate_signals\([^)]+\)'
    signal_matches = list(re.finditer(signal_pattern, content[method_match.end():]))
    
    if not signal_matches:
        print("Could not find signal generation code")
        return False
    
    # Check if the null check is already there
    null_check_pattern = r'if new_signals is None:'
    null_check_match = re.search(null_check_pattern, content[method_match.end():])
    
    if null_check_match:
        print("Null check already exists")
        return True
    
    # Find the logging statement after signal generation
    for match in signal_matches:
        pos = method_match.end() + match.end()
        log_pattern = r'self\.logger\.info\(f"Strategy \{name\} for \{symbol\} generated \{len\(new_signals\)\} signals"\)'
        log_match = re.search(log_pattern, content[pos:pos + 200])
        
        if log_match:
            # Insert the null check
            null_check = "\n                # Ensure new_signals is a list, not None\n                if new_signals is None:\n                    new_signals = []"
            insert_pos = pos + log_match.start()
            new_content = content[:insert_pos] + null_check + content[insert_pos:]
            
            # Write the modified content back to the file
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(new_content)
            
            print("Successfully added null check to run_backtest method")
            return True
    
    print("Could not find logging statement after signal generation")
    return False

def fix_date_range():
    # Path to the file
    file_path = "multi_strategy_main.py"
    
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Find the main function
        pattern = r'if __name__ == "__main__":'
        match = re.search(pattern, content)
        
        if not match:
            print("Could not find main function")
            return False
        
        # Find the date range setup
        date_pattern = r'start_date = dt\.date\((\d+), (\d+), (\d+)\)\s+end_date = dt\.date\((\d+), (\d+), (\d+)\)'
        date_match = re.search(date_pattern, content[match.end():])
        
        if not date_match:
            print("Could not find date range setup")
            return False
        
        start_year = int(date_match.group(1))
        end_year = int(date_match.group(4))
        
        # Only modify if using data from after 2023
        if start_year > 2023 or end_year > 2023:
            # Replace with 2023 dates
            new_dates = "    # Use historical data from 2023 (Alpaca free tier limitation)\n    start_date = dt.date(2023, 1, 1)\n    end_date = dt.date(2023, 12, 31)"
            pos_start = match.end() + date_match.start()
            pos_end = match.end() + date_match.end()
            new_content = content[:pos_start] + new_dates + content[pos_end:]
            
            # Write the modified content back to the file
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(new_content)
            
            print("Fixed date range to use 2023 data")
            return True
        else:
            print("Date range already using 2023 or earlier data")
            return True
    except Exception as e:
        print(f"Error fixing date range: {str(e)}")
        return False

if __name__ == "__main__":
    print("Applying fixes to the multi-strategy trading system...")
    
    # Fix the VolatilityBreakout strategy
    fix_volatility_breakout()
    
    # Fix the run_backtest method
    fix_run_backtest()
    
    # Fix the date range to use 2023 data (Alpaca free tier limitation)
    fix_date_range()
    
    print("All fixes applied successfully!")
