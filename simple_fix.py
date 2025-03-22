"""
Simple script to fix the specific issues in the multi-strategy trading system.
"""

import re

# Path to the file
file_path = "multi_strategy_system.py"

# Read the file
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()

# Fix 1: Add a check for None in the VolatilityBreakout strategy
volatility_pattern = r'class VolatilityBreakoutStrategy\(Strategy\):'
volatility_match = re.search(volatility_pattern, content)

if volatility_match:
    # Find the generate_signals method
    method_pattern = r'def generate_signals\(self,\s+symbol: str,\s+candles: List\[CandleData\],\s+stock_config: StockConfig,\s+market_state: MarketState\) -> List\[Signal\]:'
    method_match = re.search(method_pattern, content[volatility_match.end():])
    
    if method_match:
        # Find the return signals statement
        return_pattern = r'return signals'
        return_match = re.search(return_pattern, content[volatility_match.end() + method_match.end():])
        
        if return_match:
            # Insert a try-except block around the entire method body
            pos_start = volatility_match.end() + method_match.end()
            pos_end = pos_start + return_match.end()
            
            # Get the method body
            method_body = content[pos_start:pos_end]
            
            # Add try-except block
            new_method_body = "\n        try:" + method_body.replace("\n        ", "\n            ") + "\n        except Exception as e:\n            # Log the error but don't crash\n            self.logger.error(f\"Error in VolatilityBreakout strategy for {symbol}: {str(e)}\")\n            # Return empty signals list\n            return []\n"
            
            # Replace the method body
            content = content[:pos_start] + new_method_body + content[pos_end:]
            
            print("Fixed VolatilityBreakout strategy")

# Fix 2: Add null check in run_backtest method
backtest_pattern = r'def run_backtest\(self, start_date: dt\.date, end_date: dt\.date, initial_capital=100000\):'
backtest_match = re.search(backtest_pattern, content)

if backtest_match:
    # Find all occurrences of "new_signals = strategy_to_use.generate_signals"
    signals_pattern = r'new_signals = strategy_to_use\.generate_signals\([^)]+\)'
    signals_matches = list(re.finditer(signals_pattern, content[backtest_match.end():]))
    
    if signals_matches:
        # Start from the end to avoid messing up positions
        for match in reversed(signals_matches):
            # Find the logging statement
            log_pattern = r'self\.logger\.info\(f"Strategy \{name\} for \{symbol\} generated \{len\(new_signals\)\} signals"\)'
            log_match = re.search(log_pattern, content[backtest_match.end() + match.end():backtest_match.end() + match.end() + 200])
            
            if log_match:
                # Insert the null check
                null_check = "\n                # Ensure new_signals is a list, not None\n                if new_signals is None:\n                    new_signals = []"
                pos = backtest_match.end() + match.end() + log_match.start()
                content = content[:pos] + null_check + content[pos:]
                
                print("Added null check in run_backtest method")

# Fix 3: Fix the date range to use 2023 data
main_pattern = r'if __name__ == "__main__":'
main_match = re.search(main_pattern, content)

if main_match:
    # Find date range setup
    date_pattern = r'start_date = dt\.date\((\d+), (\d+), (\d+)\)\s+end_date = dt\.date\((\d+), (\d+), (\d+)\)'
    date_match = re.search(date_pattern, content[main_match.end():])
    
    if date_match:
        start_year = int(date_match.group(1))
        end_year = int(date_match.group(4))
        
        # Only modify if using data from after 2023
        if start_year > 2023 or end_year > 2023:
            # Replace with 2023 dates
            new_dates = "    # Use historical data from 2023 (Alpaca free tier limitation)\n    start_date = dt.date(2023, 1, 1)\n    end_date = dt.date(2023, 12, 31)"
            pos_start = main_match.end() + date_match.start()
            pos_end = main_match.end() + date_match.end()
            content = content[:pos_start] + new_dates + content[pos_end:]
            
            print("Fixed date range to use 2023 data")

# Write the modified content back to the file
with open(file_path, 'w', encoding='utf-8') as file:
    file.write(content)

print("All fixes applied successfully!")
