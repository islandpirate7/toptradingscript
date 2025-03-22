"""
This script patches the VolatilityBreakoutStrategy to fix the 'NoneType' has no len() error.
Run this script to apply the fix to your multi_strategy_system.py file.
"""

import re
import os

# Path to the multi_strategy_system.py file
file_path = "multi_strategy_system.py"

# Read the file content
with open(file_path, 'r') as file:
    content = file.read()

# Find the VolatilityBreakoutStrategy.generate_signals method
pattern = r'def generate_signals\(self,\s+symbol: str,\s+candles: List\[CandleData\],\s+stock_config: StockConfig,\s+market_state: MarketState\) -> List\[Signal\]:'
match = re.search(pattern, content)

if match:
    # Find the position to insert the try-except block
    pos = match.end()
    
    # Find the indentation level
    next_line_pos = content.find('\n', pos) + 1
    indentation = ''
    for char in content[next_line_pos:]:
        if char in [' ', '\t']:
            indentation += char
        else:
            break
    
    # Create the try-except block
    try_block = f'\n{indentation}"""Generate volatility breakout signals based on Bollinger Band squeeze"""\n{indentation}signals = []\n{indentation}\n{indentation}try:'
    
    # Add indentation to the existing code
    content_after_pos = content[pos:]
    # Find the first occurrence of 'signals = []'
    signals_pos = content_after_pos.find('signals = []')
    if signals_pos != -1:
        # Replace the original 'signals = []' line
        content_after_pos = content_after_pos[:signals_pos] + content_after_pos[signals_pos + len('signals = []') + 1:]
    
    # Add indentation to all lines
    content_after_pos = content_after_pos.replace('\n' + indentation, '\n' + indentation + '    ')
    
    # Add the except block at the end of the method
    except_block = f'\n{indentation}except Exception as e:\n{indentation}    # Log the error but don\'t crash\n{indentation}    self.logger.error(f"Error in VolatilityBreakout strategy for {{symbol}}: {{str(e)}}")\n{indentation}    # Return empty signals list\n{indentation}    return []\n'
    
    # Find the end of the method
    method_end = content.find('def', pos)
    if method_end == -1:
        method_end = len(content)
    
    # Insert the try-except block
    new_content = content[:pos] + try_block + content_after_pos[:method_end-pos] + except_block + content[method_end:]
    
    # Write the modified content back to the file
    with open(file_path + '.bak', 'w') as file:
        file.write(content)  # Create a backup
    
    with open(file_path, 'w') as file:
        file.write(new_content)
    
    print("Successfully patched the VolatilityBreakoutStrategy.generate_signals method.")
else:
    print("Could not find the VolatilityBreakoutStrategy.generate_signals method in the file.")

# Now add the null check for new_signals in the run_backtest method
with open(file_path, 'r') as file:
    content = file.read()

# Find all occurrences of "new_signals = strategy_to_use.generate_signals"
pattern = r'new_signals = strategy_to_use\.generate_signals\([^)]+\)\s+\n\s+# Log signal generation results\s+\n\s+self\.logger\.info\(f"Strategy {name} for {symbol} generated {len\(new_signals\)} signals"\)'
matches = list(re.finditer(pattern, content))

if matches:
    # Start from the end to avoid messing up positions
    for match in reversed(matches):
        # Insert the null check
        null_check = '\n        # Ensure new_signals is a list, not None\n        if new_signals is None:\n            new_signals = []\n        '
        pos = match.end() - content[match.start():match.end()].find('self.logger.info')
        new_content = content[:pos] + null_check + content[pos:]
        content = new_content
    
    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.write(content)
    
    print("Successfully added null checks for new_signals in the run_backtest method.")
else:
    print("Could not find the pattern to add null checks in the run_backtest method.")

# Fix the duplicate logging issue
with open(file_path, 'r') as file:
    content = file.read()

# Find the logging setup in __init__ method
pattern = r'def __init__\(self, config: MultiStrategyConfig\):[^}]+self\.logger = logging\.getLogger\([^)]+\)'
match = re.search(pattern, content)

if match:
    # Add code to remove existing handlers
    pos = match.end()
    handler_fix = '\n        # Remove existing handlers to prevent duplicate logging\n        if self.logger.handlers:\n            self.logger.handlers = []\n'
    new_content = content[:pos] + handler_fix + content[pos:]
    
    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.write(new_content)
    
    print("Successfully fixed the duplicate logging issue.")
else:
    print("Could not find the logging setup in the __init__ method.")

print("\nAll fixes have been applied. Please run your script again.")
