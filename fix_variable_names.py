"""
Script to fix variable name inconsistencies in final_sp500_strategy.py
This script will replace all instances of midcap_symbols with midcap_signals in the signal handling section
"""

import re

# Read the file
with open('final_sp500_strategy.py', 'r') as f:
    content = f.read()

# Define the line ranges for the signal handling section (run_backtest function)
start_line = 3290
end_line = 3345

# Split the content into lines
lines = content.split('\n')

# Process the lines in the signal handling section
for i in range(start_line, min(end_line + 1, len(lines))):
    # Replace midcap_symbols with midcap_signals in the signal handling section
    if 'midcap_symbols' in lines[i] and not ('get_midcap_symbols' in lines[i] or 'Added' in lines[i] or 'universe' in lines[i]):
        lines[i] = lines[i].replace('midcap_symbols', 'midcap_signals')

# Join the lines back into content
modified_content = '\n'.join(lines)

# Write the modified content back to the file
with open('final_sp500_strategy.py', 'w') as f:
    f.write(modified_content)

print("Variable name inconsistencies fixed successfully!")
