"""
Script to fix all variable name inconsistencies in final_sp500_strategy.py
This script will ensure consistent naming between midcap_signals and largecap_signals
"""

# Read the file
with open('final_sp500_strategy.py', 'r') as f:
    content = f.read()

# Fix largecap_symbols to largecap_signals
content = content.replace('len(largecap_symbols)', 'len(largecap_signals)')

# Fix midcap_symbols to midcap_signals in the signal handling section
content = content.replace('mid_cap_count = min(mid_cap_count, len(midcap_symbols))', 
                         'mid_cap_count = min(mid_cap_count, len(midcap_signals))')
content = content.replace('additional_mid_cap = min(mid_cap_count + (int(max_signals * (large_cap_percentage / 100)) - large_cap_count), len(midcap_symbols))', 
                         'additional_mid_cap = min(mid_cap_count + (int(max_signals * (large_cap_percentage / 100)) - large_cap_count), len(midcap_signals))')
content = content.replace('midcap_symbols = sorted(midcap_symbols', 
                         'midcap_signals = sorted(midcap_signals')
content = content.replace('selected_mid_cap = midcap_symbols', 
                         'selected_mid_cap = midcap_signals')
content = content.replace('len(midcap_symbols)} mid-cap)', 
                         'len(midcap_signals)} mid-cap)')

# Write the modified content back to the file
with open('final_sp500_strategy.py', 'w') as f:
    f.write(content)

print("All variable name inconsistencies fixed successfully!")
