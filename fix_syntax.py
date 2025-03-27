with open('final_sp500_strategy.py', 'r') as f:
    content = f.read()

# Remove everything after the first triple backtick
if '```' in content:
    content = content.split('```')[0]

with open('final_sp500_strategy.py', 'w') as f:
    f.write(content)

print("File fixed successfully!")
