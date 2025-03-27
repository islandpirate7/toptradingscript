import re

def fix_all_issues():
    """
    Fix all indentation and f-string issues in final_sp500_strategy.py
    """
    file_path = 'final_sp500_strategy.py'
    
    # Read the file content
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Split into lines for processing
    lines = content.split('\n')
    
    # Fix indentation issues
    indentation_fixed = 0
    i = 0
    while i < len(lines):
        # Check for if statements followed by improperly indented blocks
        if re.match(r'^\s*if\s+.*:$', lines[i]):
            # Check if the next line is properly indented
            if i + 1 < len(lines) and lines[i+1].strip() and not re.match(r'^\s{12}', lines[i+1]):
                # Get the base indentation of the if statement
                base_indent = len(lines[i]) - len(lines[i].lstrip())
                target_indent = ' ' * (base_indent + 4)
                
                # Fix the indentation of the next line and potentially following lines
                j = i + 1
                while j < len(lines) and lines[j].strip() and not re.match(r'^\s*(?:if|else:|elif)', lines[j]):
                    if not re.match(r'^\s{12}', lines[j]):
                        lines[j] = target_indent + lines[j].lstrip()
                        indentation_fixed += 1
                        print(f"Fixed indentation on line {j+1}")
                    j += 1
        i += 1
    
    # Fix f-string issues
    fstring_fixed = 0
    for i in range(len(lines)):
        # Look for problematic f-string patterns
        if 'log_file_handle.write(f"{datetime.now()} - INFO - " + str(f"' in lines[i]:
            # Extract the message content
            try:
                message_start = lines[i].index('str(f"') + 6
                message_content = lines[i][message_start:].strip()
                
                # Remove any problematic parts
                message_content = message_content.replace(' + "\\n")', '')
                message_content = message_content.replace(' + "\\n"', '')
                
                # Create a fixed version
                indent = re.match(r'^\s*', lines[i]).group(0)
                fixed_line = f'{indent}log_file_handle.write(f"{{datetime.now()}} - INFO - {message_content}\\n")'
                lines[i] = fixed_line
                
                fstring_fixed += 1
                print(f"Fixed f-string on line {i+1}")
            except Exception as e:
                print(f"Error fixing f-string on line {i+1}: {str(e)}")
    
    # Join the lines back into content
    fixed_content = '\n'.join(lines)
    
    # Write the fixed content back to the file
    with open(file_path, 'w') as file:
        file.write(fixed_content)
    
    print(f"Fixed {indentation_fixed} indentation issues and {fstring_fixed} f-string issues in {file_path}")
    print("Please run a backtest to verify that the fixes are working correctly.")

if __name__ == "__main__":
    fix_all_issues()
