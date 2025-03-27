def fix_indentation_issues():
    """Fix indentation issues in final_sp500_strategy.py"""
    file_path = 'final_sp500_strategy.py'
    
    # Read the file line by line
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Find and fix indentation issues
    fixed_count = 0
    i = 0
    while i < len(lines):
        # Check for if statements followed by improperly indented blocks
        if lines[i].strip().startswith('if ') and lines[i].strip().endswith(':'):
            # Check if the next line is properly indented
            if i + 1 < len(lines) and not lines[i + 1].startswith('            '):
                # Fix the indentation of the next line and potentially following lines
                j = i + 1
                while j < len(lines) and not lines[j].strip().startswith(('if ', 'else:', 'elif ')):
                    if lines[j].strip() and not lines[j].startswith('            '):
                        lines[j] = '            ' + lines[j].lstrip()
                        fixed_count += 1
                        print(f"Fixed indentation on line {j+1}")
                    j += 1
        i += 1
    
    # Write the fixed content back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)
    
    print(f"Fixed {fixed_count} indentation issues in final_sp500_strategy.py")

if __name__ == "__main__":
    fix_indentation_issues()
