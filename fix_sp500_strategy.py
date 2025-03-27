import os
import re

def fix_run_backtest_function():
    """Fix the duplicate return statement in the run_backtest function and add a fallback summary."""
    file_path = 'final_sp500_strategy.py'
    
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Find the exception handler in run_backtest function
    pattern = r'except Exception as e:\s+logger\.error\(f"Error running backtest: {str\(e\)}"\)\s+traceback\.print_exc\(\)\s+return None, \[\]\s+return None, \[\]'
    
    # Replace with fixed code that includes a fallback summary
    replacement = '''except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        traceback.print_exc()
        
        # Create a fallback summary with minimal information
        fallback_summary = {
            'start_date': start_date,
            'end_date': end_date,
            'total_signals': 0,
            'long_signals': 0,
            'avg_score': 0,
            'avg_long_score': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'avg_holding_period': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'initial_capital': initial_capital,
            'final_capital': initial_capital,
            'total_return': 0,
            'error': str(e)
        }
        
        return fallback_summary, []'''
    
    # Apply the fix
    fixed_content = re.sub(pattern, replacement, content)
    
    # Write the fixed content back to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(fixed_content)
    
    print(f"Fixed run_backtest function in {file_path}")

if __name__ == "__main__":
    fix_run_backtest_function()
