#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix the weekly selection parameter in run_comprehensive_backtest.py
"""

def fix_weekly_selection_parameter():
    # Read the file
    with open('run_comprehensive_backtest.py', 'r') as f:
        content = f.read()
    
    # Fix the argument parser section
    parser_section = "        parser.add_argument('--continuous_capital', action='store_true', help='Use continuous capital across quarters')"
    weekly_selection_param = "        parser.add_argument('--weekly_selection', action='store_true', help='Enable weekly stock selection refresh')"
    
    if weekly_selection_param not in content:
        content = content.replace(parser_section, parser_section + "\n" + weekly_selection_param)
    
    # Fix the function definitions
    content = content.replace(
        "def run_comprehensive_backtest(quarter, max_signals=100, initial_capital=300, multiple_runs=False, num_runs=5, continuous_capital=False, weekly_selection=args.weekly_selection, weekly_selection=False):",
        "def run_comprehensive_backtest(quarter, max_signals=100, initial_capital=300, multiple_runs=False, num_runs=5, continuous_capital=False, weekly_selection=False):"
    )
    
    content = content.replace(
        "def run_all_quarters_backtest(max_signals=100, initial_capital=300, multiple_runs=False, num_runs=5, continuous_capital=False, weekly_selection=args.weekly_selection, weekly_selection=False):",
        "def run_all_quarters_backtest(max_signals=100, initial_capital=300, multiple_runs=False, num_runs=5, continuous_capital=False, weekly_selection=False):"
    )
    
    # Fix the function calls
    content = content.replace(
        "run_backtest(start_date, end_date, mode='backtest', max_signals=max_signals, initial_capital=initial_capital, random_seed=random_seed, weekly_selection=args.weekly_selection, weekly_selection=weekly_selection)",
        "run_backtest(start_date, end_date, mode='backtest', max_signals=max_signals, initial_capital=initial_capital, random_seed=random_seed, weekly_selection=weekly_selection)"
    )
    
    # Fix the run_comprehensive_backtest calls
    content = content.replace(
        "run_comprehensive_backtest(quarter, max_signals=args.max_signals, initial_capital=args.initial_capital, multiple_runs=args.multiple_runs, num_runs=args.num_runs, continuous_capital=args.continuous_capital, weekly_selection=args.weekly_selection, weekly_selection=args.weekly_selection)",
        "run_comprehensive_backtest(quarter, max_signals=args.max_signals, initial_capital=args.initial_capital, multiple_runs=args.multiple_runs, num_runs=args.num_runs, continuous_capital=args.continuous_capital, weekly_selection=args.weekly_selection)"
    )
    
    # Fix the run_all_quarters_backtest calls
    content = content.replace(
        "run_all_quarters_backtest(max_signals=args.max_signals, initial_capital=args.initial_capital, multiple_runs=args.multiple_runs, num_runs=args.num_runs, continuous_capital=args.continuous_capital, weekly_selection=args.weekly_selection, weekly_selection=args.weekly_selection)",
        "run_all_quarters_backtest(max_signals=args.max_signals, initial_capital=args.initial_capital, multiple_runs=args.multiple_runs, num_runs=args.num_runs, continuous_capital=args.continuous_capital, weekly_selection=args.weekly_selection)"
    )
    
    # Write the updated content back to the file
    with open('run_comprehensive_backtest.py', 'w') as f:
        f.write(content)
    
    print("Fixed weekly selection parameter in run_comprehensive_backtest.py")

if __name__ == "__main__":
    fix_weekly_selection_parameter()
