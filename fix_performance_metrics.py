#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix the display_performance_metrics function in run_comprehensive_backtest_fixed.py
"""

def fix_performance_metrics():
    # Read the file
    with open('run_comprehensive_backtest_fixed.py', 'r') as f:
        content = f.readlines()
    
    # Find the display_performance_metrics function
    start_line = 0
    end_line = 0
    
    for i, line in enumerate(content):
        if "def display_performance_metrics(summary):" in line:
            start_line = i
        if start_line > 0 and "print(\"===== END METRICS =====\\n\")" in line:
            end_line = i + 1
            break
    
    if start_line == 0 or end_line == 0:
        print("Could not find the display_performance_metrics function")
        return
    
    # Replace the function with a fixed version
    new_function = [
        "def display_performance_metrics(summary):\n",
        "    \"\"\"Display performance metrics from a backtest summary\"\"\"\n",
        "    if not summary:\n",
        "        print(\"No performance metrics available\")\n",
        "        return\n",
        "    \n",
        "    # Convert tuple to dictionary if necessary\n",
        "    if isinstance(summary, tuple):\n",
        "        # The summary tuple contains the dictionary as the first element\n",
        "        summary_dict = summary[0] if len(summary) > 0 else {}\n",
        "    else:\n",
        "        summary_dict = summary\n",
        "    \n",
        "    print(\"\\n===== PERFORMANCE METRICS =====\")\n",
        "    print(f\"Win Rate: {summary_dict.get('win_rate', 0):.2f}%\")\n",
        "    print(f\"Profit Factor: {summary_dict.get('profit_factor', 0):.2f}\")\n",
        "    print(f\"Average Win: ${summary_dict.get('avg_win', 0):.2f}\")\n",
        "    print(f\"Average Loss: ${summary_dict.get('avg_loss', 0):.2f}\")\n",
        "    print(f\"Average Holding Period: {summary_dict.get('avg_holding_period', 0):.1f} days\")\n",
        "    \n",
        "    # Check if tier_metrics is available and has long/short win rates\n",
        "    if 'tier_metrics' in summary_dict:\n",
        "        tier_metrics = summary_dict['tier_metrics']\n",
        "        if isinstance(tier_metrics, dict):\n",
        "            # Check for long_win_rate directly in the tier_metrics dictionary\n",
        "            for tier_name, tier_data in tier_metrics.items():\n",
        "                if isinstance(tier_data, dict) and 'long_win_rate' in tier_data:\n",
        "                    print(f\"LONG Win Rate: {tier_data['long_win_rate']:.2f}%\")\n",
        "                    break\n",
        "    \n",
        "    # Display max drawdown if available\n",
        "    if 'max_drawdown' in summary_dict:\n",
        "        print(f\"Max Drawdown: {summary_dict.get('max_drawdown', 0):.2f}%\")\n",
        "    \n",
        "    # Display Sharpe and Sortino ratios if available\n",
        "    if 'sharpe_ratio' in summary_dict:\n",
        "        print(f\"Sharpe Ratio: {summary_dict.get('sharpe_ratio', 0):.2f}\")\n",
        "    if 'sortino_ratio' in summary_dict:\n",
        "        print(f\"Sortino Ratio: {summary_dict.get('sortino_ratio', 0):.2f}\")\n",
        "    \n",
        "    # Display final capital and total return if available\n",
        "    if 'final_capital' in summary_dict and 'initial_capital' in summary_dict:\n",
        "        initial_capital = summary_dict.get('initial_capital', 0)\n",
        "        final_capital = summary_dict.get('final_capital', 0)\n",
        "        total_return = ((final_capital / initial_capital) - 1) * 100 if initial_capital > 0 else 0\n",
        "        print(f\"Initial Capital: ${initial_capital:.2f}\")\n",
        "        print(f\"Final Capital: ${final_capital:.2f}\")\n",
        "        print(f\"Total Return: {total_return:.2f}%\")\n",
        "    \n",
        "    print(\"===== END METRICS =====\\n\")\n"
    ]
    
    # Replace the old function with the new one
    content = content[:start_line] + new_function + content[end_line:]
    
    # Write the updated content back to the file
    with open('run_comprehensive_backtest_fixed.py', 'w') as f:
        f.writelines(content)
    
    print("Fixed display_performance_metrics function in run_comprehensive_backtest_fixed.py")

if __name__ == "__main__":
    fix_performance_metrics()
