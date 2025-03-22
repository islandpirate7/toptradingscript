import os
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Define the ATR combinations that were tested
atr_combinations = [
    {"stop_loss": 1.5, "take_profit": 3.0, "name": "Conservative"},
    {"stop_loss": 2.0, "take_profit": 3.0, "name": "Balanced"},
    {"stop_loss": 2.0, "take_profit": 4.0, "name": "Aggressive"},
    {"stop_loss": 2.5, "take_profit": 3.5, "name": "Wide Stop"},
    {"stop_loss": 1.5, "take_profit": 2.5, "name": "Tight Range"}
]

# Define market periods that were tested
market_periods = [
    {
        "name": "Bull Market",
        "start_date": "2023-01-01",
        "end_date": "2023-03-31"
    },
    {
        "name": "Consolidation",
        "start_date": "2023-04-01",
        "end_date": "2023-05-15"
    },
    {
        "name": "Volatile Period",
        "start_date": "2023-05-16",
        "end_date": "2023-06-30"
    }
]

# Function to run a direct backtest and get results
def run_direct_backtest(start_date, end_date, stop_loss_atr, take_profit_atr):
    """Run a direct backtest with specific parameters and return the results"""
    import subprocess
    import tempfile
    import yaml
    
    # Create a temporary config file with the specified ATR settings
    with open('configuration_11.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Ensure strategies section exists
    if 'strategies' not in config:
        config['strategies'] = {}
    
    # Ensure MeanReversion section exists
    if 'MeanReversion' not in config['strategies']:
        config['strategies']['MeanReversion'] = {}
    
    # Set ATR multipliers
    config['strategies']['MeanReversion']['stop_loss_atr'] = stop_loss_atr
    config['strategies']['MeanReversion']['take_profit_atr'] = take_profit_atr
    
    # Create a temporary file for the config
    with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as temp_file:
        temp_config_path = temp_file.name
        yaml.dump(config, temp_file)
    
    # Create a temporary file for the results
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_results_file:
        temp_results_path = temp_results_file.name
    
    try:
        # Run the backtest command with output to JSON
        cmd = [
            "python", "multi_strategy_system.py",
            "--config", temp_config_path,
            "--backtest",
            "--start_date", start_date,
            "--end_date", end_date,
            "--strategy", "MeanReversion",
            "--output", temp_results_path
        ]
        
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, capture_output=True)
        
        # Check if results file exists and has content
        if os.path.exists(temp_results_path) and os.path.getsize(temp_results_path) > 0:
            with open(temp_results_path, 'r') as f:
                try:
                    results = json.load(f)
                    return results
                except json.JSONDecodeError:
                    print(f"Error: Could not parse results JSON from {temp_results_path}")
                    return None
        else:
            print(f"Error: No results file created at {temp_results_path}")
            return None
    
    finally:
        # Clean up temporary files
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
        if os.path.exists(temp_results_path):
            os.remove(temp_results_path)

# Function to analyze backtest results
def analyze_results():
    """Run backtests with different ATR settings and analyze the results"""
    results = []
    
    # Create results directory
    results_dir = "atr_analysis_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    for period in market_periods:
        print(f"\nTesting period: {period['name']} ({period['start_date']} to {period['end_date']})")
        
        period_results = []
        
        for atr_combo in atr_combinations:
            print(f"  Testing ATR combination: {atr_combo['name']} (SL: {atr_combo['stop_loss']}, TP: {atr_combo['take_profit']})")
            
            # Run backtest
            backtest_results = run_direct_backtest(
                period['start_date'],
                period['end_date'],
                atr_combo['stop_loss'],
                atr_combo['take_profit']
            )
            
            if backtest_results:
                # Extract key metrics
                metrics = {
                    'period': period['name'],
                    'atr_combo': atr_combo['name'],
                    'stop_loss_atr': atr_combo['stop_loss'],
                    'take_profit_atr': atr_combo['take_profit'],
                    'final_equity': backtest_results.get('final_equity', 0),
                    'total_return': backtest_results.get('total_return', 0),
                    'win_rate': backtest_results.get('win_rate', 0),
                    'profit_factor': backtest_results.get('profit_factor', 0),
                    'max_drawdown': backtest_results.get('max_drawdown', 0),
                    'sharpe_ratio': backtest_results.get('sharpe_ratio', 0),
                    'total_trades': backtest_results.get('total_trades', 0)
                }
                
                period_results.append(metrics)
                results.append(metrics)
                
                print(f"  Results: Return={metrics['total_return']:.2f}%, Win Rate={metrics['win_rate']:.2f}%, Profit Factor={metrics['profit_factor']:.2f}")
            else:
                print(f"  No results obtained for {atr_combo['name']} in {period['name']}")
        
        # If we have results for this period, save them
        if period_results:
            # Convert to DataFrame for easier analysis
            df = pd.DataFrame(period_results)
            
            # Sort by total return
            df = df.sort_values('total_return', ascending=False)
            
            # Save to CSV
            csv_file = f"{results_dir}/{period['name']}_results.csv"
            df.to_csv(csv_file, index=False)
            print(f"  Saved period results to {csv_file}")
            
            # Create visualization
            create_period_visualization(df, period['name'], results_dir)
    
    # If we have overall results, analyze them
    if results:
        # Convert to DataFrame
        all_results_df = pd.DataFrame(results)
        
        # Save all results to CSV
        all_csv_file = f"{results_dir}/all_results.csv"
        all_results_df.to_csv(all_csv_file, index=False)
        print(f"\nSaved all results to {all_csv_file}")
        
        # Create overall visualization
        create_overall_visualization(all_results_df, results_dir)
        
        # Generate recommendations
        generate_recommendations(all_results_df, results_dir)

# Function to create visualization for a specific period
def create_period_visualization(df, period_name, results_dir):
    """Create visualizations for a specific market period"""
    plt.figure(figsize=(12, 8))
    
    # Plot total return by ATR combo
    plt.subplot(2, 2, 1)
    plt.bar(df['atr_combo'], df['total_return'])
    plt.title(f'Total Return (%) - {period_name}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Plot win rate by ATR combo
    plt.subplot(2, 2, 2)
    plt.bar(df['atr_combo'], df['win_rate'])
    plt.title(f'Win Rate (%) - {period_name}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Plot profit factor by ATR combo
    plt.subplot(2, 2, 3)
    plt.bar(df['atr_combo'], df['profit_factor'])
    plt.title(f'Profit Factor - {period_name}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Plot max drawdown by ATR combo
    plt.subplot(2, 2, 4)
    plt.bar(df['atr_combo'], df['max_drawdown'])
    plt.title(f'Max Drawdown (%) - {period_name}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f"{results_dir}/{period_name}_visualization.png")
    plt.close()

# Function to create overall visualization
def create_overall_visualization(df, results_dir):
    """Create visualizations for all market periods"""
    plt.figure(figsize=(15, 10))
    
    # Group by period and ATR combo
    grouped = df.groupby(['period', 'atr_combo']).mean().reset_index()
    
    # Plot total return by period and ATR combo
    periods = df['period'].unique()
    atr_combos = df['atr_combo'].unique()
    
    # Set up the figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot total return
    for i, period in enumerate(periods):
        period_data = grouped[grouped['period'] == period]
        axes[0, 0].bar([x + i*0.15 for x in range(len(atr_combos))], 
                     period_data['total_return'], 
                     width=0.15, 
                     label=period)
    
    axes[0, 0].set_title('Total Return (%)')
    axes[0, 0].set_xticks([x + 0.3 for x in range(len(atr_combos))])
    axes[0, 0].set_xticklabels(atr_combos, rotation=45)
    axes[0, 0].legend()
    
    # Plot win rate
    for i, period in enumerate(periods):
        period_data = grouped[grouped['period'] == period]
        axes[0, 1].bar([x + i*0.15 for x in range(len(atr_combos))], 
                     period_data['win_rate'], 
                     width=0.15, 
                     label=period)
    
    axes[0, 1].set_title('Win Rate (%)')
    axes[0, 1].set_xticks([x + 0.3 for x in range(len(atr_combos))])
    axes[0, 1].set_xticklabels(atr_combos, rotation=45)
    axes[0, 1].legend()
    
    # Plot profit factor
    for i, period in enumerate(periods):
        period_data = grouped[grouped['period'] == period]
        axes[1, 0].bar([x + i*0.15 for x in range(len(atr_combos))], 
                     period_data['profit_factor'], 
                     width=0.15, 
                     label=period)
    
    axes[1, 0].set_title('Profit Factor')
    axes[1, 0].set_xticks([x + 0.3 for x in range(len(atr_combos))])
    axes[1, 0].set_xticklabels(atr_combos, rotation=45)
    axes[1, 0].legend()
    
    # Plot max drawdown
    for i, period in enumerate(periods):
        period_data = grouped[grouped['period'] == period]
        axes[1, 1].bar([x + i*0.15 for x in range(len(atr_combos))], 
                     period_data['max_drawdown'], 
                     width=0.15, 
                     label=period)
    
    axes[1, 1].set_title('Max Drawdown (%)')
    axes[1, 1].set_xticks([x + 0.3 for x in range(len(atr_combos))])
    axes[1, 1].set_xticklabels(atr_combos, rotation=45)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/overall_visualization.png")
    plt.close()

# Function to generate recommendations
def generate_recommendations(df, results_dir):
    """Generate recommendations based on the backtest results"""
    # Group by period and ATR combo
    grouped = df.groupby(['period', 'atr_combo']).mean().reset_index()
    
    # Find best ATR combo for each period based on total return
    best_return = grouped.loc[grouped.groupby('period')['total_return'].idxmax()]
    
    # Find best ATR combo for each period based on risk-adjusted return (return / max_drawdown)
    # Avoid division by zero
    grouped['risk_adjusted'] = grouped.apply(
        lambda x: x['total_return'] / max(0.01, x['max_drawdown']) if x['max_drawdown'] > 0 else x['total_return'], 
        axis=1
    )
    best_risk_adjusted = grouped.loc[grouped.groupby('period')['risk_adjusted'].idxmax()]
    
    # Find most consistent ATR combo across all periods
    consistency_scores = {}
    for combo in df['atr_combo'].unique():
        combo_data = grouped[grouped['atr_combo'] == combo]
        # Calculate coefficient of variation (lower is more consistent)
        if len(combo_data) > 0 and combo_data['total_return'].mean() > 0:
            cv = combo_data['total_return'].std() / combo_data['total_return'].mean()
            consistency_scores[combo] = cv
    
    most_consistent = min(consistency_scores.items(), key=lambda x: x[1])[0] if consistency_scores else None
    
    # Generate recommendations text
    recommendations = f"""
# MeanReversion Strategy ATR Optimization Results
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Best ATR Settings by Market Regime (Based on Total Return)

"""
    for _, row in best_return.iterrows():
        recommendations += f"- **{row['period']}**: {row['atr_combo']} (SL: {row['stop_loss_atr']}, TP: {row['take_profit_atr']}) - Return: {row['total_return']:.2f}%, Win Rate: {row['win_rate']:.2f}%, Profit Factor: {row['profit_factor']:.2f}\n"
    
    recommendations += f"""
## Best ATR Settings by Market Regime (Based on Risk-Adjusted Return)

"""
    for _, row in best_risk_adjusted.iterrows():
        recommendations += f"- **{row['period']}**: {row['atr_combo']} (SL: {row['stop_loss_atr']}, TP: {row['take_profit_atr']}) - Risk-Adjusted Return: {row['risk_adjusted']:.2f}, Return: {row['total_return']:.2f}%, Max Drawdown: {row['max_drawdown']:.2f}%\n"
    
    if most_consistent:
        combo_data = grouped[grouped['atr_combo'] == most_consistent]
        avg_return = combo_data['total_return'].mean()
        avg_win_rate = combo_data['win_rate'].mean()
        avg_profit_factor = combo_data['profit_factor'].mean()
        
        # Get the ATR values
        combo_info = next((c for c in atr_combinations if c['name'] == most_consistent), None)
        sl_atr = combo_info['stop_loss'] if combo_info else "N/A"
        tp_atr = combo_info['take_profit'] if combo_info else "N/A"
        
        recommendations += f"""
## Most Consistent ATR Setting Across All Market Regimes

- **{most_consistent}** (SL: {sl_atr}, TP: {tp_atr}) - Average Return: {avg_return:.2f}%, Average Win Rate: {avg_win_rate:.2f}%, Average Profit Factor: {avg_profit_factor:.2f}
"""
    
    recommendations += f"""
## Overall Recommendation

Based on the backtest results across different market regimes, we recommend the following ATR multiplier settings for the MeanReversion strategy:

"""
    
    # Make overall recommendation based on consistency and performance
    if most_consistent:
        combo_info = next((c for c in atr_combinations if c['name'] == most_consistent), None)
        if combo_info:
            recommendations += f"- **Stop Loss ATR Multiplier**: {combo_info['stop_loss']}\n"
            recommendations += f"- **Take Profit ATR Multiplier**: {combo_info['take_profit']}\n\n"
    
    recommendations += """
This recommendation provides a good balance between performance and consistency across different market conditions. However, for optimal results, consider adjusting the ATR multipliers based on the current market regime:

- In bull markets, a more aggressive approach might be beneficial
- In consolidation periods, a balanced approach works best
- In volatile periods, a more conservative approach with wider stops can help avoid premature exits

## Next Steps

1. Implement these optimized ATR multipliers in the MeanReversion strategy configuration
2. Consider implementing dynamic ATR multipliers that adjust based on detected market regime
3. Continue monitoring performance and fine-tune as needed
"""
    
    # Save recommendations to file
    with open(f"{results_dir}/recommendations.md", "w") as f:
        f.write(recommendations)
    
    print(f"\nRecommendations saved to {results_dir}/recommendations.md")

if __name__ == "__main__":
    analyze_results()
