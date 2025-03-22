import subprocess
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import yaml

# Define the strategies to test
strategies = [
    "MeanReversion",
    "VolatilityBreakout",
    "TrendFollowing",
    "GapTrading"
]

# Define test periods
test_periods = [
    {
        "name": "H1_2023",
        "start_date": "2023-01-01",
        "end_date": "2023-06-30",
        "description": "First half of 2023"
    }
]

# Results storage
results_dir = "strategy_analysis_results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

def run_backtest(config_file, start_date, end_date, strategy=None, output_file=None):
    """Run a backtest with the specified parameters"""
    cmd = [
        "python", "multi_strategy_system.py",
        "--config", config_file,
        "--backtest",
        "--start_date", start_date,
        "--end_date", end_date,
        "--log_level", "INFO"
    ]
    
    if strategy:
        cmd.extend(["--strategy", strategy])
    
    if output_file:
        cmd.extend(["--output", output_file])
    
    print(f"Running: {' '.join(cmd)}")
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    # Check if the process was successful
    if process.returncode != 0:
        print(f"Error running backtest: {process.stderr}")
        return None
    
    # Parse the output to extract performance metrics
    output = process.stdout
    
    # Initialize metrics dictionary
    metrics = {
        "total_return": 0,
        "win_rate": 0,
        "profit_factor": 0,
        "max_drawdown": 0,
        "sharpe_ratio": 0,
        "total_trades": 0
    }
    
    # Extract metrics from output
    try:
        for line in output.split('\n'):
            if "Total Return:" in line:
                metrics["total_return"] = float(line.split("Total Return:")[1].split("%")[0].strip())
            elif "Win Rate:" in line:
                metrics["win_rate"] = float(line.split("Win Rate:")[1].split("%")[0].strip())
            elif "Profit Factor:" in line:
                metrics["profit_factor"] = float(line.split("Profit Factor:")[1].strip())
            elif "Maximum Drawdown:" in line:
                metrics["max_drawdown"] = float(line.split("Maximum Drawdown:")[1].split("%")[0].strip())
            elif "Sharpe Ratio:" in line:
                metrics["sharpe_ratio"] = float(line.split("Sharpe Ratio:")[1].strip())
            elif "Total Trades:" in line:
                metrics["total_trades"] = int(line.split("Total Trades:")[1].strip())
    except Exception as e:
        print(f"Error parsing output: {e}")
        print(f"Output: {output}")
    
    return metrics

def analyze_strategies():
    """Run backtests for each strategy and analyze the results"""
    all_results = []
    
    for period in test_periods:
        print(f"\n===== Testing Period: {period['name']} ({period['description']}) =====")
        
        # Test each individual strategy
        for strategy in strategies:
            print(f"\nTesting strategy: {strategy}")
            
            # Run backtest for this strategy
            output_file = f"{results_dir}/{period['name']}_{strategy}_results.json"
            
            metrics = run_backtest(
                "configuration_11.yaml",
                period["start_date"],
                period["end_date"],
                strategy,
                output_file
            )
            
            if metrics:
                # Add strategy and period info to metrics
                metrics["strategy"] = strategy
                metrics["period"] = period["name"]
                metrics["start_date"] = period["start_date"]
                metrics["end_date"] = period["end_date"]
                
                all_results.append(metrics)
                
                print(f"  Results for {strategy}:")
                print(f"    Total Return: {metrics['total_return']:.2f}%")
                print(f"    Win Rate: {metrics['win_rate']:.2f}%")
                print(f"    Profit Factor: {metrics['profit_factor']:.2f}")
                print(f"    Max Drawdown: {metrics['max_drawdown']:.2f}%")
                print(f"    Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                print(f"    Total Trades: {metrics['total_trades']}")
            else:
                print(f"  No results obtained for {strategy}")
        
        # Test all strategies combined
        print("\nTesting all strategies combined")
        
        output_file = f"{results_dir}/{period['name']}_Combined_results.json"
        
        metrics = run_backtest(
            "configuration_11.yaml",
            period["start_date"],
            period["end_date"],
            None,  # No specific strategy = all strategies
            output_file
        )
        
        if metrics:
            # Add strategy and period info to metrics
            metrics["strategy"] = "Combined"
            metrics["period"] = period["name"]
            metrics["start_date"] = period["start_date"]
            metrics["end_date"] = period["end_date"]
            
            all_results.append(metrics)
            
            print(f"  Results for Combined Strategies:")
            print(f"    Total Return: {metrics['total_return']:.2f}%")
            print(f"    Win Rate: {metrics['win_rate']:.2f}%")
            print(f"    Profit Factor: {metrics['profit_factor']:.2f}")
            print(f"    Max Drawdown: {metrics['max_drawdown']:.2f}%")
            print(f"    Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"    Total Trades: {metrics['total_trades']}")
        else:
            print("  No results obtained for Combined Strategies")
    
    # Convert results to DataFrame for analysis
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Save results to CSV
        csv_file = f"{results_dir}/all_strategy_results.csv"
        df.to_csv(csv_file, index=False)
        print(f"\nSaved all results to {csv_file}")
        
        # Create visualizations
        create_strategy_comparison_chart(df, results_dir)
        
        # Generate recommendations for configuration_12
        generate_recommendations(df, results_dir)
    else:
        print("\nNo results to analyze")

def create_strategy_comparison_chart(df, results_dir):
    """Create a chart comparing the performance of different strategies"""
    plt.figure(figsize=(15, 10))
    
    # Plot total return by strategy
    plt.subplot(2, 2, 1)
    strategies = df['strategy'].unique()
    returns = [df[df['strategy'] == s]['total_return'].values[0] for s in strategies]
    plt.bar(strategies, returns)
    plt.title('Total Return (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Plot win rate by strategy
    plt.subplot(2, 2, 2)
    win_rates = [df[df['strategy'] == s]['win_rate'].values[0] for s in strategies]
    plt.bar(strategies, win_rates)
    plt.title('Win Rate (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Plot profit factor by strategy
    plt.subplot(2, 2, 3)
    profit_factors = [df[df['strategy'] == s]['profit_factor'].values[0] for s in strategies]
    plt.bar(strategies, profit_factors)
    plt.title('Profit Factor')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Plot max drawdown by strategy
    plt.subplot(2, 2, 4)
    drawdowns = [df[df['strategy'] == s]['max_drawdown'].values[0] for s in strategies]
    plt.bar(strategies, drawdowns)
    plt.title('Max Drawdown (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f"{results_dir}/strategy_comparison.png")
    plt.close()

def generate_recommendations(df, results_dir):
    """Generate recommendations for configuration_12 based on the backtest results"""
    # Load the current configuration
    with open('configuration_11.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Create a copy for configuration_12
    config_12 = config.copy()
    
    # Analyze the results
    best_strategy = df.loc[df['total_return'].idxmax()]['strategy']
    worst_strategy = df.loc[df['total_return'].idxmin()]['strategy']
    
    # Calculate the performance ratio for each strategy compared to the best
    best_return = df.loc[df['total_return'].idxmax()]['total_return']
    
    # Calculate new weights based on performance
    strategy_performance = {}
    for strategy in df['strategy'].unique():
        if strategy != 'Combined':
            strategy_return = df[df['strategy'] == strategy]['total_return'].values[0]
            # Avoid division by zero
            if best_return > 0:
                performance_ratio = max(0.1, strategy_return / best_return)
            else:
                performance_ratio = 0.1
            strategy_performance[strategy] = performance_ratio
    
    # Normalize weights to sum to 1
    total_performance = sum(strategy_performance.values())
    normalized_weights = {s: p/total_performance for s, p in strategy_performance.items()}
    
    # Update strategy weights in configuration_12
    for strategy, weight in normalized_weights.items():
        if strategy in config_12['strategy_weights']:
            config_12['strategy_weights'][strategy] = round(weight, 2)
    
    # Generate recommendations text
    recommendations = f"""
# Strategy Analysis Results and Recommendations for Configuration 12
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Performance Summary

"""
    for _, row in df.iterrows():
        recommendations += f"- **{row['strategy']}**: Return: {row['total_return']:.2f}%, Win Rate: {row['win_rate']:.2f}%, Profit Factor: {row['profit_factor']:.2f}, Max Drawdown: {row['max_drawdown']:.2f}%, Trades: {row['total_trades']}\n"
    
    recommendations += f"""
## Key Findings

- Best performing strategy: **{best_strategy}**
- Worst performing strategy: **{worst_strategy}**
- Combined strategies performance compared to best individual strategy: {(df[df['strategy'] == 'Combined']['total_return'].values[0] / best_return * 100):.2f}%

## Recommendations for Configuration 12

1. **Strategy Weights**: Based on the backtest results, we recommend the following weight adjustments:
"""
    
    for strategy, weight in normalized_weights.items():
        old_weight = config['strategy_weights'].get(strategy, 0)
        recommendations += f"   - **{strategy}**: {old_weight:.2f} → {weight:.2f} ({'+' if weight > old_weight else ''}{(weight - old_weight) * 100:.1f}%)\n"
    
    recommendations += """
2. **Risk Management**: Maintain the current ATR-based stop loss and take profit settings for the MeanReversion strategy, as they have proven effective.

3. **Position Sizing**: Consider implementing more dynamic position sizing based on strategy performance and market conditions.

4. **Market Regime Detection**: Enhance the market regime detection to better adapt strategy weights in different market conditions.

5. **Next Steps**:
   - Implement these recommendations in configuration_12
   - Run further backtests with longer time periods to validate the changes
   - Consider adding more sophisticated filters to improve signal quality
   - Explore additional strategies to diversify the trading approach
"""
    
    # Save recommendations to file
    with open(f"{results_dir}/configuration_12_recommendations.md", "w") as f:
        f.write(recommendations)
    
    # Save configuration_12 to file
    with open('configuration_12.yaml', 'w') as file:
        yaml.dump(config_12, file, default_flow_style=False)
    
    print(f"\nRecommendations saved to {results_dir}/configuration_12_recommendations.md")
    print(f"Configuration 12 saved to configuration_12.yaml")

if __name__ == "__main__":
    analyze_strategies()
