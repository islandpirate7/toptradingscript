import subprocess
import yaml
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import copy

# Define market regimes for testing
market_regimes = [
    {
        "name": "Bull Market",
        "start_date": "2023-01-01",
        "end_date": "2023-03-31",
        "description": "Strong uptrend in Q1 2023"
    },
    {
        "name": "Consolidation",
        "start_date": "2023-04-01",
        "end_date": "2023-05-15",
        "description": "Sideways market in mid-Q2 2023"
    },
    {
        "name": "Volatile Period",
        "start_date": "2023-05-16",
        "end_date": "2023-06-30",
        "description": "Higher volatility in late Q2 2023"
    },
    {
        "name": "Full Period",
        "start_date": "2023-01-01",
        "end_date": "2023-06-30",
        "description": "Complete test period"
    }
]

# Define ATR multiplier combinations to test
atr_combinations = [
    {"stop_loss": 1.5, "take_profit": 3.0, "name": "Conservative (1.5:3.0)"},
    {"stop_loss": 2.0, "take_profit": 3.0, "name": "Balanced (2.0:3.0)"},
    {"stop_loss": 2.0, "take_profit": 4.0, "name": "Aggressive (2.0:4.0)"},
    {"stop_loss": 2.5, "take_profit": 3.5, "name": "Wide Stop (2.5:3.5)"},
    {"stop_loss": 1.5, "take_profit": 2.5, "name": "Tight Range (1.5:2.5)"}
]

# Load the base configuration
with open('configuration_11.yaml', 'r') as file:
    base_config = yaml.safe_load(file)

# Create results directory if it doesn't exist
results_dir = "atr_optimization_results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Function to modify the MeanReversion strategy in the configuration
def modify_mean_reversion_strategy(config, stop_loss_multiplier, take_profit_multiplier):
    # Create a deep copy to avoid modifying the original
    modified_config = copy.deepcopy(config)
    
    # Add ATR multipliers to the MeanReversion strategy configuration
    if 'strategies' not in modified_config:
        modified_config['strategies'] = {}
    
    if 'MeanReversion' not in modified_config['strategies']:
        modified_config['strategies']['MeanReversion'] = {}
    
    modified_config['strategies']['MeanReversion']['stop_loss_atr'] = stop_loss_multiplier
    modified_config['strategies']['MeanReversion']['take_profit_atr'] = take_profit_multiplier
    
    return modified_config

# Function to run a backtest with specific parameters
def run_backtest(config, start_date, end_date, output_file):
    # Write the modified config to a temporary file
    temp_config_file = f"{results_dir}/temp_config.yaml"
    with open(temp_config_file, 'w') as file:
        yaml.dump(config, file)
    
    # Run the backtest command
    cmd = [
        "python", "multi_strategy_system.py",
        "--config", temp_config_file,
        "--backtest",
        "--start_date", start_date,
        "--end_date", end_date,
        "--strategy", "MeanReversion",
        "--output", output_file,
        "--detailed_report"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    # Check if the output file was created and load results
    if os.path.exists(output_file):
        with open(output_file, 'r') as file:
            try:
                results = json.load(file)
                return results
            except json.JSONDecodeError:
                print(f"Error: Could not parse results from {output_file}")
                return None
    else:
        print(f"Error: Output file {output_file} was not created")
        return None

# Main function to run all tests
def run_optimization():
    all_results = []
    
    # For each market regime
    for regime in market_regimes:
        print(f"\n===== Testing Market Regime: {regime['name']} ({regime['description']}) =====")
        
        # For each ATR combination
        for atr_combo in atr_combinations:
            print(f"\nTesting ATR Combination: {atr_combo['name']}")
            
            # Modify the configuration
            modified_config = modify_mean_reversion_strategy(
                base_config, 
                atr_combo['stop_loss'], 
                atr_combo['take_profit']
            )
            
            # Define output file
            output_file = f"{results_dir}/results_{regime['name'].replace(' ', '_')}_{atr_combo['name'].replace(' ', '_').replace(':', '_')}.json"
            
            # Run the backtest
            results = run_backtest(
                modified_config,
                regime['start_date'],
                regime['end_date'],
                output_file
            )
            
            if results:
                # Extract key metrics
                result_entry = {
                    "market_regime": regime['name'],
                    "regime_period": f"{regime['start_date']} to {regime['end_date']}",
                    "atr_combination": atr_combo['name'],
                    "stop_loss_multiplier": atr_combo['stop_loss'],
                    "take_profit_multiplier": atr_combo['take_profit'],
                    "results": results
                }
                
                all_results.append(result_entry)
                
                print(f"Completed test for {regime['name']} with {atr_combo['name']}")
                
                # Print key metrics
                if 'performance_metrics' in results:
                    metrics = results['performance_metrics']
                    print(f"  Total Return: {metrics.get('total_return', 'N/A')}%")
                    print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A')}")
                    print(f"  Max Drawdown: {metrics.get('max_drawdown', 'N/A')}%")
                    print(f"  Win Rate: {metrics.get('win_rate', 'N/A')}%")
                    print(f"  Profit Factor: {metrics.get('profit_factor', 'N/A')}")
            else:
                print(f"Failed to get results for {regime['name']} with {atr_combo['name']}")
    
    # Save all results to a single file
    with open(f"{results_dir}/all_optimization_results.json", 'w') as file:
        json.dump(all_results, file, indent=2)
    
    return all_results

# Function to analyze and visualize results
def analyze_results(all_results):
    if not all_results:
        print("No results to analyze")
        return
    
    # Create a DataFrame for easier analysis
    rows = []
    for result in all_results:
        if 'results' in result and 'performance_metrics' in result['results']:
            metrics = result['results']['performance_metrics']
            row = {
                "Market Regime": result['market_regime'],
                "ATR Combination": result['atr_combination'],
                "Stop Loss": result['stop_loss_multiplier'],
                "Take Profit": result['take_profit_multiplier'],
                "Total Return (%)": metrics.get('total_return', 0),
                "Sharpe Ratio": metrics.get('sharpe_ratio', 0),
                "Max Drawdown (%)": metrics.get('max_drawdown', 0),
                "Win Rate (%)": metrics.get('win_rate', 0),
                "Profit Factor": metrics.get('profit_factor', 0),
                "Number of Trades": metrics.get('total_trades', 0)
            }
            rows.append(row)
    
    if not rows:
        print("No performance metrics found in results")
        return
    
    df = pd.DataFrame(rows)
    
    # Save the DataFrame to CSV
    df.to_csv(f"{results_dir}/optimization_summary.csv", index=False)
    
    # Create visualizations
    
    # 1. Total Return by ATR Combination and Market Regime
    plt.figure(figsize=(12, 8))
    pivot = df.pivot(index="ATR Combination", columns="Market Regime", values="Total Return (%)")
    pivot.plot(kind="bar", figsize=(12, 8))
    plt.title("Total Return by ATR Combination and Market Regime")
    plt.ylabel("Total Return (%)")
    plt.xlabel("ATR Combination")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/total_return_by_atr_and_regime.png")
    
    # 2. Sharpe Ratio by ATR Combination and Market Regime
    plt.figure(figsize=(12, 8))
    pivot = df.pivot(index="ATR Combination", columns="Market Regime", values="Sharpe Ratio")
    pivot.plot(kind="bar", figsize=(12, 8))
    plt.title("Sharpe Ratio by ATR Combination and Market Regime")
    plt.ylabel("Sharpe Ratio")
    plt.xlabel("ATR Combination")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/sharpe_ratio_by_atr_and_regime.png")
    
    # 3. Win Rate by ATR Combination and Market Regime
    plt.figure(figsize=(12, 8))
    pivot = df.pivot(index="ATR Combination", columns="Market Regime", values="Win Rate (%)")
    pivot.plot(kind="bar", figsize=(12, 8))
    plt.title("Win Rate by ATR Combination and Market Regime")
    plt.ylabel("Win Rate (%)")
    plt.xlabel("ATR Combination")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/win_rate_by_atr_and_regime.png")
    
    # 4. Scatterplot of Risk-Reward (Stop Loss vs Take Profit) colored by Total Return
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        df["Stop Loss"], 
        df["Take Profit"],
        c=df["Total Return (%)"],
        s=df["Number of Trades"] * 5,  # Size based on number of trades
        alpha=0.7,
        cmap="viridis"
    )
    plt.colorbar(scatter, label="Total Return (%)")
    plt.title("Risk-Reward Analysis")
    plt.xlabel("Stop Loss ATR Multiplier")
    plt.ylabel("Take Profit ATR Multiplier")
    plt.grid(linestyle='--', alpha=0.7)
    
    # Add annotations for each point
    for i, row in df.iterrows():
        plt.annotate(
            f"{row['ATR Combination']}\n{row['Market Regime']}",
            (row["Stop Loss"], row["Take Profit"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8
        )
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/risk_reward_analysis.png")
    
    # 5. Find the best overall ATR combination
    # Group by ATR combination and calculate average metrics
    atr_performance = df.groupby("ATR Combination").agg({
        "Total Return (%)": "mean",
        "Sharpe Ratio": "mean",
        "Max Drawdown (%)": "mean",
        "Win Rate (%)": "mean",
        "Profit Factor": "mean",
        "Number of Trades": "mean"
    }).reset_index()
    
    # Calculate a composite score (higher is better)
    atr_performance["Composite Score"] = (
        atr_performance["Total Return (%)"] / atr_performance["Total Return (%)"].max() * 0.3 +
        atr_performance["Sharpe Ratio"] / atr_performance["Sharpe Ratio"].max() * 0.3 +
        (1 - atr_performance["Max Drawdown (%)"] / atr_performance["Max Drawdown (%)"].max()) * 0.2 +
        atr_performance["Win Rate (%)"] / atr_performance["Win Rate (%)"].max() * 0.1 +
        atr_performance["Profit Factor"] / atr_performance["Profit Factor"].max() * 0.1
    )
    
    # Sort by composite score
    atr_performance = atr_performance.sort_values("Composite Score", ascending=False)
    
    # Save the ATR performance summary
    atr_performance.to_csv(f"{results_dir}/atr_performance_summary.csv", index=False)
    
    # Print the best ATR combination
    best_atr = atr_performance.iloc[0]
    print("\n===== Best ATR Combination =====")
    print(f"Combination: {best_atr['ATR Combination']}")
    print(f"Average Total Return: {best_atr['Total Return (%)']:.2f}%")
    print(f"Average Sharpe Ratio: {best_atr['Sharpe Ratio']:.2f}")
    print(f"Average Max Drawdown: {best_atr['Max Drawdown (%)']:.2f}%")
    print(f"Average Win Rate: {best_atr['Win Rate (%)']:.2f}%")
    print(f"Average Profit Factor: {best_atr['Profit Factor']:.2f}")
    print(f"Composite Score: {best_atr['Composite Score']:.2f}")
    
    return atr_performance

# Run the optimization and analyze results
if __name__ == "__main__":
    print("Starting MeanReversion ATR Optimization")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    results = run_optimization()
    best_atr_combinations = analyze_results(results)
    
    print("\nOptimization complete! Results saved to:", results_dir)
    print("=" * 50)
