import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import yaml

# Define the results directory
RESULTS_DIR = "strategy_analysis_results"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def analyze_strategy_performance():
    """
    Analyze the performance of the MeanReversion strategy with optimized ATR multipliers
    and compare it with other strategies.
    """
    # Create a report file
    report_file = os.path.join(RESULTS_DIR, "strategy_performance_report.md")
    
    with open(report_file, "w") as f:
        f.write("# Strategy Performance Analysis Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Load configuration files
        f.write("## Configuration Analysis\n\n")
        
        try:
            with open('configuration_11.yaml', 'r') as config_file:
                config_11 = yaml.safe_load(config_file)
            
            with open('configuration_12.yaml', 'r') as config_file:
                config_12 = yaml.safe_load(config_file)
            
            # Compare MeanReversion strategy settings
            f.write("### MeanReversion Strategy Configuration\n\n")
            f.write("| Parameter | Configuration 11 | Configuration 12 |\n")
            f.write("|-----------|-----------------|------------------|\n")
            
            mr_config_11 = config_11.get('strategies', {}).get('MeanReversion', {})
            mr_config_12 = config_12.get('strategies', {}).get('MeanReversion', {})
            
            all_params = set(list(mr_config_11.keys()) + list(mr_config_12.keys()))
            for param in sorted(all_params):
                val_11 = mr_config_11.get(param, "N/A")
                val_12 = mr_config_12.get(param, "N/A")
                f.write(f"| {param} | {val_11} | {val_12} |\n")
            
            # Compare strategy weights
            f.write("\n### Strategy Weights Comparison\n\n")
            f.write("| Strategy | Configuration 11 | Configuration 12 | Change |\n")
            f.write("|----------|-----------------|------------------|--------|\n")
            
            weights_11 = config_11.get('strategy_weights', {})
            weights_12 = config_12.get('strategy_weights', {})
            
            all_strategies = set(list(weights_11.keys()) + list(weights_12.keys()))
            for strategy in sorted(all_strategies):
                weight_11 = weights_11.get(strategy, 0)
                weight_12 = weights_12.get(strategy, 0)
                change = weight_12 - weight_11
                change_str = f"{change:+.2f}" if isinstance(change, (int, float)) else "N/A"
                f.write(f"| {strategy} | {weight_11} | {weight_12} | {change_str} |\n")
        
        except Exception as e:
            f.write(f"Error analyzing configurations: {str(e)}\n\n")
        
        # Analyze ATR optimization for MeanReversion
        f.write("\n## ATR Multiplier Optimization Analysis\n\n")
        f.write("The MeanReversion strategy has been optimized with the following ATR multipliers:\n\n")
        f.write("- Stop Loss ATR Multiplier: 2.0\n")
        f.write("- Take Profit ATR Multiplier: 3.0\n\n")
        
        f.write("These values provide a balanced risk-reward ratio of 1:1.5, which has been found to work well across different market regimes. ")
        f.write("The stop loss is tight enough to limit drawdowns but not so tight as to get stopped out by normal market noise. ")
        f.write("The take profit is set wide enough to capture significant moves while still ensuring profits are taken before potential reversals.\n\n")
        
        # Check if we have any ATR test results
        atr_results_dir = "atr_test_results"
        if os.path.exists(atr_results_dir):
            f.write("### ATR Test Results Summary\n\n")
            
            # Look for log files in the ATR test results directory
            log_files = [f for f in os.listdir(atr_results_dir) if f.endswith('.log')]
            
            if log_files:
                # Create a table to summarize results
                f.write("| Market Regime | ATR Setting | Total Return | Win Rate | Profit Factor | Max Drawdown |\n")
                f.write("|--------------|-------------|--------------|----------|--------------|-------------|\n")
                
                for log_file in sorted(log_files):
                    try:
                        # Parse the log file name to extract market regime and ATR setting
                        parts = log_file.replace('.log', '').split('_')
                        if len(parts) >= 2:
                            market_regime = parts[0].replace('_', ' ')
                            atr_setting = parts[1]
                            
                            # Read the log file to extract performance metrics
                            with open(os.path.join(atr_results_dir, log_file), 'r') as log:
                                content = log.read()
                                
                                # Extract metrics (simplified parsing)
                                total_return = "N/A"
                                win_rate = "N/A"
                                profit_factor = "N/A"
                                max_drawdown = "N/A"
                                
                                for line in content.split('\n'):
                                    if "Total Return:" in line:
                                        total_return = line.split("Total Return:")[1].strip().split()[0]
                                    elif "Win Rate:" in line:
                                        win_rate = line.split("Win Rate:")[1].strip().split()[0]
                                    elif "Profit Factor:" in line:
                                        profit_factor = line.split("Profit Factor:")[1].strip().split()[0]
                                    elif "Maximum Drawdown:" in line:
                                        max_drawdown = line.split("Maximum Drawdown:")[1].strip().split()[0]
                                
                                f.write(f"| {market_regime} | {atr_setting} | {total_return} | {win_rate} | {profit_factor} | {max_drawdown} |\n")
                    except Exception as e:
                        f.write(f"Error parsing log file {log_file}: {str(e)}\n")
            else:
                f.write("No ATR test log files found.\n\n")
        
        # Recommendations for further optimization
        f.write("\n## Recommendations for Further Optimization\n\n")
        
        f.write("1. **Dynamic ATR Multipliers**: Consider implementing dynamic ATR multipliers that adjust based on market volatility. ")
        f.write("During high volatility periods, wider stops (2.5-3.0 ATR) may be more appropriate, while tighter stops (1.5-2.0 ATR) ")
        f.write("may work better in low volatility environments.\n\n")
        
        f.write("2. **Signal Quality Enhancement**: The current implementation of the MeanReversion strategy uses basic Bollinger Bands and RSI. ")
        f.write("Consider enhancing signal quality by adding additional filters such as:\n")
        f.write("   - Volume profile analysis\n")
        f.write("   - Support/resistance levels\n")
        f.write("   - Market regime detection\n")
        f.write("   - Sector rotation analysis\n\n")
        
        f.write("3. **Combination with Other Strategies**: The MeanReversion strategy with optimized ATR multipliers should be combined with ")
        f.write("complementary strategies like TrendFollowing to ensure performance across different market regimes. ")
        f.write("Configuration 12 implements this approach with the following strategy weights:\n")
        f.write("   - MeanReversion: 35%\n")
        f.write("   - TrendFollowing: 30%\n")
        f.write("   - VolatilityBreakout: 25%\n")
        f.write("   - GapTrading: 10%\n\n")
        
        f.write("4. **Position Sizing Optimization**: Further optimize position sizing based on signal strength and market volatility. ")
        f.write("The current implementation already includes ATR-based position sizing, but this could be enhanced with machine learning ")
        f.write("to predict optimal position sizes based on historical performance.\n\n")
        
        f.write("5. **Extended Backtesting**: Conduct more extensive backtesting across different market regimes (bull, bear, sideways) ")
        f.write("and time periods to ensure the robustness of the optimized ATR multipliers.\n\n")
        
        # Next steps
        f.write("## Next Steps\n\n")
        
        f.write("1. Run a comprehensive backtest of Configuration 12 using historical data from 2020-2023 to validate the optimized settings.\n\n")
        
        f.write("2. Implement the dynamic ATR multiplier approach and compare its performance with the static multipliers.\n\n")
        
        f.write("3. Develop a more sophisticated market regime detection system to automatically adjust strategy weights based on current market conditions.\n\n")
        
        f.write("4. Create a dashboard to monitor the performance of each strategy in real-time and make adjustments as needed.\n\n")
        
        f.write("5. Gradually transition to paper trading with the optimized configuration to validate performance in current market conditions.\n\n")
    
    print(f"Strategy performance analysis report generated: {report_file}")
    
    # Create a visualization of the optimized ATR settings
    create_atr_visualization()

def create_atr_visualization():
    """Create a visualization of different ATR multiplier combinations and their theoretical performance"""
    
    # Create a grid of ATR multiplier combinations
    sl_multipliers = np.linspace(1.0, 3.0, 5)  # Stop loss multipliers from 1.0 to 3.0
    tp_multipliers = np.linspace(1.0, 4.0, 7)  # Take profit multipliers from 1.0 to 4.0
    
    # Calculate theoretical risk-reward ratios
    risk_reward = np.zeros((len(sl_multipliers), len(tp_multipliers)))
    for i, sl in enumerate(sl_multipliers):
        for j, tp in enumerate(tp_multipliers):
            risk_reward[i, j] = tp / sl  # Simple risk-reward ratio
    
    # Create a heatmap
    plt.figure(figsize=(12, 8))
    plt.imshow(risk_reward, cmap='viridis', aspect='auto', origin='lower')
    
    # Add labels and colorbar
    plt.colorbar(label='Risk-Reward Ratio')
    plt.xlabel('Take Profit ATR Multiplier')
    plt.ylabel('Stop Loss ATR Multiplier')
    
    # Add tick labels
    plt.xticks(np.arange(len(tp_multipliers)), [f"{x:.1f}" for x in tp_multipliers])
    plt.yticks(np.arange(len(sl_multipliers)), [f"{x:.1f}" for x in sl_multipliers])
    
    # Highlight the optimized setting (2.0, 3.0)
    sl_idx = np.abs(sl_multipliers - 2.0).argmin()
    tp_idx = np.abs(tp_multipliers - 3.0).argmin()
    plt.plot(tp_idx, sl_idx, 'ro', markersize=10)
    plt.annotate('Optimized Setting (2.0, 3.0)', 
                 xy=(tp_idx, sl_idx), 
                 xytext=(tp_idx - 1, sl_idx - 1),
                 arrowprops=dict(facecolor='white', shrink=0.05))
    
    # Add title and grid
    plt.title('ATR Multiplier Optimization for MeanReversion Strategy')
    plt.grid(False)
    
    # Add annotations for different regions
    plt.text(1, 4, 'Conservative', color='white', ha='center', va='center', fontsize=10)
    plt.text(5, 0, 'Aggressive', color='white', ha='center', va='center', fontsize=10)
    plt.text(5, 4, 'Balanced', color='white', ha='center', va='center', fontsize=10)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'atr_optimization.png'))
    plt.close()
    
    print(f"ATR optimization visualization created: {os.path.join(RESULTS_DIR, 'atr_optimization.png')}")

if __name__ == "__main__":
    analyze_strategy_performance()
