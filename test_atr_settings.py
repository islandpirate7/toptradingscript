import subprocess
import yaml
import os
import datetime

# Define the ATR combinations to test
atr_combinations = [
    {"stop_loss": 1.5, "take_profit": 3.0, "name": "Conservative"},
    {"stop_loss": 2.0, "take_profit": 3.0, "name": "Balanced"},
    {"stop_loss": 2.0, "take_profit": 4.0, "name": "Aggressive"},
    {"stop_loss": 2.5, "take_profit": 3.5, "name": "Wide Stop"},
    {"stop_loss": 1.5, "take_profit": 2.5, "name": "Tight Range"}
]

# Define market periods to test
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

# Load the base configuration
with open('configuration_11.yaml', 'r') as file:
    base_config = yaml.safe_load(file)

# Create results directory
results_dir = "atr_test_results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Function to modify the configuration with specific ATR settings
def create_config_with_atr(stop_loss_atr, take_profit_atr):
    # Create a copy of the base config
    config = base_config.copy()
    
    # Ensure strategies section exists
    if 'strategies' not in config:
        config['strategies'] = {}
    
    # Ensure MeanReversion section exists
    if 'MeanReversion' not in config['strategies']:
        config['strategies']['MeanReversion'] = {}
    
    # Set ATR multipliers
    config['strategies']['MeanReversion']['stop_loss_atr'] = stop_loss_atr
    config['strategies']['MeanReversion']['take_profit_atr'] = take_profit_atr
    
    return config

# Function to run a backtest with specific settings
def run_backtest(config, start_date, end_date, log_file):
    # Create a temporary config file
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
        "--log_level", "INFO"
    ]
    
    # Redirect output to a log file
    with open(log_file, 'w') as f:
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

# Main function to run all tests
def run_atr_tests():
    print(f"Starting ATR multiplier tests at {datetime.datetime.now()}")
    
    for period in market_periods:
        print(f"\nTesting period: {period['name']} ({period['start_date']} to {period['end_date']})")
        
        for atr_combo in atr_combinations:
            print(f"  Testing ATR combination: {atr_combo['name']} (SL: {atr_combo['stop_loss']}, TP: {atr_combo['take_profit']})")
            
            # Create config with this ATR setting
            config = create_config_with_atr(atr_combo['stop_loss'], atr_combo['take_profit'])
            
            # Define log file
            log_file = f"{results_dir}/{period['name']}_{atr_combo['name']}.log"
            
            # Run backtest
            run_backtest(
                config,
                period['start_date'],
                period['end_date'],
                log_file
            )
            
            print(f"  Completed test, results saved to {log_file}")
    
    print(f"\nAll tests completed at {datetime.datetime.now()}")
    print(f"Results saved to {results_dir} directory")

if __name__ == "__main__":
    run_atr_tests()
