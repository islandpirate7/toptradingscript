import pandas as pd
import os
import re
import glob
from tabulate import tabulate

# Find the most recent backtest results files for each quarter
def find_latest_results():
    quarters = {
        'Q1': 'backtest_results_2023-01-01_to_2023-03-31_*.csv',
        'Q2': 'backtest_results_2023-04-01_to_2023-06-30_*.csv',
        'Q3': 'backtest_results_2023-07-01_to_2023-09-30_*.csv',
        'Q4': 'backtest_results_2023-10-01_to_2023-12-31_*.csv'
    }
    
    result_files = {}
    for quarter, pattern in quarters.items():
        files = glob.glob(f'backtest_results/{pattern}')
        if files:
            # Sort by timestamp in filename (most recent last)
            files.sort(key=lambda x: os.path.getmtime(x))
            result_files[quarter] = files[-1]  # Get the most recent file
    
    return result_files

# Extract performance metrics from the backtest log
def extract_performance_metrics():
    log_file = 'output.log'
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return {}
    
    with open(log_file, 'r') as f:
        log_content = f.read()
    
    # Define patterns to extract metrics for each quarter
    quarters = ['Q1_2023', 'Q2_2023', 'Q3_2023', 'Q4_2023']
    metrics = {}
    
    for quarter in quarters:
        # Find the section for this quarter
        quarter_pattern = f"Running backtest for {quarter}: .*? to .*?\n(.*?)(?=Running backtest for|$)"
        quarter_match = re.search(quarter_pattern, log_content, re.DOTALL)
        
        if not quarter_match:
            continue
        
        quarter_content = quarter_match.group(1)
        
        # Extract metrics
        metrics[quarter] = {
            'Win Rate': re.search(r'Win Rate: ([\d.]+)%', quarter_content),
            'Profit Factor': re.search(r'Profit Factor: ([\d.]+)', quarter_content),
            'Average Win': re.search(r'Average Win: \$([\d.]+)', quarter_content),
            'Average Loss': re.search(r'Average Loss: \$([\-\d.]+)', quarter_content),
            'Average Holding Period': re.search(r'Average Holding Period: ([\d.]+) days', quarter_content),
            'Initial Capital': re.search(r'Initial Capital: \$([\d.]+)', quarter_content),
            'Final Capital': re.search(r'Final Capital: \$([\d.]+)', quarter_content),
            'Total Return': re.search(r'Total Return: ([\d.]+)%', quarter_content)
        }
        
        # Convert matches to values
        for key, match in metrics[quarter].items():
            if match:
                metrics[quarter][key] = match.group(1)
            else:
                metrics[quarter][key] = 'N/A'
    
    return metrics

# Extract performance metrics from the CSV files
def extract_metrics_from_csv():
    result_files = find_latest_results()
    metrics = {}
    
    for quarter, file_path in result_files.items():
        try:
            df = pd.read_csv(file_path)
            
            # Calculate performance metrics
            total_signals = len(df)
            avg_score = df['score'].mean()
            
            metrics[quarter] = {
                'Total Signals': total_signals,
                'Average Score': f"{avg_score:.2f}",
                'Top Stocks': ', '.join(df.sort_values('score', ascending=False)['symbol'].head(5).tolist())
            }
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            metrics[quarter] = {'Error': str(e)}
    
    return metrics

# Main function
def main():
    print("Analyzing performance metrics for 2023 backtests...\n")
    
    # Get metrics from log file
    log_metrics = extract_performance_metrics()
    
    # Get metrics from CSV files
    csv_metrics = extract_metrics_from_csv()
    
    # Combine metrics
    combined_metrics = {}
    for quarter in ['Q1_2023', 'Q2_2023', 'Q3_2023', 'Q4_2023']:
        q_short = quarter.split('_')[0]
        combined_metrics[quarter] = {}
        
        # Add metrics from CSV
        if q_short in csv_metrics:
            combined_metrics[quarter].update(csv_metrics[q_short])
        
        # Add metrics from log
        if quarter in log_metrics:
            combined_metrics[quarter].update(log_metrics[quarter])
    
    # Display results in a table
    table_data = []
    headers = ['Metric', 'Q1 2023', 'Q2 2023', 'Q3 2023', 'Q4 2023']
    
    # Define the metrics to display and their order
    metrics_to_display = [
        'Total Signals', 'Average Score', 'Win Rate', 'Profit Factor',
        'Average Win', 'Average Loss', 'Average Holding Period',
        'Initial Capital', 'Final Capital', 'Total Return', 'Top Stocks'
    ]
    
    for metric in metrics_to_display:
        row = [metric]
        for quarter in ['Q1_2023', 'Q2_2023', 'Q3_2023', 'Q4_2023']:
            if quarter in combined_metrics and metric in combined_metrics[quarter]:
                row.append(combined_metrics[quarter][metric])
            else:
                row.append('N/A')
        table_data.append(row)
    
    print(tabulate(table_data, headers=headers, tablefmt='pretty'))
    
    # Summary
    print("\nSUMMARY OF BACKTEST RESULTS FOR 2023")
    print("=" * 50)
    
    # Calculate overall performance if we have all quarters
    if all(f'Q{i}_2023' in combined_metrics for i in range(1, 5)):
        try:
            initial_capital = float(combined_metrics['Q1_2023'].get('Initial Capital', 10000))
            final_capitals = []
            for i in range(1, 5):
                quarter = f'Q{i}_2023'
                if 'Final Capital' in combined_metrics[quarter]:
                    final_capitals.append(float(combined_metrics[quarter]['Final Capital']))
            
            if final_capitals:
                overall_return = (final_capitals[-1] / initial_capital - 1) * 100
                print(f"Initial Capital: ${initial_capital:.2f}")
                print(f"Final Capital: ${final_capitals[-1]:.2f}")
                print(f"Overall Return: {overall_return:.2f}%")
                
                # Calculate quarterly returns
                print("\nQuarterly Returns:")
                prev_capital = initial_capital
                for i, final_cap in enumerate(final_capitals):
                    quarterly_return = (final_cap / prev_capital - 1) * 100
                    print(f"Q{i+1} 2023: {quarterly_return:.2f}%")
                    prev_capital = final_cap
        except Exception as e:
            print(f"Error calculating overall performance: {str(e)}")

if __name__ == "__main__":
    main()
