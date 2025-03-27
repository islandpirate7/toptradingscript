import pandas as pd
import os
from tabulate import tabulate

# Define the quarters and their corresponding result files
quarters = [
    ('Q1 2023', 'backtest_results/backtest_results_2023-01-01_to_2023-03-31_20250323_171306.csv'),
    ('Q2 2023', 'backtest_results/backtest_results_2023-04-01_to_2023-06-30_20250323_171343.csv'),
    ('Q3 2023', 'backtest_results/backtest_results_2023-07-01_to_2023-09-30_20250323_171419.csv'),
    ('Q4 2023', 'backtest_results/backtest_results_2023-10-01_to_2023-12-31_20250323_171456.csv')
]

# Function to analyze a quarter's results
def analyze_quarter(quarter_name, file_path):
    print(f"\n{'='*50}")
    print(f"{quarter_name} ANALYSIS")
    print(f"{'='*50}")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    try:
        df = pd.read_csv(file_path)
        
        # Basic statistics
        print(f"\nTotal Signals: {len(df)}")
        print(f"Average Score: {df['score'].mean():.2f}")
        
        # Top 10 stocks by score
        print("\nTop 10 Stocks by Score:")
        top_stocks = df[['symbol', 'score', 'sector']].sort_values('score', ascending=False).head(10)
        print(tabulate(top_stocks, headers='keys', tablefmt='pretty', showindex=False))
        
        # Sector distribution
        print("\nSector Distribution:")
        sector_counts = df['sector'].value_counts()
        sector_data = []
        for sector, count in sector_counts.items():
            sector_data.append([sector, count, f"{count/len(df)*100:.1f}%"])
        print(tabulate(sector_data, headers=['Sector', 'Count', 'Percentage'], tablefmt='pretty', showindex=False))
        
        # Score distribution
        print("\nScore Distribution:")
        score_bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        score_labels = ['0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
        df['score_bin'] = pd.cut(df['score'], bins=score_bins, labels=score_labels, right=True)
        score_counts = df['score_bin'].value_counts().sort_index()
        score_data = []
        for score_range, count in score_counts.items():
            score_data.append([score_range, count, f"{count/len(df)*100:.1f}%"])
        print(tabulate(score_data, headers=['Score Range', 'Count', 'Percentage'], tablefmt='pretty', showindex=False))
        
    except Exception as e:
        print(f"Error analyzing {quarter_name}: {str(e)}")

# Analyze each quarter
for quarter_name, file_path in quarters:
    analyze_quarter(quarter_name, file_path)

print("\n\nDone analyzing all quarters!")
