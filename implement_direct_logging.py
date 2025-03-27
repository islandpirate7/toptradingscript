import os
import re
from datetime import datetime

def implement_direct_logging():
    """
    Implement the direct logging approach in the run_backtest function
    of final_sp500_strategy.py based on our successful test.
    """
    file_path = 'final_sp500_strategy.py'
    
    # Read the file content
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Find the run_backtest function
    run_backtest_start = content.find("def run_backtest(")
    if run_backtest_start == -1:
        print("ERROR: Could not find the run_backtest function")
        return
    
    # Find the end of the function signature
    sig_end = content.find(":", run_backtest_start)
    if sig_end == -1:
        print("ERROR: Could not find the end of the function signature")
        return
    
    # Extract the function signature
    func_sig = content[run_backtest_start:sig_end+1]
    
    # Create a new implementation of the run_backtest function
    # that uses our direct file writing approach
    new_run_backtest = f"""{func_sig}
    \"\"\"
    Run a backtest for the specified period using direct file writing for logging.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        mode (str): 'backtest' or 'live'
        initial_capital (float): Initial capital for the backtest
        random_seed (int): Random seed for reproducibility
        continuous_capital (bool): Whether to use continuous capital from previous runs
        previous_capital (float): Previous capital to start with if continuous_capital is True
        config_path (str): Path to the configuration file
        max_signals (int): Maximum number of signals to use
        min_score (float): Minimum score for a signal to be considered
        tier1_threshold (float): Threshold for Tier 1 signals
        tier2_threshold (float): Threshold for Tier 2 signals
        tier3_threshold (float): Threshold for Tier 3 signals
        largecap_allocation (float): Allocation for large-cap stocks (0-1)
        midcap_allocation (float): Allocation for mid-cap stocks (0-1)
    
    Returns:
        dict: Results of the backtest
    \"\"\"
    try:
        # Set random seed for reproducibility
        if random_seed is not None:
            random.seed(random_seed)
        
        # Load configuration
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Set up direct file writing for this backtest run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)
        
        # Create log file path
        log_file = os.path.join('logs', f"strategy_{{{timestamp}}}.log")
        
        # Open the log file for writing
        log_file_handle = open(log_file, 'w')
        
        try:
            # Log the start of the backtest
            log_file_handle.write(f"{{datetime.now()}} - INFO - Starting backtest from {{start_date}} to {{end_date}}\\n")
            log_file_handle.write(f"{{datetime.now()}} - INFO - Backtest log file created: {{log_file}}\\n")
            log_file_handle.write(f"{{datetime.now()}} - INFO - Initial capital: ${{initial_capital}}\\n")
            log_file_handle.write(f"{{datetime.now()}} - INFO - Backtest mode: {{mode}}\\n")
            log_file_handle.write(f"{{datetime.now()}} - INFO - Random seed: {{random_seed}}\\n")
            log_file_handle.write(f"{{datetime.now()}} - INFO - Continuous capital: {{continuous_capital}}\\n")
            
            if continuous_capital and previous_capital is not None:
                log_file_handle.write(f"{{datetime.now()}} - INFO - Previous capital: ${{previous_capital}}\\n")
            
            log_file_handle.write(f"{{datetime.now()}} - INFO - Running backtest from {{start_date}} to {{end_date}} with initial capital ${{initial_capital}} (Seed: {{random_seed}})\\n")
            log_file_handle.flush()
            os.fsync(log_file_handle.fileno())
            
            # Create output directories if they don't exist
            for path_key in ['backtest_results', 'plots', 'trades', 'performance']:
                os.makedirs(config['paths'][path_key], exist_ok=True)
            
            # Load Alpaca credentials
            alpaca = AlpacaAPI(
                api_key=config['alpaca']['api_key'],
                api_secret=config['alpaca']['api_secret'],
                base_url=config['alpaca']['base_url']
            )
            
            # Generate signals for the specified date range
            signals = generate_signals(start_date, end_date, config)
            
            # Split signals into large-cap and mid-cap
            largecap_signals = [s for s in signals if s['symbol'] in get_sp500_symbols()]
            midcap_signals = [s for s in signals if s['symbol'] in get_midcap_symbols()]
            
            log_file_handle.write(f"{{datetime.now()}} - INFO - Generated {{len(signals)}} total signals: {{len(largecap_signals)}} large-cap, {{len(midcap_signals)}} mid-cap\\n")
            log_file_handle.flush()
            os.fsync(log_file_handle.fileno())
            
            # Ensure a balanced mix of LONG trades
            if max_signals is not None and len(signals) > max_signals:
                log_file_handle.write(f"{{datetime.now()}} - INFO - Limiting signals to top {{max_signals}} (from {{len(signals)}})\\n")
                log_file_handle.flush()
                os.fsync(log_file_handle.fileno())
                
                # Calculate how many large-cap and mid-cap signals to include
                max_largecap = int(max_signals * largecap_allocation)
                max_midcap = max_signals - max_largecap
                
                # Sort signals by score
                largecap_signals = sorted(largecap_signals, key=lambda x: x['score'], reverse=True)
                midcap_signals = sorted(midcap_signals, key=lambda x: x['score'], reverse=True)
                
                # Select top signals from each category
                selected_large_cap = largecap_signals[:max_largecap]
                selected_mid_cap = midcap_signals[:max_midcap]
                
                # Combine and sort by score
                signals = selected_large_cap + selected_mid_cap
                signals = sorted(signals, key=lambda x: (x['score'], x['symbol']), reverse=True)
                
                log_file_handle.write(f"{{datetime.now()}} - INFO - Final signals: {{len(signals)}}\\n")
                log_file_handle.flush()
                os.fsync(log_file_handle.fileno())
            else:
                # If no max_signals specified or we have fewer signals than max, still log the signal count
                # Sort signals deterministically by score and then by symbol (for tiebreaking)
                signals = sorted(signals, key=lambda x: (x['score'], x['symbol']), reverse=True)
                log_file_handle.write(f"{{datetime.now()}} - INFO - Using all {{len(signals)}}\\n")
                log_file_handle.flush()
                os.fsync(log_file_handle.fileno())
            
            # Initialize portfolio
            portfolio = Portfolio(
                initial_capital=initial_capital if not continuous_capital else previous_capital,
                cash_allocation=config['portfolio']['cash_allocation'],
                max_positions=config['portfolio']['max_positions'],
                position_size=config['portfolio']['position_size'],
                stop_loss=config['portfolio']['stop_loss'],
                take_profit=config['portfolio']['take_profit']
            )
            
            # Set thresholds if provided
            if tier1_threshold is not None:
                portfolio.tier1_threshold = tier1_threshold
            if tier2_threshold is not None:
                portfolio.tier2_threshold = tier2_threshold
            if tier3_threshold is not None:
                portfolio.tier3_threshold = tier3_threshold
            
            # Log threshold values
            log_file_handle.write(f"{{datetime.now()}} - INFO - Tier 1 threshold: {{portfolio.tier1_threshold}}\\n")
            log_file_handle.write(f"{{datetime.now()}} - INFO - Tier 2 threshold: {{portfolio.tier2_threshold}}\\n")
            log_file_handle.write(f"{{datetime.now()}} - INFO - Tier 3 threshold: {{portfolio.tier3_threshold}}\\n")
            log_file_handle.flush()
            os.fsync(log_file_handle.fileno())
            
            # Process signals
            for signal in signals:
                # Skip signals below the minimum score threshold
                if min_score is not None and signal['score'] < min_score:
                    log_file_handle.write(f"{{datetime.now()}} - INFO - Skipping trade for {{signal['symbol']}} with score {{signal['score']:.2f}} - below minimum score threshold\\n")
                    log_file_handle.flush()
                    os.fsync(log_file_handle.fileno())
                    continue
                
                # Determine the tier based on score
                if signal['score'] >= portfolio.tier1_threshold:
                    tier = 1
                elif signal['score'] >= portfolio.tier2_threshold:
                    tier = 2
                elif signal['score'] >= portfolio.tier3_threshold:
                    tier = 3
                else:
                    log_file_handle.write(f"{{datetime.now()}} - INFO - Skipping trade for {{signal['symbol']}} with score {{signal['score']:.2f}} - below Tier 3 threshold\\n")
                    log_file_handle.flush()
                    os.fsync(log_file_handle.fileno())
                    continue
                
                # Execute the trade
                trade_result = portfolio.execute_trade(signal, tier=tier)
                
                if trade_result['success']:
                    log_file_handle.write(f"{{datetime.now()}} - INFO - Executed {{trade_result['action']}} trade for {{signal['symbol']}} with score {{signal['score']:.2f}} (Tier {{tier}})\\n")
                    log_file_handle.flush()
                    os.fsync(log_file_handle.fileno())
                else:
                    log_file_handle.write(f"{{datetime.now()}} - INFO - Failed to execute trade for {{signal['symbol']}}: {{trade_result['message']}}\\n")
                    log_file_handle.flush()
                    os.fsync(log_file_handle.fileno())
            
            # Calculate performance metrics
            performance = portfolio.calculate_performance()
            
            # Log performance
            log_file_handle.write(f"{{datetime.now()}} - INFO - Backtest completed\\n")
            log_file_handle.write(f"{{datetime.now()}} - INFO - Final portfolio value: ${{performance['final_value']:.2f}}\\n")
            log_file_handle.write(f"{{datetime.now()}} - INFO - Return: {{performance['return']:.2f}}%\\n")
            log_file_handle.write(f"{{datetime.now()}} - INFO - Annualized return: {{performance['annualized_return']:.2f}}%\\n")
            log_file_handle.write(f"{{datetime.now()}} - INFO - Sharpe ratio: {{performance['sharpe_ratio']:.2f}}\\n")
            log_file_handle.write(f"{{datetime.now()}} - INFO - Max drawdown: {{performance['max_drawdown']:.2f}}%\\n")
            log_file_handle.write(f"{{datetime.now()}} - INFO - Win rate: {{performance['win_rate']:.2f}}%\\n")
            log_file_handle.flush()
            os.fsync(log_file_handle.fileno())
            
            # Save results
            results = {{
                'portfolio': portfolio,
                'performance': performance,
                'signals': signals,
                'log_file': log_file
            }}
            
            return results
        
        finally:
            # Make sure to close the log file handle
            if 'log_file_handle' in locals() and log_file_handle:
                log_file_handle.write(f"{{datetime.now()}} - INFO - Closing log file\\n")
                log_file_handle.flush()
                os.fsync(log_file_handle.fileno())
                log_file_handle.close()
    
    except Exception as e:
        print(f"Error in run_backtest: {{str(e)}}")
        import traceback
        traceback.print_exc()
        return {{'success': False, 'error': str(e)}}
"""
    
    # Find where to end the replacement
    # Look for the next function definition after run_backtest
    next_def = content.find("\ndef ", run_backtest_start + 1)
    if next_def == -1:
        print("ERROR: Could not find the next function definition")
        return
    
    # Create a backup of the original file
    backup_file = f"{file_path}.bak"
    with open(backup_file, 'w') as file:
        file.write(content)
    print(f"Created backup of original file: {backup_file}")
    
    # Replace the run_backtest function with our new implementation
    new_content = content[:run_backtest_start] + new_run_backtest + content[next_def:]
    
    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.write(new_content)
    
    print("Successfully implemented direct logging in the run_backtest function")
    print("This version uses direct file writing with explicit flush and fsync calls")
    print("Please run a backtest to verify that the logs are now being properly created and populated")

if __name__ == "__main__":
    implement_direct_logging()
