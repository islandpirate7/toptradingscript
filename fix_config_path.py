import os
import json
import yaml
import logging
import traceback
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_backtest_results():
    """
    Checks for backtest result files with null summaries and fixes them by creating
    valid summary data.
    """
    try:
        # Get the backtest_results directory
        backtest_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_results")
        
        if not os.path.exists(backtest_dir):
            logger.error(f"Backtest directory {backtest_dir} does not exist")
            return
        
        # Get all JSON files in the directory
        json_files = [f for f in os.listdir(backtest_dir) if f.endswith('.json')]
        
        if not json_files:
            logger.info("No JSON files found in backtest_results directory")
            return
        
        fixed_count = 0
        
        for json_file in json_files:
            file_path = os.path.join(backtest_dir, json_file)
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Check if summary is missing or null
                needs_fixing = False
                
                if 'summary' not in data:
                    logger.info(f"File missing summary key: {json_file}")
                    data['summary'] = None
                    needs_fixing = True
                elif data['summary'] is None:
                    logger.info(f"Found file with null summary: {json_file}")
                    needs_fixing = True
                
                if needs_fixing:
                    # Extract information from the filename
                    # Format: backtest_Q2_2023_2023-04-01_to_2023-06-30_20250324_040107.json
                    parts = json_file.split('_')
                    
                    # Default values
                    quarter = "Unknown"
                    year_part = "Unknown"
                    start_date = ""
                    end_date = ""
                    
                    # Try to extract quarter and year if available
                    if len(parts) > 2 and parts[0] == "backtest":
                        quarter = parts[1]
                        if len(parts) > 2:
                            year_part = parts[2].split('.')[0]  # Remove .json if it's there
                    
                    # Extract dates if available
                    for i, part in enumerate(parts):
                        if part == 'to' and i > 0 and i < len(parts) - 1:
                            start_date = parts[i-1]
                            end_date = parts[i+1]
                    
                    # Create a valid summary
                    valid_summary = {
                        'quarter': f"{quarter}_{year_part}" if year_part != "Unknown" else quarter,
                        'start_date': start_date if start_date else '',
                        'end_date': end_date if end_date else '',
                        'error': 'Backtest execution failed to return valid results - fixed by fix_config_path.py',
                        'win_rate': 0,
                        'profit_factor': 0,
                        'total_return': 0,
                        'initial_capital': 300,
                        'final_capital': 300,
                        'total_signals': 0,
                        'long_signals': 0,
                        'fixed_timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
                    }
                    
                    # Update the data
                    data['summary'] = valid_summary
                    
                    # Write back to the file
                    with open(file_path, 'w') as f:
                        json.dump(data, f, default=str)
                    
                    logger.info(f"Fixed summary for {json_file}")
                    fixed_count += 1
            
            except Exception as e:
                logger.error(f"Error processing {json_file}: {str(e)}")
                logger.error(traceback.format_exc())
        
        logger.info(f"Fixed {fixed_count} files with null summaries")
    
    except Exception as e:
        logger.error(f"Error in fix_backtest_results: {str(e)}")
        logger.error(traceback.format_exc())

def fix_config_path():
    """
    Updates the sp500_config.yaml path in run_comprehensive_backtest.py to use an absolute path.
    """
    try:
        # Check if sp500_config.yaml exists
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sp500_config.yaml")
        
        if not os.path.exists(config_path):
            logger.error(f"sp500_config.yaml not found at {config_path}")
            
            # Create a basic config file if it doesn't exist
            with open(config_path, 'w') as f:
                yaml.dump({
                    'paths': {
                        'backtest_results': './backtest_results',
                        'plots': './plots',
                        'trades': './trades',
                        'performance': './performance'
                    },
                    'strategy': {
                        'max_trades_per_run': 40,
                        'position_sizing': {
                            'base_position_pct': 5
                        },
                        'midcap_stocks': {
                            'large_cap_percentage': 70,
                            'position_factor': 0.8
                        }
                    }
                }, f)
            
            logger.info(f"Created basic sp500_config.yaml at {config_path}")
    
    except Exception as e:
        logger.error(f"Error in fix_config_path: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    logger.info("Starting fix_config_path.py")
    fix_config_path()
    fix_backtest_results()
    logger.info("Completed fix_config_path.py")
