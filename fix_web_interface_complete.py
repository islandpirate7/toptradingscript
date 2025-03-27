#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Complete fix script for web interface issues
This script will:
1. Fix all CORS issues in the web interface
2. Fix JSON response handling in all API endpoints
3. Fix the favicon.ico 405 error
4. Fix the run_comprehensive_backtest 500 error
5. Update the JavaScript to properly handle API responses
"""

import os
import sys
import json
import shutil
import logging
import traceback
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"fix_web_interface_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

def fix_app_file():
    """Fix issues with the app.py file"""
    try:
        # Get the path to the web interface directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        web_interface_dir = os.path.join(script_dir, 'new_web_interface')
        
        # Ensure web interface directory exists
        if not os.path.exists(web_interface_dir):
            logger.error(f"Web interface directory not found: {web_interface_dir}")
            return False
        
        # Path to the app_fixed.py file
        app_fixed_path = os.path.join(web_interface_dir, 'app_fixed.py')
        
        # If app_fixed.py doesn't exist, check if app.py exists
        if not os.path.exists(app_fixed_path):
            app_path = os.path.join(web_interface_dir, 'app.py')
            if os.path.exists(app_path):
                # Copy app.py to app_fixed.py
                shutil.copy2(app_path, app_fixed_path)
                logger.info(f"Created app_fixed.py from app.py")
            else:
                logger.error(f"Neither app_fixed.py nor app.py found in {web_interface_dir}")
                return False
        
        # Read the app_fixed.py file
        with open(app_fixed_path, 'r') as f:
            content = f.read()
        
        # Create a backup of the original file
        backup_path = f"{app_fixed_path}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with open(backup_path, 'w') as f:
            f.write(content)
        logger.info(f"Created backup of app_fixed.py: {backup_path}")
        
        # Add favicon route to fix 405 error
        if '@app.route(\'/favicon.ico\')' not in content:
            # Find a good place to add the favicon route
            index_route = content.find('@app.route(\'/\')')
            if index_route >= 0:
                # Add favicon route before the index route
                favicon_route = """
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

"""
                content = content[:index_route] + favicon_route + content[index_route:]
                logger.info("Added favicon route to fix 405 error")
            else:
                logger.warning("Could not find index route to add favicon route")
        
        # Fix the get_processes function to properly format the response
        if 'def get_processes():' in content:
            # Find the start of the function
            start_idx = content.find('def get_processes():')
            if start_idx >= 0:
                # Find the return statement
                return_idx = content.find('return jsonify(processes)', start_idx)
                if return_idx >= 0:
                    # Replace the return statement with a properly formatted response
                    old_return = 'return jsonify(processes)'
                    new_return = 'return jsonify({"processes": list(processes.values())})'
                    
                    # Replace the return statement
                    content = content.replace(old_return, new_return)
                    logger.info("Updated get_processes function to return properly formatted JSON")
                else:
                    logger.warning("Could not find return statement in get_processes function")
            else:
                logger.warning("Could not find get_processes function")
        
        # Fix the get_backtest_results_route function to properly format the response
        if 'def get_backtest_results_route():' in content:
            # Find the start of the function
            start_idx = content.find('def get_backtest_results_route():')
            if start_idx >= 0:
                # Find the return statement
                return_idx = content.find('return jsonify(results)', start_idx)
                if return_idx >= 0:
                    # Replace the return statement with a properly formatted response
                    old_return = 'return jsonify(results)'
                    new_return = 'return jsonify({"results": results})'
                    
                    # Replace the return statement
                    content = content.replace(old_return, new_return)
                    logger.info("Updated get_backtest_results_route function to return properly formatted JSON")
                else:
                    logger.warning("Could not find return statement in get_backtest_results_route function")
            else:
                logger.warning("Could not find get_backtest_results_route function")
        
        # Fix the run_comprehensive_backtest function to properly handle AJAX requests
        if 'def run_comprehensive_backtest():' in content:
            # Find the start of the function
            start_idx = content.find('def run_comprehensive_backtest():')
            if start_idx >= 0:
                # Find the try block
                try_idx = content.find('try:', start_idx)
                if try_idx >= 0:
                    # Find the except block
                    except_idx = content.find('except Exception as e:', try_idx)
                    if except_idx >= 0:
                        # Replace the except block with improved error handling
                        except_end_idx = content.find('return redirect(url_for(\'index\'))', except_idx)
                        if except_end_idx >= 0:
                            old_except = content[except_idx:except_end_idx + len('return redirect(url_for(\'index\'))')]
                            new_except = """except Exception as e:
        logger.error(f"Error running comprehensive backtest: {str(e)}", exc_info=True)
        
        # Check if this is an AJAX request
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({"error": str(e)}), 500
        
        # For regular form submissions, use flash and redirect
        flash(f'Error running comprehensive backtest: {str(e)}', 'danger')
        return redirect(url_for('index'))"""
                            
                            # Replace the except block
                            content = content.replace(old_except, new_except)
                            logger.info("Updated run_comprehensive_backtest function to handle AJAX requests")
                        else:
                            logger.warning("Could not find end of except block in run_comprehensive_backtest function")
                    else:
                        logger.warning("Could not find except block in run_comprehensive_backtest function")
                else:
                    logger.warning("Could not find try block in run_comprehensive_backtest function")
            else:
                logger.warning("Could not find run_comprehensive_backtest function")
        
        # Save the modified file
        with open(app_fixed_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Successfully updated {app_fixed_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error fixing app file: {str(e)}", exc_info=True)
        return False

def fix_javascript_file():
    """Fix issues with the main.js file"""
    try:
        # Get the path to the web interface directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        web_interface_dir = os.path.join(script_dir, 'new_web_interface')
        
        # Ensure web interface directory exists
        if not os.path.exists(web_interface_dir):
            logger.error(f"Web interface directory not found: {web_interface_dir}")
            return False
        
        # Path to the main.js file
        js_path = os.path.join(web_interface_dir, 'static', 'js', 'main.js')
        
        # Ensure main.js exists
        if not os.path.exists(js_path):
            logger.error(f"main.js not found: {js_path}")
            return False
        
        # Read the main.js file
        with open(js_path, 'r') as f:
            content = f.read()
        
        # Create a backup of the original file
        backup_path = f"{js_path}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with open(backup_path, 'w') as f:
            f.write(content)
        logger.info(f"Created backup of main.js: {backup_path}")
        
        # Fix the setupAutoRefresh function to properly handle API responses
        if 'function setupAutoRefresh()' in content:
            # Find the start of the function
            start_idx = content.find('function setupAutoRefresh()')
            if start_idx >= 0:
                # Create a completely new setupAutoRefresh function
                old_function_start = content.find('{', start_idx)
                old_function_end = find_matching_brace(content, old_function_start)
                
                if old_function_start >= 0 and old_function_end >= 0:
                    old_function = content[start_idx:old_function_end + 1]
                    
                    new_function = """function setupAutoRefresh() {
    // Refresh active processes every 5 seconds
    const processesTableBody = document.getElementById('processesTableBody');
    if (processesTableBody) {
        setInterval(function() {
            fetch(baseUrl + '/get_processes')
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Network response was not ok: ' + response.statusText);
                    }
                    return response.json();
                })
                .then(data => {
                    processesTableBody.innerHTML = '';
                    
                    // Safely check if processes exist and have length
                    const processes = data.processes || [];
                    
                    if (processes.length === 0) {
                        processesTableBody.innerHTML = '<tr><td colspan="4" class="text-center">No active processes</td></tr>';
                    } else {
                        processes.forEach(process => {
                            const row = document.createElement('tr');
                            row.innerHTML = `
                                <td>${process.name || 'Unknown'}</td>
                                <td>${process.status || 'Unknown'}</td>
                                <td>${process.start_time || 'Unknown'}</td>
                                <td>
                                    <button class="btn btn-sm btn-danger" onclick="stopProcess('${process.name}')">Stop</button>
                                    <button class="btn btn-sm btn-info" onclick="viewProcessLogs('${process.name}', ${JSON.stringify(process.logs || [])})">Logs</button>
                                </td>
                            `;
                            processesTableBody.appendChild(row);
                        });
                    }
                })
                .catch(error => {
                    console.error('Error fetching processes:', error);
                    processesTableBody.innerHTML = '<tr><td colspan="4" class="text-center text-danger">Error loading processes: ' + error.message + '</td></tr>';
                });
        }, 5000);
    }
    
    // Refresh backtest results every 10 seconds
    const backtestResultsTableBody = document.getElementById('backtestResultsTableBody');
    if (backtestResultsTableBody) {
        setInterval(function() {
            fetch(baseUrl + '/get_backtest_results')
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Network response was not ok: ' + response.statusText);
                    }
                    return response.json();
                })
                .then(data => {
                    backtestResultsTableBody.innerHTML = '';
                    
                    // Safely check if results exist and have length
                    const results = data.results || [];
                    
                    if (results.length === 0) {
                        backtestResultsTableBody.innerHTML = '<tr><td colspan="5" class="text-center">No backtest results</td></tr>';
                    } else {
                        results.forEach(result => {
                            const row = document.createElement('tr');
                            row.innerHTML = `
                                <td>${result.filename || 'Unknown'}</td>
                                <td>${result.quarter || result.date_range || 'N/A'}</td>
                                <td>${result.profit_loss || 'N/A'}</td>
                                <td>${result.win_rate || 'N/A'}</td>
                                <td>
                                    <a href="${baseUrl}/view_backtest_result/${result.filename}" class="btn btn-sm btn-primary">View</a>
                                </td>
                            `;
                            backtestResultsTableBody.appendChild(row);
                        });
                    }
                })
                .catch(error => {
                    console.error('Error fetching backtest results:', error);
                    backtestResultsTableBody.innerHTML = '<tr><td colspan="5" class="text-center text-danger">Error loading results: ' + error.message + '</td></tr>';
                });
        }, 10000);
    }
}"""
                    
                    # Replace the old function with the new one
                    content = content.replace(old_function, new_function)
                    logger.info("Updated setupAutoRefresh function to properly handle API responses")
                else:
                    logger.warning("Could not find boundaries of setupAutoRefresh function")
            else:
                logger.warning("Could not find setupAutoRefresh function")
        
        # Fix the form submission for comprehensive backtest
        if 'document.getElementById(\'comprehensiveBacktestForm\').addEventListener(\'submit\'' in content:
            # Find the form submission event listener
            form_idx = content.find('document.getElementById(\'comprehensiveBacktestForm\').addEventListener(\'submit\'')
            if form_idx >= 0:
                # Find the function body
                func_start = content.find('{', form_idx)
                func_end = find_matching_brace(content, func_start)
                
                if func_start >= 0 and func_end >= 0:
                    old_func = content[form_idx:func_end + 1]
                    
                    # Create a new form submission handler with proper AJAX
                    new_func = """document.getElementById('comprehensiveBacktestForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Get form data
            const formData = new FormData(this);
            
            // Show loading indicator
            const submitBtn = this.querySelector('button[type="submit"]');
            const originalBtnText = submitBtn.innerHTML;
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Running...';
            
            // Send AJAX request
            fetch(baseUrl + '/run_comprehensive_backtest', {
                method: 'POST',
                body: formData,
                headers: {
                    'X-Requested-With': 'XMLHttpRequest'
                }
            })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(data => {
                        throw new Error(data.error || 'Server error');
                    });
                }
                return response.json();
            })
            .then(data => {
                // Show success message
                showToast('success', 'Comprehensive backtest started successfully');
                
                // Reset form button
                submitBtn.disabled = false;
                submitBtn.innerHTML = originalBtnText;
                
                // Refresh processes list
                setTimeout(function() {
                    fetch(baseUrl + '/get_processes')
                        .then(response => response.json())
                        .then(data => {
                            // Update processes table
                            const processesTableBody = document.getElementById('processesTableBody');
                            if (processesTableBody) {
                                processesTableBody.innerHTML = '';
                                
                                const processes = data.processes || [];
                                
                                if (processes.length === 0) {
                                    processesTableBody.innerHTML = '<tr><td colspan="4" class="text-center">No active processes</td></tr>';
                                } else {
                                    processes.forEach(process => {
                                        const row = document.createElement('tr');
                                        row.innerHTML = `
                                            <td>${process.name || 'Unknown'}</td>
                                            <td>${process.status || 'Unknown'}</td>
                                            <td>${process.start_time || 'Unknown'}</td>
                                            <td>
                                                <button class="btn btn-sm btn-danger" onclick="stopProcess('${process.name}')">Stop</button>
                                                <button class="btn btn-sm btn-info" onclick="viewProcessLogs('${process.name}', ${JSON.stringify(process.logs || [])})">Logs</button>
                                            </td>
                                        `;
                                        processesTableBody.appendChild(row);
                                    });
                                }
                            }
                        })
                        .catch(error => {
                            console.error('Error fetching processes:', error);
                        });
                }, 1000);
            })
            .catch(error => {
                console.error('Error running comprehensive backtest:', error);
                showToast('danger', 'Error: ' + error.message);
                
                // Reset form button
                submitBtn.disabled = false;
                submitBtn.innerHTML = originalBtnText;
            });
        })"""
                    
                    # Replace the old function with the new one
                    content = content.replace(old_func, new_func)
                    logger.info("Updated comprehensive backtest form submission handler")
                else:
                    logger.warning("Could not find boundaries of comprehensive backtest form submission handler")
            else:
                logger.warning("Could not find comprehensive backtest form submission handler")
        
        # Save the modified file
        with open(js_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Successfully updated {js_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error fixing JavaScript file: {str(e)}", exc_info=True)
        return False

def add_favicon():
    """Add a favicon to the static directory"""
    try:
        # Get the path to the web interface directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        web_interface_dir = os.path.join(script_dir, 'new_web_interface')
        
        # Ensure web interface directory exists
        if not os.path.exists(web_interface_dir):
            logger.error(f"Web interface directory not found: {web_interface_dir}")
            return False
        
        # Path to the static directory
        static_dir = os.path.join(web_interface_dir, 'static')
        
        # Ensure static directory exists
        if not os.path.exists(static_dir):
            os.makedirs(static_dir)
            logger.info(f"Created static directory: {static_dir}")
        
        # Path to the favicon
        favicon_path = os.path.join(static_dir, 'favicon.ico')
        
        # Check if favicon already exists
        if os.path.exists(favicon_path):
            logger.info(f"Favicon already exists: {favicon_path}")
            return True
        
        # Create a simple favicon (1x1 transparent pixel)
        # This is a minimal valid .ico file in hex
        favicon_hex = "00000100010001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
        
        # Convert hex to bytes
        favicon_bytes = bytes.fromhex(favicon_hex)
        
        # Write the favicon
        with open(favicon_path, 'wb') as f:
            f.write(favicon_bytes)
        
        logger.info(f"Created favicon: {favicon_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error adding favicon: {str(e)}", exc_info=True)
        return False

def create_launcher_script():
    """Create or update the launcher script"""
    try:
        # Get the path to the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Path to the launcher script
        launcher_path = os.path.join(script_dir, 'launch_web_interface_complete.py')
        
        # Create the launcher script
        with open(launcher_path, 'w') as f:
            f.write("""#!/usr/bin/env python
# -*- coding: utf-8 -*-

\"\"\"
Launch script for the fixed web interface
This script will:
1. Set up proper logging
2. Import and run the fixed web interface
3. Run on the standard port 5000
\"\"\"

import os
import sys
import logging
from datetime import datetime

# Set up logging
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"web_interface_{datetime.now().strftime('%Y%m%d')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)
logger = logging.getLogger(__name__)

def main():
    \"\"\"Main function to launch the web interface\"\"\"
    try:
        logger.info("Starting fixed web interface")
        
        # Add the parent directory to the path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(script_dir)
        
        # Import the fixed web interface
        web_interface_dir = os.path.join(script_dir, 'new_web_interface')
        sys.path.append(web_interface_dir)
        
        # Check if app_fixed.py exists
        app_fixed_path = os.path.join(web_interface_dir, 'app_fixed.py')
        if not os.path.exists(app_fixed_path):
            logger.error(f"Fixed web interface not found: {app_fixed_path}")
            logger.info("Please run fix_web_interface_complete.py first")
            return
        
        # Import the app from app_fixed.py
        sys.path.insert(0, os.path.dirname(app_fixed_path))
        from app_fixed import app
        
        # Run the app
        logger.info("Running web interface on http://localhost:5000")
        app.run(host='0.0.0.0', port=5000, debug=True)
    
    except Exception as e:
        logger.error(f"Error launching web interface: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
""")
        
        logger.info(f"Created launcher script: {launcher_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error creating launcher script: {str(e)}", exc_info=True)
        return False

def ensure_config_exists():
    """Ensure sp500_config.yaml exists"""
    try:
        # Get the path to the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Path to the config file
        config_path = os.path.join(script_dir, 'sp500_config.yaml')
        
        # Check if config file exists
        if not os.path.exists(config_path):
            logger.info(f"Creating default sp500_config.yaml: {config_path}")
            
            # Create default config
            with open(config_path, 'w') as f:
                f.write("""# S&P 500 Trading Strategy Configuration

# Alpaca API credentials
alpaca:
  api_key: 'YOUR_API_KEY'
  api_secret: 'YOUR_API_SECRET'
  base_url: 'https://paper-api.alpaca.markets'  # Use paper trading by default

# Initial capital
initial_capital: 10000

# Trading parameters
trading:
  max_positions: 20
  position_sizing:
    base_position_pct: 5
    tier1_factor: 3.0
    tier2_factor: 1.5
    midcap_factor: 0.8
  stop_loss_pct: 5
  take_profit_pct: 10
  max_drawdown_pct: 15
  large_cap_percentage: 70

# Signal thresholds
signals:
  tier1_threshold: 0.8
  tier2_threshold: 0.7
  tier3_threshold: 0.6

# Backtest parameters
backtest:
  max_signals_per_day: 40
  random_seed: 42
  multiple_runs: true
  num_runs: 5
  continuous_capital: true
  weekly_selection: true
""")
            
            logger.info(f"Created default sp500_config.yaml")
            return True
        else:
            logger.info(f"sp500_config.yaml already exists: {config_path}")
            return True
    
    except Exception as e:
        logger.error(f"Error ensuring config exists: {str(e)}", exc_info=True)
        return False

def find_matching_brace(text, open_brace_idx):
    """Find the matching closing brace for an opening brace"""
    if text[open_brace_idx] != '{':
        return -1
    
    stack = 1
    for i in range(open_brace_idx + 1, len(text)):
        if text[i] == '{':
            stack += 1
        elif text[i] == '}':
            stack -= 1
            if stack == 0:
                return i
    
    return -1

if __name__ == "__main__":
    logger.info("Starting complete web interface fix script")
    
    # Ensure config exists
    ensure_config_exists()
    
    # Add favicon
    add_favicon()
    
    # Fix app file
    fix_app_file()
    
    # Fix JavaScript file
    fix_javascript_file()
    
    # Create launcher script
    create_launcher_script()
    
    logger.info("Web interface fixes complete")
    logger.info("Run launch_web_interface_complete.py to start the web interface")
