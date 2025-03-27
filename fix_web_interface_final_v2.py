#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix Web Interface Final V2 Script
This script fixes all remaining issues with the web interface:
1. Fixes the favicon.ico 405 error
2. Fixes the JSON response format for get_processes and get_backtest_results
3. Fixes the run_comprehensive_backtest function to properly handle AJAX requests
4. Creates a default seasonality.yaml file to prevent warnings
5. Updates the JavaScript to properly handle API responses
"""

import os
import sys
import yaml
import json
import logging
import shutil
import re
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Root directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_INTERFACE_DIR = os.path.join(ROOT_DIR, 'new_web_interface')
STATIC_DIR = os.path.join(WEB_INTERFACE_DIR, 'static')
JS_DIR = os.path.join(STATIC_DIR, 'js')
DATA_DIR = os.path.join(ROOT_DIR, 'data')

def ensure_config_exists():
    """Ensure sp500_config.yaml exists"""
    config_file = os.path.join(ROOT_DIR, 'sp500_config.yaml')
    
    if os.path.exists(config_file):
        logger.info(f"sp500_config.yaml already exists: {config_file}")
        return
    
    # Create default config
    default_config = {
        'api_key': 'YOUR_API_KEY',
        'api_secret': 'YOUR_API_SECRET',
        'base_url': 'https://paper-api.alpaca.markets',
        'data_feed': 'iex',
        'symbols_file': 'sp500_symbols.txt',
        'max_positions': 10,
        'position_size': 0.1,
        'tier1_threshold': 0.8,
        'tier2_threshold': 0.7,
        'tier3_threshold': 0.6
    }
    
    # Save default config
    with open(config_file, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False)
    
    logger.info(f"Created default sp500_config.yaml: {config_file}")

def create_seasonality_file():
    """Create seasonality.yaml file to prevent warnings"""
    seasonality_file = os.path.join(DATA_DIR, 'seasonality.yaml')
    
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if os.path.exists(seasonality_file):
        logger.info(f"seasonality.yaml already exists: {seasonality_file}")
        return
    
    # Create default seasonality data
    default_seasonality = {
        'monthly': {
            'SPY': {
                '1': 0.01,  # January
                '2': 0.005, # February
                '3': 0.008, # March
                '4': 0.012, # April
                '5': 0.003, # May
                '6': -0.002, # June
                '7': 0.01,  # July
                '8': 0.005, # August
                '9': -0.005, # September
                '10': 0.007, # October
                '11': 0.015, # November
                '12': 0.02  # December
            }
        },
        'weekly': {
            'SPY': {
                '1': 0.002, # Monday
                '2': 0.001, # Tuesday
                '3': 0.0,   # Wednesday
                '4': 0.001, # Thursday
                '5': 0.003  # Friday
            }
        }
    }
    
    # Save default seasonality data
    with open(seasonality_file, 'w') as f:
        yaml.dump(default_seasonality, f, default_flow_style=False)
    
    logger.info(f"Created default seasonality.yaml: {seasonality_file}")

def create_favicon():
    """Create a simple favicon.ico file"""
    favicon_path = os.path.join(STATIC_DIR, 'favicon.ico')
    
    if os.path.exists(favicon_path):
        logger.info(f"favicon.ico already exists: {favicon_path}")
        return
    
    # Create a simple 16x16 favicon (1x1 pixel repeated)
    favicon_data = bytes.fromhex(
        '00000100010010100000010020006804000016000000280000001000'
        '0000200000000100200000000000000000000000000000000000000000'
        '0000000000000000000000000000000000000000000000000000000000'
        '0000000000000000000000000000000000000000000000000000000000'
        '0000000000000000000000000000000000000000000000000000000000'
        '0000000000000000000000000000000000000000000000000000000000'
        '0000000000000000000000000000000000000000000000000000000000'
        '0000000000000000000000000000000000000000000000000000000000'
        '0000000000000000000000000000000000000000000000000000000000'
        '0000000000000000000000000000000000000000000000000000000000'
        '0000000000000000000000000000000000000000000000000000000000'
        '0000000000000000000000000000000000000000000000000000000000'
        '0000000000000000000000000000000000000000000000000000000000'
        '0000000000000000000000000000000000000000000000000000000000'
        '0000000000000000000000000000000000000000000000000000000000'
        '0000000000000000000000000000000000000000000000000000000000'
    )
    
    with open(favicon_path, 'wb') as f:
        f.write(favicon_data)
    
    logger.info(f"Created favicon: {favicon_path}")

def fix_app_fixed_py():
    """Fix app_fixed.py to properly handle AJAX requests and JSON responses"""
    app_fixed_path = os.path.join(WEB_INTERFACE_DIR, 'app_fixed.py')
    
    if not os.path.exists(app_fixed_path):
        logger.error(f"app_fixed.py not found: {app_fixed_path}")
        return
    
    # Create backup
    backup_path = f"{app_fixed_path}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(app_fixed_path, backup_path)
    logger.info(f"Created backup of app_fixed.py: {backup_path}")
    
    with open(app_fixed_path, 'r') as f:
        content = f.read()
    
    # Add favicon route if not already present
    if 'def favicon():' not in content:
        favicon_route = """
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')
"""
        # Insert after the imports and before the routes
        content = re.sub(
            r'(# Flask routes)',
            f'{favicon_route}\n\n\\1',
            content
        )
        logger.info("Added favicon route to fix 405 error")
    
    # Fix get_processes function to always return a JSON response with processes list
    get_processes_pattern = r'@app\.route\(\'/get_processes\'\)\s*def get_processes\(\):(.*?)return jsonify\((.*?)\)'
    if re.search(get_processes_pattern, content, re.DOTALL):
        content = re.sub(
            get_processes_pattern,
            lambda m: m.group(0).replace(
                'return jsonify(' + m.group(2) + ')',
                'return jsonify({"processes": list(processes.values())})'
            ),
            content,
            flags=re.DOTALL
        )
        logger.info("Fixed get_processes function to return proper JSON response")
    else:
        logger.warning("Could not find return statement in get_processes function")
    
    # Fix get_backtest_results_route function to always return a JSON response with results list
    get_results_pattern = r'@app\.route\(\'/get_backtest_results\'\)\s*def get_backtest_results_route\(\):(.*?)return jsonify\((.*?)\)'
    if re.search(get_results_pattern, content, re.DOTALL):
        content = re.sub(
            get_results_pattern,
            lambda m: m.group(0).replace(
                'return jsonify(' + m.group(2) + ')',
                'return jsonify({"results": results})'
            ),
            content,
            flags=re.DOTALL
        )
        logger.info("Fixed get_backtest_results_route function to return proper JSON response")
    else:
        logger.warning("Could not find return statement in get_backtest_results_route function")
    
    # Fix run_comprehensive_backtest function to handle AJAX requests
    run_backtest_pattern = r'@app\.route\(\'/run_comprehensive_backtest\', methods=\[\'POST\'\]\)\s*def run_comprehensive_backtest\(\):(.*?)except Exception as e:(.*?)return redirect\(url_for\(\'index\'\)\)'
    if re.search(run_backtest_pattern, content, re.DOTALL):
        # Add AJAX request handling
        ajax_response = """
        # Check if this is an AJAX request
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({"success": True, "message": f'Comprehensive backtest started for {", ".join(quarters_list)}'})
"""
        content = re.sub(
            r'(flash\(f\'Comprehensive backtest started for \{", ".join\(quarters_list\)\}\', \'success\'\)\s*return redirect\(url_for\(\'index\'\)\))',
            f'{ajax_response}\\1',
            content
        )
        
        # Add AJAX error handling
        ajax_error_response = """
        # Check if this is an AJAX request
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({"success": False, "error": str(e)}), 500
"""
        content = re.sub(
            r'(logger\.error\(f"Error running comprehensive backtest: \{str\(e\)\}", exc_info=True\)\s*)(flash\(f\'Error running comprehensive backtest: \{str\(e\)\}\', \'danger\'\)\s*return redirect\(url_for\(\'index\'\)\))',
            f'\\1{ajax_error_response}\\2',
            content
        )
        
        logger.info("Updated run_comprehensive_backtest function to handle AJAX requests")
    else:
        logger.warning("Could not find run_comprehensive_backtest function")
    
    # Write updated content
    with open(app_fixed_path, 'w') as f:
        f.write(content)
    
    logger.info(f"Successfully updated {app_fixed_path}")

def fix_main_js():
    """Fix main.js to properly handle API responses"""
    main_js_path = os.path.join(JS_DIR, 'main.js')
    
    if not os.path.exists(main_js_path):
        logger.error(f"main.js not found: {main_js_path}")
        return
    
    # Create backup
    backup_path = f"{main_js_path}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(main_js_path, backup_path)
    logger.info(f"Created backup of main.js: {backup_path}")
    
    with open(main_js_path, 'r') as f:
        content = f.read()
    
    # Fix setupAutoRefresh function to properly handle API responses
    setup_auto_refresh_pattern = r'function setupAutoRefresh\(\) \{(.*?)\}'
    if re.search(setup_auto_refresh_pattern, content, re.DOTALL):
        new_setup_auto_refresh = """function setupAutoRefresh() {
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
        
        content = re.sub(setup_auto_refresh_pattern, new_setup_auto_refresh, content, flags=re.DOTALL)
        logger.info("Updated setupAutoRefresh function to properly handle API responses")
    else:
        logger.warning("Could not find setupAutoRefresh function")
    
    # Fix comprehensive backtest form submission
    backtest_form_pattern = r'document\.getElementById\(\'comprehensiveBacktestForm\'\)\.addEventListener\(\'submit\', function\(e\) \{(.*?)\}\);'
    if re.search(backtest_form_pattern, content, re.DOTALL):
        new_backtest_form = """document.getElementById('comprehensiveBacktestForm').addEventListener('submit', function(e) {
            e.preventDefault();
            const formData = new FormData(this);
            
            // Show loading indicator
            const submitBtn = this.querySelector('button[type="submit"]');
            const originalText = submitBtn.innerHTML;
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Running...';
            
            fetch(baseUrl + '/run_comprehensive_backtest', {
                method: 'POST',
                body: formData,
                headers: {
                    'X-Requested-With': 'XMLHttpRequest'
                }
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Server returned status ' + response.status);
                }
                return response.json();
            })
            .then(data => {
                if (data.success) {
                    showToast('success', data.message);
                } else if (data.error) {
                    showToast('danger', data.error);
                } else {
                    showToast('success', 'Comprehensive backtest started successfully');
                }
                
                // Reset form
                submitBtn.disabled = false;
                submitBtn.innerHTML = originalText;
            })
            .catch(error => {
                console.error('Error running comprehensive backtest:', error);
                showToast('danger', 'Error running comprehensive backtest: ' + error);
                
                // Reset form
                submitBtn.disabled = false;
                submitBtn.innerHTML = originalText;
            });
        });"""
        
        content = re.sub(backtest_form_pattern, new_backtest_form, content, flags=re.DOTALL)
        logger.info("Updated comprehensive backtest form submission to handle AJAX responses")
    else:
        logger.warning("Could not find comprehensive backtest form submission")
    
    # Write updated content
    with open(main_js_path, 'w') as f:
        f.write(content)
    
    logger.info(f"Successfully updated {main_js_path}")

def create_launcher_script():
    """Create a launcher script for the fixed web interface"""
    launcher_path = os.path.join(ROOT_DIR, 'launch_web_interface_final_v2.py')
    
    launcher_content = """#!/usr/bin/env python
# -*- coding: utf-8 -*-

\"\"\"
Launch Web Interface Final V2
This script launches the fixed web interface with all issues resolved
\"\"\"

import os
import sys
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    \"\"\"Main function\"\"\"
    logger.info("Starting fixed web interface")
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Change to the web interface directory
    web_interface_dir = os.path.join(script_dir, 'new_web_interface')
    os.chdir(web_interface_dir)
    
    # Add the parent directory to the path
    sys.path.insert(0, script_dir)
    
    # Import the Flask app
    from new_web_interface.app_fixed import app
    
    # Run the app
    logger.info("Running web interface on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    main()
"""
    
    with open(launcher_path, 'w') as f:
        f.write(launcher_content)
    
    logger.info(f"Created launcher script: {launcher_path}")

def main():
    """Main function"""
    logger.info("Starting complete web interface fix script v2")
    
    # Ensure configuration exists
    ensure_config_exists()
    
    # Create seasonality file to prevent warnings
    create_seasonality_file()
    
    # Create favicon
    create_favicon()
    
    # Fix app_fixed.py
    fix_app_fixed_py()
    
    # Fix main.js
    fix_main_js()
    
    # Create launcher script
    create_launcher_script()
    
    logger.info("Web interface fixes complete")
    logger.info("Run launch_web_interface_final_v2.py to start the web interface")

if __name__ == '__main__':
    main()
