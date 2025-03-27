#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Debug and Fix Web Interface

This script:
1. Kills any existing Flask processes on port 5000
2. Fixes all known issues with the web interface
3. Adds debug logging to help identify issues
4. Launches the web interface with proper error handling
"""

import os
import sys
import logging
import yaml
import json
import shutil
import subprocess
import time
import psutil
import traceback
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for more detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/debug_web_interface_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Root directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_INTERFACE_DIR = os.path.join(ROOT_DIR, 'new_web_interface')
STATIC_DIR = os.path.join(WEB_INTERFACE_DIR, 'static')
JS_DIR = os.path.join(STATIC_DIR, 'js')
TEMPLATES_DIR = os.path.join(WEB_INTERFACE_DIR, 'templates')

def kill_processes_on_port(port):
    """Kill all processes running on the specified port"""
    logger.info(f"Killing all processes on port {port}")
    
    try:
        # Find processes using the port
        for proc in psutil.process_iter(['pid', 'name', 'connections']):
            try:
                for conn in proc.connections():
                    if conn.laddr.port == port:
                        logger.info(f"Killing process {proc.pid} ({proc.name()}) using port {port}")
                        psutil.Process(proc.pid).terminate()
                        time.sleep(1)  # Give it time to terminate gracefully
                        
                        # Force kill if still running
                        if psutil.pid_exists(proc.pid):
                            psutil.Process(proc.pid).kill()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
    except Exception as e:
        logger.warning(f"Error killing processes on port {port}: {str(e)}")
        
        # Fallback to using netstat and taskkill
        try:
            # Get list of PIDs using the port
            netstat_output = subprocess.check_output(f"netstat -ano | findstr :{port}", shell=True).decode('utf-8')
            for line in netstat_output.splitlines():
                if "LISTENING" in line:
                    pid = line.strip().split()[-1]
                    logger.info(f"Killing process {pid} using port {port}")
                    subprocess.call(f"taskkill /F /PID {pid}", shell=True)
        except Exception as e:
            logger.warning(f"Error using netstat fallback: {str(e)}")

def fix_main_js():
    """Fix the main.js file"""
    main_js_path = os.path.join(JS_DIR, 'main.js')
    logger.info(f"Fixing {main_js_path}")
    
    # Create backup
    backup_path = f"{main_js_path}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(main_js_path, backup_path)
    logger.info(f"Created backup at {backup_path}")
    
    # Fix the file
    fixed_content = """/**
 * Main JavaScript for S&P 500 Trading Strategy Web Interface
 */

// Base URL for API calls
const baseUrl = '';

// Document ready
document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    if (tooltipTriggerList) {
        tooltipTriggerList.forEach(function (tooltipTriggerEl) {
            new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }

    // Initialize popovers
    const popoverTriggerList = document.querySelectorAll('[data-bs-toggle="popover"]');
    if (popoverTriggerList) {
        popoverTriggerList.forEach(function (popoverTriggerEl) {
            new bootstrap.Popover(popoverTriggerEl);
        });
    }

    // Emergency Stop button
    const emergencyStopBtn = document.getElementById('emergencyStopBtn');
    if (emergencyStopBtn) {
        emergencyStopBtn.addEventListener('click', function() {
            // Show modal instead of confirm
            const emergencyStopModal = document.getElementById('emergencyStopModal');
            if (emergencyStopModal) {
                const modal = new bootstrap.Modal(emergencyStopModal);
                modal.show();
            }
        });
    }

    // Confirm Emergency Stop
    const confirmEmergencyStop = document.getElementById('confirmEmergencyStop');
    if (confirmEmergencyStop) {
        confirmEmergencyStop.addEventListener('click', function() {
            fetch(baseUrl + '/emergency_stop', {
                method: 'POST'
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showToast('success', data.message);
                } else {
                    showToast('danger', data.message);
                }
                
                // Hide modal
                const emergencyStopModal = document.getElementById('emergencyStopModal');
                if (emergencyStopModal) {
                    const modal = bootstrap.Modal.getInstance(emergencyStopModal);
                    modal.hide();
                }
            })
            .catch(error => {
                console.error('Error during emergency stop:', error);
                showToast('danger', 'Error during emergency stop: ' + error);
                
                // Hide modal
                const emergencyStopModal = document.getElementById('emergencyStopModal');
                if (emergencyStopModal) {
                    const modal = bootstrap.Modal.getInstance(emergencyStopModal);
                    modal.hide();
                }
            });
        });
    }

    // Setup auto-refresh for processes and backtest results
    setupAutoRefresh();
});

/**
 * Set up auto-refresh for processes and backtest results tables
 */
function setupAutoRefresh() {
    // Refresh active processes every 5 seconds
    const processesTableBody = document.getElementById('processesTableBody');
    if (processesTableBody) {
        const refreshProcesses = function() {
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
        };
        setInterval(refreshProcesses, 5000);
        // Call immediately
        refreshProcesses();
    }
    
    // Refresh backtest results every 10 seconds
    const backtestResultsTableBody = document.getElementById('backtestResultsTableBody');
    if (backtestResultsTableBody) {
        const refreshBacktestResults = function() {
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
        };
        setInterval(refreshBacktestResults, 10000);
        // Call immediately
        refreshBacktestResults();
    }
}

/**
 * Show toast notification
 * 
 * @param {string} type - Toast type (success, danger, warning, info)
 * @param {string} message - Toast message
 */
function showToast(type, message) {
    // Create toast container if it doesn't exist
    let toastContainer = document.getElementById('toastContainer');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.id = 'toastContainer';
        toastContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        document.body.appendChild(toastContainer);
    }
    
    // Create toast
    const toastId = 'toast-' + Date.now();
    const toast = document.createElement('div');
    toast.className = `toast align-items-center text-white bg-${type} border-0`;
    toast.id = toastId;
    toast.setAttribute('role', 'alert');
    toast.setAttribute('aria-live', 'assertive');
    toast.setAttribute('aria-atomic', 'true');
    
    // Toast content
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
        </div>
    `;
    
    // Add toast to container
    toastContainer.appendChild(toast);
    
    // Initialize and show toast
    const bsToast = new bootstrap.Toast(toast);
    bsToast.show();
    
    // Remove toast after it's hidden
    toast.addEventListener('hidden.bs.toast', function() {
        toast.remove();
    });
}

/**
 * Format date string
 * 
 * @param {string} dateString - Date string
 * @returns {string} - Formatted date string
 */
function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
}

/**
 * Format number as currency
 * 
 * @param {number} value - Number to format
 * @returns {string} - Formatted currency string
 */
function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(value);
}

/**
 * Format number as percentage
 * 
 * @param {number} value - Number to format
 * @returns {string} - Formatted percentage string
 */
function formatPercent(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'percent',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }).format(value / 100);
}

/**
 * Stop a process
 * 
 * @param {string} processName - Name of the process to stop
 */
function stopProcess(processName) {
    if (confirm(`Are you sure you want to stop ${processName}?`)) {
        fetch(baseUrl + '/stop_process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `process_name=${encodeURIComponent(processName)}`
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showToast('success', data.message);
            } else {
                showToast('danger', data.message);
            }
        })
        .catch(error => {
            console.error('Error stopping process:', error);
            showToast('danger', 'Error stopping process: ' + error);
        });
    }
}

/**
 * View process logs
 * 
 * @param {string} processName - Name of the process
 * @param {Array} logs - Array of log lines
 */
function viewProcessLogs(processName, logs) {
    // Create modal if it doesn't exist
    let logsModal = document.getElementById('logsModal');
    if (!logsModal) {
        logsModal = document.createElement('div');
        logsModal.id = 'logsModal';
        logsModal.className = 'modal fade';
        logsModal.setAttribute('tabindex', '-1');
        logsModal.setAttribute('aria-labelledby', 'logsModalLabel');
        logsModal.setAttribute('aria-hidden', 'true');
        
        logsModal.innerHTML = `
            <div class="modal-dialog modal-dialog-scrollable modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title" id="logsModalLabel">Process Logs</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body">
                        <pre id="logsContent" class="bg-dark text-light p-3" style="max-height: 500px; overflow-y: auto;"></pre>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(logsModal);
    }
    
    // Set modal title
    const logsModalLabel = document.getElementById('logsModalLabel');
    if (logsModalLabel) {
        logsModalLabel.textContent = `Logs for ${processName}`;
    }
    
    // Set logs content
    const logsContent = document.getElementById('logsContent');
    if (logsContent) {
        if (Array.isArray(logs) && logs.length > 0) {
            logsContent.textContent = logs.join('\n');
        } else {
            logsContent.textContent = 'No logs available';
        }
    }
    
    // Show modal
    const modal = new bootstrap.Modal(logsModal);
    modal.show();
}"""
    
    # Write the fixed content
    with open(main_js_path, 'w') as f:
        f.write(fixed_content)
    
    logger.info(f"Successfully fixed {main_js_path}")
    return True

def fix_index_html():
    """Fix the index.html file"""
    index_html_path = os.path.join(TEMPLATES_DIR, 'index.html')
    logger.info(f"Fixing {index_html_path}")
    
    # Create backup
    backup_path = f"{index_html_path}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(index_html_path, backup_path)
    logger.info(f"Created backup at {backup_path}")
    
    try:
        # Read the file
        with open(index_html_path, 'r') as f:
            content = f.read()
        
        # Fix the updateBacktestResultsTable function
        if "function updateBacktestResultsTable(" in content:
            logger.info("Fixing updateBacktestResultsTable function")
            
            # Replace the function definition
            content = content.replace(
                "function updateBacktestResultsTable(results) {",
                "function updateBacktestResultsTable(data) {"
            )
            
            # Add the results extraction line
            content = content.replace(
                "// Check if there are any results",
                "// Get results array from data object\n            const results = data.results || [];\n            \n            // Check if there are any results"
            )
            
            # Write the fixed content
            with open(index_html_path, 'w') as f:
                f.write(content)
            
            logger.info(f"Successfully fixed {index_html_path}")
            return True
        else:
            logger.info("updateBacktestResultsTable function not found or already fixed")
            return True
    except Exception as e:
        logger.error(f"Error fixing {index_html_path}: {str(e)}")
        # Restore from backup
        shutil.copy2(backup_path, index_html_path)
        logger.info(f"Restored from backup")
        return False

def fix_app_py():
    """Fix the app_fixed.py file"""
    app_py_path = os.path.join(WEB_INTERFACE_DIR, 'app_fixed.py')
    logger.info(f"Fixing {app_py_path}")
    
    # Create backup
    backup_path = f"{app_py_path}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(app_py_path, backup_path)
    logger.info(f"Created backup at {backup_path}")
    
    try:
        # Read the file
        with open(app_py_path, 'r') as f:
            content = f.readlines()
        
        # Fix the file
        fixed_content = []
        in_get_processes = False
        in_get_backtest_results = False
        in_run_comprehensive_backtest = False
        
        for line in content:
            # Fix get_processes function
            if '@app.route(\'/get_processes\')' in line:
                in_get_processes = True
                fixed_content.append(line)
            elif in_get_processes and 'return jsonify' in line:
                fixed_content.append('        return jsonify({"processes": list(processes.values())})\n')
                in_get_processes = False
            # Fix get_backtest_results_route function
            elif '@app.route(\'/get_backtest_results\')' in line:
                in_get_backtest_results = True
                fixed_content.append(line)
            elif in_get_backtest_results and 'return jsonify' in line and 'results' in line:
                fixed_content.append('            return jsonify({"results": results})\n')
                in_get_backtest_results = False
            # Fix run_comprehensive_backtest function
            elif '@app.route(\'/run_comprehensive_backtest\'' in line:
                in_run_comprehensive_backtest = True
                fixed_content.append(line)
            elif in_run_comprehensive_backtest and 'flash(' in line and not line.strip().startswith('#'):
                # Ensure proper indentation for flash statements
                if not line.startswith('        '):
                    fixed_content.append('        ' + line.lstrip())
                else:
                    fixed_content.append(line)
            # Add any other lines
            else:
                fixed_content.append(line)
        
        # Add cache-busting headers
        cache_headers = """
# Add cache-busting headers to prevent browser caching of static files
@app.after_request
def add_header(response):
    \"\"\"Add headers to prevent caching\"\"\"
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response
"""
        
        # Find the right spot to insert the cache headers
        for i, line in enumerate(fixed_content):
            if '@app.route' in line and i > 10:
                fixed_content.insert(i, cache_headers)
                break
        
        # Write the fixed content
        with open(app_py_path, 'w') as f:
            f.writelines(fixed_content)
        
        logger.info(f"Successfully fixed {app_py_path}")
        return True
    except Exception as e:
        logger.error(f"Error fixing {app_py_path}: {str(e)}")
        # Restore from backup
        shutil.copy2(backup_path, app_py_path)
        logger.info(f"Restored from backup")
        return False

def create_launcher_script():
    """Create a launcher script for the web interface"""
    launcher_path = os.path.join(ROOT_DIR, 'launch_web_interface_debug.py')
    logger.info(f"Creating launcher script at {launcher_path}")
    
    launcher_content = """#!/usr/bin/env python
# -*- coding: utf-8 -*-

\"\"\"
Launch Web Interface Debug

This script launches the web interface with debug logging
\"\"\"

import os
import sys
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/web_interface_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

def main():
    \"\"\"Main function\"\"\"
    logger.info("Starting web interface with debug logging")
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Add the parent directory to the path
    sys.path.insert(0, script_dir)
    
    try:
        # Import the Flask app
        from new_web_interface.app_fixed import app
        
        # Run the app with debug logging
        logger.info("Running web interface on http://localhost:5000")
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    except Exception as e:
        logger.error(f"Error running web interface: {str(e)}", exc_info=True)
        return False
    
    return True

if __name__ == "__main__":
    main()
"""
    
    with open(launcher_path, 'w') as f:
        f.write(launcher_content)
    
    logger.info(f"Successfully created launcher script at {launcher_path}")
    return True

def launch_web_interface():
    """Launch the web interface"""
    logger.info("Launching web interface")
    
    try:
        # Run the launcher script
        launcher_path = os.path.join(ROOT_DIR, 'launch_web_interface_debug.py')
        
        # Use subprocess to run the launcher script
        process = subprocess.Popen(
            [sys.executable, launcher_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Wait for the server to start
        logger.info("Waiting for server to start...")
        server_started = False
        start_time = time.time()
        timeout = 30  # 30 seconds timeout
        
        while time.time() - start_time < timeout:
            line = process.stdout.readline()
            if line:
                logger.info(f"Server output: {line.strip()}")
                if "Running on http://127.0.0.1:5000" in line:
                    server_started = True
                    break
            
            # Check if process is still running
            if process.poll() is not None:
                logger.error(f"Server process exited with code {process.returncode}")
                break
            
            time.sleep(0.1)
        
        if server_started:
            logger.info("Server started successfully!")
            logger.info("Web interface is available at http://localhost:5000")
            
            # Open the web interface in the default browser
            try:
                import webbrowser
                webbrowser.open("http://localhost:5000")
                logger.info("Opened web interface in browser")
            except Exception as e:
                logger.warning(f"Could not open browser: {str(e)}")
            
            return True
        else:
            logger.error("Server failed to start within timeout period")
            return False
    except Exception as e:
        logger.error(f"Error launching web interface: {str(e)}")
        return False

def main():
    """Main function"""
    logger.info("Starting debug and fix web interface script")
    
    # Kill any existing processes on port 5000
    kill_processes_on_port(5000)
    
    # Fix the files
    if not fix_main_js():
        logger.error("Failed to fix main.js")
        return False
    
    if not fix_index_html():
        logger.error("Failed to fix index.html")
        return False
    
    if not fix_app_py():
        logger.error("Failed to fix app_fixed.py")
        return False
    
    if not create_launcher_script():
        logger.error("Failed to create launcher script")
        return False
    
    # Launch the web interface
    if not launch_web_interface():
        logger.error("Failed to launch web interface")
        return False
    
    logger.info("Web interface debug and fix complete")
    return True

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        logger.error(traceback.format_exc())
