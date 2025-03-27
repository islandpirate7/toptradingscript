/**
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
                    if (modal) {
                        modal.hide();
                    }
                }
            })
            .catch(error => {
                console.error('Error during emergency stop:', error);
                showToast('danger', 'Error during emergency stop: ' + error.message);
            });
        });
    }

    // Setup forms
    setupForms();
});

/**
 * Set up forms
 */
function setupForms() {
    // Paper Trading Form
    const paperTradingForm = document.getElementById('paperTradingForm');
    if (paperTradingForm) {
        paperTradingForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            
            fetch('/run_paper_trading', {
                method: 'POST',
                body: formData
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
                console.error('Error starting paper trading:', error);
                showToast('danger', 'Error starting paper trading: ' + error.message);
            });
        });
    }
    
    // Live Trading Form
    const liveTradingForm = document.getElementById('liveTradingForm');
    if (liveTradingForm) {
        liveTradingForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            
            fetch('/run_live_trading', {
                method: 'POST',
                body: formData
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
                console.error('Error starting live trading:', error);
                showToast('danger', 'Error starting live trading: ' + error.message);
            });
        });
    }
    
    // Backtest Form
    const backtestForm = document.getElementById('backtestForm');
    if (backtestForm) {
        backtestForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            
            fetch('/run_backtest', {
                method: 'POST',
                body: formData
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
                console.error('Error running backtest:', error);
                showToast('danger', 'Error running backtest: ' + error.message);
            });
        });
    }
    
    // Comprehensive Backtest Form
    const comprehensiveBacktestForm = document.getElementById('comprehensiveBacktestForm');
    if (comprehensiveBacktestForm) {
        comprehensiveBacktestForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Get form data
            const quartersInput = document.getElementById('quarters').value;
            if (!quartersInput) {
                showToast('danger', 'Please enter at least one quarter to test');
                return;
            }
            
            // Create FormData object
            const formData = new FormData();
            
            // Add quarters as a comma-separated string
            formData.append('quarters', quartersInput);
            
            // Send request
            fetch('/run_comprehensive_backtest', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showToast('success', data.message);
                } else {
                    showToast('danger', data.message || 'Error running comprehensive backtest');
                }
            })
            .catch(error => {
                console.error('Error running comprehensive backtest:', error);
                showToast('danger', 'Error running comprehensive backtest: ' + error.message);
            });
        });
    }
    
    // Fetch processes and backtest results once (no auto-refresh)
    if (document.getElementById('processesTable')) {
        fetchProcesses();
    }
    
    if (document.getElementById('backtestResultsTable')) {
        fetchBacktestResults();
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
    const toast = document.createElement('div');
    toast.className = `toast align-items-center text-white bg-${type} border-0`;
    toast.setAttribute('role', 'alert');
    toast.setAttribute('aria-live', 'assertive');
    toast.setAttribute('aria-atomic', 'true');
    
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
    
    // Show toast
    const bsToast = new bootstrap.Toast(toast, {
        autohide: true,
        delay: 5000
    });
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
            showToast('danger', 'Error stopping process: ' + error.message);
        });
    }
}

/**
 * View process logs
 *
 * @param {string} processName - Name of the process
 * @param {Array|string} logs - Array of log lines or string
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
        } else if (typeof logs === 'string') {
            logsContent.textContent = logs;
        } else {
            logsContent.textContent = 'No logs available';
        }
    }
    
    // Show modal
    const modal = new bootstrap.Modal(logsModal);
    modal.show();
}

/**
 * Fetch active processes
 */
function fetchProcesses() {
    fetch(baseUrl + '/get_processes')
        .then(response => response.json())
        .then(data => {
            updateProcessesTable(data.processes || []);
        })
        .catch(error => {
            console.error('Error fetching processes:', error);
        });
}

/**
 * Update processes table
 * 
 * @param {Array} processes - List of processes
 */
function updateProcessesTable(processes) {
    const table = document.getElementById('processesTable');
    if (!table) return;
    
    const tbody = table.querySelector('tbody');
    if (!tbody) return;
    
    // Clear table
    tbody.innerHTML = '';
    
    // Check if there are any processes
    if (processes.length === 0) {
        tbody.innerHTML = '<tr><td colspan="4" class="text-center">No active processes</td></tr>';
        return;
    }
    
    // Add processes to table
    for (const process of processes) {
        const row = document.createElement('tr');
        
        // Add cells
        row.innerHTML = `
            <td>${process.name}</td>
            <td><span class="badge ${process.status === 'Running' ? 'bg-success' : 'bg-secondary'}">${process.status}</span></td>
            <td>${process.start_time}</td>
            <td>
                <button class="btn btn-sm btn-info view-logs" data-process="${process.name}">View Logs</button>
                ${process.status === 'Running' ? `<button class="btn btn-sm btn-danger stop-process" data-process="${process.name}">Stop</button>` : ''}
            </td>
        `;
        
        tbody.appendChild(row);
    }
    
    // Add event listeners to buttons
    document.querySelectorAll('.stop-process').forEach(button => {
        button.addEventListener('click', function() {
            const processName = this.getAttribute('data-process');
            stopProcess(processName);
        });
    });
    
    document.querySelectorAll('.view-logs').forEach(button => {
        button.addEventListener('click', function() {
            const processName = this.getAttribute('data-process');
            const process = processes.find(p => p.name === processName);
            if (process) {
                viewProcessLogs(processName, process.logs || []);
            }
        });
    });
}

/**
 * Fetch backtest results
 */
function fetchBacktestResults() {
    fetch(baseUrl + '/get_backtest_results')
        .then(response => response.json())
        .then(data => {
            updateBacktestResultsTable(data);
        })
        .catch(error => {
            console.error('Error fetching backtest results:', error);
        });
}

/**
 * Update backtest results table
 * 
 * @param {Object} data - Backtest results data
 */
function updateBacktestResultsTable(data) {
    const table = document.getElementById('backtestResultsTable');
    if (!table) return;
    
    const tbody = table.querySelector('tbody');
    if (!tbody) return;
    
    // Clear table
    tbody.innerHTML = '';
    
    // Get results array from data object
    const results = data.results || [];
    
    // Check if there are any results
    if (results.length === 0) {
        tbody.innerHTML = '<tr><td colspan="4" class="text-center">No backtest results available</td></tr>';
        return;
    }
    
    // Add results to table
    for (const result of results) {
        const row = document.createElement('tr');
        
        // Format quarter/date range
        let dateRange = result.quarter || '';
        if (result.date_range) {
            dateRange += dateRange ? ` (${result.date_range})` : result.date_range;
        }
        
        // Add cells
        row.innerHTML = `
            <td>${result.filename}</td>
            <td>${dateRange}</td>
            <td>${result.modified || ''}</td>
            <td>
                <a href="/view_backtest_result/${result.filename}" class="btn btn-sm btn-primary" target="_blank">View</a>
            </td>
        `;
        
        tbody.appendChild(row);
    }
}