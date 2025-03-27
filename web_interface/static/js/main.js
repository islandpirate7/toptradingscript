/**
 * Main JavaScript file for S&P 500 Trading Strategy Web Interface
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Emergency Stop Button
    const emergencyStopBtn = document.getElementById('emergencyStopBtn');
    if (emergencyStopBtn) {
        emergencyStopBtn.addEventListener('click', function() {
            const emergencyStopModal = new bootstrap.Modal(document.getElementById('emergencyStopModal'));
            emergencyStopModal.show();
        });
    }

    // Restart Server Button
    const restartServerBtn = document.getElementById('restartServerBtn');
    if (restartServerBtn) {
        restartServerBtn.addEventListener('click', function() {
            const restartServerModal = new bootstrap.Modal(document.getElementById('restartServerModal'));
            restartServerModal.show();
        });
    }

    // Confirm Restart Server Button
    const confirmRestartServerBtn = document.getElementById('confirmRestartServer');
    if (confirmRestartServerBtn) {
        confirmRestartServerBtn.addEventListener('click', function() {
            // Show loading state
            this.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Restarting...';
            this.disabled = true;
            
            // Close the modal first to avoid issues with the page reloading
            const modal = bootstrap.Modal.getInstance(document.getElementById('restartServerModal'));
            if (modal) {
                modal.hide();
            }
            
            // Show notification that restart is in progress
            showNotification('Server restart initiated. The page will reload shortly...', 'warning');
            
            fetch('/restart_server', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({})
            })
            .then(response => {
                // The server might restart before sending a response
                // So we'll handle both successful responses and errors
                if (response.ok) {
                    try {
                        return response.json();
                    } catch (e) {
                        // If parsing fails, we'll assume the server is restarting
                        return { success: true, message: "Server is restarting" };
                    }
                } else {
                    // If we get an error response, the server might be restarting
                    return { success: true, message: "Server might be restarting" };
                }
            })
            .then(data => {
                // Set a timer to reload the page after a delay
                setTimeout(function() {
                    showNotification('Attempting to reconnect...', 'info');
                    
                    // Function to check if server is back up
                    function checkServerStatus() {
                        fetch('/', { 
                            method: 'GET',
                            cache: 'no-store'  // Prevent caching
                        })
                        .then(response => {
                            if (response.ok) {
                                // Server is back up, reload the page
                                showNotification('Server is back online! Reloading page...', 'success');
                                setTimeout(() => window.location.reload(), 1000);
                            } else {
                                // Server returned an error, try again
                                setTimeout(checkServerStatus, 1000);
                            }
                        })
                        .catch(() => {
                            // Server still down, try again in 1 second
                            showNotification('Server still restarting, trying again...', 'info');
                            setTimeout(checkServerStatus, 1000);
                        });
                    }
                    
                    // Start checking server status after a delay
                    setTimeout(checkServerStatus, 3000);
                }, 2000);
            })
            .catch(error => {
                console.error('Error:', error);
                
                // Even if we get an error, the server might be restarting
                // So we'll still try to reconnect
                showNotification('Server might be restarting. Attempting to reconnect...', 'warning');
                
                // Function to check if server is back up
                function checkServerStatus() {
                    fetch('/', { 
                        method: 'GET',
                        cache: 'no-store'  // Prevent caching
                    })
                    .then(response => {
                        if (response.ok) {
                            // Server is back up, reload the page
                            showNotification('Server is back online! Reloading page...', 'success');
                            setTimeout(() => window.location.reload(), 1000);
                        } else {
                            // Server returned an error, try again
                            setTimeout(checkServerStatus, 1000);
                        }
                    })
                    .catch(() => {
                        // Server still down, try again in 1 second
                        showNotification('Server still restarting, trying again...', 'info');
                        setTimeout(checkServerStatus, 1000);
                    });
                }
                
                // Start checking server status after a delay
                setTimeout(checkServerStatus, 5000);
            });
        });
    }

    // Confirm Emergency Stop Button
    const confirmEmergencyStopBtn = document.getElementById('confirmEmergencyStop');
    if (confirmEmergencyStopBtn) {
        confirmEmergencyStopBtn.addEventListener('click', function() {
            // Show loading state
            this.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
            this.disabled = true;
            
            fetch('/emergency_stop', {
                method: 'POST'
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showNotification('Emergency stop completed successfully', 'success');
                    setTimeout(function() {
                        window.location.reload();
                    }, 2000);
                } else {
                    showNotification('Error: ' + data.message, 'danger');
                    // Reset button state
                    confirmEmergencyStopBtn.innerHTML = 'Confirm Emergency Stop';
                    confirmEmergencyStopBtn.disabled = false;
                }
            })
            .catch(error => {
                console.error('Error:', error);
                showNotification('An error occurred during emergency stop', 'danger');
                // Reset button state
                confirmEmergencyStopBtn.innerHTML = 'Confirm Emergency Stop';
                confirmEmergencyStopBtn.disabled = false;
            });
        });
    }

    // Process Status Refresh
    function refreshProcessStatus() {
        const activeProcessesDiv = document.getElementById('activeProcesses');
        if (activeProcessesDiv) {
            fetch('/get_processes')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    if (Object.keys(data.processes).length > 0) {
                        let html = `
                        <div class="table-responsive">
                            <table class="table table-striped">
                                <thead>
                                    <tr>
                                        <th>Process Name</th>
                                        <th>Status</th>
                                        <th>Actions</th>
                                    </tr>
                                </thead>
                                <tbody>`;
                        
                        for (const [name, status] of Object.entries(data.processes)) {
                            const statusClass = status === 'running' ? 'success' : status === 'starting' ? 'warning' : 'danger';
                            html += `
                            <tr>
                                <td>${name}</td>
                                <td>
                                    <span class="badge bg-${statusClass}">
                                        ${status}
                                    </span>
                                </td>
                                <td>
                                    <button class="btn btn-sm btn-danger stop-process" data-process="${name}">Stop</button>
                                    <button class="btn btn-sm btn-secondary view-logs" data-process="${name}">View Logs</button>
                                </td>
                            </tr>`;
                        }
                        
                        html += `
                                </tbody>
                            </table>
                        </div>`;
                        
                        activeProcessesDiv.innerHTML = html;
                        
                        // Reattach event listeners
                        attachProcessButtonListeners();
                    } else {
                        activeProcessesDiv.innerHTML = '<p class="text-muted">No active processes</p>';
                    }
                }
            })
            .catch(error => {
                console.error('Error refreshing process status:', error);
            });
        }
    }

    // Attach event listeners to process buttons
    function attachProcessButtonListeners() {
        // Stop Process Buttons
        document.querySelectorAll('.stop-process').forEach(button => {
            button.addEventListener('click', function() {
                const processName = this.getAttribute('data-process');
                if (confirm(`Are you sure you want to stop process ${processName}?`)) {
                    // Show loading state
                    this.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>';
                    this.disabled = true;
                    
                    fetch(`/stop_process/${processName}`, {
                        method: 'POST'
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            showNotification(data.message, 'success');
                            setTimeout(function() {
                                refreshProcessStatus();
                            }, 1000);
                        } else {
                            showNotification('Error: ' + data.message, 'danger');
                            // Reset button state
                            this.innerHTML = 'Stop';
                            this.disabled = false;
                        }
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        showNotification('An error occurred while stopping the process', 'danger');
                        // Reset button state
                        this.innerHTML = 'Stop';
                        this.disabled = false;
                    });
                }
            });
        });

        // View Logs Buttons
        document.querySelectorAll('.view-logs').forEach(button => {
            button.addEventListener('click', function() {
                const processName = this.getAttribute('data-process');
                // Show loading state
                this.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>';
                this.disabled = true;
                
                fetch(`/process_logs/${processName}`)
                .then(response => response.json())
                .then(data => {
                    // Reset button state
                    this.innerHTML = 'View Logs';
                    this.disabled = false;
                    
                    if (data.success) {
                        const logsContent = document.getElementById('logsContent');
                        logsContent.textContent = data.logs.join('\n');
                        const logsModal = new bootstrap.Modal(document.getElementById('logsModal'));
                        logsModal.show();
                    } else {
                        showNotification('Error: ' + data.message, 'danger');
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    showNotification('An error occurred while fetching logs', 'danger');
                    // Reset button state
                    this.innerHTML = 'View Logs';
                    this.disabled = false;
                });
            });
        });
    }

    // Initial attachment of process button listeners
    attachProcessButtonListeners();

    // Positions Refresh
    function refreshPositions() {
        const openPositionsDiv = document.getElementById('openPositions');
        if (openPositionsDiv) {
            fetch('/get_positions')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    if (data.positions.length > 0) {
                        let html = `
                        <div class="table-responsive">
                            <table class="table table-striped">
                                <thead>
                                    <tr>
                                        <th>Symbol</th>
                                        <th>Quantity</th>
                                        <th>Entry Price</th>
                                        <th>Current Price</th>
                                        <th>P/L</th>
                                    </tr>
                                </thead>
                                <tbody>`;
                        
                        data.positions.forEach(position => {
                            const plClass = parseFloat(position.unrealized_pl) > 0 ? 'text-success' : 'text-danger';
                            html += `
                            <tr>
                                <td>${position.symbol}</td>
                                <td>${position.qty}</td>
                                <td>$${position.entry_price}</td>
                                <td>$${position.current_price}</td>
                                <td class="${plClass}">
                                    $${position.unrealized_pl} (${position.unrealized_plpc}%)
                                </td>
                            </tr>`;
                        });
                        
                        html += `
                                </tbody>
                            </table>
                        </div>`;
                        
                        openPositionsDiv.innerHTML = html;
                    } else {
                        openPositionsDiv.innerHTML = '<p class="text-muted">No open positions</p>';
                    }
                }
            })
            .catch(error => {
                console.error('Error refreshing positions:', error);
            });
        }
    }

    // Backtest Results Refresh
    function refreshBacktestResults() {
        const backtestResultsDiv = document.getElementById('backtestResults');
        if (backtestResultsDiv) {
            fetch('/get_backtest_results')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    if (data.results.length > 0) {
                        let html = `
                        <div class="table-responsive">
                            <table class="table table-striped">
                                <thead>
                                    <tr>
                                        <th>File</th>
                                        <th>Date</th>
                                        <th>Actions</th>
                                    </tr>
                                </thead>
                                <tbody>`;
                        
                        data.results.slice(0, 5).forEach(result => {
                            html += `
                            <tr>
                                <td>${result.name}</td>
                                <td>${result.date}</td>
                                <td>
                                    <a href="${result.path}" class="btn btn-sm btn-primary" target="_blank">View</a>
                                </td>
                            </tr>`;
                        });
                        
                        html += `
                                </tbody>
                            </table>
                        </div>`;
                        
                        backtestResultsDiv.innerHTML = html;
                    } else {
                        backtestResultsDiv.innerHTML = '<p class="text-muted">No backtest results</p>';
                    }
                }
            })
            .catch(error => {
                console.error('Error refreshing backtest results:', error);
            });
        }
    }

    // Form Submission Handlers
    const paperTradingForm = document.getElementById('paperTradingForm');
    if (paperTradingForm) {
        paperTradingForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const submitBtn = this.querySelector('button[type="submit"]');
            
            // Show loading state
            submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Starting...';
            submitBtn.disabled = true;
            
            const formData = new FormData(this);
            fetch('/run_paper_trading', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                // Reset button state
                submitBtn.innerHTML = 'Start Paper Trading';
                submitBtn.disabled = false;
                
                if (data.success) {
                    showNotification(data.message, 'success');
                    setTimeout(function() {
                        refreshProcessStatus();
                    }, 1000);
                } else {
                    showNotification('Error: ' + data.message, 'danger');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                showNotification('An error occurred while starting paper trading', 'danger');
                // Reset button state
                submitBtn.innerHTML = 'Start Paper Trading';
                submitBtn.disabled = false;
            });
        });
    }

    const simulationForm = document.getElementById('simulationForm');
    if (simulationForm) {
        simulationForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const submitBtn = this.querySelector('button[type="submit"]');
            
            // Show loading state
            submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Starting...';
            submitBtn.disabled = true;
            
            const formData = new FormData(this);
            fetch('/run_simulation', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                // Reset button state
                submitBtn.innerHTML = 'Start Simulation';
                submitBtn.disabled = false;
                
                if (data.success) {
                    showNotification(data.message, 'success');
                    setTimeout(function() {
                        refreshProcessStatus();
                    }, 1000);
                } else {
                    showNotification('Error: ' + data.message, 'danger');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                showNotification('An error occurred while starting simulation', 'danger');
                // Reset button state
                submitBtn.innerHTML = 'Start Simulation';
                submitBtn.disabled = false;
            });
        });
    }

    const backtestForm = document.getElementById('backtestForm');
    if (backtestForm) {
        backtestForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const submitBtn = this.querySelector('button[type="submit"]');
            
            // Show loading state
            submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Starting...';
            submitBtn.disabled = true;
            
            const formData = new FormData(this);
            fetch('/run_backtest', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                // Reset button state
                submitBtn.innerHTML = 'Start Backtest';
                submitBtn.disabled = false;
                
                if (data.success) {
                    showNotification(data.message, 'success');
                    setTimeout(function() {
                        refreshProcessStatus();
                    }, 1000);
                } else {
                    showNotification('Error: ' + data.message, 'danger');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                showNotification('An error occurred while starting backtest', 'danger');
                // Reset button state
                submitBtn.innerHTML = 'Start Backtest';
                submitBtn.disabled = false;
            });
        });
    }

    // Configuration Form
    const configForm = document.getElementById('configForm');
    if (configForm) {
        configForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const submitBtn = this.querySelector('button[type="submit"]');
            
            // Show loading state
            submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Saving...';
            submitBtn.disabled = true;
            
            const formData = new FormData(this);
            fetch('/update_config', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                // Reset button state
                submitBtn.innerHTML = 'Save Configuration';
                submitBtn.disabled = false;
                
                if (data.success) {
                    showNotification('Configuration updated successfully', 'success');
                } else {
                    showNotification('Error: ' + data.message, 'danger');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                showNotification('An error occurred while updating configuration', 'danger');
                // Reset button state
                submitBtn.innerHTML = 'Save Configuration';
                submitBtn.disabled = false;
            });
        });
    }

    // Notification system
    function showNotification(message, type) {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `toast align-items-center text-white bg-${type} border-0`;
        notification.setAttribute('role', 'alert');
        notification.setAttribute('aria-live', 'assertive');
        notification.setAttribute('aria-atomic', 'true');
        
        notification.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
        `;
        
        // Add to container (create if doesn't exist)
        let toastContainer = document.querySelector('.toast-container');
        if (!toastContainer) {
            toastContainer = document.createElement('div');
            toastContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
            document.body.appendChild(toastContainer);
        }
        
        toastContainer.appendChild(notification);
        
        // Initialize and show the toast
        const toast = new bootstrap.Toast(notification, {
            autohide: true,
            delay: 5000
        });
        toast.show();
        
        // Remove from DOM after hiding
        notification.addEventListener('hidden.bs.toast', function() {
            notification.remove();
        });
    }

    // Set up periodic refresh
    if (document.getElementById('activeProcesses') || document.getElementById('openPositions') || document.getElementById('backtestResults')) {
        // Initial refresh
        refreshProcessStatus();
        refreshPositions();
        refreshBacktestResults();
        
        // Set up interval for periodic refresh (every 30 seconds)
        setInterval(function() {
            refreshProcessStatus();
            refreshPositions();
            refreshBacktestResults();
        }, 30000);
    }
});
