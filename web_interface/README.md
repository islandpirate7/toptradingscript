# S&P 500 Trading Strategy Web Interface

This web interface provides a user-friendly way to control and monitor the S&P 500 trading strategy. It allows you to run backtests, paper trading, and market simulations, as well as configure strategy parameters and monitor open positions.

## Features

- **Dashboard**: View active processes, open positions, and recent backtest results
- **Configuration**: Modify strategy parameters through a user-friendly interface
- **Backtesting**: Run backtests for specific quarters with customizable parameters
- **Paper Trading**: Execute the strategy in paper trading mode using Alpaca API
- **Market Simulation**: Simulate market behavior for testing without requiring actual market connectivity
- **Emergency Stop**: Quickly stop all processes and close all positions in case of emergency

## Installation

1. Ensure all dependencies are installed:
   ```
   pip install -r requirements.txt
   ```

2. Configure your Alpaca API credentials in a JSON file (for paper trading).

## Usage

### Starting the Web Interface

Run the launcher script:

```
python start_web_interface.py
```

Optional arguments:
- `--host`: Host to run the web interface on (default: 127.0.0.1)
- `--port`: Port to run the web interface on (default: 5000)
- `--debug`: Run in debug mode

Example:
```
python start_web_interface.py --host 0.0.0.0 --port 8080
```

### Accessing the Web Interface

Once started, open your web browser and navigate to:
```
http://localhost:5000
```
(or the host/port you specified)

### Using the Dashboard

1. **Running Paper Trading**:
   - Set the maximum number of signals
   - Set the duration in hours
   - Set the check interval in minutes
   - Click "Start Paper Trading"

2. **Running Market Simulation**:
   - Set the number of days to simulate
   - Set the initial capital
   - Set the maximum number of signals
   - Set the check interval in days
   - Click "Start Simulation"

3. **Running Backtests**:
   - Enter the quarters to test (e.g., 2023Q1,2023Q2)
   - Set the number of runs
   - Set the random seed for reproducibility
   - Click "Start Backtest"

4. **Managing Processes**:
   - View active processes in the "Active Processes" section
   - Stop a process by clicking the "Stop" button
   - View logs by clicking the "View Logs" button

5. **Monitoring Positions**:
   - View open positions in the "Open Positions" section
   - See entry price, current price, and P/L for each position

6. **Emergency Stop**:
   - Click the "Emergency Stop" button in the navigation bar
   - Confirm the action in the confirmation dialog

### Configuring the Strategy

1. Navigate to the "Configuration" page
2. Modify parameters in the different tabs:
   - General Settings
   - Signal Generation
   - Position Sizing
   - Stop Loss
   - Mid-Cap Integration
   - Paths
3. Click "Save Configuration" to apply changes

## Deploying to Vercel via GitHub

1. Push your code to GitHub:
   ```
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/yourrepository.git
   git push -u origin main
   ```

2. Sign up for a Vercel account at https://vercel.com

3. Connect your GitHub repository to Vercel:
   - Click "Import Project"
   - Select "Import Git Repository"
   - Enter your repository URL or select it from the list
   - Configure the project settings:
     - Framework Preset: Other
     - Build Command: `pip install -r requirements.txt`
     - Output Directory: `web_interface`
     - Install Command: `pip install -r requirements.txt`

4. Deploy the application:
   - Click "Deploy"
   - Wait for the deployment to complete
   - Access your application at the provided URL

## Handling Open Positions During Shutdowns

When the application shuts down, the emergency stop procedure is automatically triggered to ensure all positions are properly handled:

1. All running processes are stopped
2. All open positions are closed
3. A log of the emergency stop is created

To manually handle open positions:
1. Click the "Emergency Stop" button before shutting down
2. Verify all positions are closed in the "Open Positions" section
3. Shut down the application

## Troubleshooting

- **Web interface not starting**: Check that all dependencies are installed and that the port is not in use
- **Paper trading not working**: Verify your Alpaca API credentials are correctly configured
- **Backtests not running**: Ensure the data files are available in the specified paths
- **Configuration not saving**: Check file permissions for the configuration file

## License

This project is licensed under the MIT License - see the LICENSE file for details.
