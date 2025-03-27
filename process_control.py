#!/usr/bin/env python
"""
Process Control for Trading System
---------------------------------
This module provides functionality to start, stop, and monitor trading processes.
It ensures graceful shutdown and proper resource cleanup.
"""

import os
import sys
import time
import signal
import psutil
import argparse
import json
import yaml
import logging
import threading
import subprocess
from datetime import datetime, timedelta
import atexit

# Import our custom modules
from trading_logger import get_logger
from error_handler import get_error_handler, error_context, SystemError

# Get logger
logger = get_logger("process_control")
error_handler = get_error_handler()

# Global variables
PROCESS_REGISTRY = {}
PROCESS_INFO_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', 'processes.json')
SHUTDOWN_FLAG = threading.Event()

def load_config(config_file='sp500_config.yaml'):
    """Load configuration from YAML file"""
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        error = SystemError(
            f"Error loading configuration: {str(e)}",
            severity="ERROR",
            details={"file": config_file}
        )
        error_handler.handle_error(error)
        return {}

def save_process_info():
    """Save process information to file"""
    os.makedirs(os.path.dirname(PROCESS_INFO_FILE), exist_ok=True)
    
    # Convert datetime objects to strings
    serializable_registry = {}
    for process_id, info in PROCESS_REGISTRY.items():
        serializable_info = info.copy()
        if 'start_time' in serializable_info and isinstance(serializable_info['start_time'], datetime):
            serializable_info['start_time'] = serializable_info['start_time'].isoformat()
        serializable_registry[process_id] = serializable_info
    
    with open(PROCESS_INFO_FILE, 'w') as f:
        json.dump(serializable_registry, f, indent=2)

def load_process_info():
    """Load process information from file"""
    global PROCESS_REGISTRY
    
    if os.path.exists(PROCESS_INFO_FILE):
        try:
            with open(PROCESS_INFO_FILE, 'r') as f:
                loaded_registry = json.load(f)
            
            # Convert string timestamps back to datetime objects
            for process_id, info in loaded_registry.items():
                if 'start_time' in info and isinstance(info['start_time'], str):
                    try:
                        info['start_time'] = datetime.fromisoformat(info['start_time'])
                    except ValueError:
                        info['start_time'] = datetime.now()
            
            PROCESS_REGISTRY = loaded_registry
            
            # Verify processes are still running
            for process_id in list(PROCESS_REGISTRY.keys()):
                if not is_process_running(process_id):
                    logger.warning(f"Process {process_id} is no longer running, removing from registry")
                    del PROCESS_REGISTRY[process_id]
            
            save_process_info()
        except Exception as e:
            logger.error(f"Error loading process info: {str(e)}")
            PROCESS_REGISTRY = {}

def is_process_running(process_id):
    """Check if a process is still running"""
    if not process_id or not process_id.isdigit():
        return False
    
    try:
        pid = int(process_id)
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False
    except Exception as e:
        logger.error(f"Error checking process status: {str(e)}")
        return False

def start_process(process_type, args=None):
    """Start a trading process"""
    with error_context({"operation": "start_process", "process_type": process_type}):
        if args is None:
            args = {}
        
        logger.info(f"Starting {process_type} process with args: {args}")
        
        # Determine command based on process type
        if process_type == 'backtest':
            cmd = [sys.executable, 'trading_cli.py', 'backtest']
            
            # Add arguments
            if 'start_date' in args:
                cmd.extend(['--start-date', args['start_date']])
            if 'end_date' in args:
                cmd.extend(['--end-date', args['end_date']])
            if 'initial_capital' in args:
                cmd.extend(['--initial-capital', str(args['initial_capital'])])
            if 'max_signals' in args:
                cmd.extend(['--max-signals', str(args['max_signals'])])
            if 'random_seed' in args:
                cmd.extend(['--random-seed', str(args['random_seed'])])
            
        elif process_type == 'paper_trading':
            cmd = [sys.executable, 'trading_cli.py', 'paper']
            
            # Add arguments
            if 'initial_capital' in args:
                cmd.extend(['--initial-capital', str(args['initial_capital'])])
            if 'max_signals' in args:
                cmd.extend(['--max-signals', str(args['max_signals'])])
            
        elif process_type == 'live_trading':
            cmd = [sys.executable, 'trading_cli.py', 'live']
            
            # Add arguments
            if 'initial_capital' in args:
                cmd.extend(['--initial-capital', str(args['initial_capital'])])
            if 'max_signals' in args:
                cmd.extend(['--max-signals', str(args['max_signals'])])
            
        else:
            logger.error(f"Unknown process type: {process_type}")
            return None
        
        try:
            # Start the process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Register the process
            process_id = str(process.pid)
            PROCESS_REGISTRY[process_id] = {
                'type': process_type,
                'pid': process_id,
                'command': ' '.join(cmd),
                'start_time': datetime.now(),
                'status': 'running',
                'args': args
            }
            
            # Save process information
            save_process_info()
            
            logger.info(f"Started {process_type} process with PID {process_id}")
            
            # Start a thread to monitor the process output
            threading.Thread(
                target=monitor_process_output,
                args=(process, process_id),
                daemon=True
            ).start()
            
            return process_id
            
        except Exception as e:
            error = SystemError(
                f"Error starting {process_type} process: {str(e)}",
                severity="ERROR",
                details={"process_type": process_type, "args": args}
            )
            error_handler.handle_error(error)
            logger.error(f"Error starting {process_type} process: {str(e)}")
            return None

def monitor_process_output(process, process_id):
    """Monitor process output and log it"""
    try:
        # Create log file for process output
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', 'processes')
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"{PROCESS_REGISTRY[process_id]['type']}_{process_id}.log")
        
        with open(log_file, 'w') as f:
            # Write process information
            f.write(f"Process: {PROCESS_REGISTRY[process_id]['type']}\n")
            f.write(f"PID: {process_id}\n")
            f.write(f"Command: {PROCESS_REGISTRY[process_id]['command']}\n")
            f.write(f"Start Time: {PROCESS_REGISTRY[process_id]['start_time']}\n")
            f.write(f"Args: {PROCESS_REGISTRY[process_id]['args']}\n")
            f.write("\n--- Process Output ---\n\n")
            f.flush()
            
            # Monitor stdout
            for line in process.stdout:
                f.write(line)
                f.flush()
        
        # Process has ended
        return_code = process.wait()
        
        # Update process status
        if process_id in PROCESS_REGISTRY:
            PROCESS_REGISTRY[process_id]['status'] = 'completed' if return_code == 0 else 'failed'
            PROCESS_REGISTRY[process_id]['return_code'] = return_code
            PROCESS_REGISTRY[process_id]['end_time'] = datetime.now()
            save_process_info()
        
        logger.info(f"Process {process_id} has ended with return code {return_code}")
        
    except Exception as e:
        logger.error(f"Error monitoring process {process_id}: {str(e)}")

def stop_process(process_id, timeout=30):
    """Stop a trading process"""
    with error_context({"operation": "stop_process", "process_id": process_id}):
        if process_id not in PROCESS_REGISTRY:
            logger.warning(f"Process {process_id} not found in registry")
            return False
        
        logger.info(f"Stopping process {process_id} ({PROCESS_REGISTRY[process_id]['type']})")
        
        try:
            # Get the process
            pid = int(process_id)
            process = psutil.Process(pid)
            
            # Send SIGTERM signal for graceful shutdown
            process.terminate()
            
            # Wait for process to terminate
            try:
                process.wait(timeout=timeout)
                logger.info(f"Process {process_id} terminated gracefully")
                
                # Update process status
                PROCESS_REGISTRY[process_id]['status'] = 'stopped'
                PROCESS_REGISTRY[process_id]['end_time'] = datetime.now()
                save_process_info()
                
                return True
            except psutil.TimeoutExpired:
                # Process didn't terminate within timeout, force kill
                logger.warning(f"Process {process_id} did not terminate within {timeout} seconds, force killing")
                process.kill()
                
                # Update process status
                PROCESS_REGISTRY[process_id]['status'] = 'killed'
                PROCESS_REGISTRY[process_id]['end_time'] = datetime.now()
                save_process_info()
                
                return True
            
        except psutil.NoSuchProcess:
            logger.warning(f"Process {process_id} no longer exists")
            
            # Update process status
            PROCESS_REGISTRY[process_id]['status'] = 'not_found'
            PROCESS_REGISTRY[process_id]['end_time'] = datetime.now()
            save_process_info()
            
            return False
        except Exception as e:
            error = SystemError(
                f"Error stopping process {process_id}: {str(e)}",
                severity="ERROR",
                details={"process_id": process_id}
            )
            error_handler.handle_error(error)
            logger.error(f"Error stopping process {process_id}: {str(e)}")
            return False

def stop_all_processes(timeout=30):
    """Stop all trading processes"""
    with error_context({"operation": "stop_all_processes"}):
        logger.info("Stopping all trading processes")
        
        success = True
        for process_id in list(PROCESS_REGISTRY.keys()):
            if PROCESS_REGISTRY[process_id]['status'] == 'running':
                if not stop_process(process_id, timeout):
                    success = False
        
        return success

def get_process_info(process_id):
    """Get information about a process"""
    if process_id not in PROCESS_REGISTRY:
        return None
    
    info = PROCESS_REGISTRY[process_id].copy()
    
    # Check if process is still running
    if info['status'] == 'running' and not is_process_running(process_id):
        info['status'] = 'terminated'
        PROCESS_REGISTRY[process_id]['status'] = 'terminated'
        PROCESS_REGISTRY[process_id]['end_time'] = datetime.now()
        save_process_info()
    
    return info

def list_processes():
    """List all trading processes"""
    # Load process info from file
    load_process_info()
    
    # Check if processes are still running
    for process_id in list(PROCESS_REGISTRY.keys()):
        if PROCESS_REGISTRY[process_id]['status'] == 'running' and not is_process_running(process_id):
            PROCESS_REGISTRY[process_id]['status'] = 'terminated'
            PROCESS_REGISTRY[process_id]['end_time'] = datetime.now()
    
    # Save updated process info
    save_process_info()
    
    return PROCESS_REGISTRY

def cleanup_old_processes(days=7):
    """Clean up old process records"""
    cutoff_date = datetime.now() - timedelta(days=days)
    
    for process_id in list(PROCESS_REGISTRY.keys()):
        if 'end_time' in PROCESS_REGISTRY[process_id] and PROCESS_REGISTRY[process_id]['end_time'] < cutoff_date:
            logger.info(f"Removing old process record: {process_id}")
            del PROCESS_REGISTRY[process_id]
    
    save_process_info()

def signal_handler(sig, frame):
    """Handle signals for graceful shutdown"""
    logger.info(f"Received signal {sig}, initiating shutdown")
    SHUTDOWN_FLAG.set()
    stop_all_processes()
    sys.exit(0)

def register_signal_handlers():
    """Register signal handlers for graceful shutdown"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def main():
    """Main function"""
    # Register signal handlers
    register_signal_handlers()
    
    # Register cleanup function
    atexit.register(stop_all_processes)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process Control for Trading System')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start a trading process')
    start_parser.add_argument('process_type', choices=['backtest', 'paper_trading', 'live_trading'],
                             help='Type of process to start')
    start_parser.add_argument('--start-date', help='Start date for backtest (YYYY-MM-DD)')
    start_parser.add_argument('--end-date', help='End date for backtest (YYYY-MM-DD)')
    start_parser.add_argument('--initial-capital', type=float, help='Initial capital')
    start_parser.add_argument('--max-signals', type=int, help='Maximum number of signals')
    start_parser.add_argument('--random-seed', type=int, help='Random seed for reproducibility')
    
    # Stop command
    stop_parser = subparsers.add_parser('stop', help='Stop a trading process')
    stop_parser.add_argument('process_id', help='Process ID to stop')
    stop_parser.add_argument('--timeout', type=int, default=30,
                            help='Timeout in seconds for graceful shutdown')
    
    # Stop all command
    stop_all_parser = subparsers.add_parser('stop-all', help='Stop all trading processes')
    stop_all_parser.add_argument('--timeout', type=int, default=30,
                               help='Timeout in seconds for graceful shutdown')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all trading processes')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Get information about a process')
    info_parser.add_argument('process_id', help='Process ID to get information about')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old process records')
    cleanup_parser.add_argument('--days', type=int, default=7,
                              help='Number of days to keep process records')
    
    args = parser.parse_args()
    
    # Load process info from file
    load_process_info()
    
    # Execute command
    if args.command == 'start':
        # Prepare arguments
        process_args = {}
        if hasattr(args, 'start_date') and args.start_date:
            process_args['start_date'] = args.start_date
        if hasattr(args, 'end_date') and args.end_date:
            process_args['end_date'] = args.end_date
        if hasattr(args, 'initial_capital') and args.initial_capital:
            process_args['initial_capital'] = args.initial_capital
        if hasattr(args, 'max_signals') and args.max_signals:
            process_args['max_signals'] = args.max_signals
        if hasattr(args, 'random_seed') and args.random_seed:
            process_args['random_seed'] = args.random_seed
        
        # Start process
        process_id = start_process(args.process_type, process_args)
        if process_id:
            print(f"Started {args.process_type} process with ID {process_id}")
        else:
            print(f"Failed to start {args.process_type} process")
            return 1
    
    elif args.command == 'stop':
        # Stop process
        if stop_process(args.process_id, args.timeout):
            print(f"Stopped process {args.process_id}")
        else:
            print(f"Failed to stop process {args.process_id}")
            return 1
    
    elif args.command == 'stop-all':
        # Stop all processes
        if stop_all_processes(args.timeout):
            print("Stopped all processes")
        else:
            print("Failed to stop some processes")
            return 1
    
    elif args.command == 'list':
        # List processes
        processes = list_processes()
        if not processes:
            print("No processes found")
        else:
            print(f"Found {len(processes)} processes:")
            for pid, info in processes.items():
                status = info['status']
                process_type = info['type']
                start_time = info['start_time'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(info['start_time'], datetime) else info['start_time']
                
                if status == 'running' and not is_process_running(pid):
                    status = 'terminated'
                
                print(f"  {pid}: {process_type} ({status}) - Started at {start_time}")
    
    elif args.command == 'info':
        # Get process info
        info = get_process_info(args.process_id)
        if not info:
            print(f"Process {args.process_id} not found")
            return 1
        else:
            print(f"Process Information for {args.process_id}:")
            print(f"  Type: {info['type']}")
            print(f"  Status: {info['status']}")
            print(f"  Command: {info['command']}")
            print(f"  Start Time: {info['start_time']}")
            if 'end_time' in info:
                print(f"  End Time: {info['end_time']}")
                if 'start_time' in info and isinstance(info['start_time'], datetime) and isinstance(info['end_time'], datetime):
                    duration = info['end_time'] - info['start_time']
                    print(f"  Duration: {duration}")
            if 'return_code' in info:
                print(f"  Return Code: {info['return_code']}")
            print(f"  Arguments: {info['args']}")
    
    elif args.command == 'cleanup':
        # Clean up old process records
        cleanup_old_processes(args.days)
        print(f"Cleaned up process records older than {args.days} days")
    
    else:
        parser.print_help()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
