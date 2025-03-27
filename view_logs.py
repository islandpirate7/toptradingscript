#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
View log files from the logs directory
"""

import os
import sys
import glob
import argparse
from datetime import datetime

def list_log_files(log_dir='logs'):
    """List all log files in the logs directory"""
    if not os.path.exists(log_dir):
        print(f"Log directory '{log_dir}' does not exist.")
        return []
    
    log_files = glob.glob(os.path.join(log_dir, '*.log'))
    log_files.sort(key=os.path.getmtime, reverse=True)
    
    return log_files

def print_log_files(log_files):
    """Print a list of log files with their creation time"""
    if not log_files:
        print("No log files found.")
        return
    
    print(f"Found {len(log_files)} log files:")
    for i, log_file in enumerate(log_files):
        mtime = datetime.fromtimestamp(os.path.getmtime(log_file))
        size = os.path.getsize(log_file)
        print(f"{i+1}. {os.path.basename(log_file)} - {mtime.strftime('%Y-%m-%d %H:%M:%S')} - {size/1024:.1f} KB")

def view_log_file(log_file, lines=None):
    """View the contents of a log file"""
    if not os.path.exists(log_file):
        print(f"Log file '{log_file}' does not exist.")
        return
    
    print(f"Viewing log file: {log_file}")
    print("-" * 80)
    
    with open(log_file, 'r') as f:
        if lines:
            # Read the last N lines
            content = f.readlines()
            if len(content) > lines:
                print(f"... (showing last {lines} of {len(content)} lines) ...")
                content = content[-lines:]
            print(''.join(content), end='')
        else:
            # Read the entire file
            print(f.read(), end='')
    
    print("-" * 80)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='View log files from the logs directory')
    parser.add_argument('-l', '--list', action='store_true', help='List all log files')
    parser.add_argument('-n', '--number', type=int, default=None, help='Number of log file to view (from list)')
    parser.add_argument('-f', '--file', type=str, default=None, help='Path to log file to view')
    parser.add_argument('--lines', type=int, default=None, help='Number of lines to view (from the end)')
    parser.add_argument('-d', '--dir', type=str, default='logs', help='Log directory')
    
    args = parser.parse_args()
    
    log_files = list_log_files(args.dir)
    
    if args.list or (not args.number and not args.file):
        print_log_files(log_files)
        return
    
    if args.number:
        if not log_files:
            print("No log files found.")
            return
        
        if args.number < 1 or args.number > len(log_files):
            print(f"Invalid log file number. Please choose a number between 1 and {len(log_files)}.")
            return
        
        view_log_file(log_files[args.number - 1], args.lines)
    
    if args.file:
        view_log_file(args.file, args.lines)

if __name__ == "__main__":
    main()
