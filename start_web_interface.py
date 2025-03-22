#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Launcher for S&P 500 Trading Strategy Web Interface
"""

import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description='Start the S&P 500 Trading Strategy Web Interface')
    parser.add_argument('--host', default='127.0.0.1', help='Host to run the web interface on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the web interface on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    args = parser.parse_args()
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the web interface app.py
    app_path = os.path.join(script_dir, 'web_interface', 'app.py')
    
    # Ensure the web_interface directory exists
    web_interface_dir = os.path.join(script_dir, 'web_interface')
    if not os.path.exists(web_interface_dir):
        os.makedirs(web_interface_dir)
        print(f"Created directory: {web_interface_dir}")
    
    # Ensure the templates and static directories exist
    templates_dir = os.path.join(web_interface_dir, 'templates')
    static_dir = os.path.join(web_interface_dir, 'static')
    css_dir = os.path.join(static_dir, 'css')
    js_dir = os.path.join(static_dir, 'js')
    
    for directory in [templates_dir, static_dir, css_dir, js_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
    
    # Check if app.py exists
    if not os.path.exists(app_path):
        print(f"Error: {app_path} does not exist.")
        print("Please ensure the web interface files are properly set up.")
        sys.exit(1)
    
    # Start the Flask app
    print(f"Starting web interface on http://{args.host}:{args.port}")
    cmd = [sys.executable, app_path]
    
    # Add environment variables for Flask
    env = os.environ.copy()
    env['FLASK_APP'] = app_path
    env['FLASK_HOST'] = args.host
    env['FLASK_PORT'] = str(args.port)
    
    if args.debug:
        env['FLASK_ENV'] = 'development'
        env['FLASK_DEBUG'] = '1'
        print("Running in debug mode")
    
    try:
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        print("\nWeb interface stopped")

if __name__ == "__main__":
    main()
