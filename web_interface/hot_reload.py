#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hot Reload Module for S&P 500 Trading Strategy Web Interface
Monitors configuration files for changes and triggers reload
"""

import os
import time
import threading
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)

class ConfigFileHandler(FileSystemEventHandler):
    """Handler for config file change events"""
    
    def __init__(self, callback, file_patterns=None):
        """Initialize the handler with a callback function
        
        Args:
            callback: Function to call when a config file changes
            file_patterns: List of file patterns to watch (e.g., ['*.yaml', '*.json'])
        """
        self.callback = callback
        self.file_patterns = file_patterns or ['*.yaml', '*.json']
        self.last_modified = {}
        self.cooldown = 1.0  # Cooldown period in seconds to avoid multiple reloads
        
    def on_modified(self, event):
        """Called when a file is modified"""
        if not event.is_directory:
            file_path = event.src_path
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # Check if the file extension matches our patterns
            if any(file_path.endswith(pattern.replace('*', '')) for pattern in self.file_patterns):
                current_time = time.time()
                
                # Check if we're in the cooldown period
                if file_path in self.last_modified and current_time - self.last_modified[file_path] < self.cooldown:
                    return
                
                self.last_modified[file_path] = current_time
                logger.info(f"Config file changed: {file_path}")
                self.callback(file_path)

class HotReloader:
    """Hot reloader for configuration files"""
    
    def __init__(self, directories, callback, file_patterns=None):
        """Initialize the hot reloader
        
        Args:
            directories: List of directories to watch
            callback: Function to call when a config file changes
            file_patterns: List of file patterns to watch
        """
        self.directories = directories if isinstance(directories, list) else [directories]
        self.callback = callback
        self.file_patterns = file_patterns
        self.observer = None
        self.running = False
        
    def start(self):
        """Start watching for file changes"""
        if self.running:
            return
            
        self.observer = Observer()
        handler = ConfigFileHandler(self.callback, self.file_patterns)
        
        for directory in self.directories:
            if os.path.exists(directory):
                self.observer.schedule(handler, directory, recursive=True)
                logger.info(f"Watching directory for changes: {directory}")
            else:
                logger.warning(f"Directory does not exist: {directory}")
        
        self.observer.start()
        self.running = True
        logger.info("Hot reload started")
        
    def stop(self):
        """Stop watching for file changes"""
        if self.observer and self.running:
            self.observer.stop()
            self.observer.join()
            self.running = False
            logger.info("Hot reload stopped")

def start_hot_reload(app, config_file):
    """Start hot reload for a Flask app
    
    Args:
        app: Flask application
        config_file: Path to the main config file
    """
    config_dir = os.path.dirname(config_file)
    
    def reload_config(file_path):
        """Reload configuration when a file changes"""
        import sys
        from importlib import reload
        
        # Get the app module
        from web_interface import app as app_module
        
        # Reload the module
        reload(app_module)
        
        # Call load_config
        app_module.load_config()
        logger.info("Configuration reloaded successfully")
    
    reloader = HotReloader(config_dir, reload_config, ['*.yaml', '*.json'])
    
    # Start in a separate thread
    thread = threading.Thread(target=reloader.start)
    thread.daemon = True
    thread.start()
    
    # Store the reloader in the app context
    app.hot_reloader = reloader
    
    return reloader
