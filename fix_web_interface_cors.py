#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix script for web interface CORS and JSON response issues
This script will:
1. Update the Flask app to properly handle CORS requests
2. Fix JSON response handling in the web interface
3. Ensure proper error handling for API endpoints
"""

import os
import json
import logging
import traceback
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"fix_web_interface_cors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

def fix_web_interface():
    """Fix issues with the web interface"""
    try:
        # Get the path to the web interface directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        web_interface_dir = os.path.join(script_dir, 'new_web_interface')
        
        # Ensure web interface directory exists
        if not os.path.exists(web_interface_dir):
            logger.error(f"Web interface directory not found: {web_interface_dir}")
            return False
        
        # Fix the app.py file to handle CORS properly
        app_py_path = os.path.join(web_interface_dir, 'app.py')
        
        if os.path.exists(app_py_path):
            logger.info(f"Fixing app.py: {app_py_path}")
            
            # Read the file
            with open(app_py_path, 'r') as f:
                content = f.read()
            
            # Create a backup of the original file
            backup_path = f"{app_py_path}.bak"
            with open(backup_path, 'w') as f:
                f.write(content)
            logger.info(f"Created backup of app.py: {backup_path}")
            
            # Update the CORS headers function
            if '@app.after_request\ndef add_cors_headers(response):' in content:
                # Replace the existing CORS function with an improved version
                old_cors_code = """@app.after_request
def add_cors_headers(response):
    """
                
                new_cors_code = """@app.after_request
def add_cors_headers(response):
    """Add CORS headers to all responses"""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    return response

@app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
@app.route('/<path:path>', methods=['OPTIONS'])
def handle_options(path):
    """Handle OPTIONS requests"""
    response = app.make_default_options_response()
    add_cors_headers(response)
    return response
"""
                
                # Find the start and end of the existing CORS function
                start_idx = content.find('@app.after_request\ndef add_cors_headers(response):')
                end_idx = content.find('@app.route', start_idx)
                
                if start_idx >= 0 and end_idx >= 0:
                    # Replace the existing CORS function
                    new_content = content[:start_idx] + new_cors_code + content[end_idx:]
                    
                    # Save the modified file
                    with open(app_py_path, 'w') as f:
                        f.write(new_content)
                    
                    logger.info("Updated CORS headers in app.py")
                else:
                    logger.warning("Could not find CORS function boundaries in app.py")
            else:
                # Add new CORS headers function
                new_cors_code = """
@app.after_request
def add_cors_headers(response):
    """Add CORS headers to all responses"""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    return response

@app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
@app.route('/<path:path>', methods=['OPTIONS'])
def handle_options(path):
    """Handle OPTIONS requests"""
    response = app.make_default_options_response()
    add_cors_headers(response)
    return response
"""
                # Find the line after the imports
                import_end = content.find('# Initialize logging')
                if import_end > 0:
                    # Insert CORS code after imports
                    new_content = content[:import_end] + new_cors_code + content[import_end:]
                    
                    # Save the modified file
                    with open(app_py_path, 'w') as f:
                        f.write(new_content)
                    
                    logger.info("Added CORS headers to app.py")
                else:
                    logger.warning("Could not find suitable location to add CORS headers in app.py")
            
            # Fix the run_comprehensive_backtest function to properly handle errors
            if '@app.route(\'/run_comprehensive_backtest\', methods=[\'POST\'])' in content:
                # Find the start of the function
                start_idx = content.find('@app.route(\'/run_comprehensive_backtest\', methods=[\'POST\'])')
                if start_idx >= 0:
                    # Find the function definition
                    func_start = content.find('def run_comprehensive_backtest():', start_idx)
                    if func_start >= 0:
                        # Find the try block
                        try_start = content.find('try:', func_start)
                        if try_start >= 0:
                            # Find the except block
                            except_start = content.find('except Exception as e:', try_start)
                            if except_start >= 0:
                                # Find the end of the except block
                                except_end = content.find('return redirect(url_for(\'index\'))', except_start)
                                if except_end >= 0:
                                    # Replace the except block with improved error handling
                                    old_except = content[except_start:except_end + len('return redirect(url_for(\'index\'))')]
                                    new_except = """except Exception as e:
        logger.error(f"Error running comprehensive backtest: {str(e)}", exc_info=True)
        error_message = f'Error running comprehensive backtest: {str(e)}'
        
        # Check if this is an AJAX request
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'error': error_message}), 500
        
        # For regular form submissions, use flash and redirect
        flash(error_message, 'danger')
        return redirect(url_for('index'))"""
                                    
                                    # Replace the except block
                                    new_content = content.replace(old_except, new_except)
                                    
                                    # Save the modified file
                                    with open(app_py_path, 'w') as f:
                                        f.write(new_content)
                                    
                                    logger.info("Updated error handling in run_comprehensive_backtest function")
                                else:
                                    logger.warning("Could not find end of except block in run_comprehensive_backtest function")
                            else:
                                logger.warning("Could not find except block in run_comprehensive_backtest function")
                        else:
                            logger.warning("Could not find try block in run_comprehensive_backtest function")
                    else:
                        logger.warning("Could not find run_comprehensive_backtest function definition")
                else:
                    logger.warning("Could not find run_comprehensive_backtest route")
            
            # Fix the get_backtest_results function to properly handle errors
            if '@app.route(\'/get_backtest_results\')' in content:
                # Find the start of the function
                start_idx = content.find('@app.route(\'/get_backtest_results\')')
                if start_idx >= 0:
                    # Find the function definition
                    func_start = content.find('def get_backtest_results_route():', start_idx)
                    if func_start >= 0:
                        # Replace the entire function with improved error handling
                        func_end = content.find('@app.route', func_start)
                        if func_end >= 0:
                            old_func = content[start_idx:func_end]
                            new_func = """@app.route('/get_backtest_results')
def get_backtest_results_route():
    """Get backtest results"""
    try:
        results = get_backtest_results()
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error getting backtest results: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

"""
                            # Replace the function
                            new_content = content.replace(old_func, new_func)
                            
                            # Save the modified file
                            with open(app_py_path, 'w') as f:
                                f.write(new_content)
                            
                            logger.info("Updated error handling in get_backtest_results function")
                        else:
                            logger.warning("Could not find end of get_backtest_results_route function")
                    else:
                        logger.warning("Could not find get_backtest_results_route function definition")
                else:
                    logger.warning("Could not find get_backtest_results route")
            
            # Create a new fixed version of the web interface app
            fixed_app_py_path = os.path.join(web_interface_dir, 'app_fixed.py')
            with open(app_py_path, 'r') as f:
                content = f.read()
            
            with open(fixed_app_py_path, 'w') as f:
                f.write(content)
            
            logger.info(f"Created fixed version of app.py: {fixed_app_py_path}")
        
        logger.info("Web interface fix completed")
        return True
    
    except Exception as e:
        logger.error(f"Error fixing web interface: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("Starting web interface CORS fix script")
    
    # Fix web interface
    if fix_web_interface():
        logger.info("Successfully fixed web interface")
    else:
        logger.error("Failed to fix web interface")
    
    logger.info("Fix script completed")
