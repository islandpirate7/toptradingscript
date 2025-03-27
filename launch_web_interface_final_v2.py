#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Launch Web Interface Final V2
This script launches the fixed web interface with all issues resolved
"""

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
    """Main function"""
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
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)

if __name__ == '__main__':
    main()
