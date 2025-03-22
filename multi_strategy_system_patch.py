"""
Patch for MultiStrategySystem to fix the trade_history issue
"""
import sys
import os
import logging
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MultiStrategySystemPatch")

def apply_patch():
    """Apply patches to the MultiStrategySystem class to fix issues"""
    try:
        # Import the module
        from multi_strategy_system import MultiStrategySystem
        
        # Save the original method
        original_generate_backtest_data = MultiStrategySystem._generate_backtest_data
        
        # Define the patched method
        def patched_generate_backtest_data(self, start_date, end_date):
            """Patched version of _generate_backtest_data that initializes trade_history"""
            # Initialize trade_history if it doesn't exist
            if not hasattr(self, 'trade_history'):
                self.trade_history = []
            
            # Create a local reference to self.trade_history to fix the NameError
            def fixed_generate_backtest_data(self, start_date, end_date):
                # Get local reference to avoid NameError
                trade_history = self.trade_history
                
                # Call the original method with the local variable in scope
                return original_generate_backtest_data(self, start_date, end_date)
            
            return fixed_generate_backtest_data(self, start_date, end_date)
        
        # Apply the patch
        MultiStrategySystem._generate_backtest_data = patched_generate_backtest_data
        
        # Also patch the run_backtest method to initialize trade_history
        original_run_backtest = MultiStrategySystem.run_backtest
        
        def patched_run_backtest(self, start_date, end_date, initial_capital=100000):
            """Patched version of run_backtest that initializes trade_history"""
            # Initialize trade_history
            self.trade_history = []
            
            # Call the original method
            return original_run_backtest(self, start_date, end_date, initial_capital)
        
        # Apply the patch
        MultiStrategySystem.run_backtest = patched_run_backtest
        
        logger.info("Successfully applied patch to MultiStrategySystem")
        return True
    except Exception as e:
        logger.error(f"Failed to apply patch: {e}")
        return False

if __name__ == "__main__":
    if apply_patch():
        print("Patch applied successfully")
    else:
        print("Failed to apply patch")
        sys.exit(1)
