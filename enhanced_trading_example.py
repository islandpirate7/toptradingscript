#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Trading Example
-----------------------
This script demonstrates how to use the enhanced trading functions with the existing multi-strategy system.
It shows how to integrate adaptive position sizing and ML-based strategy selection.
"""

import sys
import logging
import datetime as dt
import numpy as np
import traceback
import copy
from typing import List, Dict, Any

# Import the multi-strategy system
from multi_strategy_system import MultiStrategySystem, SystemConfig, Signal, MarketRegime

# Import enhanced trading functions
from enhanced_trading_functions import (
    calculate_adaptive_position_size,
    filter_signals,
    generate_ml_signals
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('enhanced_trading.log')
    ]
)

logger = logging.getLogger("EnhancedTrading")

def main():
    """Main function to demonstrate enhanced trading features"""
    logger.info("Starting Enhanced Trading Example")
    
    # Initialize the multi-strategy system with configuration
    # (This would typically load from a config file)
    system = MultiStrategySystem(SystemConfig())
    
    # Example of how to use the enhanced position sizing
    def enhanced_position_sizing_example():
        logger.info("Demonstrating Enhanced Position Sizing")
        
        # Create a sample signal
        signal = Signal(
            symbol="AAPL",
            direction=1,  # 1 for buy, -1 for sell
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=160.0,
            expiration=dt.datetime.now() + dt.timedelta(days=5),
            strategy="TrendFollowing"
        )
        
        # Add score and metadata to the signal
        signal.score = 0.85
        signal.metadata = {
            "predicted_performance": 0.12,
            "regime_weight": 1.2
        }
        
        # Calculate position size using the enhanced method
        position_size = calculate_adaptive_position_size(
            signal=signal,
            market_state=system.market_state,
            candle_data=system.candle_data,
            current_equity=system.current_equity,
            position_sizing_config=system.position_sizing_config,
            logger=logger
        )
        
        logger.info(f"Enhanced position size for {signal.symbol}: ${position_size:.2f}")
        
        # Compare with original method
        original_position_size = system._calculate_position_size(signal)
        logger.info(f"Original position size for {signal.symbol}: ${original_position_size:.2f}")
        
        return position_size
    
    # Example of how to use ML-based signal generation
    def enhanced_signal_generation_example():
        logger.info("Demonstrating ML-Based Signal Generation")
        
        # Generate signals using ML-based strategy selection
        all_signals = generate_ml_signals(
            stocks=system.config.stocks,
            strategies=system.strategies,
            candle_data=system.candle_data,
            market_state=system.market_state,
            ml_strategy_selector=system.ml_strategy_selector,
            logger=logger
        )
        
        logger.info(f"Generated {len(all_signals)} signals using ML-based strategy selection")
        
        # Apply enhanced quality filters
        filtered_signals = filter_signals(
            signals=all_signals,
            candle_data=system.candle_data,
            config=system.config,
            signal_quality_filters=system.signal_quality_filters,
            logger=logger
        )
        
        logger.info(f"After filtering: {len(filtered_signals)} signals passed quality filters")
        
        return filtered_signals
    
    # Example of how to integrate with the existing system
    def integrate_with_system():
        logger.info("Demonstrating Integration with Existing System")
        
        # Store original methods
        original_generate_signals = system._generate_signals
        original_calculate_position_size = system._calculate_position_size
        
        # Override methods with enhanced versions
        def enhanced_generate_signals(self):
            logger.info("Using enhanced signal generation")
            
            # Clear previous signals
            self.signals = []
            
            # Generate signals using ML-based strategy selection
            all_signals = generate_ml_signals(
                self.config.stocks,
                self.strategies,
                self.candle_data,
                self.market_state,
                self.ml_strategy_selector,
                self.logger
            )
            
            # Apply quality filters
            filtered_signals = filter_signals(
                all_signals,
                self.candle_data,
                self.config,
                self.signal_quality_filters,
                self.logger
            )
            
            # Add filtered signals to the system
            self.signals.extend(filtered_signals)
            
            # Log signal generation summary
            self.logger.info(f"Generated {len(all_signals)} signals, {len(filtered_signals)} passed quality filters")
        
        def enhanced_calculate_position_size(self, signal):
            logger.info("Using enhanced position sizing")
            
            return calculate_adaptive_position_size(
                signal=signal,
                market_state=self.market_state,
                candle_data=self.candle_data,
                current_equity=self.current_equity,
                position_sizing_config=self.position_sizing_config,
                logger=self.logger
            )
        
        # Apply the enhanced methods
        system._generate_signals = enhanced_generate_signals.__get__(system)
        system._calculate_position_size = enhanced_calculate_position_size.__get__(system)
        
        logger.info("Enhanced methods have been integrated with the system")
        
        # Run the system with enhanced methods
        # (This would typically call system.start() or system.run_backtest())
        
        # Restore original methods if needed
        # system._generate_signals = original_generate_signals
        # system._calculate_position_size = original_calculate_position_size
    
    # Run the examples
    try:
        # Initialize the system
        system.market_state = system.market_analyzer.analyze_market(
            market_data=system.market_data,
            vix_data=system.vix_data
        )
        
        # Run the examples
        enhanced_position_sizing_example()
        enhanced_signal_generation_example()
        integrate_with_system()
        
        logger.info("Enhanced Trading Example completed successfully")
    except Exception as e:
        logger.error(f"Error in Enhanced Trading Example: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
