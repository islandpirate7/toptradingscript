#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test TA-Lib Installation
------------------------
This script tests if TA-Lib is properly installed and working.
"""

import numpy as np
import pandas as pd
import talib
import matplotlib.pyplot as plt

def test_talib():
    """Test if TA-Lib is properly installed and working"""
    print("Testing TA-Lib installation...")
    
    # Create sample data
    data = np.random.random(100)
    
    # Try some TA-Lib functions
    try:
        # Calculate SMA
        sma = talib.SMA(data, timeperiod=14)
        print("SMA calculation successful")
        
        # Calculate RSI
        rsi = talib.RSI(data, timeperiod=14)
        print("RSI calculation successful")
        
        # Calculate MACD
        macd, macdsignal, macdhist = talib.MACD(data)
        print("MACD calculation successful")
        
        # Calculate Bollinger Bands
        upper, middle, lower = talib.BBANDS(data, timeperiod=20)
        print("Bollinger Bands calculation successful")
        
        print("\nTA-Lib is properly installed and working!")
        return True
    
    except Exception as e:
        print(f"\nError testing TA-Lib: {e}")
        return False

if __name__ == "__main__":
    test_talib()
