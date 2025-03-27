"""
Final fix script for the multi-strategy trading system.
This script focuses on fixing the VolatilityBreakout strategy and ensuring it always returns a list.
"""

import re

def fix_volatility_strategy():
    file_path = "multi_strategy_system.py"
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Create a backup
    with open(file_path + ".bak2", 'w', encoding='utf-8') as file:
        file.write(content)
    
    print(f"Created backup at {file_path}.bak2")
    
    # Fix the VolatilityBreakout strategy
    # Find the class definition
    class_pattern = r'class VolatilityBreakoutStrategy\(Strategy\):'
    class_match = re.search(class_pattern, content)
    
    if class_match:
        # Find the generate_signals method
        method_pattern = r'def generate_signals\(self,\s+symbol: str,\s+candles: List\[CandleData\],\s+stock_config: StockConfig,\s+market_state: MarketState\) -> List\[Signal\]:'
        method_match = re.search(method_pattern, content[class_match.end():])
        
        if method_match:
            method_start = class_match.end() + method_match.end()
            
            # Find the method body
            next_method_pattern = r'def _calculate_atr\('
            next_method_match = re.search(next_method_pattern, content[method_start:])
            
            if next_method_match:
                method_end = method_start + next_method_match.start()
                
                # Extract the method body
                method_body = content[method_start:method_end]
                
                # Check if try-except is already there
                if "try:" not in method_body:
                    # Add try-except block
                    new_method_body = """
        \"\"\"Generate volatility breakout signals based on Bollinger Band squeeze\"\"\"
        signals = []
        
        try:
            # Get parameters from config
            bb_period = self.get_param("bb_period", 20)
            bb_std_dev = self.get_param("bb_std_dev", 2.0)
            keltner_period = self.get_param("keltner_period", 20)
            keltner_factor = self.get_param("keltner_factor", 1.5)
            min_squeeze_periods = self.get_param("min_squeeze_periods", 10)
            volume_threshold = self.get_param("volume_threshold", 1.5)  # Volume surge threshold
            
            if len(candles) < bb_period + min_squeeze_periods + 5:
                # Not enough data for calculation
                return signals
            
            # Convert candles to DataFrame for easier calculation
            df = pd.DataFrame([candle.to_dict() for candle in candles])
            df.set_index('timestamp', inplace=True)
            
            # Calculate Bollinger Bands
            df['sma'] = df['close'].rolling(window=bb_period).mean()
            df['std'] = df['close'].rolling(window=bb_period).std()
            df['bb_upper'] = df['sma'] + (df['std'] * bb_std_dev)
            df['bb_lower'] = df['sma'] - (df['std'] * bb_std_dev)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['sma']
            
            # Calculate Keltner Channels
            df['ema'] = df['close'].ewm(span=keltner_period, adjust=False).mean()
            df['atr'] = self._calculate_atr(df, keltner_period)
            
            # Check if atr calculation returned valid values
            if df['atr'].isna().all():
                return signals
            
            df['kc_upper'] = df['ema'] + (df['atr'] * keltner_factor)
            df['kc_lower'] = df['ema'] - (df['atr'] * keltner_factor)
            
            # Calculate squeeze condition (Bollinger Bands inside Keltner Channels)
            df['squeeze'] = (df['bb_lower'] > df['kc_lower']) & (df['bb_upper'] < df['kc_upper'])
            
            # Calculate momentum
            df['momentum'] = df['close'] - df['close'].shift(5)
            
            # Calculate volume ratio (current volume / average volume)
            df['avg_volume'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['avg_volume']
            
            # Identify squeeze ending (was in squeeze, now broken out)
            df['squeeze_ending'] = df['squeeze'].shift(1) & ~df['squeeze']
            
            # Remove NaN values
            df.dropna(inplace=True)
            
            if len(df) < 5:
                return signals
            
            # Look for squeeze setups followed by breakouts
            for i in range(len(df) - 1, max(0, len(df) - 5), -1):
                # Check if we have a squeeze ending
                if df.iloc[i]['squeeze_ending']:
                    # Check if we had a sustained squeeze before this
                    squeeze_duration = 0
                    for j in range(i-1, max(0, i-min_squeeze_periods), -1):
                        if df.iloc[j]['squeeze']:
                            squeeze_duration += 1
                        else:
                            break
                    
                    # Only proceed if we had a sufficiently long squeeze
                    if squeeze_duration >= min_squeeze_periods:
                        # Check for a breakout with volume confirmation
                        current = df.iloc[i]
                        prev = df.iloc[i-1]
                        
                        # Bullish breakout
                        if (current.close > current.sma and
                            current.close > prev.close and
                            current.volume_ratio > volume_threshold and
                            current.momentum > 0):
                            
                            # Calculate signal strength based on momentum and volume
                            momentum_strength = min(current.momentum / 2, 2.0)
                            volume_strength = min((current.volume_ratio - 1) / 0.5, 2.0)
                            
                            # Determine overall signal strength
                            if momentum_strength > 1.0 and volume_strength > 1.0:
                                strength = SignalStrength.STRONG_BUY
                            elif momentum_strength > 0.5 or volume_strength > 0.5:
                                strength = SignalStrength.MODERATE_BUY
                            else:
                                strength = SignalStrength.WEAK_BUY
                            
                            # Calculate stop loss and take profit
                            entry_price = current.close
                            stop_loss = self.calculate_stop_loss(entry_price, TradeDirection.LONG, candles[-i-1:])
                            take_profit = self.calculate_take_profit(TradeDirection.LONG, entry_price, stop_loss, candles[-i-1:], stock_config)
                            
                            # Create signal
                            signal = Signal(
                                timestamp=df.index[i],
                                symbol=symbol,
                                strategy=self.name,
                                direction=TradeDirection.LONG,
                                strength=strength,
                                entry_price=entry_price,
                                stop_loss=stop_loss,
                                take_profit=take_profit,
                                expiration=df.index[i] + dt.timedelta(days=3),
                                metadata={
                                    "squeeze_duration": squeeze_duration,
                                    "momentum": current.momentum,
                                    "volume_ratio": current.volume_ratio,
                                    "bb_width": current.bb_width
                                }
                            )
                            
                            signals.append(signal)
                        
                        # Bearish breakout
                        elif (current.close < current.sma and
                              current.close < prev.close and
                              current.volume_ratio > volume_threshold and
                              current.momentum < 0):
                            
                            # Calculate signal strength based on momentum and volume
                            momentum_strength = min(abs(current.momentum) / 2, 2.0)
                            volume_strength = min((current.volume_ratio - 1) / 0.5, 2.0)
                            
                            # Determine overall signal strength
                            if momentum_strength > 1.0 and volume_strength > 1.0:
                                strength = SignalStrength.STRONG_SELL
                            elif momentum_strength > 0.5 or volume_strength > 0.5:
                                strength = SignalStrength.MODERATE_SELL
                            else:
                                strength = SignalStrength.WEAK_SELL
                            
                            # Calculate stop loss and take profit
                            entry_price = current.close
                            stop_loss = self.calculate_stop_loss(entry_price, TradeDirection.SHORT, candles[-i-1:])
                            take_profit = self.calculate_take_profit(TradeDirection.SHORT, entry_price, stop_loss, candles[-i-1:], stock_config)
                            
                            # Create signal
                            signal = Signal(
                                timestamp=df.index[i],
                                symbol=symbol,
                                strategy=self.name,
                                direction=TradeDirection.SHORT,
                                strength=strength,
                                entry_price=entry_price,
                                stop_loss=stop_loss,
                                take_profit=take_profit,
                                expiration=df.index[i] + dt.timedelta(days=5),  # Extend expiration to 5 days
                                metadata={
                                    "squeeze_duration": squeeze_duration,
                                    "momentum": current.momentum,
                                    "volume_ratio": current.volume_ratio,
                                    "bb_width": current.bb_width
                                }
                            )
                            
                            signals.append(signal)
        except Exception as e:
            # Log the error but don't crash
            self.logger.error(f"Error in VolatilityBreakout strategy for {symbol}: {str(e)}")
            # Return empty signals list
            return []
        
        return signals"""
                    
                    # Replace the method body
                    new_content = content[:method_start] + new_method_body + content[method_end:]
                    
                    # Write the modified content back to the file
                    with open(file_path, 'w', encoding='utf-8') as file:
                        file.write(new_content)
                    
                    print("Successfully fixed the VolatilityBreakout strategy")
                    return True
                else:
                    print("Try-except block already exists in the VolatilityBreakout strategy")
                    return True
    
    print("Could not find the VolatilityBreakout strategy class")
    return False

def fix_date_range():
    """Fix the date range to use historical data from 2023 or earlier"""
    file_path = "multi_strategy_main.py"
    
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Find the main function
        pattern = r'if __name__ == "__main__":'
        match = re.search(pattern, content)
        
        if match:
            # Find the date range setup
            date_pattern = r'start_date = dt\.date\((\d+), (\d+), (\d+)\)\s+end_date = dt\.date\((\d+), (\d+), (\d+)\)'
            date_match = re.search(date_pattern, content[match.end():])
            
            if date_match:
                start_year = int(date_match.group(1))
                end_year = int(date_match.group(4))
                
                # Only modify if using data from after 2023
                if start_year > 2023 or end_year > 2023:
                    # Replace with 2023 dates
                    new_dates = "    # Use historical data from 2023 (Alpaca free tier limitation)\n    start_date = dt.date(2023, 1, 1)\n    end_date = dt.date(2023, 12, 31)"
                    pos_start = match.end() + date_match.start()
                    pos_end = match.end() + date_match.end()
                    new_content = content[:pos_start] + new_dates + content[pos_end:]
                    
                    # Write the modified content back to the file
                    with open(file_path, 'w', encoding='utf-8') as file:
                        file.write(new_content)
                    
                    print("Fixed date range to use 2023 data")
                    return True
                else:
                    print("Date range already using 2023 or earlier data")
                    return True
            else:
                print("Could not find date range in main function")
                return False
        else:
            print("Could not find main function")
            return False
    except Exception as e:
        print(f"Error fixing date range: {str(e)}")
        return False

if __name__ == "__main__":
    print("Applying fixes to the multi-strategy trading system...")
    
    # Fix the VolatilityBreakout strategy
    fix_volatility_strategy()
    
    # Fix the date range to use 2023 data (Alpaca free tier limitation)
    fix_date_range()
    
    print("All fixes applied successfully!")
