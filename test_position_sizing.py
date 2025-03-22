import os
import yaml
import logging
import pandas as pd
from datetime import datetime
from final_sp500_strategy import SP500Strategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger()

def test_tiered_position_sizing():
    """
    Test the tiered position sizing functionality with sample signals
    """
    # Load configuration
    with open('sp500_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    logger.info(f"Loaded configuration with tiered position sizing:")
    logger.info(f"Tier 1 (score >= {config['strategy']['position_sizing']['tier1']['min_score']}): {config['strategy']['position_sizing']['tier1']['multiplier']}x multiplier")
    logger.info(f"Tier 2 (score >= {config['strategy']['position_sizing']['tier2']['min_score']}): {config['strategy']['position_sizing']['tier2']['multiplier']}x multiplier")
    logger.info(f"Tier 3 (score >= {config['strategy']['position_sizing']['tier3']['min_score']}): {config['strategy']['position_sizing']['tier3']['multiplier']}x multiplier")
    
    # Create strategy instance
    strategy = SP500Strategy(None, config=config, mode='paper')
    
    # Create sample signals with different scores
    sample_signals = [
        {'symbol': 'AAPL', 'direction': 'LONG', 'score': 0.95, 'price': 200.0, 'sector': 'Technology', 'market_regime': 'BULLISH'},
        {'symbol': 'MSFT', 'direction': 'LONG', 'score': 0.85, 'price': 300.0, 'sector': 'Technology', 'market_regime': 'BULLISH'},
        {'symbol': 'AMZN', 'direction': 'LONG', 'score': 0.75, 'price': 150.0, 'sector': 'Consumer Discretionary', 'market_regime': 'BULLISH'},
        {'symbol': 'GOOGL', 'direction': 'LONG', 'score': 0.65, 'price': 120.0, 'sector': 'Communication Services', 'market_regime': 'BULLISH'},
        {'symbol': 'META', 'direction': 'SHORT', 'score': 0.95, 'price': 400.0, 'sector': 'Communication Services', 'market_regime': 'BULLISH'},
        {'symbol': 'NFLX', 'direction': 'SHORT', 'score': 0.85, 'price': 500.0, 'sector': 'Communication Services', 'market_regime': 'BULLISH'},
        {'symbol': 'TSLA', 'direction': 'SHORT', 'score': 0.75, 'price': 250.0, 'sector': 'Consumer Discretionary', 'market_regime': 'BULLISH'},
        {'symbol': 'JPM', 'direction': 'SHORT', 'score': 0.65, 'price': 150.0, 'sector': 'Financials', 'market_regime': 'BULLISH'},
    ]
    
    # Test position sizing
    available_buying_power = 100000
    logger.info(f"Testing position sizing with available buying power: ${available_buying_power:.2f}")
    
    # Calculate position sizes
    results = []
    for signal in sample_signals:
        position_size = strategy.calculate_position_size(signal, available_buying_power)
        
        # Calculate shares
        shares = int(position_size / signal['price'])
        
        # Store results
        results.append({
            'symbol': signal['symbol'],
            'direction': signal['direction'],
            'score': signal['score'],
            'sector': signal['sector'],
            'price': signal['price'],
            'position_size': position_size,
            'shares': shares,
            'capital_pct': position_size / available_buying_power * 100
        })
    
    # Create DataFrame and display results
    results_df = pd.DataFrame(results)
    
    # Sort by position size (descending)
    results_df = results_df.sort_values('position_size', ascending=False)
    
    # Print results
    logger.info("\nPosition Sizing Results:")
    for _, row in results_df.iterrows():
        logger.info(f"{row['symbol']} ({row['direction']}): Score {row['score']:.2f}, Size ${row['position_size']:.2f}, Shares {row['shares']}, {row['capital_pct']:.2f}% of capital")
    
    # Summarize by tier
    logger.info("\nPosition Sizing by Tier:")
    tier1 = results_df[results_df['score'] >= config['strategy']['position_sizing']['tier1']['min_score']]
    tier2 = results_df[(results_df['score'] >= config['strategy']['position_sizing']['tier2']['min_score']) & (results_df['score'] < config['strategy']['position_sizing']['tier1']['min_score'])]
    tier3 = results_df[(results_df['score'] >= config['strategy']['position_sizing']['tier3']['min_score']) & (results_df['score'] < config['strategy']['position_sizing']['tier2']['min_score'])]
    other = results_df[results_df['score'] < config['strategy']['position_sizing']['tier3']['min_score']]
    
    logger.info(f"Tier 1 (≥{config['strategy']['position_sizing']['tier1']['min_score']}): {len(tier1)} signals, Avg Size ${tier1['position_size'].mean():.2f}")
    logger.info(f"Tier 2 (≥{config['strategy']['position_sizing']['tier2']['min_score']}): {len(tier2)} signals, Avg Size ${tier2['position_size'].mean():.2f}")
    logger.info(f"Tier 3 (≥{config['strategy']['position_sizing']['tier3']['min_score']}): {len(tier3)} signals, Avg Size ${tier3['position_size'].mean():.2f}")
    logger.info(f"Other (<{config['strategy']['position_sizing']['tier3']['min_score']}): {len(other)} signals, Avg Size ${other['position_size'].mean():.2f}")
    
    # Test prioritization
    logger.info("\nTesting signal prioritization...")
    prioritized_signals = strategy.prioritize_signals(sample_signals)
    
    # Print prioritized signals
    logger.info("\nPrioritized Signals:")
    for i, signal in enumerate(prioritized_signals):
        logger.info(f"{i+1}. {signal['symbol']} ({signal['direction']}): Base Score {signal['score']:.2f}, Priority {signal.get('priority', 'N/A'):.2f}")
    
    # Save results to CSV
    os.makedirs("./results/analysis", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(f"./results/analysis/position_sizing_test_{timestamp}.csv", index=False)
    logger.info(f"\nResults saved to ./results/analysis/position_sizing_test_{timestamp}.csv")

if __name__ == "__main__":
    test_tiered_position_sizing()
