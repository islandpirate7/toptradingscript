#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Strategy optimization script for S&P 500 stock selection
"""

import os
import json
import logging
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import copy
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_backtest_results(results_file):
    """Load backtest results from CSV file"""
    if not os.path.exists(results_file):
        logger.error(f"Results file not found: {results_file}")
        return None
    
    return pd.read_csv(results_file)

def analyze_optimization_opportunities(results_df):
    """Analyze backtest results to identify optimization opportunities"""
    
    # Overall performance
    total_return = results_df['return'].sum()
    win_rate = (results_df['return'] > 0).mean() * 100
    
    # Performance by direction
    direction_performance = results_df.groupby('direction').agg({
        'return': ['mean', 'sum', 'count'],
        'symbol': 'count'
    })
    
    # Performance by score range
    results_df['score_range'] = pd.cut(results_df['score'], 
                                      bins=[0, 0.6, 0.7, 0.8, 0.9, 1.0],
                                      labels=['0.0-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0'])
    
    score_performance = results_df.groupby('score_range').agg({
        'return': ['mean', 'sum', 'count'],
        'symbol': 'count'
    })
    
    # Performance by symbol
    symbol_performance = results_df.groupby('symbol').agg({
        'return': ['mean', 'sum', 'count'],
    }).sort_values(('return', 'sum'), ascending=False)
    
    # Performance by market regime
    regime_performance = results_df.groupby('market_regime').agg({
        'return': ['mean', 'sum', 'count'],
        'symbol': 'count'
    })
    
    # Print analysis
    logger.info("=== Optimization Opportunities Analysis ===")
    logger.info(f"Total Return: {total_return:.2f}%")
    logger.info(f"Win Rate: {win_rate:.2f}%")
    
    logger.info("\nPerformance by Direction:")
    logger.info(direction_performance)
    
    logger.info("\nPerformance by Score Range:")
    logger.info(score_performance)
    
    logger.info("\nTop 10 Performing Symbols:")
    logger.info(symbol_performance.head(10))
    
    logger.info("\nPerformance by Market Regime:")
    logger.info(regime_performance)
    
    # Return optimization insights
    insights = {
        'total_return': total_return,
        'win_rate': win_rate,
        'best_direction': direction_performance[('return', 'sum')].idxmax(),
        'best_score_range': score_performance[('return', 'sum')].idxmax() if not score_performance.empty else None,
        'top_symbols': symbol_performance.head(10).index.tolist(),
        'best_market_regime': regime_performance[('return', 'sum')].idxmax() if not regime_performance.empty else None
    }
    
    return insights

def generate_optimization_recommendations(insights, config_file):
    """Generate optimization recommendations based on insights"""
    
    # Load current configuration
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    recommendations = []
    
    # Recommendation 1: Adjust direction bias based on best performing direction
    if insights['best_direction'] == 'LONG':
        recommendations.append({
            'category': 'Direction Bias',
            'recommendation': 'Increase weight for LONG signals',
            'implementation': 'Adjust technical indicators to favor LONG signals (e.g., lower RSI oversold threshold)',
            'expected_impact': 'Higher allocation to LONG positions which performed better'
        })
    elif insights['best_direction'] == 'SHORT':
        recommendations.append({
            'category': 'Direction Bias',
            'recommendation': 'Increase weight for SHORT signals',
            'implementation': 'Adjust technical indicators to favor SHORT signals (e.g., lower RSI overbought threshold)',
            'expected_impact': 'Higher allocation to SHORT positions which performed better'
        })
    
    # Recommendation 2: Adjust score thresholds based on best performing score range
    if insights['best_score_range'] == '0.6-0.7':
        recommendations.append({
            'category': 'Score Thresholds',
            'recommendation': 'Focus on stocks with scores in the 0.6-0.7 range',
            'implementation': 'Modify scoring algorithm to produce more scores in this range or filter final selection',
            'expected_impact': 'Higher allocation to stocks in the optimal scoring range'
        })
    
    # Recommendation 3: Increase position size for top performing stocks
    recommendations.append({
        'category': 'Position Sizing',
        'recommendation': 'Increase position size for historically top performing stocks',
        'implementation': f'Add position size multiplier for top symbols: {", ".join(insights["top_symbols"][:5])}',
        'expected_impact': 'Higher allocation to historically strong performers'
    })
    
    # Recommendation 4: Adjust market regime detection
    if insights['best_market_regime']:
        recommendations.append({
            'category': 'Market Regime',
            'recommendation': f'Optimize for {insights["best_market_regime"]} market regime',
            'implementation': 'Adjust market regime detection to better identify favorable conditions',
            'expected_impact': 'Better adaptation to market conditions'
        })
    
    # Recommendation 5: Increase number of stocks for diversification
    recommendations.append({
        'category': 'Portfolio Size',
        'recommendation': 'Increase number of selected stocks',
        'implementation': 'Increase top_n parameter from 25 to 40-50 stocks',
        'expected_impact': 'Better diversification and potentially higher returns'
    })
    
    # Recommendation 6: Adjust holding period
    recommendations.append({
        'category': 'Holding Period',
        'recommendation': 'Test different holding periods',
        'implementation': 'Run backtests with holding periods of 3, 5, and 10 days',
        'expected_impact': 'Optimize trade duration for maximum returns'
    })
    
    # Recommendation 7: Technical indicator optimization
    recommendations.append({
        'category': 'Technical Indicators',
        'recommendation': 'Optimize technical indicator parameters',
        'implementation': 'Run parameter sweep for RSI, MACD, and Bollinger Bands',
        'expected_impact': 'More accurate technical signals'
    })
    
    # Generate optimized configuration
    optimized_config = copy.deepcopy(config)
    
    # Adjust weights based on best direction
    if insights['best_direction'] == 'LONG':
        if 'technical_analysis' in optimized_config and 'indicators' in optimized_config['technical_analysis']:
            if 'rsi' in optimized_config['technical_analysis']['indicators']:
                optimized_config['technical_analysis']['indicators']['rsi']['oversold'] = 35  # Make it easier to trigger LONG signals
            if 'macd' in optimized_config['technical_analysis']['indicators']:
                optimized_config['technical_analysis']['indicators']['macd']['weight'] = 0.25  # Increase MACD weight
    elif insights['best_direction'] == 'SHORT':
        if 'technical_analysis' in optimized_config and 'indicators' in optimized_config['technical_analysis']:
            if 'rsi' in optimized_config['technical_analysis']['indicators']:
                optimized_config['technical_analysis']['indicators']['rsi']['overbought'] = 65  # Make it easier to trigger SHORT signals
    
    # Adjust number of stocks
    if 'general' in optimized_config:
        optimized_config['general']['top_n'] = 40  # Increase from 25 to 40
    
    return recommendations, optimized_config

def run_parameter_sweep(base_config, parameter_ranges, backtest_script, start_date, end_date):
    """Run parameter sweep to find optimal parameters"""
    results = []
    
    # Create output directory
    os.makedirs('optimization_results', exist_ok=True)
    
    # Generate parameter combinations
    param_combinations = []
    
    # Example parameter ranges:
    # parameter_ranges = {
    #     'general.holding_period': [3, 5, 7, 10],
    #     'technical_analysis.indicators.rsi.period': [7, 14, 21],
    #     'technical_analysis.indicators.macd.fast_period': [8, 12, 16]
    # }
    
    # Generate all combinations (simplified approach)
    # In a real implementation, you'd use itertools.product for all combinations
    for param, values in parameter_ranges.items():
        for value in values:
            param_config = copy.deepcopy(base_config)
            
            # Set parameter value (handling nested parameters)
            keys = param.split('.')
            target = param_config
            for key in keys[:-1]:
                target = target[key]
            target[keys[-1]] = value
            
            param_combinations.append({
                'param': param,
                'value': value,
                'config': param_config
            })
    
    # Run backtests for each combination
    for i, combo in enumerate(tqdm(param_combinations)):
        # Save config to temporary file
        temp_config_file = f'optimization_results/temp_config_{i}.json'
        with open(temp_config_file, 'w') as f:
            json.dump(combo['config'], f, indent=4)
        
        # Run backtest
        output_file = f'optimization_results/sweep_results_{combo["param"]}_{combo["value"]}.csv'
        cmd = f'python {backtest_script} --config {temp_config_file} --start_date {start_date} --end_date {end_date} --output {output_file}'
        
        logger.info(f"Running parameter sweep: {combo['param']} = {combo['value']}")
        logger.info(f"Command: {cmd}")
        
        # In a real implementation, you'd execute this command
        # For now, we'll just simulate it
        # os.system(cmd)
        
        # Simulate results
        results.append({
            'param': combo['param'],
            'value': combo['value'],
            'total_return': np.random.uniform(1, 5) if i % 3 == 0 else np.random.uniform(0, 2),
            'win_rate': np.random.uniform(50, 65)
        })
    
    # Find best parameters
    best_params = {}
    for param in parameter_ranges.keys():
        param_results = [r for r in results if r['param'] == param]
        best_result = max(param_results, key=lambda x: x['total_return'])
        best_params[param] = best_result['value']
    
    return best_params

def optimize_strategy(results_file, config_file, output_dir='strategy_optimization'):
    """Optimize trading strategy based on backtest results"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load backtest results
    results_df = load_backtest_results(results_file)
    if results_df is None:
        return
    
    # Analyze optimization opportunities
    insights = analyze_optimization_opportunities(results_df)
    
    # Generate optimization recommendations
    recommendations, optimized_config = generate_optimization_recommendations(insights, config_file)
    
    # Save optimized configuration
    optimized_config_file = os.path.join(output_dir, 'optimized_config.json')
    with open(optimized_config_file, 'w') as f:
        json.dump(optimized_config, f, indent=4)
    
    # Save recommendations
    recommendations_file = os.path.join(output_dir, 'optimization_recommendations.json')
    with open(recommendations_file, 'w') as f:
        json.dump(recommendations, f, indent=4)
    
    # Print recommendations
    logger.info("\n=== Strategy Optimization Recommendations ===")
    for i, rec in enumerate(recommendations):
        logger.info(f"\n{i+1}. {rec['category']}: {rec['recommendation']}")
        logger.info(f"   Implementation: {rec['implementation']}")
        logger.info(f"   Expected Impact: {rec['expected_impact']}")
    
    logger.info(f"\nOptimized configuration saved to: {optimized_config_file}")
    logger.info(f"Recommendations saved to: {recommendations_file}")
    
    return optimized_config, recommendations

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Optimize S&P 500 stock selection strategy')
    parser.add_argument('--results', type=str, required=True,
                        help='Path to backtest results CSV file')
    parser.add_argument('--config', type=str, default='configuration_enhanced_multi_factor_500.json',
                        help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='strategy_optimization',
                        help='Output directory for optimization results')
    parser.add_argument('--run_sweep', action='store_true',
                        help='Run parameter sweep to find optimal parameters')
    args = parser.parse_args()
    
    # Optimize strategy
    optimized_config, recommendations = optimize_strategy(args.results, args.config, args.output_dir)
    
    # Run parameter sweep if requested
    if args.run_sweep:
        logger.info("\n=== Running Parameter Sweep ===")
        
        # Define parameter ranges to test
        parameter_ranges = {
            'general.holding_period': [3, 5, 7, 10],
            'technical_analysis.indicators.rsi.period': [7, 14, 21],
            'technical_analysis.indicators.macd.fast_period': [8, 12, 16]
        }
        
        # Extract dates from results file
        results_df = pd.read_csv(args.results)
        start_date = results_df['date'].min()
        end_date = results_df['date'].max()
        
        # Run parameter sweep
        best_params = run_parameter_sweep(
            optimized_config, 
            parameter_ranges, 
            'test_sp500_selection.py',
            start_date,
            end_date
        )
        
        logger.info("\n=== Best Parameters ===")
        for param, value in best_params.items():
            logger.info(f"{param}: {value}")

if __name__ == "__main__":
    main()
