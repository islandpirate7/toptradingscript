#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimization Visualizer
----------------------
This script provides visualization tools for analyzing optimization results
and understanding the factors affecting trading system performance.
"""

import os
import sys
import logging
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import traceback
from typing import List, Dict, Any, Tuple
import argparse
import json
from matplotlib.gridspec import GridSpec

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('optimization_visualizer.log')
    ]
)

logger = logging.getLogger("OptimizationVisualizer")

def load_optimization_results(results_file):
    """
    Load optimization results from file
    
    Args:
        results_file: Path to results file (JSON or YAML)
        
    Returns:
        Dict: Optimization results
    """
    try:
        file_ext = os.path.splitext(results_file)[1].lower()
        
        if file_ext == '.json':
            with open(results_file, 'r') as file:
                results = json.load(file)
        elif file_ext in ['.yaml', '.yml']:
            with open(results_file, 'r') as file:
                results = yaml.safe_load(file)
        else:
            logger.error(f"Unsupported file format: {file_ext}")
            return None
        
        logger.info(f"Loaded optimization results from {results_file}")
        return results
    
    except Exception as e:
        logger.error(f"Error loading optimization results: {str(e)}")
        return None

def visualize_sharpe_ratio_factors(results, output_file="sharpe_ratio_factors.png"):
    """
    Visualize factors affecting Sharpe ratio
    
    Args:
        results: Optimization results
        output_file: Output file for visualization
    """
    logger.info("Visualizing Sharpe ratio factors")
    
    try:
        # Extract Sharpe ratio factors
        factors = results.get('sharpe_ratio_factors', {})
        
        if not factors:
            logger.error("No Sharpe ratio factors found in results")
            return
        
        # Create figure
        plt.figure(figsize=(14, 10))
        
        # Create GridSpec for layout
        gs = GridSpec(2, 2, figure=plt.gcf())
        
        # Plot 1: Correlation heatmap
        ax1 = plt.subplot(gs[0, 0])
        
        # Extract correlation data
        corr_data = factors.get('correlations', {})
        if corr_data:
            # Convert to DataFrame
            corr_df = pd.DataFrame(corr_data)
            
            # Plot heatmap
            sns.heatmap(corr_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax1)
            ax1.set_title('Correlation with Sharpe Ratio')
        else:
            ax1.text(0.5, 0.5, "No correlation data available", 
                    horizontalalignment='center', verticalalignment='center')
        
        # Plot 2: Factor importance
        ax2 = plt.subplot(gs[0, 1])
        
        # Extract importance data
        importance_data = factors.get('importance', {})
        if importance_data:
            # Convert to DataFrame
            importance_df = pd.DataFrame({
                'Factor': list(importance_data.keys()),
                'Importance': list(importance_data.values())
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            # Plot bar chart
            sns.barplot(x='Importance', y='Factor', data=importance_df, ax=ax2)
            ax2.set_title('Factor Importance for Sharpe Ratio')
        else:
            ax2.text(0.5, 0.5, "No importance data available", 
                    horizontalalignment='center', verticalalignment='center')
        
        # Plot 3: Sharpe ratio vs. Win Rate
        ax3 = plt.subplot(gs[1, 0])
        
        # Extract win rate data
        win_rate_data = factors.get('win_rate_impact', {})
        if win_rate_data:
            # Convert to DataFrame
            win_rate_df = pd.DataFrame({
                'Win Rate': list(win_rate_data.keys()),
                'Sharpe Ratio': list(win_rate_data.values())
            })
            
            # Plot scatter with regression line
            sns.regplot(x='Win Rate', y='Sharpe Ratio', data=win_rate_df, ax=ax3)
            ax3.set_title('Sharpe Ratio vs. Win Rate')
        else:
            ax3.text(0.5, 0.5, "No win rate impact data available", 
                    horizontalalignment='center', verticalalignment='center')
        
        # Plot 4: Sharpe ratio vs. Trading Frequency
        ax4 = plt.subplot(gs[1, 1])
        
        # Extract trading frequency data
        freq_data = factors.get('trading_frequency_impact', {})
        if freq_data:
            # Convert to DataFrame
            freq_df = pd.DataFrame({
                'Trading Frequency': list(freq_data.keys()),
                'Sharpe Ratio': list(freq_data.values())
            })
            
            # Plot scatter with regression line
            sns.regplot(x='Trading Frequency', y='Sharpe Ratio', data=freq_df, ax=ax4)
            ax4.set_title('Sharpe Ratio vs. Trading Frequency')
        else:
            ax4.text(0.5, 0.5, "No trading frequency impact data available", 
                    horizontalalignment='center', verticalalignment='center')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_file, dpi=300)
        logger.info(f"Sharpe ratio factors visualization saved to {output_file}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error visualizing Sharpe ratio factors: {str(e)}")
        logger.error(traceback.format_exc())

def visualize_ml_strategy_optimization(results, output_file="ml_strategy_optimization.png"):
    """
    Visualize ML strategy optimization results
    
    Args:
        results: Optimization results
        output_file: Output file for visualization
    """
    logger.info("Visualizing ML strategy optimization results")
    
    try:
        # Extract ML strategy optimization results
        ml_results = results.get('ml_strategy_optimization', {})
        
        if not ml_results:
            logger.error("No ML strategy optimization results found")
            return
        
        # Create figure
        plt.figure(figsize=(14, 10))
        
        # Create GridSpec for layout
        gs = GridSpec(2, 2, figure=plt.gcf())
        
        # Plot 1: Hyperparameter importance
        ax1 = plt.subplot(gs[0, 0])
        
        # Extract hyperparameter importance data
        param_importance = ml_results.get('param_importance', {})
        if param_importance:
            # Convert to DataFrame
            param_df = pd.DataFrame({
                'Parameter': list(param_importance.keys()),
                'Importance': list(param_importance.values())
            })
            
            # Sort by importance
            param_df = param_df.sort_values('Importance', ascending=False)
            
            # Plot bar chart
            sns.barplot(x='Importance', y='Parameter', data=param_df, ax=ax1)
            ax1.set_title('Hyperparameter Importance')
        else:
            ax1.text(0.5, 0.5, "No hyperparameter importance data available", 
                    horizontalalignment='center', verticalalignment='center')
        
        # Plot 2: Model performance comparison
        ax2 = plt.subplot(gs[0, 1])
        
        # Extract model performance data
        model_perf = ml_results.get('model_performance', {})
        if model_perf:
            # Convert to DataFrame
            perf_df = pd.DataFrame({
                'Model': list(model_perf.keys()),
                'Score': list(model_perf.values())
            })
            
            # Sort by score
            perf_df = perf_df.sort_values('Score', ascending=False)
            
            # Plot bar chart
            sns.barplot(x='Score', y='Model', data=perf_df, ax=ax2)
            ax2.set_title('Model Performance Comparison')
        else:
            ax2.text(0.5, 0.5, "No model performance data available", 
                    horizontalalignment='center', verticalalignment='center')
        
        # Plot 3: Feature importance
        ax3 = plt.subplot(gs[1, 0])
        
        # Extract feature importance data
        feature_importance = ml_results.get('feature_importance', {})
        if feature_importance:
            # Convert to DataFrame
            feature_df = pd.DataFrame({
                'Feature': list(feature_importance.keys()),
                'Importance': list(feature_importance.values())
            })
            
            # Sort by importance
            feature_df = feature_df.sort_values('Importance', ascending=False)
            
            # Plot bar chart
            sns.barplot(x='Importance', y='Feature', data=feature_df, ax=ax3)
            ax3.set_title('Feature Importance')
        else:
            ax3.text(0.5, 0.5, "No feature importance data available", 
                    horizontalalignment='center', verticalalignment='center')
        
        # Plot 4: Learning curve
        ax4 = plt.subplot(gs[1, 1])
        
        # Extract learning curve data
        learning_curve = ml_results.get('learning_curve', {})
        if learning_curve:
            # Convert to DataFrame
            train_sizes = learning_curve.get('train_sizes', [])
            train_scores = learning_curve.get('train_scores', [])
            test_scores = learning_curve.get('test_scores', [])
            
            # Plot learning curve
            ax4.plot(train_sizes, train_scores, 'o-', color='r', label='Training score')
            ax4.plot(train_sizes, test_scores, 'o-', color='g', label='Cross-validation score')
            ax4.set_title('Learning Curve')
            ax4.set_xlabel('Training examples')
            ax4.set_ylabel('Score')
            ax4.legend(loc='best')
        else:
            ax4.text(0.5, 0.5, "No learning curve data available", 
                    horizontalalignment='center', verticalalignment='center')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_file, dpi=300)
        logger.info(f"ML strategy optimization visualization saved to {output_file}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error visualizing ML strategy optimization: {str(e)}")
        logger.error(traceback.format_exc())

def visualize_signal_filter_optimization(results, output_file="signal_filter_optimization.png"):
    """
    Visualize signal filter optimization results
    
    Args:
        results: Optimization results
        output_file: Output file for visualization
    """
    logger.info("Visualizing signal filter optimization results")
    
    try:
        # Extract signal filter optimization results
        filter_results = results.get('signal_filter_optimization', {})
        
        if not filter_results:
            logger.error("No signal filter optimization results found")
            return
        
        # Create figure
        plt.figure(figsize=(14, 10))
        
        # Create GridSpec for layout
        gs = GridSpec(2, 2, figure=plt.gcf())
        
        # Plot 1: Parameter impact on win rate
        ax1 = plt.subplot(gs[0, 0])
        
        # Extract parameter impact data
        param_impact = filter_results.get('param_impact_win_rate', {})
        if param_impact:
            # Convert to DataFrame
            impact_df = pd.DataFrame({
                'Parameter': list(param_impact.keys()),
                'Impact': list(param_impact.values())
            })
            
            # Sort by impact
            impact_df = impact_df.sort_values('Impact', ascending=False)
            
            # Plot bar chart
            sns.barplot(x='Impact', y='Parameter', data=impact_df, ax=ax1)
            ax1.set_title('Parameter Impact on Win Rate')
        else:
            ax1.text(0.5, 0.5, "No parameter impact data available", 
                    horizontalalignment='center', verticalalignment='center')
        
        # Plot 2: Signal quality distribution
        ax2 = plt.subplot(gs[0, 1])
        
        # Extract signal quality data
        quality_data = filter_results.get('signal_quality_distribution', {})
        if quality_data:
            # Convert to DataFrame
            quality_df = pd.DataFrame({
                'Quality': list(quality_data.keys()),
                'Frequency': list(quality_data.values())
            })
            
            # Plot histogram
            sns.barplot(x='Quality', y='Frequency', data=quality_df, ax=ax2)
            ax2.set_title('Signal Quality Distribution')
        else:
            ax2.text(0.5, 0.5, "No signal quality distribution data available", 
                    horizontalalignment='center', verticalalignment='center')
        
        # Plot 3: Filter threshold vs. performance
        ax3 = plt.subplot(gs[1, 0])
        
        # Extract threshold data
        threshold_data = filter_results.get('threshold_performance', {})
        if threshold_data:
            # Convert to DataFrame
            threshold_df = pd.DataFrame({
                'Threshold': list(threshold_data.keys()),
                'Performance': list(threshold_data.values())
            })
            
            # Plot line chart
            sns.lineplot(x='Threshold', y='Performance', data=threshold_df, ax=ax3)
            ax3.set_title('Filter Threshold vs. Performance')
        else:
            ax3.text(0.5, 0.5, "No threshold performance data available", 
                    horizontalalignment='center', verticalalignment='center')
        
        # Plot 4: Strategy-specific filter performance
        ax4 = plt.subplot(gs[1, 1])
        
        # Extract strategy-specific data
        strategy_data = filter_results.get('strategy_filter_performance', {})
        if strategy_data:
            # Convert to DataFrame
            strategies = []
            performances = []
            
            for strategy, perf in strategy_data.items():
                strategies.append(strategy)
                performances.append(perf)
            
            strategy_df = pd.DataFrame({
                'Strategy': strategies,
                'Performance': performances
            })
            
            # Sort by performance
            strategy_df = strategy_df.sort_values('Performance', ascending=False)
            
            # Plot bar chart
            sns.barplot(x='Performance', y='Strategy', data=strategy_df, ax=ax4)
            ax4.set_title('Strategy-Specific Filter Performance')
        else:
            ax4.text(0.5, 0.5, "No strategy-specific filter performance data available", 
                    horizontalalignment='center', verticalalignment='center')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_file, dpi=300)
        logger.info(f"Signal filter optimization visualization saved to {output_file}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error visualizing signal filter optimization: {str(e)}")
        logger.error(traceback.format_exc())

def visualize_position_sizing_optimization(results, output_file="position_sizing_optimization.png"):
    """
    Visualize position sizing optimization results
    
    Args:
        results: Optimization results
        output_file: Output file for visualization
    """
    logger.info("Visualizing position sizing optimization results")
    
    try:
        # Extract position sizing optimization results
        position_results = results.get('position_sizing_optimization', {})
        
        if not position_results:
            logger.error("No position sizing optimization results found")
            return
        
        # Create figure
        plt.figure(figsize=(14, 10))
        
        # Create GridSpec for layout
        gs = GridSpec(2, 2, figure=plt.gcf())
        
        # Plot 1: Parameter impact on Sharpe ratio
        ax1 = plt.subplot(gs[0, 0])
        
        # Extract parameter impact data
        param_impact = position_results.get('param_impact_sharpe', {})
        if param_impact:
            # Convert to DataFrame
            impact_df = pd.DataFrame({
                'Parameter': list(param_impact.keys()),
                'Impact': list(param_impact.values())
            })
            
            # Sort by impact
            impact_df = impact_df.sort_values('Impact', ascending=False)
            
            # Plot bar chart
            sns.barplot(x='Impact', y='Parameter', data=impact_df, ax=ax1)
            ax1.set_title('Parameter Impact on Sharpe Ratio')
        else:
            ax1.text(0.5, 0.5, "No parameter impact data available", 
                    horizontalalignment='center', verticalalignment='center')
        
        # Plot 2: Risk per trade vs. performance
        ax2 = plt.subplot(gs[0, 1])
        
        # Extract risk per trade data
        risk_data = position_results.get('risk_per_trade_performance', {})
        if risk_data:
            # Convert to DataFrame
            risk_df = pd.DataFrame({
                'Risk Per Trade': list(risk_data.keys()),
                'Performance': list(risk_data.values())
            })
            
            # Plot line chart
            sns.lineplot(x='Risk Per Trade', y='Performance', data=risk_df, ax=ax2)
            ax2.set_title('Risk Per Trade vs. Performance')
        else:
            ax2.text(0.5, 0.5, "No risk per trade performance data available", 
                    horizontalalignment='center', verticalalignment='center')
        
        # Plot 3: Position size distribution
        ax3 = plt.subplot(gs[1, 0])
        
        # Extract position size data
        size_data = position_results.get('position_size_distribution', {})
        if size_data:
            # Convert to DataFrame
            size_df = pd.DataFrame({
                'Position Size': list(size_data.keys()),
                'Frequency': list(size_data.values())
            })
            
            # Plot histogram
            sns.barplot(x='Position Size', y='Frequency', data=size_df, ax=ax3)
            ax3.set_title('Position Size Distribution')
        else:
            ax3.text(0.5, 0.5, "No position size distribution data available", 
                    horizontalalignment='center', verticalalignment='center')
        
        # Plot 4: Strategy-specific position sizing performance
        ax4 = plt.subplot(gs[1, 1])
        
        # Extract strategy-specific data
        strategy_data = position_results.get('strategy_position_performance', {})
        if strategy_data:
            # Convert to DataFrame
            strategies = []
            performances = []
            
            for strategy, perf in strategy_data.items():
                strategies.append(strategy)
                performances.append(perf)
            
            strategy_df = pd.DataFrame({
                'Strategy': strategies,
                'Performance': performances
            })
            
            # Sort by performance
            strategy_df = strategy_df.sort_values('Performance', ascending=False)
            
            # Plot bar chart
            sns.barplot(x='Performance', y='Strategy', data=strategy_df, ax=ax4)
            ax4.set_title('Strategy-Specific Position Sizing Performance')
        else:
            ax4.text(0.5, 0.5, "No strategy-specific position sizing performance data available", 
                    horizontalalignment='center', verticalalignment='center')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_file, dpi=300)
        logger.info(f"Position sizing optimization visualization saved to {output_file}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error visualizing position sizing optimization: {str(e)}")
        logger.error(traceback.format_exc())

def visualize_optimization_results(results_file, output_dir="."):
    """
    Visualize optimization results
    
    Args:
        results_file: Path to results file
        output_dir: Output directory for visualizations
    """
    logger.info(f"Visualizing optimization results from {results_file}")
    
    # Load optimization results
    results = load_optimization_results(results_file)
    
    if not results:
        logger.error("Failed to load optimization results")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize Sharpe ratio factors
    visualize_sharpe_ratio_factors(
        results=results,
        output_file=os.path.join(output_dir, "sharpe_ratio_factors.png")
    )
    
    # Visualize ML strategy optimization
    visualize_ml_strategy_optimization(
        results=results,
        output_file=os.path.join(output_dir, "ml_strategy_optimization.png")
    )
    
    # Visualize signal filter optimization
    visualize_signal_filter_optimization(
        results=results,
        output_file=os.path.join(output_dir, "signal_filter_optimization.png")
    )
    
    # Visualize position sizing optimization
    visualize_position_sizing_optimization(
        results=results,
        output_file=os.path.join(output_dir, "position_sizing_optimization.png")
    )
    
    logger.info("Visualization of optimization results completed")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Optimization Visualizer')
    
    parser.add_argument('--results', type=str, required=True,
                        help='Path to optimization results file (JSON or YAML)')
    
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Output directory for visualizations')
    
    return parser.parse_args()

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_arguments()
    
    try:
        # Visualize optimization results
        visualize_optimization_results(
            results_file=args.results,
            output_dir=args.output_dir
        )
        
    except Exception as e:
        logger.error(f"Error in optimization visualizer: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
