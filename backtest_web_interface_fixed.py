#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Backtest Web Interface (Fixed)
---------------------
This module implements a web interface for the backtest functionality using the updated backtest engine.
"""

import os
import yaml
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import logging
import json
import traceback

# Import our updated backtest implementation
from run_comprehensive_backtest_updated import run_backtest_for_web

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', f"web_interface_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load configuration
try:
    with open('sp500_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    logger.info("Successfully loaded configuration from sp500_config.yaml")
except Exception as e:
    logger.error(f"Error loading configuration: {str(e)}")
    # Create a default configuration if the file doesn't exist
    config = {
        'paths': {
            'backtest_results': 'backtest_results',
            'plots': 'plots',
            'trades': 'trades',
            'performance': 'performance'
        },
        'strategy': {
            'max_trades_per_run': 40,
            'position_sizing': {
                'base_position_pct': 5
            }
        }
    }
    # Ensure the directories exist
    for path_key in config['paths']:
        os.makedirs(config['paths'][path_key], exist_ok=True)
    
    # Save the default configuration
    try:
        with open('sp500_config.yaml', 'w') as f:
            yaml.dump(config, f)
        logger.info("Created default configuration file sp500_config.yaml")
    except Exception as e:
        logger.error(f"Error creating default configuration: {str(e)}")

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Trading Strategy Backtest"

# Define the layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Trading Strategy Backtest", className="text-center my-4"),
            html.Hr(),
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Backtest Parameters"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Initial Capital"),
                            dbc.Input(id="initial-capital", type="number", value=10000, placeholder="Enter initial capital"),
                        ], width=6),
                        dbc.Col([
                            html.Label("Max Signals"),
                            dbc.Input(id="max-signals", type="number", value=50, placeholder="Enter max signals"),
                        ], width=6),
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("Start Date"),
                            dcc.DatePickerSingle(
                                id="start-date",
                                date=datetime(2023, 1, 1),
                                display_format="YYYY-MM-DD",
                            ),
                        ], width=6),
                        dbc.Col([
                            html.Label("End Date"),
                            dcc.DatePickerSingle(
                                id="end-date",
                                date=datetime(2023, 3, 31),
                                display_format="YYYY-MM-DD",
                            ),
                        ], width=6),
                    ], className="mt-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("Weekly Selection"),
                            dbc.Checklist(
                                options=[{"label": "Enable Weekly Stock Selection Refresh", "value": 'True'}],
                                value=['True'],
                                id="weekly-selection",
                                switch=True,
                            ),
                        ], width=12),
                    ], className="mt-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("Run Backtest", id="run-backtest", color="primary", className="mt-3 w-100"),
                        ], width=12),
                    ]),
                ]),
            ]),
        ], width=12),
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Div(id="backtest-results", className="mt-4"),
        ], width=12),
    ]),
    
    dbc.Row([
        dbc.Col([
            dcc.Loading(
                id="loading",
                type="circle",
                children=[
                    dcc.Graph(id="equity-curve", className="mt-4"),
                    dcc.Graph(id="drawdown-curve", className="mt-4"),
                ],
            ),
        ], width=12),
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Div(id="positions-table", className="mt-4"),
        ], width=12),
    ]),
    
], fluid=True)

@app.callback(
    [
        Output("backtest-results", "children"),
        Output("equity-curve", "figure"),
        Output("drawdown-curve", "figure"),
        Output("positions-table", "children"),
    ],
    [Input("run-backtest", "n_clicks")],
    [
        State("initial-capital", "value"),
        State("start-date", "date"),
        State("end-date", "date"),
        State("max-signals", "value"),
        State("weekly-selection", "value"),
    ],
    prevent_initial_call=True,
)
def run_backtest_callback(n_clicks, initial_capital, start_date, end_date, max_signals, weekly_selection):
    if n_clicks is None:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    logger.info(f"Running backtest with parameters: initial_capital={initial_capital}, start_date={start_date}, end_date={end_date}, max_signals={max_signals}, weekly_selection={weekly_selection}")
    
    try:
        # Run backtest using our updated function
        result = run_backtest_for_web(
            start_date=start_date,
            end_date=end_date,
            max_signals=max_signals,
            initial_capital=initial_capital,
            weekly_selection=weekly_selection == ['True'] if isinstance(weekly_selection, list) else bool(weekly_selection)
        )
        
        # Extract metrics and trades
        metrics = result['summary']
        trades = result['trades']
        
        # Create results card
        results_card = dbc.Card([
            dbc.CardHeader("Backtest Results"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H5("Win Rate"),
                        html.H3(f"{metrics.get('win_rate', 0):.2f}%"),
                    ], width=4),
                    dbc.Col([
                        html.H5("Profit Factor"),
                        html.H3(f"{float(metrics.get('profit_factor', 0)):.2f}"),
                    ], width=4),
                    dbc.Col([
                        html.H5("Total Return"),
                        html.H3(f"{metrics.get('total_return', 0):.2f}%"),
                    ], width=4),
                ]),
                dbc.Row([
                    dbc.Col([
                        html.H5("Net Profit"),
                        html.H3(f"${metrics.get('net_profit', 0):.2f}"),
                    ], width=4),
                    dbc.Col([
                        html.H5("Avg Win"),
                        html.H3(f"${metrics.get('avg_profit_per_winner', 0):.2f}"),
                    ], width=4),
                    dbc.Col([
                        html.H5("Avg Loss"),
                        html.H3(f"${metrics.get('avg_loss_per_loser', 0):.2f}"),
                    ], width=4),
                ]),
                dbc.Row([
                    dbc.Col([
                        html.H5("Initial Capital"),
                        html.H3(f"${metrics.get('initial_capital', 0):.2f}"),
                    ], width=4),
                    dbc.Col([
                        html.H5("Final Capital"),
                        html.H3(f"${metrics.get('final_capital', 0):.2f}"),
                    ], width=4),
                    dbc.Col([
                        html.H5("# Trades"),
                        html.H3(f"{metrics.get('num_trades', 0)}"),
                    ], width=4),
                ]),
            ]),
        ])
        
        # Create equity curve
        if trades:
            # Convert trades to DataFrame for easier processing
            trades_df = pd.DataFrame(trades)
            
            # Sort trades by entry date
            if 'entry_date' in trades_df.columns:
                trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
                trades_df = trades_df.sort_values('entry_date')
            
            # Create equity curve data
            dates = []
            equity = []
            drawdowns = []
            
            current_equity = initial_capital
            max_equity = current_equity
            
            # Add initial point
            if 'entry_date' in trades_df.columns and len(trades_df) > 0:
                dates.append(pd.to_datetime(start_date))
                equity.append(current_equity)
                drawdowns.append(0)
                
                # Add points for each trade
                for _, trade in trades_df.iterrows():
                    if 'profit_loss' in trade:
                        current_equity += trade['profit_loss']
                        max_equity = max(max_equity, current_equity)
                        drawdown = (current_equity - max_equity) / max_equity * 100
                        
                        dates.append(pd.to_datetime(trade['exit_date'] if 'exit_date' in trade else trade['entry_date']))
                        equity.append(current_equity)
                        drawdowns.append(drawdown)
            
            # Create equity curve figure
            equity_fig = go.Figure()
            equity_fig.add_trace(go.Scatter(
                x=dates,
                y=equity,
                mode='lines',
                name='Equity',
                line=dict(color='blue', width=2)
            ))
            
            equity_fig.update_layout(
                title='Equity Curve',
                xaxis_title='Date',
                yaxis_title='Equity ($)',
                template='plotly_white'
            )
            
            # Create drawdown curve figure
            drawdown_fig = go.Figure()
            drawdown_fig.add_trace(go.Scatter(
                x=dates,
                y=drawdowns,
                mode='lines',
                name='Drawdown',
                line=dict(color='red', width=2),
                fill='tozeroy'
            ))
            
            drawdown_fig.update_layout(
                title='Drawdown Curve',
                xaxis_title='Date',
                yaxis_title='Drawdown (%)',
                template='plotly_white',
                yaxis=dict(autorange="reversed")  # Invert y-axis for drawdowns
            )
            
            # Create positions table
            if len(trades_df) > 0:
                # Format the trades DataFrame for display
                display_df = trades_df.copy()
                
                # Format columns for display
                if 'entry_date' in display_df.columns:
                    display_df['entry_date'] = pd.to_datetime(display_df['entry_date']).dt.strftime('%Y-%m-%d')
                if 'exit_date' in display_df.columns:
                    display_df['exit_date'] = pd.to_datetime(display_df['exit_date']).dt.strftime('%Y-%m-%d')
                if 'profit_loss' in display_df.columns:
                    display_df['profit_loss'] = display_df['profit_loss'].map('${:.2f}'.format)
                if 'profit_loss_pct' in display_df.columns:
                    display_df['profit_loss_pct'] = display_df['profit_loss_pct'].map('{:.2f}%'.format)
                
                # Select columns to display
                display_columns = [
                    'symbol', 'direction', 'entry_date', 'exit_date', 
                    'entry_price', 'exit_price', 'shares', 'profit_loss', 
                    'profit_loss_pct', 'is_winner'
                ]
                
                # Filter to only include columns that exist
                display_columns = [col for col in display_columns if col in display_df.columns]
                
                # Create the table
                positions_table = dbc.Table.from_dataframe(
                    display_df[display_columns].head(50),  # Limit to first 50 trades
                    striped=True,
                    bordered=True,
                    hover=True,
                    responsive=True
                )
                
                positions_table = dbc.Card([
                    dbc.CardHeader("Trade Positions (First 50)"),
                    dbc.CardBody([positions_table])
                ])
            else:
                positions_table = html.Div("No positions available")
        else:
            # Create empty figures if no trades
            equity_fig = go.Figure()
            equity_fig.update_layout(
                title='Equity Curve',
                xaxis_title='Date',
                yaxis_title='Equity ($)',
                template='plotly_white'
            )
            
            drawdown_fig = go.Figure()
            drawdown_fig.update_layout(
                title='Drawdown Curve',
                xaxis_title='Date',
                yaxis_title='Drawdown (%)',
                template='plotly_white',
                yaxis=dict(autorange="reversed")
            )
            
            positions_table = html.Div("No positions available")
        
        return results_card, equity_fig, drawdown_fig, positions_table
        
    except Exception as e:
        logger.error(f"Error in backtest: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Create error card
        error_card = dbc.Card([
            dbc.CardHeader("Backtest Error"),
            dbc.CardBody([
                html.P(f"An error occurred during the backtest: {str(e)}"),
                html.P("Please check the logs for details.")
            ])
        ])
        
        # Create empty figures
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title='No Data Available',
            template='plotly_white'
        )
        
        return error_card, empty_fig, empty_fig, html.Div("No positions available")

if __name__ == '__main__':
    app.run(debug=True, port=8050)
