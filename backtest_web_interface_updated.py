#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Backtest Web Interface (Updated)
---------------------
This module implements a web interface for the backtest functionality using the updated strategy.
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

# Import our updated backtest implementation
from backtest_engine_updated import run_backtest

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
with open('sp500_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

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
    
    # Run backtest
    summary, signals = run_backtest(
        start_date=start_date,
        end_date=end_date,
        mode='backtest',
        max_signals=max_signals,
        initial_capital=initial_capital,
        random_seed=42,
        weekly_selection=weekly_selection == ['True'] if isinstance(weekly_selection, list) else bool(weekly_selection)
    )
    
    # Extract performance metrics from summary
    if summary is None:
        # Handle error case
        results_card = dbc.Card([
            dbc.CardHeader("Backtest Error"),
            dbc.CardBody([
                html.P("An error occurred during the backtest. Please check the logs for details.")
            ])
        ])
        return results_card, {}, {}, html.Div("No positions available")
        
    # Extract performance metrics from summary
    metrics = summary
    
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
                    html.H5("Average Win"),
                    html.H3(f"${float(metrics.get('avg_win', 0)):.2f}"),
                ], width=4),
                dbc.Col([
                    html.H5("Average Loss"),
                    html.H3(f"${float(metrics.get('avg_loss', 0)):.2f}"),
                ], width=4),
                dbc.Col([
                    html.H5("Avg Holding Period"),
                    html.H3(f"{metrics.get('avg_holding_period', 0):.1f} days"),
                ], width=4),
            ], className="mt-3"),
            dbc.Row([
                dbc.Col([
                    html.H5("Initial Capital"),
                    html.H3(f"${metrics.get('initial_capital', 0):.2f}"),
                ], width=6),
                dbc.Col([
                    html.H5("Final Capital"),
                    html.H3(f"${metrics.get('final_capital', 0):.2f}"),
                ], width=6),
            ], className="mt-3"),
        ]),
    ])
    
    # Create equity curve figure
    # For simplicity, we'll just use a placeholder equity curve based on the final capital
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    initial_capital = metrics.get('initial_capital', 10000)
    final_capital = metrics.get('final_capital', initial_capital)
    
    # Simple linear growth model for visualization
    equity_values = np.linspace(initial_capital, final_capital, len(dates))
    equity_data = pd.DataFrame({
        'timestamp': dates,
        'equity': equity_values
    })
    
    equity_fig = go.Figure()
    equity_fig.add_trace(go.Scatter(
        x=equity_data['timestamp'],
        y=equity_data['equity'],
        mode='lines',
        name='Equity Curve',
        line=dict(color='blue', width=2),
    ))
    
    equity_fig.update_layout(
        title='Equity Curve (Estimated)',
        xaxis_title='Date',
        yaxis_title='Portfolio Value ($)',
        template='plotly_white',
        height=500,
    )
    
    # Create drawdown curve figure
    equity_data['peak'] = equity_data['equity'].cummax()
    equity_data['drawdown'] = (equity_data['equity'] / equity_data['peak'] - 1) * 100
    
    drawdown_fig = go.Figure()
    drawdown_fig.add_trace(go.Scatter(
        x=equity_data['timestamp'],
        y=equity_data['drawdown'],
        mode='lines',
        name='Drawdown',
        line=dict(color='red', width=2),
        fill='tozeroy',
    ))
    
    drawdown_fig.update_layout(
        title='Drawdown Curve (Estimated)',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        template='plotly_white',
        height=500,
    )
    
    # Create positions table
    # Since we don't have direct access to the positions from our backtest,
    # we'll create a placeholder table with the tier metrics
    tier_metrics = metrics.get('tier_metrics', {})
    
    tier_data = []
    for tier_name, tier_info in tier_metrics.items():
        tier_data.append({
            'Tier': tier_name,
            'Win Rate': f"{tier_info.get('win_rate', 0):.2f}%",
            'Avg P&L': f"${float(tier_info.get('avg_pl', 0)):.2f}",
            'Trade Count': tier_info.get('trade_count', 0),
            'Long Win Rate': f"{tier_info.get('long_win_rate', 0):.2f}%",
            'Long Count': tier_info.get('long_count', 0),
        })
    
    positions_table = html.Div([
        dbc.Card([
            dbc.CardHeader("Tier Performance"),
            dbc.CardBody([
                html.Div([
                    dbc.Table.from_dataframe(
                        pd.DataFrame(tier_data),
                        striped=True,
                        bordered=True,
                        hover=True,
                        responsive=True,
                    ) if tier_data else html.P("No tier data available"),
                ]),
            ]),
        ]),
    ])
    
    return results_card, equity_fig, drawdown_fig, positions_table

if __name__ == '__main__':
    app.run(debug=True, port=8050)
