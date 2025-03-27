#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Backtest Web Interface
---------------------
This module implements a web interface for the backtest functionality.
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

# Import our backtest implementation
from fixed_backtest_final import run_backtest, get_sp500_symbols
from alpaca_api import AlpacaAPI

# Load configuration
with open('sp500_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize API
api = AlpacaAPI(
    api_key=config['alpaca']['api_key'],
    api_secret=config['alpaca']['api_secret'],
    base_url=config['alpaca']['base_url'],
    data_url=config['alpaca']['data_url']
)

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
                            html.Label("Strategy Name"),
                            dbc.Input(id="strategy-name", type="text", value="sp500_strategy", placeholder="Enter strategy name"),
                        ], width=6),
                        dbc.Col([
                            html.Label("Initial Capital"),
                            dbc.Input(id="initial-capital", type="number", value=10000, placeholder="Enter initial capital"),
                        ], width=6),
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("Start Date"),
                            dcc.DatePickerSingle(
                                id="start-date",
                                date=datetime.now() - timedelta(days=30),
                                display_format="YYYY-MM-DD",
                            ),
                        ], width=6),
                        dbc.Col([
                            html.Label("End Date"),
                            dcc.DatePickerSingle(
                                id="end-date",
                                date=datetime.now(),
                                display_format="YYYY-MM-DD",
                            ),
                        ], width=6),
                    ], className="mt-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("Max Signals"),
                            dbc.Input(id="max-signals", type="number", value=5, placeholder="Enter max signals"),
                        ], width=4),
                        dbc.Col([
                            html.Label("Min Score"),
                            dbc.Input(id="min-score", type="number", value=0.6, placeholder="Enter min score", step=0.1, min=0, max=1),
                        ], width=4),
                        dbc.Col([
                            html.Label("Universe Size"),
                            dbc.Input(id="universe-size", type="number", value=10, placeholder="Enter universe size"),
                        ], width=4),
                    ], className="mt-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("Tier 1 Threshold"),
                            dbc.Input(id="tier1-threshold", type="number", value=0.8, placeholder="Enter tier 1 threshold", step=0.1, min=0, max=1),
                        ], width=4),
                        dbc.Col([
                            html.Label("Tier 2 Threshold"),
                            dbc.Input(id="tier2-threshold", type="number", value=0.7, placeholder="Enter tier 2 threshold", step=0.1, min=0, max=1),
                        ], width=4),
                        dbc.Col([
                            html.Label("Tier 3 Threshold"),
                            dbc.Input(id="tier3-threshold", type="number", value=0.6, placeholder="Enter tier 3 threshold", step=0.1, min=0, max=1),
                        ], width=4),
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
        State("strategy-name", "value"),
        State("initial-capital", "value"),
        State("start-date", "date"),
        State("end-date", "date"),
        State("max-signals", "value"),
        State("min-score", "value"),
        State("universe-size", "value"),
        State("tier1-threshold", "value"),
        State("tier2-threshold", "value"),
        State("tier3-threshold", "value"),
    ],
    prevent_initial_call=True,
)
def run_backtest_callback(n_clicks, strategy_name, initial_capital, start_date, end_date, max_signals, min_score, universe_size, tier1_threshold, tier2_threshold, tier3_threshold):
    # Get universe of symbols
    universe = get_sp500_symbols()[:universe_size]
    
    # Run backtest
    results = run_backtest(
        api=api,
        strategy_name=strategy_name,
        universe=universe,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        max_signals=max_signals,
        min_score=min_score,
        tier1_threshold=tier1_threshold,
        tier2_threshold=tier2_threshold,
        tier3_threshold=tier3_threshold
    )
    
    # Extract performance metrics
    performance = results['performance']
    portfolio = results['portfolio']
    
    # Create results card
    results_card = dbc.Card([
        dbc.CardHeader("Backtest Results"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H5("Final Portfolio Value"),
                    html.H3(f"${performance['final_equity']:.2f}"),
                ], width=4),
                dbc.Col([
                    html.H5("Return"),
                    html.H3(f"{performance['return_pct']:.2f}%"),
                ], width=4),
                dbc.Col([
                    html.H5("Annualized Return"),
                    html.H3(f"{performance['annualized_return']:.2f}%"),
                ], width=4),
            ]),
            dbc.Row([
                dbc.Col([
                    html.H5("Sharpe Ratio"),
                    html.H3(f"{performance['sharpe_ratio']:.2f}"),
                ], width=4),
                dbc.Col([
                    html.H5("Max Drawdown"),
                    html.H3(f"{performance['max_drawdown']:.2f}%"),
                ], width=4),
                dbc.Col([
                    html.H5("Win Rate"),
                    html.H3(f"{performance['win_rate']:.2f}%"),
                ], width=4),
            ], className="mt-3"),
        ]),
    ])
    
    # Create equity curve figure
    equity_data = pd.DataFrame(portfolio.equity_curve)
    equity_data['timestamp'] = pd.to_datetime(equity_data['timestamp'])
    
    equity_fig = go.Figure()
    equity_fig.add_trace(go.Scatter(
        x=equity_data['timestamp'],
        y=equity_data['equity'],
        mode='lines',
        name='Equity Curve',
        line=dict(color='blue', width=2),
    ))
    
    equity_fig.update_layout(
        title='Equity Curve',
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
        title='Drawdown Curve',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        template='plotly_white',
        height=500,
    )
    
    # Create positions table
    open_positions = []
    for symbol, position in portfolio.open_positions.items():
        open_positions.append({
            'Symbol': symbol,
            'Entry Price': f"${position.entry_price:.2f}",
            'Entry Date': position.entry_time.strftime('%Y-%m-%d'),
            'Direction': position.direction,
            'Size': position.position_size,
            'Tier': position.tier if hasattr(position, 'tier') else 'N/A',
        })
    
    closed_positions = []
    for position in portfolio.closed_positions[-10:]:  # Show last 10 closed positions
        closed_positions.append({
            'Symbol': position.symbol,
            'Entry Price': f"${position.entry_price:.2f}",
            'Exit Price': f"${position.exit_price:.2f}",
            'Entry Date': position.entry_time.strftime('%Y-%m-%d'),
            'Exit Date': position.exit_time.strftime('%Y-%m-%d'),
            'Direction': position.direction,
            'P&L': f"${position.pnl:.2f}",
            'P&L %': f"{position.pnl_pct:.2f}%",
            'Exit Reason': position.exit_reason,
        })
    
    positions_table = html.Div([
        dbc.Card([
            dbc.CardHeader("Open Positions"),
            dbc.CardBody([
                html.Div([
                    dbc.Table.from_dataframe(
                        pd.DataFrame(open_positions),
                        striped=True,
                        bordered=True,
                        hover=True,
                        responsive=True,
                    ) if open_positions else html.P("No open positions"),
                ]),
            ]),
        ]),
        
        dbc.Card([
            dbc.CardHeader("Recent Closed Positions"),
            dbc.CardBody([
                html.Div([
                    dbc.Table.from_dataframe(
                        pd.DataFrame(closed_positions),
                        striped=True,
                        bordered=True,
                        hover=True,
                        responsive=True,
                    ) if closed_positions else html.P("No closed positions"),
                ]),
            ]),
        ], className="mt-4"),
    ])
    
    return results_card, equity_fig, drawdown_fig, positions_table

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
