import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import ast
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

# ── PostgreSQL connection ────────────────────────────────
DATABASE_URL = (
    f"postgresql://{os.getenv('DB_USER')}:"
    f"{os.getenv('DB_PASSWORD')}@"
    f"{os.getenv('DB_HOST')}:"
    f"{os.getenv('DB_PORT')}/"
    f"{os.getenv('DB_NAME')}"
)

engine = create_engine(DATABASE_URL)

# ── Empty dataframe fallback ─────────────────────────────
EMPTY_DF = pd.DataFrame(columns=[
    'id', 'location', 'psi_score', 'severity',
    'total_objects', 'class_counts', 'image_path', 'timestamp'
])

def load_data():
    try:
        with engine.connect() as conn:
            df = pd.read_sql('SELECT * FROM detections', conn)
        return df
    except Exception as e:
        print(f"DB error: {e}")
        return EMPTY_DF.copy()

# ── Empty figure helper ──────────────────────────────────
def empty_fig(msg="No data yet"):
    fig = go.Figure()
    fig.add_annotation(
        text=msg,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color="#7a9bae")
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    return fig

# ── Theme ────────────────────────────────────────────────
THEME = {
    'paper_bgcolor': '#051e38',
    'plot_bgcolor':  '#020c18',
    'font':          dict(color='#e8f4f8', family='DM Sans'),
    'colorway':      ['#00d4ff', '#00ff9d', '#ffb347', '#ff4f6b', '#7b61ff']
}

def apply_theme(fig):
    fig.update_layout(
        paper_bgcolor=THEME['paper_bgcolor'],
        plot_bgcolor=THEME['plot_bgcolor'],
        font=THEME['font'],
        colorway=THEME['colorway'],
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
    )
    return fig

# ── Dash app ─────────────────────────────────────────────
app_dash = dash.Dash(
    __name__,
    requests_pathname_prefix='/dashboard/'
)

app_dash.layout = html.Div(
    style={
        'backgroundColor': '#020c18',
        'minHeight': '100vh',
        'padding': '24px',
        'fontFamily': 'DM Sans, sans-serif'
    },
    children=[
        # Header
        html.Div([
            html.H1(
                "Marine Debris Dashboard",
                style={
                    'color': '#00d4ff',
                    'fontFamily': 'Space Mono, monospace',
                    'fontSize': '24px',
                    'marginBottom': '4px'
                }
            ),
            html.P(
                "Real-time plastic pollution severity index",
                style={'color': '#7a9bae', 'fontSize': '14px'}
            )
        ], style={'marginBottom': '32px'}),

        # Auto refresh
        dcc.Interval(id='refresh', interval=30000, n_intervals=0),

        # Summary stats row
        html.Div(id='summary-stats', style={
            'display': 'grid',
            'gridTemplateColumns': 'repeat(4, 1fr)',
            'gap': '16px',
            'marginBottom': '24px'
        }),

        # Charts row 1
        html.Div([
            html.Div(dcc.Graph(id='psi-bar'),   style={'flex': '1'}),
            html.Div(dcc.Graph(id='class-pie'), style={'flex': '1'}),
        ], style={'display': 'flex', 'gap': '16px', 'marginBottom': '16px'}),

        # Charts row 2
        html.Div([
            html.Div(dcc.Graph(id='psi-timeline'),   style={'flex': '2'}),
            html.Div(dcc.Graph(id='severity-chart'), style={'flex': '1'}),
        ], style={'display': 'flex', 'gap': '16px'}),
    ]
)
# ── Callbacks ────────────────────────────────────────────

# Callback 1: Summary stats only
@app_dash.callback(
    Output('summary-stats', 'children'),
    Input('refresh', 'n_intervals')
)
def update_stats(_):
    df = load_data()

    def stat_card(label, value, color='#00d4ff'):
        return html.Div([
            html.Div(str(value), style={
                'fontSize': '28px',
                'fontWeight': '700',
                'color': color,
                'fontFamily': 'Space Mono, monospace'
            }),
            html.Div(label, style={
                'fontSize': '11px',
                'color': '#7a9bae',
                'textTransform': 'uppercase',
                'letterSpacing': '1px'
            })
        ], style={
            'background': 'rgba(5,30,56,0.7)',
            'border': '1px solid rgba(0,212,255,0.15)',
            'borderRadius': '12px',
            'padding': '20px',
            'textAlign': 'center'
        })

    if df.empty:
        return [
            stat_card('Total Scans', 0),
            stat_card('Locations', 0),
            stat_card('Avg PSI', 0),
            stat_card('Total Objects', 0),
        ]

    return [
        stat_card('Total Scans',   len(df)),
        stat_card('Locations',     df['location'].nunique()),
        stat_card('Avg PSI',       round(df['psi_score'].mean(), 4), '#ffb347'),
        stat_card('Total Objects', int(df['total_objects'].sum()), '#00ff9d'),
    ]


# Callback 2: Charts only
@app_dash.callback(
    Output('psi-bar',        'figure'),
    Output('class-pie',      'figure'),
    Output('psi-timeline',   'figure'),
    Output('severity-chart', 'figure'),
    Input('refresh', 'n_intervals')
)
def update_charts(_):
    df = load_data()

    if df.empty:
        return empty_fig(), empty_fig(), empty_fig(), empty_fig()

    # PSI bar
    psi_fig = px.bar(
        df.groupby('location')['psi_score'].mean().reset_index(),
        x='location', y='psi_score',
        color='psi_score',
        color_continuous_scale='RdYlGn_r',
        title='Average PSI Score by Location',
        labels={'psi_score': 'PSI Score', 'location': 'Location'}
    )
    apply_theme(psi_fig)

    # Class pie
    all_counts = {}
    for row in df['class_counts']:
        try:
            counts = ast.literal_eval(row) if isinstance(row, str) else row
            for cls, cnt in counts.items():
                all_counts[cls] = all_counts.get(cls, 0) + cnt
        except:
            continue

    pie_fig = px.pie(
        values=list(all_counts.values()),
        names=[n.replace('_', ' ') for n in all_counts.keys()],
        title='Plastic Type Distribution',
        hole=0.4
    ) if all_counts else empty_fig('No class data')
    if all_counts:
        apply_theme(pie_fig)

    # Timeline
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    timeline_fig = px.line(
        df.sort_values('timestamp'),
        x='timestamp', y='psi_score',
        color='location',
        title='PSI Score Over Time',
        markers=True,
        labels={'psi_score': 'PSI Score', 'timestamp': 'Time'}
    )
    apply_theme(timeline_fig)

    # Severity
    severity_fig = px.histogram(
        df,
        x='severity',
        color='severity',
        category_orders={'severity': ['Low','Moderate','High','Critical']},
        color_discrete_map={
            'Low': '#00ff9d', 'Moderate': '#ffb347',
            'High': '#ff4f6b', 'Critical': '#ff0000'
        },
        title='Severity Distribution'
    )
    apply_theme(severity_fig)

    return psi_fig, pie_fig, timeline_fig, severity_fig