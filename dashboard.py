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

BASE_LAYOUT = dict(
    paper_bgcolor='#051e38',
    plot_bgcolor='#020c18',
    font=dict(color='#e8f4f8', family='DM Sans', size=11),
    colorway=['#00d4ff', '#00ff9d', '#ffb347', '#ff4f6b', '#7b61ff'],
    margin=dict(l=40, r=40, t=40, b=120),
    autosize=True,
    xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
    yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
    legend=dict(
        orientation='h',
        yanchor='top',
        y=-0.25,
        xanchor='center',
        x=0.5,
        font=dict(size=10),
        bgcolor='rgba(5,30,56,0.7)',
        bordercolor='rgba(0,212,255,0.2)',
        borderwidth=1,
        itemwidth=30,
        tracegroupgap=4,
    ),
)

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


# ── Dash app ─────────────────────────────────────────────
app_dash = dash.Dash(
    __name__,
    requests_pathname_prefix='/dashboard/'
)

app_dash.layout = html.Div(
    style={
        'backgroundColor': '#020c18',
        'minHeight': '100vh',
        'padding': '20px',
        'overflowY': 'auto',
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
            html.Div(dcc.Graph(id='psi-bar',    style={'height': '380px'}), style={'flex': '1', 'minWidth': '0'}),
            html.Div(dcc.Graph(id='class-pie',  style={'height': '380px'}), style={'flex': '1', 'minWidth': '0'}),
        ], style={'display': 'flex', 'gap': '16px', 'marginBottom': '16px'}),
        # Charts row 2
        html.Div([
            html.Div(dcc.Graph(id='psi-timeline',   style={'height': '320px'}), style={'flex': '2', 'minWidth': '0'}),
            html.Div(dcc.Graph(id='severity-chart', style={'height': '320px'}), style={'flex': '1', 'minWidth': '0'}),
        ], style={'display': 'flex', 'gap': '16px', 'marginBottom': '16px'}),
    ]
)
# Callbacks

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
    psi_fig.update_layout(**BASE_LAYOUT)

    # Class pie
    all_counts = {}
    for row in df['class_counts']:
        try:
            counts = ast.literal_eval(row) if isinstance(row, str) else row
            for cls, cnt in counts.items():
                all_counts[cls] = all_counts.get(cls, 0) + cnt
        except:
            continue

    pie_fig = go.Figure(go.Pie(
        values=list(all_counts.values()),
        labels=[n.replace('_', ' ') for n in all_counts.keys()],
        title='Plastic Types Distribution',
        hole=0.4,
        textposition='inside',
        textinfo='label+percent',
        insidetextorientation='radial',
        marker=dict(
            colors=['#00d4ff','#00ff9d','#ffb347','#ff4f6b',
                    '#7b61ff','#fd79a8','#74b9ff','#55efc4','#fdcb6e'],
            line=dict(color='#020c18', width=1.5)
        ),
        showlegend=False,             # ← keys hidden, info is inside slices
    )) if all_counts else empty_fig('No class data')
    if all_counts:
            pie_fig.update_layout(**BASE_LAYOUT)
            pie_fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))

            
    # Timeline
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    timeline_fig = px.line(
        df.sort_values('timestamp'),
        x='timestamp', y='psi_score',
        color='location',
        title='PSI Score Over Time',
        markers=True,
        labels={'psi_score': 'PSI Score', 'timestamp': ''}
    )
    timeline_fig.update_layout(**BASE_LAYOUT)
    timeline_fig.update_layout(
    legend=dict(
        title=dict(text='Location', font=dict(size=10)),  
        orientation='h',
        yanchor='top',
        y=-0.18,             
        xanchor='center',
        x=0.5,
        font=dict(size=10),
        bgcolor='rgba(5,30,56,0.7)',
        bordercolor='rgba(0,212,255,0.2)',
        borderwidth=1,
    ),
    margin=dict(l=40, r=40, t=40, b=100), 
    xaxis=dict(
        gridcolor='rgba(255,255,255,0.05)',
        title=''      
        )   
    )

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
        title='Severity Distribution',
        labels={'severity': 'Severity', 'count': 'Count'}
        )
    severity_fig.update_layout(**BASE_LAYOUT)
    severity_fig.update_layout(
        showlegend=False,            # ← x-axis already shows the labels, legend is duplicate
        margin=dict(l=40, r=20, t=40, b=40),  # ← tighter bottom, no legend space needed
        xaxis=dict(
            title='',                # ← x-axis label is self-explanatory, remove clutter
            gridcolor='rgba(255,255,255,0.05)'
        ),
        yaxis=dict(
            title='Count',
            gridcolor='rgba(255,255,255,0.05)'
        )
    )

    return psi_fig, pie_fig, timeline_fig, severity_fig