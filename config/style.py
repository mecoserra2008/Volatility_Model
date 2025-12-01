"""
Custom CSS styling for the Portfolio Dashboard
"""

CUSTOM_CSS = """
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #4a4a4a;
    }
    .stMetric label {
        color: #fafafa !important;
        font-weight: 500;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.5rem !important;
    }
    .stMetric [data-testid="stMetricDelta"] {
        color: #fafafa !important;
    }
    div[data-testid="column"] {
        background-color: #1e1e1e;
        padding: 10px;
        border-radius: 8px;
    }
    h1 {
        color: #1f77b4;
    }
    h2 {
        color: #2ca02c;
    }
    h3 {
        color: #e0e0e0;
    }
    </style>
"""
