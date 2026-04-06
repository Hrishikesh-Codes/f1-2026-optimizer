"""
F1 2026 Optimizer — Design System

Injects custom CSS for the Ferrari red / near-black dark theme.
Call inject_custom_css() at the top of every page's render() function.
"""
import streamlit as st


def inject_custom_css() -> None:
    """Inject the F1 2026 dark theme CSS into the Streamlit app."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

    /* -- Global -- */
    .stApp { background-color: #0A0A0A; }
    .block-container { padding-top: 1.5rem; }

    /* -- Typography -- */
    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #FFFFFF !important;
        letter-spacing: -0.5px;
    }
    p, li, span, div { color: #CCCCCC; }

    /* -- Sidebar -- */
    [data-testid="stSidebar"] {
        background-color: #141414 !important;
        border-right: 2px solid #DC0000;
    }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #FFFFFF !important; }

    /* -- Metric cards -- */
    [data-testid="metric-container"] {
        background-color: #141414;
        border-left: 3px solid #DC0000;
        padding: 12px 16px;
        border-radius: 2px;
        margin: 4px 0;
    }
    [data-testid="metric-container"] label {
        color: #888888 !important;
        font-size: 11px !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-family: 'Space Grotesk', sans-serif;
    }
    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-family: 'Courier New', monospace;
        font-size: 26px !important;
        font-weight: 700;
    }
    [data-testid="stMetricDelta"] { font-size: 13px !important; }

    /* -- Primary button -- */
    .stButton > button {
        background-color: #DC0000 !important;
        color: #FFFFFF !important;
        border: none !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        border-radius: 2px !important;
        padding: 0.5rem 1.5rem !important;
    }
    .stButton > button:hover { background-color: #AA0000 !important; }
    .stButton > button:active { background-color: #880000 !important; }

    /* -- Form inputs -- */
    .stSelectbox label, .stSlider label,
    .stRadio label, .stCheckbox label,
    .stNumberInput label, .stTextInput label {
        color: #888888 !important;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-family: 'Space Grotesk', sans-serif;
    }
    .stSelectbox > div > div {
        background-color: #141414 !important;
        border-color: #333333 !important;
        color: #FFFFFF !important;
    }
    .stTextInput input, .stNumberInput input {
        background-color: #141414 !important;
        color: #FFFFFF !important;
        border-color: #333333 !important;
    }

    /* -- DataFrames -- */
    [data-testid="stDataFrame"] {
        border: 1px solid #222222;
        border-radius: 2px;
    }
    /* DataFrames — force legible text inside Arrow tables */
    .dvn-scroller { background-color: #141414 !important; }
    .dvn-scroller .cell-wrap span,
    .dvn-scroller .cell-wrap div,
    [data-testid="stDataFrame"] span,
    [data-testid="stDataFrame"] div { color: #ffffff !important; }
    [data-testid="stDataFrame"] [data-testid="glideDataEditor"] { background: #141414 !important; }

    /* -- Divider -- */
    hr { border-color: #222222 !important; margin: 1.5rem 0 !important; }

    /* -- Alerts -- */
    .stAlert {
        border-left: 3px solid #DC0000 !important;
        background-color: #141414 !important;
        border-radius: 2px !important;
    }
    div[data-baseweb="notification"] { background-color: #141414 !important; }

    /* -- Progress bar -- */
    .stProgress > div > div { background-color: #DC0000 !important; }
    .stProgress { background-color: #222222 !important; }

    /* -- Tabs -- */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #141414;
        border-bottom: 1px solid #333333;
        gap: 0;
    }
    .stTabs [data-baseweb="tab"] {
        color: #888888 !important;
        padding: 8px 20px !important;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #FFFFFF !important;
        border-bottom: 2px solid #DC0000 !important;
        background-color: #1A1A1A !important;
    }

    /* -- Expander -- */
    details { border: 1px solid #222222 !important; border-radius: 2px !important; }
    summary { color: #888888 !important; font-size: 13px !important; }

    /* -- Pit wall panel -- */
    .pit-wall-panel {
        background-color: #0D0D0D;
        border: 1px solid #333333;
        border-left: 4px solid #DC0000;
        border-radius: 2px;
        padding: 20px 24px;
        font-family: 'Courier New', monospace;
        margin: 12px 0;
    }
    .pit-wall-panel .action-text {
        color: #DC0000;
        font-size: 22px;
        font-weight: 700;
        letter-spacing: 3px;
        text-transform: uppercase;
    }
    .pit-wall-panel .sub-text {
        color: #888888;
        font-size: 12px;
        margin-top: 6px;
        letter-spacing: 1px;
    }
    .pit-wall-panel .oom-text {
        color: #FFD700;
        font-size: 13px;
        margin-top: 8px;
        letter-spacing: 1px;
    }

    /* -- Blinking cursor -- */
    .blink {
        animation: blink-anim 1s step-end infinite;
        color: #DC0000;
    }
    @keyframes blink-anim {
        0%, 100% { opacity: 1; }
        50% { opacity: 0; }
    }

    /* -- Caption / small text -- */
    .stCaption { color: #666666 !important; font-size: 12px !important; }

    /* -- Spinner -- */
    .stSpinner > div { border-top-color: #DC0000 !important; }

    /* -- Mobile responsive -- */
    @media (max-width: 768px) {
        .block-container { padding: 0.5rem !important; }
        h1 { font-size: 20px !important; }
        [data-testid="metric-container"] { padding: 8px 10px; }
    }
    </style>
    """, unsafe_allow_html=True)
