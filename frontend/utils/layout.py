import streamlit as st
from pathlib import Path

def load_custom_css():
    """Load custom CSS for the entire app"""
    st.markdown("""
        <style>
        /* Global Variables */
        :root {
            --bg-primary: #0a0e1a;
            --bg-secondary: #121829;
            --accent-purple: #7c3aed;
            --accent-blue: #3b82f6;
            --lunar-gray: #9ca3af;
        }

        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Global styles */
        .main {
            background-color: var(--bg-primary);
        }

        .stApp {
            background-color: var(--bg-primary);
        }

        /* Glass card effect */
        .glass-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 1rem;
            padding: 1.5rem;
        }

        /* Gradient text */
        .gradient-text {
            background: linear-gradient(135deg, #7c3aed 0%, #3b82f6 50%, #06b6d4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        /* Stars background animation */
        .stars-bg {
            background-image:
                radial-gradient(2px 2px at 20px 30px, #eee, transparent),
                radial-gradient(2px 2px at 40px 70px, rgba(255,255,255,0.8), transparent),
                radial-gradient(1px 1px at 90px 40px, #fff, transparent),
                radial-gradient(2px 2px at 160px 120px, rgba(255,255,255,0.9), transparent),
                radial-gradient(1px 1px at 230px 80px, #fff, transparent),
                radial-gradient(2px 2px at 300px 150px, rgba(255,255,255,0.7), transparent);
            background-size: 350px 200px;
            animation: twinkle 8s ease-in-out infinite alternate;
        }

        @keyframes twinkle {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }

        /* Custom button styles */
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #a855f7 0%, #3b82f6 100%) !important;
            color: white !important;
            border: none !important;
            box-shadow: 0 10px 25px -5px rgba(168, 85, 247, 0.25);
            transition: all 0.3s ease;
        }

        .stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #9333ea 0%, #2563eb 100%) !important;
            box-shadow: 0 20px 40px -5px rgba(168, 85, 247, 0.4);
            transform: translateY(-2px);
        }

        /* Navigation styling */
        [data-testid="stSidebar"] {
            background: rgba(18, 24, 41, 0.95);
            backdrop-filter: blur(12px);
        }

        /* Footer */
        .custom-footer {
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            padding: 2rem 1rem;
            text-align: center;
            color: #6b7280;
            font-size: 0.875rem;
            margin-top: 4rem;
        }
        </style>
    """, unsafe_allow_html=True)

def render_footer():
    """Render the app footer"""
    st.markdown("""
        <div class="custom-footer">
            <p>LunarCraterAge Classifier — Powered by LROCNet Moon Classifier Dataset</p>
            <p style="margin-top: 0.5rem;">Lunar Reconnaissance Orbiter Imagery Analysis</p>
        </div>
    """, unsafe_allow_html=True)

def setup_page_config(page_title="LunarCraterAge Classifier", page_icon="🌙"):
    """Setup common page configuration"""
    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'About': "LunarCraterAge Classifier - AI-powered lunar crater age prediction"
        }
    )

def init_layout(page_title="LunarCraterAge Classifier", page_icon="🌙"):
    """Initialize the complete layout for a page"""
    setup_page_config(page_title, page_icon)
    load_custom_css()
