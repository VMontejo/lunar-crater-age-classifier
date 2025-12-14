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
            color: white;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: rgba(18, 24, 41, 0.95);
            backdrop-filter: blur(12px);
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
            color: white;
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

        .stButton > button[kind="secondary"] {
            background: transparent !important;
            color: #9ca3af !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            transition: all 0.3s ease;
        }

        .stButton > button[kind="secondary"]:hover {
            background: rgba(255, 255, 255, 0.1) !important;
            color: white !important;
            border-color: rgba(255, 255, 255, 0.3) !important;
        }

        /* File uploader */
        [data-testid="stFileUploader"] {
            background: rgba(255, 255, 255, 0.02);
            border: 2px dashed rgba(139, 92, 246, 0.3);
            border-radius: 1rem;
        }

        /* Info boxes */
        .stAlert {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(139, 92, 246, 0.2);
            color: white;
        }

        /* Metric containers */
        [data-testid="stMetricValue"] {
            color: white;
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

def render_header():
    """Render the app header with logo"""
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 1rem; padding: 1rem 0 2rem 0;">
            <div style="width: 2.5rem; height: 2.5rem; border-radius: 50%;
                        background: linear-gradient(135deg, #7c3aed 0%, #3b82f6 100%);
                        display: flex; align-items: center; justify-content: center;">
                <span style="font-size: 1.25rem;">🌙</span>
            </div>
            <span style="font-weight: 700; font-size: 1.25rem; color: white;">
                LunarChron
            </span>
        </div>
    """, unsafe_allow_html=True)

def render_sidebar_navigation():
    """Render sidebar navigation menu"""
    with st.sidebar:
        st.markdown("""
            <div style="text-align: center; padding: 1rem 0 2rem 0;">
                <div style="width: 4rem; height: 4rem; margin: 0 auto; border-radius: 50%;
                            background: linear-gradient(135deg, #7c3aed 0%, #3b82f6 100%);
                            display: flex; align-items: center; justify-content: center;">
                    <span style="font-size: 2rem;">🌙</span>
                </div>
                <h2 style="margin-top: 1rem; color: white; font-size: 1.5rem;">LunarChron</h2>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🧭 Navigation")

        # Navigation buttons
        if st.button("🏠 Home", use_container_width=True, key="nav_home"):
            st.switch_page("main.py")

        if st.button("🔬 Classify", use_container_width=True, key="nav_classify"):
            st.switch_page("pages/classify.py")

        if st.button("ℹ️ About", use_container_width=True, key="nav_about"):
            st.switch_page("pages/about.py")

        st.markdown("---")

        # Stats in sidebar
        st.markdown("### 📊 Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Images", "5,000+", delta=None)
        with col2:
            st.metric("Classes", "3", delta=None)

def render_footer():
    """Render the app footer"""
    st.markdown("""
        <div class="custom-footer">
            <p>LunarAge Classifier — Powered by LROCNet Moon Classifier Dataset</p>
            <p style="margin-top: 0.5rem;">Lunar Reconnaissance Orbiter Imagery Analysis</p>
        </div>
    """, unsafe_allow_html=True)

def setup_page_config(page_title="LunarChron Classifier", page_icon="🌙"):
    """Setup common page configuration"""
    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'About': "LunarAge Classifier - AI-powered lunar crater age prediction"
        }
    )

def init_layout(page_title="LunarChron Classifier", page_icon="🌙", show_sidebar=True):
    """Initialize the complete layout for a page"""
    setup_page_config(page_title, page_icon)
    load_custom_css()

    if show_sidebar:
        render_sidebar_navigation()

    render_header()
