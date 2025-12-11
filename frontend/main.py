import streamlit as st
from pathlib import Path

# Backend Configuration
BACKEND_URL = st.secrets.get("BACKEND_URL", "http://localhost:8000")  # Will use docker service name in production

# Shared CSS styles
def load_styles():
    """Load shared CSS styles"""
    st.markdown("""
        <style>
        /* Base Styles */
        .hero-container {
            min-height: 0vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 2rem 1rem;
            position: relative;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            border-radius: 9999px;
            background: rgba(139, 92, 246, 0.1);
            border: 1px solid rgba(139, 92, 246, 0.3);
            color: #c084fc;
            font-size: 0.875rem;
            margin-bottom: 2rem;
        }

        /* Typography */
        .main-heading {
            font-size: clamp(2rem, 5vw, 4rem);
            font-weight: 700;
            line-height: 1.2;
            margin-bottom: 1.5rem;
        }

        .gradient-text {
            background: linear-gradient(135deg, #a855f7 0%, #3b82f6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .subtitle {
            font-size: 1.25rem;
            color: #9ca3af;
            max-width: 42rem;
            margin: 0 auto 2.5rem;
            line-height: 1.75;
        }

        /* Stats Section */
        .stats-container {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 2rem;
            max-width: 36rem;
            margin: 3rem auto 0;
            padding-top: 3rem;
        }

        .stat-item { text-align: center; }

        .stat-value {
            font-size: 1.875rem;
            font-weight: 700;
            background: linear-gradient(135deg, #a855f7 0%, #3b82f6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .stat-label {
            font-size: 0.875rem;
            color: #6b7280;
            margin-top: 0.25rem;
        }

        /* Button Styles */
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

        .stButton > button[kind="primary"]:active {
            transform: translateY(0px);
        }

        /* Responsive */
        @media (max-width: 640px) {
            .stats-container {
                grid-template-columns: 1fr;
                gap: 1.5rem;
            }
        }
        </style>
    """, unsafe_allow_html=True)

def render_badge(text, icon="✨"):
    """Render a badge component"""
    st.markdown(f'<div class="badge">{icon} {text}</div>', unsafe_allow_html=True)


def render_heading(main_text, gradient_text):
    """Render main heading with gradient"""
    st.markdown(f"""
        <h1 class="main-heading">
            <span style="color: white;">{main_text}</span><br/>
            <span class="gradient-text">{gradient_text}</span>
        </h1>
    """, unsafe_allow_html=True)


def render_stats():
    """Render statistics section"""
    stats = [
        {"value": "5,000+", "label": "Labeled Images"},
        {"value": "3", "label": "Classifications"},
        {"value": "227×277", "label": "Image Size"}
    ]

    stats_html = "".join([
        f'<div class="stat-item"><div class="stat-value">{s["value"]}</div>'
        f'<div class="stat-label">{s["label"]}</div></div>'
        for s in stats
    ])

    st.markdown(f'<div class="stats-container">{stats_html}</div>', unsafe_allow_html=True)


def hero_section():
    """Hero section for the Lunar Crater Age Classifier"""
    st.markdown('<div class="hero-container">', unsafe_allow_html=True)

    render_badge("AI-Powered Lunar Crater Age Classifier")
    render_heading("Discover the", "Age of Lunar Craters")

    st.markdown("""
        <p class="subtitle">
            Upload lunar surface imagery and let our AI classify craters as fresh,
            old, or non-crater regions with precise age estimations.
        </p>
    """, unsafe_allow_html=True)

    # CTA Buttons
    _, col_center, _ = st.columns([1, 2, 1])
    with col_center:
        btn_col1, btn_col2 = st.columns(2)

        with btn_col1:
            if st.button("🚀 Start Classifying", use_container_width=True, type="primary"):
                st.switch_page("pages/classify.py")

        with btn_col2:
            if st.button("📖 Learn More", use_container_width=True):
                st.switch_page("pages/about.py")

    render_stats()
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="LunarCrater Classifier",
        page_icon="🌙",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Store backend URL in session state
    if 'backend_url' not in st.session_state:
        st.session_state.backend_url = BACKEND_URL

    load_styles()
    hero_section()
    st.markdown("<br><br>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
