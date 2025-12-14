import streamlit as st
from pathlib import Path
from utils.layout import init_layout, render_footer
from utils.navigation import render_sidebar_navigation

# Initialize session state
if 'backend_url' not in st.session_state:
    st.session_state.backend_url = st.secrets['BACKEND_URL']

# Initialize layout
init_layout(page_title="Home", page_icon="🌙")

# Render navigation
render_sidebar_navigation()

def render_badge(text, icon="✨"):
    """Render a badge component"""
    st.markdown(f"""
        <div style="display: inline-flex; align-items: center; gap: 0.5rem;
                    padding: 0.5rem 1rem; border-radius: 9999px;
                    background: rgba(139, 92, 246, 0.1);
                    border: 1px solid rgba(139, 92, 246, 0.3);
                    color: #c084fc; font-size: 0.875rem; margin-bottom: 2rem;">
            {icon} {text}
        </div>
    """, unsafe_allow_html=True)

def render_heading(main_text, gradient_text):
    """Render main heading with gradient"""
    st.markdown(f"""
        <h1 style="font-size: clamp(2rem, 5vw, 4rem); font-weight: 700;
                   line-height: 1.2; margin-bottom: 1.5rem; text-align: center;">
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

    cols = st.columns(3)
    for col, stat in zip(cols, stats):
        with col:
            st.markdown(f"""
                <div style="text-align: center;">
                    <div class="gradient-text" style="font-size: 1.875rem; font-weight: 700;">
                        {stat['value']}
                    </div>
                    <div style="font-size: 0.875rem; color: #6b7280; margin-top: 0.25rem;">
                        {stat['label']}
                    </div>
                </div>
            """, unsafe_allow_html=True)

def hero_section():
    """Hero section for the Lunar Crater Age Classifier"""
    render_badge("AI-Powered Lunar Crater Age Classifier")
    render_heading("Discover the", "Age of Lunar Craters")

    st.markdown("""
        <p style="font-size: 1.25rem; color: #9ca3af; max-width: 42rem;
                  margin: 0 auto 2.5rem; line-height: 1.75; text-align: center;">
            Upload lunar surface imagery and let our AI classify craters as fresh,
            old, or non-crater regions with precise age estimations.
        </p>
    """, unsafe_allow_html=True)

    # CTA Buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        btn_col1, btn_col2 = st.columns(2)

        with btn_col1:
            if st.button("🚀 Start Classifying", use_container_width=True, type="primary"):
                st.switch_page("pages/classify.py")

        with btn_col2:
            if st.button("📖 Learn More", use_container_width=True):
                st.switch_page("pages/about.py")

    st.markdown("<br><br>", unsafe_allow_html=True)
    render_stats()

def main():
    """Main application entry point"""
    hero_section()
    st.markdown("<br><br>", unsafe_allow_html=True)
    render_footer()

if __name__ == "__main__":
    main()
