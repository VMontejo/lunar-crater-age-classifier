import streamlit as st

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
                <h2 style="margin-top: 1rem; color: white;">LunarCraterAge Classifier</h2>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Get current page
        try:
            current_page = st.navigation.current_page.name if hasattr(st, 'navigation') else None
        except:
            current_page = None

        # Navigation buttons
        if st.button("🏠 Home", use_container_width=True,
                     type="primary" if current_page == "main" else "secondary"):
            st.switch_page("main.py")

        if st.button("🔬 Classify", use_container_width=True,
                     type="primary" if current_page == "classify" else "secondary"):
            st.switch_page("pages/classify.py")

        if st.button("ℹ️ About", use_container_width=True,
                     type="primary" if current_page == "about" else "secondary"):
            st.switch_page("pages/about.py")

        st.markdown("---")

        # Stats in sidebar
        st.markdown("### Who are we")
        st.markdown("#### our names")
        st.markdown("#### our names")
        st.markdown("#### our names")
        st.markdown("#### our names")
        st.markdown("#### our names")
