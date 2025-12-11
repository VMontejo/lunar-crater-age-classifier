import streamlit as st
from pathlib import Path

def hero_section():
    """
    Hero section for the Lunar Crater Age Classifier
    """

    # Custom CSS for styling
    st.markdown("""
        <style>
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
            text-align: left;
            color: #9ca3af;
            max-width: 42rem;
            margin: 0 auto 2.5rem;
            line-height: 1.75;
        }

        /* Custom Primary Button Gradient */
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

        .stats-container {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 2rem;
            max-width: 36rem;
            margin: 3rem auto 0;
            padding-top: 3rem;
        }

        .stat-item {
            text-align: center;
        }

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

        /* Features Section Styles */
        .features-section {
            padding: 6rem 1rem;
            max-width: 72rem;
            margin: 0 auto;
        }

        .features-header {
            text-align: center;
            margin-bottom: 4rem;
        }

        .features-subtitle {
            color: #9ca3af;
            max-width: 36rem;
            margin: 1rem auto 0;
        }

        .feature-card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 1rem;
            padding: 1.5rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(139, 92, 246, 0.2);
            transition: all 0.3s ease;
            height: 100%;
        }

        .feature-card:hover {
            background: rgba(255, 255, 255, 0.08);
            transform: translateY(-4px);
        }

        .feature-icon {
            width: 3rem;
            height: 3rem;
            border-radius: 0.75rem;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 1rem;
            font-size: 1.5rem;
            transition: transform 0.3s ease;
        }

        .feature-card:hover .feature-icon {
            transform: scale(1.1);
        }

        .feature-icon-yellow {
            background: linear-gradient(135deg, #eab308 0%, #f97316 100%);
        }

        .feature-icon-green {
            background: linear-gradient(135deg, #22c55e 0%, #10b981 100%);
        }

        .feature-icon-purple {
            background: linear-gradient(135deg, #a855f7 0%, #ec4899 100%);
        }

        .feature-icon-blue {
            background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%);
        }

        .feature-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: white;
            margin-bottom: 0.5rem;
        }

        .feature-description {
            color: #9ca3af;
            line-height: 1.75;
        }

        /* Responsive adjustments */
        @media (max-width: 640px) {
            .stats-container {
                grid-template-columns: 1fr;
                gap: 1.5rem;
            }
        }
        </style>
    """, unsafe_allow_html=True)

    # Hero content
    st.markdown('<div class="hero-container">', unsafe_allow_html=True)

    # Badge
    st.markdown("""
        <div class="badge">
            ✨ AI-Powered Lunar Crater Age Classifier
        </div>
    """, unsafe_allow_html=True)

    # Main heading
    st.markdown("""
        <h1 class="main-heading">
            <span style="color: white;">Discover the</span><br/>
            <span class="gradient-text">Age of Lunar Craters</span>
        </h1>
    """, unsafe_allow_html=True)

    # Subtitle
    st.markdown("""
        <p class="subtitle">
            Upload lunar surface imagery and let our AI classify craters as fresh,
            old, or non-crater regions with precise age estimations.
        </p>
    """, unsafe_allow_html=True)

    # CTA Buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        button_col1, button_col2 = st.columns(2)

        with button_col1:
            if st.button("🚀 Start Classifying", use_container_width=True, type="primary"):
                st.switch_page("pages/classify.py")

        with button_col2:
            if st.button("📖 Learn More", use_container_width=True):
                st.switch_page("pages/about.py")

    # Stats section
    st.markdown("""
        <div class="stats-container">
            <div class="stat-item">
                <div class="stat-value">5,000+</div>
                <div class="stat-label">Labeled Images</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">3</div>
                <div class="stat-label">Classifications</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">227×277</div>
                <div class="stat-label">Image Size</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

def features_section():
    """
    Features section showcasing the key capabilities
    """

    features = [
        {
            "icon": "⚡",
            "title": "Instant Classification",
            "description": "Upload an image and receive classification results in seconds with our optimized AI model.",
            "color": "yellow"
        },
        {
            "icon": "🎯",
            "title": "High Accuracy",
            "description": "Trained on 5,000+ labeled lunar images from the Lunar Reconnaissance Orbiter.",
            "color": "green"
        },
        {
            "icon": "🕐",
            "title": "Age Estimation",
            "description": "Get estimated crater ages distinguishing between fresh and ancient impact sites.",
            "color": "purple"
        },
    ]

    # Header
    st.markdown("""
        <div class="features-header">
            <h2 class="main-heading" style="font-size: 2.5rem;">
                <span class="gradient-text">Powerful Features</span>
            </h2>
        </div>
    """, unsafe_allow_html=True)

    # Features Grid
    col1, col2 = st.columns(2)

    for idx, feature in enumerate(features):
        col = col1 if idx % 2 == 0 else col2

        with col:
            st.markdown(f"""
                <div class="feature-card">
                    <div class="feature-icon feature-icon-{feature['color']}">
                        {feature['icon']}
                    </div>
                    <h3 class="feature-title">{feature['title']}</h3>
                    <p class="feature-description">{feature['description']}</p>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Main page logic
if __name__ == "__main__":
    st.set_page_config(
        page_title="LunarCrater Classifier",
        page_icon="🌙",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Render hero section
    hero_section()

    # Add some spacing
    st.markdown("<br><br>", unsafe_allow_html=True)

    # Render features section
    features_section()
