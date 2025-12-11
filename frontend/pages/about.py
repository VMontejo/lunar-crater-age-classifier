import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.layout import init_layout, render_footer
from utils.navigation import render_sidebar_navigation

# Initialize layout
init_layout(page_title="About", page_icon="ℹ️")

# Render navigation
render_sidebar_navigation()

# Constants
DATASET_INFO = [
    {
        "icon": "🔭",
        "label": "Source",
        "value": "Lunar Reconnaissance Orbiter Imagery",
        "sublabel": "LROCNet Moon Classifier Dataset"
    },
    {
        "icon": "💾",
        "label": "Dataset Size",
        "value": "5,000+ labeled images",
        "sublabel": "Pre-sorted train/validation/test sets"
    },
    {
        "icon": "🖼️",
        "label": "Image Specs",
        "value": "227×277 pixels",
        "sublabel": "Lunar surface cutouts"
    },
    {
        "icon": "📊",
        "label": "Classes",
        "value": "3 categories",
        "sublabel": "Fresh crater, Old crater, None"
    }
]

CLASS_DISTRIBUTION = [
    {"label": "Fresh Crater", "percentage": 11, "color": "purple", "icon": "✨"},
    {"label": "Old Crater", "percentage": 18, "color": "blue", "icon": "⭕"},
    {"label": "None", "percentage": 71, "color": "gray", "icon": "🚫"}
]

COLOR_MAP = {
    "purple": "#a855f7",
    "blue": "#60a5fa",
    "gray": "#9ca3af"
}

METHODOLOGIES = [
    {
        "title": "Fresh Craters",
        "color": "purple",
        "description": """Characterized by sharp, well-defined rims and bright ejecta blankets. These craters are
            typically less than 1 billion years old and show minimal signs of degradation from
            space weathering and subsequent impacts."""
    },
    {
        "title": "Old Craters",
        "color": "blue",
        "description": """Ancient impact sites with degraded, subdued rims and infilled floors. These craters
            are typically 1-4 billion years old and have been significantly modified by subsequent
            impacts and lunar geological processes."""
    },
    {
        "title": "No Crater (None)",
        "color": "gray",
        "description": """Lunar surface regions without identifiable crater features. This includes mare basalt
            plains, highland terrain, and areas where craters may be too small or degraded to classify."""
    }
]

CSS_STYLES = """
    <style>
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
        margin-bottom: 1.5rem;
    }

    .gradient-text {
        background: linear-gradient(135deg, #a855f7 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: 700;
    }

    .subtitle {
        color: #9ca3af;
        max-width: 42rem;
        margin: 1rem auto 0;
        line-height: 1.75;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 1rem;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(139, 92, 246, 0.2);
        transition: all 0.3s ease;
    }

    .glass-card:hover {
        background: rgba(255, 255, 255, 0.08);
    }

    .icon-container {
        width: 2.5rem;
        height: 2.5rem;
        border-radius: 0.5rem;
        background: rgba(168, 85, 247, 0.2);
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }

    .progress-bar-container {
        height: 0.75rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 9999px;
        overflow: hidden;
        margin-top: 0.5rem;
    }

    .progress-bar {
        height: 100%;
        border-radius: 9999px;
        animation: grow 1s ease-out;
    }

    .progress-bar-purple {
        background: linear-gradient(to right, #9333ea, #a855f7);
    }

    .progress-bar-blue {
        background: linear-gradient(to right, #2563eb, #60a5fa);
    }

    .progress-bar-gray {
        background: linear-gradient(to right, #4b5563, #9ca3af);
    }

    @keyframes grow {
        from { width: 0; }
    }

    .methodology-card {
        border-left: 3px solid;
        padding-left: 1rem;
        margin-bottom: 1rem;
    }

    .methodology-purple { border-color: #a855f7; }
    .methodology-blue { border-color: #3b82f6; }
    .methodology-gray { border-color: #6b7280; }

    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: white;
        margin-bottom: 1.5rem;
    }

    .data-label {
        font-size: 0.875rem;
        color: #6b7280;
        margin-bottom: 0.25rem;
    }

    .data-value {
        font-size: 1.125rem;
        font-weight: 600;
        color: white;
        margin-bottom: 0.25rem;
    }

    .data-sublabel {
        font-size: 0.875rem;
        color: #6b7280;
    }
    </style>
"""

def render_header():
    """Render the page header with title and subtitle."""
    st.markdown("""
        <div class="main-heading">
            <div class="badge">
                📖 About the Project
            </div>
            <h1 class="gradient-text">Understanding Lunar Crater Age Classifier</h1>
            <p class="subtitle">
                Our AI model is trained on high-quality lunar surface imagery from NASA's
                Lunar Reconnaissance Orbiter to classify craters and estimate their geological age.
            </p>
        </div>
    """, unsafe_allow_html=True)

def render_info_card(item):
    """Render a single information card."""
    return f"""
        <div class="glass-card">
            <div style="display: flex; align-items: flex-start; gap: 1rem;">
                <div class="icon-container">
                    <span style="font-size: 1.25rem;">{item['icon']}</span>
                </div>
                <div>
                    <p class="data-label">{item['label']}</p>
                    <p class="data-value">{item['value']}</p>
                    <p class="data-sublabel">{item['sublabel']}</p>
                </div>
            </div>
        </div>
    """


def render_dataset_overview():
    """Render the dataset overview section."""
    st.markdown('<h2 class="section-title">Dataset Overview</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    for idx, item in enumerate(DATASET_INFO):
        col = col1 if idx % 2 == 0 else col2
        with col:
            st.markdown(render_info_card(item), unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)


def render_class_distribution():
    """Render the class distribution section with progress bars."""
    st.markdown('<h2 class="section-title">Class Distribution</h2>', unsafe_allow_html=True)

    for item in CLASS_DISTRIBUTION:
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown(f"""
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1rem;">{item['icon']}</span>
                    <span style="color: #d1d5db;">{item['label']}</span>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
                <span style="font-weight: 600; color: {COLOR_MAP[item['color']]};">
                    {item['percentage']}%
                </span>
            """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class="progress-bar-container">
                <div class="progress-bar progress-bar-{item['color']}"
                     style="width: {item['percentage']}%;"></div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

def render_methodology():
    """Render the classification methodology section."""
    st.markdown('<h2 class="section-title">Classification Methodology</h2>', unsafe_allow_html=True)

    for method in METHODOLOGIES:
        st.markdown(f"""
            <div class="methodology-card methodology-{method['color']}">
                <h3 style="font-weight: 600; color: white; margin-bottom: 0.5rem;">
                    {method['title']}
                </h3>
                <p style="color: #9ca3af; font-size: 0.875rem; line-height: 1.75; margin: 0;">
                    {method['description']}
                </p>
            </div>
        """, unsafe_allow_html=True)

def render_data_source():
    """Render the data source section."""
    st.markdown('<h2 class="section-title">Data Source</h2>', unsafe_allow_html=True)

    st.markdown("""
        <p style="color: #9ca3af; line-height: 1.75; margin-bottom: 1.5rem;">
            The training data is sourced from the publicly available LROCNet Moon Classifier dataset,
            which contains 5,000 labeled lunar surface images captured by NASA's Lunar Reconnaissance
            Orbiter Camera (LROC).
        </p>
    """, unsafe_allow_html=True)

    st.link_button(
        "🔗 View Dataset on Zenodo",
        "https://zenodo.org/records/7041842",
        use_container_width=False
    )

    st.markdown("<br><br>", unsafe_allow_html=True)

    with st.expander("📚 Reference & Citation"):
        st.markdown("""
            **LROCNet: Detecting Impact Ejecta and Older Craters on the Lunar Surface**

            *Emily Dunkel, Steven Lu, Kevin Grimes, James McAuley, and Kiri Wagstaff*
            Jet Propulsion Laboratory, California Institute of Technology

            ---

            **Abstract:**
            NASA's Planetary Data System (PDS) contains data collected by missions to explore our solar system.
            This includes Lunar Reconnaissance Orbiter (LRO), which has collected as much data as all other
            planetary missions combined. Working with the PDS Cartography and Imaging Sciences Node (IMG),
            we developed LROCNet, a deep learning classifier for imagery from LRO's Narrow Angle Cameras (NACs).

            ---

            **Dataset Details:**
            - **Total Images:** 5,000 labeled lunar surface images
            - **Image Resolution:** 227×277 pixels
            - **Classes:** Fresh crater (11%), Old crater (18%), None (71%)
            - **Source:** Lunar Reconnaissance Orbiter Camera (LROC)
            - **Availability:** [Zenodo.org](https://zenodo.org/records/7041842)

            ---

            **Model Performance:**
            - **Overall Accuracy:** 82% on test set
            - **Fresh Crater Detection:** 64% accuracy
            - **Old Crater Detection:** 80% accuracy
            - **No Crater Detection:** 86% accuracy
            - **Human Label Agreement:** 83%

            The model's accuracy is comparable to human labelers, as labeling is challenging
            due to the lack of clear class boundaries in some cases.
        """)

def about_page():
    """Main about page for the Lunar Crater Age Classifier."""
    st.markdown(CSS_STYLES, unsafe_allow_html=True)

    render_header()
    render_dataset_overview()
    st.markdown("<br>", unsafe_allow_html=True)

    render_class_distribution()
    st.markdown("<br>", unsafe_allow_html=True)

    render_methodology()
    st.markdown("<br>", unsafe_allow_html=True)

    render_footer()

if __name__ == "__main__":
    st.set_page_config(
        page_title="About - LunarCrater",
        page_icon="ℹ️",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    about_page()
