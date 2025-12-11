import streamlit as st
import time
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image
import numpy as np

def classify_page():
    """
    Classify page for lunar crater age classification
    """

    # Custom CSS
    st.markdown("""
        <style>
        .classify-header {
            text-align: center;
            margin-bottom: 3rem;
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
            max-width: 36rem;
            margin: 1rem auto 0;
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

        .info-card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 1rem;
            padding: 1.5rem;
            text-align: center;
            border: 1px solid rgba(139, 92, 246, 0.2);
            backdrop-filter: blur(10px);
        }

        .info-card-purple {
            border-color: rgba(168, 85, 247, 0.2);
        }

        .info-card-blue {
            border-color: rgba(59, 130, 246, 0.2);
        }

        .info-card-gray {
            border-color: rgba(156, 163, 175, 0.2);
        }

        .result-container {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 1rem;
            padding: 2rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(139, 92, 246, 0.2);
        }

        .confidence-badge {
            display: inline-block;
            padding: 0.5rem 1rem;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.875rem;
        }

        .confidence-high {
            background: rgba(34, 197, 94, 0.2);
            color: #86efac;
        }

        .confidence-medium {
            background: rgba(251, 191, 36, 0.2);
            color: #fde68a;
        }

        .confidence-low {
            background: rgba(239, 68, 68, 0.2);
            color: #fca5a5;
        }

        .upload-area {
            border: 2px dashed rgba(139, 92, 246, 0.3);
            border-radius: 1rem;
            padding: 3rem 2rem;
            text-align: center;
            background: rgba(255, 255, 255, 0.02);
            transition: all 0.3s ease;
        }

        .upload-area:hover {
            border-color: rgba(168, 85, 247, 0.5);
            background: rgba(255, 255, 255, 0.05);
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'classification_result' not in st.session_state:
        st.session_state.classification_result = None
    if 'image_preview' not in st.session_state:
        st.session_state.image_preview = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False

    # Header
    st.markdown("""
        <div class="main-heading">
            <h1 class="gradient-text">Classify Lunar Crater Image</h1>
            <p class="subtitle">
                Upload a lunar surface image to identify craters and estimate their geological age
                <p>
                </p>
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Main content area
    if st.session_state.classification_result is None:
        render_upload_section()
    else:
        render_result_section()

    # Info cards (only show when no result and not processing)
    if st.session_state.classification_result is None and not st.session_state.processing:
        render_info_cards()


def render_upload_section():
    """Render the image upload section"""

    st.markdown('<div class="upload-area">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop your lunar crater image here or click to browse",
        type=['png', 'jpg', 'jpeg', 'tif', 'tiff'],
        help="Upload a lunar surface image (227×277 pixels recommended)",
        label_visibility="collapsed"
    )

    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        # Show preview
        image = Image.open(uploaded_file)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Preview", use_container_width=True)

        # Classify button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🔬 Classify Image", use_container_width=True, type="primary"):
                classify_image(uploaded_file, image)


def classify_image(uploaded_file, image):
    """Handle image classification"""

    st.session_state.processing = True

    # Store image preview
    st.session_state.image_preview = image

    # Show progress
    with st.spinner("🌙 Analyzing lunar surface..."):
        # Simulate processing time
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.02)
            progress_bar.progress(i + 1)

        # TODO: Replace this with your actual model inference
        # result = our_model.predict(image)

        # Simulated classification result
        # In production, this would call your trained model
        result = simulate_classification(image)

        st.session_state.classification_result = result
        st.session_state.processing = False

    st.rerun()

# Replace the simulate_classification() function with our actual model inference:
def simulate_classification(image):
    """
    Simulate classification result
    TODO: Replace with actual model inference
    """
    import random

    # Simulate realistic distribution from the paper
    rand = random.random()

    if rand < 0.11:  # 11% fresh craters
        classification = "fresh_crater"
        display_name = "Fresh Crater"
        confidence = random.randint(70, 95)
        estimated_age = f"{random.randint(100, 900)} million years"
        color = "purple"
    elif rand < 0.29:  # 18% old craters (0.11 + 0.18)
        classification = "old_crater"
        display_name = "Old Crater"
        confidence = random.randint(75, 92)
        estimated_age = f"{random.uniform(1.0, 3.5):.1f} billion years"
        color = "blue"
    else:  # 71% no crater
        classification = "none"
        display_name = "No Crater Detected"
        confidence = random.randint(80, 98)
        estimated_age = None
        color = "gray"

    return {
        "classification": classification,
        "display_name": display_name,
        "confidence": confidence,
        "estimated_age": estimated_age,
        "color": color
    }


def render_result_section():
    """Render the classification result"""

    result = st.session_state.classification_result
    image = st.session_state.image_preview

    # Image and results side by side
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, use_container_width=True, caption="Analyzed Image")

    with col2:
        st.markdown(f"### Classification Result")

        # Classification type
        st.markdown(f"#### {result['display_name']}")

        # Confidence badge
        confidence = result['confidence']
        if confidence >= 85:
            badge_class = "confidence-high"
        elif confidence >= 70:
            badge_class = "confidence-medium"
        else:
            badge_class = "confidence-low"

        st.markdown(f"""
            <div class="confidence-badge {badge_class}">
                Confidence: {confidence}%
            </div>
        """, unsafe_allow_html=True)

        # Estimated age if applicable
        if result['estimated_age']:
            st.markdown(f"**Estimated Age:** {result['estimated_age']}")

        # Additional details
        st.markdown("---")
        st.markdown("##### Details")

        if result['classification'] == 'fresh_crater':
            st.info("""
                🌟 **Fresh Crater Detected**

                This crater shows signs of recent impact with visible ejecta
                material. The sharp edges and bright rays indicate it formed
                relatively recently in geological terms.
            """)
        elif result['classification'] == 'old_crater':
            st.info("""
                ⏳ **Old Crater Detected**

                This crater shows signs of degradation with softened edges
                and filled interior. The lack of visible ejecta suggests
                it formed over a billion years ago.
            """)
        else:
            st.info("""
                🌑 **No Crater Detected**

                The analyzed region appears to be lunar surface terrain
                without a significant crater formation.
            """)

    st.markdown('</div>', unsafe_allow_html=True)

    # Action buttons
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2= st.columns([1, 1])

    with col1:
        if st.button("🔄 Classify Another Image", use_container_width=True):
            st.session_state.classification_result = None
            st.session_state.image_preview = None
            st.rerun()

    with col2:
        # Download results as JSON
        import json
        result_json = json.dumps(result, indent=2)
        st.download_button(
            label="💾 Download Results",
            data=result_json,
            file_name="classification_result.json",
            mime="application/json",
            use_container_width=True
        )

def render_info_cards():
    """Render information cards about crater types"""

    st.markdown("<br><br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
            <div class="info-card info-card-purple">
                <p style="font-weight: 500; color: #c084fc; margin-bottom: 0.5rem;">
                    Fresh Crater
                </p>
                <p style="font-size: 0.875rem; color: #6b7280; margin: 0;">
                    < 1 billion years
                </p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="info-card info-card-blue">
                <p style="font-weight: 500; color: #60a5fa; margin-bottom: 0.5rem;">
                    Old Crater
                </p>
                <p style="font-size: 0.875rem; color: #6b7280; margin: 0;">
                    > 1 billion years
                </p>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class="info-card info-card-gray">
                <p style="font-weight: 500; color: #9ca3af; margin-bottom: 0.5rem;">
                    No Crater
                </p>
                <p style="font-size: 0.875rem; color: #6b7280; margin: 0;">
                    Surface terrain
                </p>
            </div>
        """, unsafe_allow_html=True)


# Main execution
if __name__ == "__main__":
    st.set_page_config(
        page_title="Classify - LunarAge Classifier",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    classify_page()
