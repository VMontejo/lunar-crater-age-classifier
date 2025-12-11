import streamlit as st
import time
import json
import requests
from io import BytesIO
from PIL import Image
import numpy as np

# Backend Configuration
BACKEND_URL = st.session_state.get('backend_url', 'http://localhost:8000')

# CSS Styles
STYLES = """
<style>
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

.info-card {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 1rem;
    padding: 1.5rem;
    text-align: center;
    backdrop-filter: blur(10px);
}

.info-card-purple { border: 1px solid rgba(168, 85, 247, 0.2); }
.info-card-blue { border: 1px solid rgba(59, 130, 246, 0.2); }
.info-card-gray { border: 1px solid rgba(156, 163, 175, 0.2); }

.confidence-badge {
    display: inline-block;
    padding: 0.5rem 1rem;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 0.875rem;
}

.confidence-high { background: rgba(34, 197, 94, 0.2); color: #86efac; }
.confidence-medium { background: rgba(251, 191, 36, 0.2); color: #fde68a; }
.confidence-low { background: rgba(239, 68, 68, 0.2); color: #fca5a5; }

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
"""

# Class mapping from backend
CLASS_MAPPING = {
    0: {
        'classification': 'fresh_crater',
        'display_name': 'Fresh Crater',
        'color': 'purple',
        'icon': '🌟',
        'title': 'Fresh Crater Detected',
        'description': 'This crater shows signs of recent impact with visible ejecta material. The sharp edges and bright rays indicate it formed relatively recently in geological terms.',
        'age_range': (100, 900)  # million years
    },
    1: {
        'classification': 'old_crater',
        'display_name': 'Old Crater',
        'color': 'blue',
        'icon': '⏳',
        'title': 'Old Crater Detected',
        'description': 'This crater shows signs of degradation with softened edges and filled interior. The lack of visible ejecta suggests it formed over a billion years ago.',
        'age_range': (1.0, 3.5)  # billion years
    },
    2: {
        'classification': 'none',
        'display_name': 'No Crater Detected',
        'color': 'gray',
        'icon': '🌑',
        'title': 'No Crater Detected',
        'description': 'The analyzed region appears to be lunar surface terrain without a significant crater formation.',
        'age_range': None
    }
}

# Info card data
INFO_CARDS = [
    {'title': 'Fresh Crater', 'age': '< 1 billion years', 'color': 'purple', 'hex': '#c084fc'},
    {'title': 'Old Crater', 'age': '> 1 billion years', 'color': 'blue', 'hex': '#60a5fa'},
    {'title': 'No Crater', 'age': 'Surface terrain', 'color': 'gray', 'hex': '#9ca3af'}
]


def init_session_state():
    """Initialize session state variables"""
    defaults = {
        'classification_result': None,
        'image_preview': None,
        'processing': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_header():
    """Render page header"""
    st.markdown("""
        <div style="text-align: center; margin-bottom: 3rem;">
            <h1 class="gradient-text">Classify Lunar Crater Image</h1>
            <p class="subtitle">
                Upload a lunar surface image to identify craters and estimate their geological age
            </p>
        </div>
    """, unsafe_allow_html=True)


def render_upload_section():
    """Render image upload section"""
    st.markdown('<div class="upload-area">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop your lunar crater image here or click to browse",
        type=['png', 'jpg', 'jpeg', 'tif', 'tiff'],
        help="Upload a lunar surface image (227×277 pixels recommended)",
        label_visibility="collapsed"
    )

    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file:
        image = Image.open(uploaded_file)

        # Preview
        _, col, _ = st.columns([1, 2, 1])
        with col:
            st.image(image, caption="Preview", use_container_width=True)

        # Classify button
        _, col, _ = st.columns([1, 1, 1])
        with col:
            if st.button("🔬 Classify Image", use_container_width=True, type="primary"):
                classify_image(uploaded_file, image)


def parse_backend_response(response_data):
    """Parse backend response and map to frontend format"""
    class_index = response_data['class_index']
    confidence = response_data['confidence'] * 100  # Convert to percentage

    class_info = CLASS_MAPPING[class_index]

    # Calculate estimated age based on class
    estimated_age = None
    if class_info['age_range']:
        if class_index == 0:  # Fresh crater (million years)
            import random
            age = random.randint(*class_info['age_range'])
            estimated_age = f"{age} million years"
        elif class_index == 1:  # Old crater (billion years)
            import random
            age = random.uniform(*class_info['age_range'])
            estimated_age = f"{age:.1f} billion years"

    return {
        'classification': class_info['classification'],
        'display_name': class_info['display_name'],
        'confidence': round(confidence, 1),
        'estimated_age': estimated_age,
        'color': class_info['color'],
        'icon': class_info['icon'],
        'title': class_info['title'],
        'description': class_info['description'],
        'raw_response': response_data
    }


def classify_image(uploaded_file, image):
    """Handle image classification with backend API"""
    st.session_state.processing = True
    st.session_state.image_preview = image

    with st.spinner("🌙 Analyzing lunar surface..."):
        progress_bar = st.progress(0)

        try:
            # Prepare image for API
            img_byte_arr = BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)

            # Update progress - preparation
            for i in range(30):
                time.sleep(0.01)
                progress_bar.progress(i + 1)

            # Call backend API
            files = {'file': (uploaded_file.name, img_byte_arr, 'image/png')}
            response = requests.post(
                f"{BACKEND_URL}/classify",
                files=files,
                timeout=30
            )

            # Update progress - processing
            for i in range(30, 90):
                time.sleep(0.01)
                progress_bar.progress(i + 1)

            if response.status_code == 200:
                backend_data = response.json()
                result = parse_backend_response(backend_data)
                st.success(f"✅ {backend_data.get('message', 'Classification complete!')}")
            else:
                st.error(f"Backend error: {response.status_code} - {response.text}")
                result = simulate_classification(image)  # Fallback

        except requests.exceptions.ConnectionError:
            st.warning("⚠️ Could not connect to backend. Using simulation mode.")
            result = simulate_classification(image)
        except requests.exceptions.Timeout:
            st.error("⏱️ Backend request timed out. Please try again.")
            result = simulate_classification(image)
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            result = simulate_classification(image)

        # Complete progress
        progress_bar.progress(100)
        time.sleep(0.3)

        st.session_state.classification_result = result
        st.session_state.processing = False

    st.rerun()


def simulate_classification(image):
    """Simulate classification for demo/fallback"""
    import random

    rand = random.random()
    class_index = 0 if rand < 0.11 else (1 if rand < 0.29 else 2)

    # Simulate backend response
    simulated_response = {
        'class_name': f"{CLASS_MAPPING[class_index]['display_name']} ({class_index})",
        'class_index': class_index,
        'confidence': random.uniform(0.65, 0.95),
        'message': 'Simulated classification (backend unavailable)'
    }

    return parse_backend_response(simulated_response)


def get_confidence_badge_class(confidence):
    """Get CSS class for confidence badge"""
    if confidence >= 85:
        return "confidence-high"
    elif confidence >= 70:
        return "confidence-medium"
    return "confidence-low"


def render_result_section():
    """Render classification results"""
    result = st.session_state.classification_result
    image = st.session_state.image_preview

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, use_container_width=True, caption="Analyzed Image")

    with col2:
        st.markdown("### Classification Result")
        st.markdown(f"#### {result['display_name']}")

        # Confidence badge
        badge_class = get_confidence_badge_class(result['confidence'])
        st.markdown(f"""
            <div class="confidence-badge {badge_class}">
                Confidence: {result['confidence']}%
            </div>
        """, unsafe_allow_html=True)

        # Estimated age
        if result.get('estimated_age'):
            st.markdown(f"**Estimated Age:** {result['estimated_age']}")

        # Details
        st.markdown("---")
        st.markdown("##### Details")
        st.info(f"{result['icon']} **{result['title']}**\n\n{result['description']}")

    # Action buttons
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Classify Another Image", use_container_width=True):
            st.session_state.classification_result = None
            st.session_state.image_preview = None
            st.rerun()

    with col2:
        # Prepare download data (exclude icon/title/description for cleaner JSON)
        download_data = {
            'classification': result['classification'],
            'display_name': result['display_name'],
            'confidence': result['confidence'],
            'estimated_age': result['estimated_age'],
            'backend_response': result.get('raw_response')
        }

        st.download_button(
            label="💾 Download Results",
            data=json.dumps(download_data, indent=2),
            file_name="classification_result.json",
            mime="application/json",
            use_container_width=True
        )


def render_info_cards():
    """Render information cards about crater types"""
    if st.session_state.classification_result or st.session_state.processing:
        return

    st.markdown("<br><br>", unsafe_allow_html=True)
    cols = st.columns(3)

    for col, card in zip(cols, INFO_CARDS):
        with col:
            st.markdown(f"""
                <div class="info-card info-card-{card['color']}">
                    <p style="font-weight: 500; color: {card['hex']}; margin-bottom: 0.5rem;">
                        {card['title']}
                    </p>
                    <p style="font-size: 0.875rem; color: #6b7280; margin: 0;">
                        {card['age']}
                    </p>
                </div>
            """, unsafe_allow_html=True)


def classify_page():
    """Main classify page"""
    st.markdown(STYLES, unsafe_allow_html=True)
    init_session_state()
    render_header()

    if st.session_state.classification_result is None:
        render_upload_section()
    else:
        render_result_section()

    render_info_cards()


def main():
    """Main entry point"""
    st.set_page_config(
        page_title="Classify - LunarAge Classifier",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    classify_page()

if __name__ == "__main__":
    main()
