import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Chest X-Ray Screening AI",
    page_icon="🫁",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #2E86AB, #4FC3F7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #9CA3AF;
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }
    .result-card {
        background: #1C2430;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #2A3441;
    }
    .risk-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 0.8rem;
    }
    .risk-low { background: #1B4332; color: #4ADE80; }
    .risk-medium { background: #4A3B0A; color: #FBBF24; }
    .risk-high { background: #4A1515; color: #F87171; }

    .metric-row {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-bottom: 1px solid #2A3441;
    }
    .metric-label { color: #9CA3AF; font-size: 0.9rem; }
    .metric-value { font-weight: 600; }

    div[data-testid="stFileUploader"] {
        border: 2px dashed #2E86AB;
        border-radius: 12px;
        padding: 1rem;
    }

    .footer-disclaimer {
        text-align: center;
        color: #6B7280;
        font-size: 0.8rem;
        padding: 1.5rem 0;
        border-top: 1px solid #2A3441;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

RECOMMENDED_ACTION = {
    "Low": "No immediate concern flagged. Continue routine care and consult a doctor if symptoms develop.",
    "Medium": "Consider scheduling a check-up with a doctor for further evaluation.",
    "High": "We recommend consulting a doctor or radiologist promptly for a proper evaluation.",
}

RISK_CLASS = {"Low": "risk-low", "Medium": "risk-medium", "High": "risk-high"}
RISK_EMOJI = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}


def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>🫁 Chest X-Ray Screening AI</h1>
    </div>
    <p class="subtitle">AI-powered pneumonia pattern detection with visual explainability</p>
    """, unsafe_allow_html=True)


def render_footer_disclaimer():
    st.markdown("""
    <div class="footer-disclaimer">
        🏥 This tool is a research/awareness prototype built for Data Odyssey 2026.<br>
        It is <b>not</b> a certified medical device and must not be used for actual diagnosis
        or treatment decisions.<br>
        If you are experiencing symptoms, please seek care from a licensed healthcare provider.
    </div>
    """, unsafe_allow_html=True)


# --- Disclaimer gate ---
if "disclaimer_accepted" not in st.session_state:
    st.session_state.disclaimer_accepted = False

if not st.session_state.disclaimer_accepted:
    render_header()
    st.info(
        "Before continuing, please confirm you understand this tool provides "
        "**risk awareness only**, not a medical diagnosis."
    )
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("✅ I understand — Continue", use_container_width=True):
            st.session_state.disclaimer_accepted = True
            st.rerun()
    st.stop()

# --- Main app ---
render_header()

uploaded_file = st.file_uploader(
    "📤 Upload a chest X-ray image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, PNG"
)

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(uploaded_file, caption="Uploaded X-Ray", use_container_width=True)
    with col2:
        st.markdown("### Ready to analyze")
        st.write(f"**File:** {uploaded_file.name}")
        st.write(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
        analyze_clicked = st.button("🔍 Analyze Image", use_container_width=True, type="primary")

    if analyze_clicked:
        with st.spinner("Analyzing image..."):
            response = requests.post(
                f"{API_URL}/predict",
                files={"file": (uploaded_file.name, uploaded_file.getvalue())}
            )

            if response.status_code != 200:
                error_msg = response.json().get("error", "Unknown error occurred.")
                st.error(f"⚠️ {error_msg}")
                st.info("Try uploading a clear, in-focus chest X-ray image in JPG or PNG format.")
                st.stop()

            st.session_state["last_result"] = response.json()

    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        risk = result["risk_level"]

        st.markdown(f"""
        <div class="result-card">
            <span class="risk-badge {RISK_CLASS[risk]}">{RISK_EMOJI[risk]} {risk} Risk</span>
            <div class="metric-row">
                <span class="metric-label">Pneumonia-associated probability</span>
                <span class="metric-value">{result['probability']:.1%}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Model confidence</span>
                <span class="metric-value">{result['confidence_level']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.caption(result['confidence_note'])
        st.markdown(f"**📋 Recommended next step:** {RECOMMENDED_ACTION[risk]}")
        st.info(result["explanation"])

        st.markdown("###  Grad-CAM Explanation")
        st.caption("Warmer colors (red/yellow) indicate regions the model focused on most.")
        overlay_response = requests.get(f"{API_URL}{result['overlay_url']}")
        with open("temp_overlay.png", "wb") as f:
            f.write(overlay_response.content)
        st.image("temp_overlay.png", use_container_width=True)

        report_response = requests.get(f"{API_URL}/report/{result['file_id']}")
        if report_response.status_code == 200:
            st.download_button(
                label="📄 Download Full Report (PDF)",
                data=report_response.content,
                file_name="xray_screening_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )

# --- Comparison section ---
with st.expander("🔄 Compare Two X-Rays"):
    st.write("Upload two images (e.g., before/after treatment) to compare risk scores side by side.")

    col1, col2 = st.columns(2)
    with col1:
        file_a = st.file_uploader("Image A", type=["jpg", "jpeg", "png"], key="compare_a")
        if file_a:
            st.image(file_a, caption="Image A", use_container_width=True)
    with col2:
        file_b = st.file_uploader("Image B", type=["jpg", "jpeg", "png"], key="compare_b")
        if file_b:
            st.image(file_b, caption="Image B", use_container_width=True)

    if file_a and file_b and st.button("⚖️ Compare", use_container_width=True):
        with st.spinner("Analyzing both images..."):
            resp_a = requests.post(f"{API_URL}/predict", files={"file": (file_a.name, file_a.getvalue())})
            resp_b = requests.post(f"{API_URL}/predict", files={"file": (file_b.name, file_b.getvalue())})

            if resp_a.status_code == 200 and resp_b.status_code == 200:
                result_a, result_b = resp_a.json(), resp_b.json()

                colA, colB = st.columns(2)
                with colA:
                    st.metric("Risk Level (A)", result_a["risk_level"], f"{result_a['probability']:.1%}")
                    overlay_a = requests.get(f"{API_URL}{result_a['overlay_url']}")
                    with open("temp_overlay_a.png", "wb") as f:
                        f.write(overlay_a.content)
                    st.image("temp_overlay_a.png", use_container_width=True)

                with colB:
                    st.metric("Risk Level (B)", result_b["risk_level"], f"{result_b['probability']:.1%}")
                    overlay_b = requests.get(f"{API_URL}{result_b['overlay_url']}")
                    with open("temp_overlay_b.png", "wb") as f:
                        f.write(overlay_b.content)
                    st.image("temp_overlay_b.png", use_container_width=True)

                delta = result_b['probability'] - result_a['probability']
                if abs(delta) > 0.1:
                    direction = "increased" if delta > 0 else "decreased"
                    st.info(f"Pneumonia-associated probability {direction} by {abs(delta):.1%} between the two images.")
                else:
                    st.info("No significant change detected between the two images.")
            else:
                st.error("One or both analyses failed. Please check both images.")

render_footer_disclaimer()