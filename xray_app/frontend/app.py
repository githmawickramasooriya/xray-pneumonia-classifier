import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

RECOMMENDED_ACTION = {
    "Low": "No immediate concern flagged. Continue routine care and consult a doctor if symptoms develop.",
    "Medium": "Consider scheduling a check-up with a doctor for further evaluation.",
    "High": "We recommend consulting a doctor or radiologist promptly for a proper evaluation.",
}


def render_footer_disclaimer():
    st.divider()
    st.markdown(
        """
        <div style="text-align:center; color:#888; font-size:0.85rem; padding: 1rem 0;">
        🏥 This tool is a research/awareness prototype built for Data Odyssey 2026.<br>
        It is <b>not</b> a certified medical device and must not be used for actual diagnosis
        or treatment decisions.<br>
        If you are experiencing symptoms, please seek care from a licensed healthcare provider.
        </div>
        """,
        unsafe_allow_html=True
    )


# --- Mandatory acknowledgment gate ---
if "disclaimer_accepted" not in st.session_state:
    st.session_state.disclaimer_accepted = False

if not st.session_state.disclaimer_accepted:
    st.title("Chest X-Ray Pneumonia Screening Aid")
    st.info(
        "Before continuing, please confirm you understand this tool provides "
        "**risk awareness only**, not a medical diagnosis."
    )
    if st.button("I understand — Continue"):
        st.session_state.disclaimer_accepted = True
        st.rerun()
    st.stop()

# --- Main app (only reached after acknowledgment) ---
st.title("Chest X-Ray Pneumonia Screening Aid")
st.write(
    "Upload a chest X-ray image to see what patterns an AI model associates with pneumonia. "
    "This tool is for educational/screening purposes only and is not a substitute for "
    "professional medical diagnosis."
)

uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded X-Ray", width=300)

    if st.button("Analyze"):
        with st.spinner("Analyzing..."):
            response = requests.post(
                f"{API_URL}/predict",
                files={"file": (uploaded_file.name, uploaded_file.getvalue())}
            )

            if response.status_code != 200:
                error_msg = response.json().get("error", "Unknown error occurred.")
                st.error(f"⚠️ {error_msg}")
                st.info("Try uploading a clear, in-focus chest X-ray image in JPG or PNG format.")
                st.stop()

            result = response.json()
            st.subheader(f"Risk Level: {result['risk_level']}")
            st.write(f"Probability of pneumonia-associated patterns: {result['probability']:.1%}")
            st.markdown(f"**Recommended next step:** {RECOMMENDED_ACTION[result['risk_level']]}")
            st.info(result["explanation"])

            overlay_response = requests.get(f"{API_URL}{result['overlay_url']}")
            with open("temp_overlay.png", "wb") as f:
                f.write(overlay_response.content)
            st.image("temp_overlay.png", caption="Grad-CAM Explanation", width=400)

render_footer_disclaimer()