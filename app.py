import streamlit as st
import numpy as np
from datetime import datetime
import io
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import pickle

# FIXED PDF IMPORT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))

# ---------------- PAGE ----------------
st.set_page_config(page_title="Parkinson Detection", layout="centered")

# ---------------- STYLE ----------------
st.markdown("""<style>
[data-testid="stHeader"] {display: none;}
.block-container {padding-top: 0rem; margin-top: 0rem;}
.stApp {
    background: linear-gradient(rgba(13,71,161,0.5), rgba(13,71,161,0.5)),
    url("https://images.unsplash.com/photo-1586773860418-d37222d8fce3");
    background-size: cover;
    background-position: center;
}
.title-box {
    background: white;
    padding: 15px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    color: #0d47a1;
}
.stButton>button, .stDownloadButton>button {
    background-color: #ff5722 !important;
    color: white !important;
    font-weight: bold;
    border-radius: 10px;
}
h1,h2,h3,label,p {color: black !important;}
.footer {text-align: right; font-size: 13px; margin-top: 20px; color: black;}
</style>""", unsafe_allow_html=True)

# ---------------- SAFE PDF ----------------
def create_pdf(name, date, result, fig):
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer)
        styles = getSampleStyleSheet()

        content = []
        content.append(Paragraph("Parkinson Detection Report", styles['Title']))
        content.append(Spacer(1, 20))
        content.append(Paragraph(f"Name: {name}", styles['Normal']))
        content.append(Paragraph(f"Date: {str(date)}", styles['Normal']))
        content.append(Paragraph(f"Result: {result}", styles['Normal']))
        content.append(Spacer(1, 20))

        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', bbox_inches='tight')
        img_buffer.seek(0)

        content.append(Image(img_buffer, width=250, height=120))

        doc.build(content)
        buffer.seek(0)
        return buffer.getvalue()

    except Exception as e:
        return None  # prevents crash

# ---------------- UI ----------------
st.markdown('<div class="title-box">🧠 Parkinson Detection System</div>', unsafe_allow_html=True)

name = st.text_input("Patient Name")
date = st.date_input("Date", datetime.today())

option = st.radio("Choose Input Method", ["Manual Input", "Voice Input"])

# ---------------- MANUAL ----------------
if option == "Manual Input":

    jitter = st.number_input("Jitter", 0.0, 0.05, 0.005)
    shimmer = st.number_input("Shimmer", 0.0, 0.1, 0.03)
    ppe = st.number_input("PPE", 0.0, 0.5, 0.1)

    if st.button("🔍 Predict"):

        if name.strip() == "":
            st.warning("Enter patient name")
        else:
            prediction = model.predict([[jitter, shimmer, ppe]])[0]
            result = "No Parkinson" if prediction == 0 else "Parkinson Detected"

            fig, ax = plt.subplots(figsize=(2.5,1.2))
            ax.bar(["J","S","P"], [jitter, shimmer, ppe])

            st.session_state["result"] = result
            st.session_state["fig"] = fig
            st.session_state["pdf"] = create_pdf(name, date, result, fig)

# SHOW RESULT
if "result" in st.session_state:
    st.success("Prediction Complete")
    st.write("Result:", st.session_state["result"])
    st.pyplot(st.session_state["fig"])

    if st.session_state["pdf"] is not None:
        st.download_button(
            "⬇ Download Report",
            st.session_state["pdf"],
            file_name="report.pdf",
            mime="application/pdf"
        )
    else:
        st.warning("PDF generation failed")

# ---------------- VOICE ----------------
elif option == "Voice Input":

    st.write("🎤 Click to record 5 seconds")

    if st.button("🎙 Record"):

        fs = 44100
        audio = sd.rec(int(5 * fs), samplerate=fs, channels=1)
        sd.wait()

        write("voice.wav", fs, audio)

        y, sr = librosa.load("voice.wav")

        jitter = np.std(y)
        shimmer = np.mean(np.abs(y))
        ppe = np.var(y)

        prediction = model.predict([[jitter, shimmer, ppe]])[0]
        result = "No Parkinson" if prediction == 0 else "Parkinson Detected"

        fig, ax = plt.subplots(figsize=(2.5,1.2))
        ax.plot(y[:1000])

        st.session_state["result_voice"] = result
        st.session_state["fig_voice"] = fig
        st.session_state["pdf_voice"] = create_pdf(name, date, result, fig)

if "result_voice" in st.session_state:
    st.success("Voice Analysis Done")
    st.write("Result:", st.session_state["result_voice"])
    st.pyplot(st.session_state["fig_voice"])

    if st.session_state["pdf_voice"] is not None:
        st.download_button(
            "⬇ Download Voice Report",
            st.session_state["pdf_voice"],
            file_name="voice_report.pdf",
            mime="application/pdf"
        )
    else:
        st.warning("PDF generation failed")

# ---------------- FOOTER ----------------
st.markdown('<div class="footer">Developed by Nimalrajasegar</div>', unsafe_allow_html=True)
