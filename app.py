import io
import pickle
from datetime import datetime

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sounddevice as sd
import streamlit as st
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer
from scipy.io.wavfile import write


FEATURE_LABELS = ["Jitter", "Shimmer", "PPE"]
FEATURE_COLUMNS = ["MDVP:Jitter(%)", "MDVP:Shimmer", "PPE"]


@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)

    return model, scaler


@st.cache_data
def load_reference_data():
    data = pd.read_csv("data.csv")
    if "name" in data.columns:
        data = data.drop(columns=["name"])
    data["NDI"] = (
        data["MDVP:Jitter(%)"] + data["MDVP:Shimmer"] + data["PPE"]
    ) / 3
    return data


def calculate_ndi(jitter, shimmer, ppe):
    return (jitter + shimmer + ppe) / 3


def classify_ndi(ndi_value):
    if ndi_value < 0.12:
        return "Normal"
    if ndi_value < 0.16:
        return "Mild"
    return "Severe"


def risk_label(probability, prediction):
    if prediction == 0:
        return "Low", "No Parkinson pattern detected"
    if probability >= 80:
        return "High", "Parkinson pattern detected"
    if probability >= 55:
        return "Moderate", "Parkinson pattern detected"
    return "Review", "Borderline Parkinson pattern"


def predict_condition(model, scaler, jitter, shimmer, ppe):
    raw_features = pd.DataFrame([[jitter, shimmer, ppe]], columns=FEATURE_COLUMNS)
    scaled_features = scaler.transform(raw_features)
    prediction = int(model.predict(scaled_features)[0])
    probabilities = model.predict_proba(scaled_features)[0]
    confidence = float(probabilities[prediction] * 100)
    ndi_value = calculate_ndi(jitter, shimmer, ppe)
    ndi_condition = classify_ndi(ndi_value)
    risk, summary = risk_label(confidence, prediction)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "ndi": ndi_value,
        "ndi_condition": ndi_condition,
        "risk": risk,
        "summary": summary,
    }


def apply_chart_theme(ax):
    ax.set_facecolor("#ffffff")
    ax.figure.set_facecolor("#ffffff")
    ax.grid(axis="y", color="#e6eaf0", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d8dee8")
    ax.spines["bottom"].set_color("#d8dee8")
    ax.tick_params(colors="#475467", labelsize=9)
    ax.title.set_color("#172033")
    ax.yaxis.label.set_color("#475467")


def create_feature_chart(values, reference_data):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    x = np.arange(len(FEATURE_LABELS))
    reference = [
        reference_data["MDVP:Jitter(%)"].median(),
        reference_data["MDVP:Shimmer"].median(),
        reference_data["PPE"].median(),
    ]

    ax.bar(x - 0.18, reference, width=0.34, label="Dataset median", color="#d8dee8")
    ax.bar(x + 0.18, values, width=0.34, label="Current sample", color="#147d73")
    ax.set_xticks(x)
    ax.set_xticklabels(FEATURE_LABELS)
    ax.set_title("Feature comparison")
    ax.set_ylabel("Value")
    ax.legend(frameon=False, loc="upper left")
    apply_chart_theme(ax)
    fig.tight_layout()
    return fig


def create_wave_chart(audio):
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.plot(audio[:3000], color="#2952cc", linewidth=1)
    ax.set_title("Recorded voice waveform")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    apply_chart_theme(ax)
    fig.tight_layout()
    return fig


def create_pdf(name, report_date, result, values, fig):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    content = [
        Paragraph("Neural Degradation Screening Report", styles["Title"]),
        Spacer(1, 10),
        Paragraph(f"Patient: {name or 'Not provided'}", styles["Normal"]),
        Paragraph(f"Date: {report_date}", styles["Normal"]),
        Paragraph(f"Screening result: {result['summary']}", styles["Normal"]),
        Paragraph(f"Risk level: {result['risk']}", styles["Normal"]),
        Paragraph(f"Model confidence: {result['confidence']:.2f}%", styles["Normal"]),
        Paragraph(f"NDI: {result['ndi']:.4f} ({result['ndi_condition']})", styles["Normal"]),
        Spacer(1, 8),
        Paragraph(
            f"Features: Jitter={values[0]:.5f}, Shimmer={values[1]:.5f}, PPE={values[2]:.5f}",
            styles["Normal"],
        ),
        Spacer(1, 16),
    ]

    image_buffer = io.BytesIO()
    fig.savefig(image_buffer, format="png", bbox_inches="tight")
    image_buffer.seek(0)
    content.append(Image(image_buffer, width=380, height=190))
    content.append(Spacer(1, 14))
    content.append(
        Paragraph(
            "This AI result is decision support only and is not a clinical diagnosis.",
            styles["Italic"],
        )
    )

    doc.build(content)
    buffer.seek(0)
    return buffer.getvalue()


def inject_styles():
    st.markdown(
        """
        <style>
        :root {
            --bg: #f4f6f9;
            --surface: #ffffff;
            --ink: #172033;
            --muted: #667085;
            --quiet: #98a2b3;
            --line: #d8dee8;
            --nav: #101828;
            --teal: #147d73;
            --blue: #2952cc;
            --amber: #b45309;
            --red: #b42318;
            --green: #087443;
        }

        [data-testid="stHeader"] {
            background: rgba(244, 246, 249, 0.86);
            backdrop-filter: blur(8px);
        }

        .stApp {
            background: var(--bg);
            color: var(--ink);
        }

        .block-container {
            max-width: 1240px;
            padding: 1.3rem 2rem 3rem;
        }

        [data-testid="stSidebar"] {
            background: var(--nav);
            border-right: 1px solid #1d2939;
        }

        [data-testid="stSidebar"] * {
            color: #f9fafb;
        }

        [data-testid="stSidebar"] [data-testid="stMetricValue"] {
            font-size: 1.25rem;
        }

        .app-header {
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            gap: 1rem;
            border-bottom: 1px solid var(--line);
            padding-bottom: 1rem;
            margin-bottom: 1.1rem;
        }

        .eyebrow {
            color: var(--teal);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .app-header h1 {
            font-size: clamp(1.9rem, 3vw, 3rem);
            line-height: 1.05;
            letter-spacing: 0;
            margin: 0;
        }

        .app-header p {
            color: var(--muted);
            font-size: 0.98rem;
            max-width: 760px;
            margin: 0.55rem 0 0;
        }

        .system-pill {
            align-items: center;
            background: #ecfdf3;
            border: 1px solid #abefc6;
            border-radius: 999px;
            color: #067647;
            display: inline-flex;
            font-size: 0.82rem;
            font-weight: 800;
            gap: 0.45rem;
            padding: 0.45rem 0.75rem;
            white-space: nowrap;
        }

        .status-dot {
            background: #17b26a;
            border-radius: 999px;
            display: inline-block;
            height: 0.55rem;
            width: 0.55rem;
        }

        .workspace-grid {
            display: grid;
            grid-template-columns: 1fr 320px;
            gap: 1rem;
            margin-bottom: 1rem;
        }

        .panel {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem;
        }

        .panel-title {
            color: var(--ink);
            font-size: 1rem;
            font-weight: 800;
            line-height: 1.25;
            margin: 0 0 0.25rem;
        }

        .panel-copy {
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.45;
            margin: 0;
        }

        .data-strip {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin-top: 0.9rem;
        }

        .strip-item {
            background: #f8fafc;
            border: 1px solid #e6eaf0;
            border-radius: 8px;
            padding: 0.75rem;
        }

        .strip-item span {
            color: var(--muted);
            display: block;
            font-size: 0.74rem;
            margin-bottom: 0.28rem;
        }

        .strip-item strong {
            color: var(--ink);
            display: block;
            font-size: 1rem;
        }

        .metric-row {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 1rem 0;
        }

        .metric-card {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            min-height: 104px;
            padding: 0.9rem;
        }

        .metric-card span {
            color: var(--muted);
            display: block;
            font-size: 0.78rem;
            margin-bottom: 0.45rem;
        }

        .metric-card strong {
            color: var(--ink);
            display: block;
            font-size: 1.25rem;
            line-height: 1.12;
        }

        .risk-low {
            border-top: 4px solid var(--green);
        }

        .risk-review,
        .risk-moderate {
            border-top: 4px solid var(--amber);
        }

        .risk-high {
            border-top: 4px solid var(--red);
        }

        .notice {
            background: #fffbeb;
            border: 1px solid #fedf89;
            border-radius: 8px;
            color: #854a0e;
            font-size: 0.9rem;
            line-height: 1.45;
            padding: 0.85rem 1rem;
        }

        .result-summary {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            margin-top: 0.9rem;
            padding: 1rem;
        }

        .result-summary strong {
            color: var(--ink);
        }

        .result-summary p {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.45;
            margin: 0.35rem 0 0;
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 6px;
            min-height: 2.75rem;
            font-weight: 800;
        }

        .stTabs [data-baseweb="tab-list"] {
            background: #e9edf3;
            border-radius: 8px;
            gap: 0.25rem;
            padding: 0.25rem;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 6px;
            padding: 0.5rem 0.9rem;
        }

        div[data-testid="stForm"] {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem;
        }

        div[data-testid="stAlert"] {
            border-radius: 8px;
        }

        @media (max-width: 920px) {
            .workspace-grid {
                grid-template-columns: 1fr;
            }

            .metric-row {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }

            .app-header {
                align-items: flex-start;
                flex-direction: column;
            }
        }

        @media (max-width: 560px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .data-strip,
            .metric-row {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    st.markdown(
        """
        <header class="app-header">
            <div>
                <div class="eyebrow">Clinical decision support workspace</div>
                <h1>Neural Degradation Screening</h1>
                <p>
                    Capture voice-derived features, compare them against the reference dataset,
                    and export a concise screening report for follow-up review.
                </p>
            </div>
            <div class="system-pill"><span class="status-dot"></span> Model ready</div>
        </header>
        """,
        unsafe_allow_html=True,
    )


def render_context_panel(reference_data):
    total_records = len(reference_data)
    positive_rate = reference_data["status"].mean() * 100
    median_ndi = reference_data["NDI"].median()

    st.markdown(
        f"""
        <div class="workspace-grid">
            <section class="panel">
                <div class="panel-title">Assessment setup</div>
                <p class="panel-copy">
                    This workspace uses the project model artifacts already stored in this folder.
                    Inputs are limited to the same three features used during training so the
                    frontend stays aligned with the pipeline.
                </p>
                <div class="data-strip">
                    <div class="strip-item">
                        <span>Reference records</span>
                        <strong>{total_records}</strong>
                    </div>
                    <div class="strip-item">
                        <span>Positive class share</span>
                        <strong>{positive_rate:.1f}%</strong>
                    </div>
                    <div class="strip-item">
                        <span>Median NDI</span>
                        <strong>{median_ndi:.4f}</strong>
                    </div>
                </div>
            </section>
            <aside class="panel">
                <div class="panel-title">Responsible use</div>
                <p class="panel-copy">
                    Results are screening support, not diagnosis. Use the output to guide
                    discussion, repeat assessment, or clinical referral when appropriate.
                </p>
            </aside>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(result):
    risk_class = result["risk"].lower()
    st.markdown(
        f"""
        <div class="metric-row">
            <div class="metric-card risk-{risk_class}">
                <span>Screening result</span>
                <strong>{result['summary']}</strong>
            </div>
            <div class="metric-card risk-{risk_class}">
                <span>Risk level</span>
                <strong>{result['risk']}</strong>
            </div>
            <div class="metric-card">
                <span>Model confidence</span>
                <strong>{result['confidence']:.1f}%</strong>
            </div>
            <div class="metric-card">
                <span>NDI score</span>
                <strong>{result['ndi']:.4f}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result(name, report_date, result, values, chart):
    render_metrics(result)

    if result["prediction"] == 0:
        st.success("No Parkinson-like pattern was detected in this sample.")
    else:
        st.error("A Parkinson-like pattern was detected. This should be reviewed by a qualified clinician.")

    st.markdown(
        f"""
        <section class="result-summary">
            <div class="panel-title">Review notes</div>
            <p>
                <strong>NDI category:</strong> {result['ndi_condition']}.
                The submitted features were scaled with the saved project scaler before prediction.
                Confidence reflects the model class probability for this specific sample.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.pyplot(chart, use_container_width=True)

    graph_buffer = io.BytesIO()
    chart.savefig(graph_buffer, format="png", bbox_inches="tight")
    graph_buffer.seek(0)

    pdf = create_pdf(name, report_date, result, values, chart)

    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            "Download chart",
            graph_buffer,
            file_name="feature_comparison.png",
            mime="image/png",
            use_container_width=True,
        )
    with col_b:
        st.download_button(
            "Download report",
            pdf,
            file_name="screening_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


def patient_details():
    with st.form("patient_details"):
        st.markdown('<div class="panel-title">Patient details</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns([2, 1])
        with col_a:
            name = st.text_input("Patient name", placeholder="Enter patient name")
        with col_b:
            report_date = st.date_input("Screening date", datetime.today())
        submitted = st.form_submit_button("Save details", use_container_width=True)

    if submitted:
        st.toast("Patient details updated.")

    return name, report_date


def manual_screening(model, scaler, reference_data, name, report_date):
    with st.form("manual_screening"):
        st.markdown('<div class="panel-title">Manual feature entry</div>', unsafe_allow_html=True)
        st.caption("Enter values for the exact three features used by the trained model.")

        col_1, col_2, col_3 = st.columns(3)
        with col_1:
            jitter = st.number_input(
                "Jitter (%)",
                min_value=0.0,
                max_value=0.10,
                value=0.006,
                step=0.001,
                format="%.5f",
            )
        with col_2:
            shimmer = st.number_input(
                "Shimmer",
                min_value=0.0,
                max_value=0.20,
                value=0.030,
                step=0.001,
                format="%.5f",
            )
        with col_3:
            ppe = st.number_input(
                "PPE",
                min_value=0.0,
                max_value=0.60,
                value=0.120,
                step=0.005,
                format="%.5f",
            )

        run = st.form_submit_button("Run screening", type="primary", use_container_width=True)

    if run:
        values = [jitter, shimmer, ppe]
        result = predict_condition(model, scaler, jitter, shimmer, ppe)
        chart = create_feature_chart(values, reference_data)
        render_result(name, report_date, result, values, chart)


def voice_screening(model, scaler, reference_data, name, report_date):
    with st.form("voice_screening"):
        st.markdown('<div class="panel-title">Voice capture</div>', unsafe_allow_html=True)
        st.caption("Records from the active microphone and extracts approximate feature values.")
        duration = st.slider("Recording duration", min_value=3, max_value=8, value=5, step=1)
        run = st.form_submit_button("Record and analyze", type="primary", use_container_width=True)

    if run:
        with st.spinner("Recording audio and extracting features..."):
            sample_rate = 44100
            audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
            sd.wait()
            write("voice.wav", sample_rate, audio)

            y, _ = librosa.load("voice.wav", sr=sample_rate)
            jitter = float(np.mean(np.abs(np.diff(y))))
            shimmer = float(np.mean(librosa.feature.rms(y=y)))
            ppe = float(np.var(librosa.feature.spectral_flatness(y=y)))

        values = [jitter, shimmer, ppe]
        result = predict_condition(model, scaler, jitter, shimmer, ppe)
        feature_chart = create_feature_chart(values, reference_data)
        wave_chart = create_wave_chart(y)

        render_result(name, report_date, result, values, feature_chart)
        st.pyplot(wave_chart, use_container_width=True)


def about_page(reference_data):
    st.markdown('<section class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Project overview</div>', unsafe_allow_html=True)
    st.write(
        "This application is a focused frontend for the existing Parkinson screening "
        "pipeline. It loads the saved XGBoost model and StandardScaler, accepts the "
        "three trained features, and produces a documented screening result."
    )
    st.dataframe(
        reference_data[FEATURE_COLUMNS + ["NDI", "status"]].describe().round(5),
        use_container_width=True,
    )
    st.markdown("</section>", unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="Neural Degradation Screening",
        page_icon="ND",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()

    model, scaler = load_artifacts()
    reference_data = load_reference_data()

    with st.sidebar:
        st.title("ND Screening")
        page = st.radio("Navigation", ["Screening", "Reference"], label_visibility="collapsed")
        st.divider()
        st.metric("Model inputs", "3")
        st.metric("Dataset rows", f"{len(reference_data)}")
        st.metric("Report export", "PDF / PNG")

    render_header()

    if page == "Reference":
        about_page(reference_data)
        return

    render_context_panel(reference_data)
    name, report_date = patient_details()

    st.markdown(
        """
        <div class="notice">
            This tool supports screening workflow only. Do not use it as the sole basis
            for diagnosis, medication decisions, or emergency assessment.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    manual_tab, voice_tab = st.tabs(["Manual input", "Voice capture"])
    with manual_tab:
        manual_screening(model, scaler, reference_data, name, report_date)
    with voice_tab:
        voice_screening(model, scaler, reference_data, name, report_date)


if __name__ == "__main__":
    main()
