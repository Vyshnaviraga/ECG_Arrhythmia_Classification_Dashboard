import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tensorflow as tf
import os
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ECG Arrhythmia Classifier",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Theme State
# ─────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

D = st.session_state.dark_mode

# ─────────────────────────────────────────────
# Theme Tokens — explicit per-mode, no ambiguity
# ─────────────────────────────────────────────
if D:
    T = {
        "bg":           "#0d1117",
        "surface":      "#161b22",
        "surface2":     "#1c2128",
        "border":       "#30363d",
        "text":         "#E6EDF3",
        "text2":        "#B8C1CC",
        "muted":        "#8B949E",
        "accent":       "#58A6FF",
        "grid":         "#21262d",
        "input_bg":     "#0d1117",
        "input_text":   "#E6EDF3",
        "plot_bg":      "rgba(0,0,0,0)",
        "table_bg":     "#161b22",
        "table_text":   "#E6EDF3",
        "table_border": "#30363d",
        "label_color":  "#8B949E",
        # Clinical box
        "cbox_bg":      "#161b22",
        "cbox_border":  "#30363d",
        "cbox_title":   "#58A6FF",
        "cbox_text":    "#B8C1CC",
        # Bar chart inactive
        "bar_inactive": "#21262d",
        # Risk
        "risk_low_bg":   "#0d1f0d", "risk_low_bd":  "#238636", "risk_low_txt":  "#22C55E",
        "risk_med_bg":   "#1c1700", "risk_med_bd":  "#9e6a03", "risk_med_txt":  "#EAB308",
        "risk_hi_bg":    "#1f0d0d", "risk_hi_bd":   "#da3633", "risk_hi_txt":   "#EF4444",
    }
else:
    T = {
        "bg":           "#F8FAFC",
        "surface":      "#FFFFFF",
        "surface2":     "#F1F5F9",
        "border":       "#D1D5DB",
        "text":         "#000000",
        "text2":        "#111827",
        "muted":        "#374151",
        "accent":       "#2563EB",
        "grid":         "#E5E7EB",
        "input_bg":     "#FFFFFF",
        "input_text":   "#111827",
        "plot_bg":      "rgba(0,0,0,0)",
        "table_bg":     "#FFFFFF",
        "table_text":   "#111827",
        "table_border": "#D1D5DB",
        "label_color":  "#374151",
        # Clinical box
        "cbox_bg":      "#FFFFFF",
        "cbox_border":  "#D1D5DB",
        "cbox_title":   "#2563EB",
        "cbox_text":    "#374151",
        # Bar chart inactive
        "bar_inactive": "#E5E7EB",
        # Risk
        "risk_low_bg":   "#DCFCE7", "risk_low_bd":  "#16A34A", "risk_low_txt":  "#15803D",
        "risk_med_bg":   "#FEF9C3", "risk_med_bd":  "#CA8A04", "risk_med_txt":  "#854D0E",
        "risk_hi_bg":    "#FEE2E2", "risk_hi_bd":   "#DC2626", "risk_hi_txt":   "#991B1B",
    }

# ─────────────────────────────────────────────
# Class Colors — identical in both themes, high contrast
# ─────────────────────────────────────────────
CLASS_COLOR = {
    0: "#22C55E",
    1: "#EAB308",
    2: "#EF4444",
    3: "#F97316",
    4: "#A1A1AA",
}
CLASS_FILL = {
    0: "rgba(34,197,94,0.10)",
    1: "rgba(234,179,8,0.10)",
    2: "rgba(239,68,68,0.10)",
    3: "rgba(249,115,22,0.10)",
    4: "rgba(161,161,170,0.10)",
}

# ─────────────────────────────────────────────
# CSS — full specificity overrides, no bleed
# ─────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}
.stApp {{
    background-color: {T["bg"]} !important;
    color: {T["text"]} !important;
}}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background-color: {T["surface"]} !important;
    border-right: 1px solid {T["border"]} !important;
}}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] label {{
    color: {T["text"]} !important;
}}

/* ── Selectbox Customizations (Ref 1) ── */
div[data-baseweb="select"] > div {{
    background-color: {T["input_bg"]} !important;
    border-color: {T["border"]} !important;
    min-height: 42px !important; /* Slightly increased height */
}}
div[data-baseweb="select"] span,
div[data-baseweb="select"] div {{
    color: {T["input_text"]} !important;
    font-size: 14px !important; /* Increased dropdown font size */
    font-weight: 600 !important; /* Bolder selected text */
}}
/* Dropdown menu list */
ul[data-baseweb="menu"],
li[data-baseweb="menu-item"] {{
    background-color: {T["input_bg"]} !important;
    color: {T["input_text"]} !important;
    font-size: 14px !important;
}}
li[data-baseweb="menu-item"]:hover {{
    background-color: {T["surface2"]} !important;
}}

/* ── Slider ── */
.stSlider label,
.stSlider span {{
    color: {T["text2"]} !important;
}}
div[data-testid="stSlider"] p {{
    color: {T["text2"]} !important;
}}

/* ── Buttons ── */
.stButton > button {{
    background-color: {T["surface2"]} !important;
    border: 1px solid {T["border"]} !important;
    color: {T["text"]} !important;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 500;
}}
.stButton > button:hover {{
    background-color: {T["border"]} !important;
    color: {T["text"]} !important;
}}

/* ── Dataframe / Table ── */
.stDataFrame,
.stDataFrame table,
.stDataFrame thead,
.stDataFrame tbody,
.stDataFrame tr,
.stDataFrame th,
.stDataFrame td {{
    background-color: {T["table_bg"]} !important;
    color: {T["table_text"]} !important;
    border-color: {T["table_border"]} !important;
}}
.stDataFrame th {{
    font-weight: 700 !important;
    color: {T["muted"]} !important;
    background-color: {T["surface2"]} !important;
}}
/* iframe-based dataframe fallback */
iframe {{
    background-color: {T["table_bg"]} !important;
}}

/* ── General text overrides (main area) ── */
.main p, .main span, .main div, .main label {{
    color: {T["text"]};
}}

/* ── Cards ── */
.metric-card {{
    background: {T["surface"]};
    border: 1px solid {T["border"]};
    border-radius: 10px;
    padding: 16px 18px;
    text-align: center;
}}
.metric-label {{
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {T["muted"]};
    margin-bottom: 6px;
}}
.metric-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 22px;
    font-weight: 700;
    color: {T["text"]};
}}
.metric-sub {{
    font-size: 12px;
    color: {T["text2"]};
    margin-top: 4px;
}}

/* ── Risk badges ── */
.risk-low {{
    background: {T["risk_low_bg"]};
    border: 1px solid {T["risk_low_bd"]};
    color: {T["risk_low_txt"]};
    border-radius: 8px; padding: 14px 20px;
    font-size: 15px; font-weight: 600; text-align: center;
}}
.risk-medium {{
    background: {T["risk_med_bg"]};
    border: 1px solid {T["risk_med_bd"]};
    color: {T["risk_med_txt"]};
    border-radius: 8px; padding: 14px 20px;
    font-size: 15px; font-weight: 600; text-align: center;
}}
.risk-high {{
    background: {T["risk_hi_bg"]};
    border: 1px solid {T["risk_hi_bd"]};
    color: {T["risk_hi_txt"]};
    border-radius: 8px; padding: 14px 20px;
    font-size: 15px; font-weight: 600; text-align: center;
}}

/* ── Section headers ── */
.section-header {{
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {T["text"]};
    border-bottom: 1px solid {T["border"]};
    padding-bottom: 8px;
    margin-bottom: 16px;
    margin-top: 4px;
}}

/* ── Clinical boxes ── */
.clinical-box {{
    background: {T["cbox_bg"]};
    border: 1px solid {T["cbox_border"]};
    border-left: 3px solid {T["accent"]};
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 12px;
}}
.clinical-title {{
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {T["cbox_title"] if "cbox_title" in T else T["accent"]};
    margin-bottom: 7px;
}}
.clinical-text {{
    font-size: 13px;
    color: {T["cbox_text"]};
    line-height: 1.65;
}}

/* ── Info rows (sidebar) ── */
.info-row {{
    display: flex;
    justify-content: space-between;
    padding: 7px 0;
    border-bottom: 1px solid {T["border"]};
    font-size: 13px;
}}
.info-key {{ color: {T["muted"]}; }}
.info-val {{
    color: {T["text"]};
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
}}

/* ── Sidebar labels (Ref 5) ── */
.sidebar-label {{
    font-size: 13px; /* Slightly higher section heading size */
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {T["text"]};
    margin: 18px 0 10px 0; /* Improved spacing consistency */
}}

/* ── Hero ── */
.hero-title {{
    font-size: 28px;
    font-weight: 700;
    color: {T["text"]};
    letter-spacing: -0.3px;
    line-height: 1.2;
}}
.hero-badge {{
    display: inline-block;
    background: {T["surface2"]};
    border: 1px solid {T["border"]};
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 12px;
    font-weight: 600;
    color: {T["text2"]};
    margin: 8px 4px 0 0;
}}

/* ── Disclaimer (Ref 2) ── */
.disclaimer {{
    background: {T["surface2"]};
    border: 1px solid {T["border"]};
    border-radius: 8px;
    padding: 16px 20px; /* Increased padding */
    font-size: 14px;    /* Increased text font-size */
    color: {T["text2"]};
}}

/* ── Explainability checkmarks ── */
.explain-item {{
    padding: 5px 0;
    font-size: 13px;
    color: {T["cbox_text"]};
}}

#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
.stDeployButton {{display: none;}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
CLASS_NAMES = {
    0: "Normal Beat",
    1: "Supraventricular Beat",
    2: "Ventricular Beat",
    3: "Fusion Beat",
    4: "Unknown Beat"
}
CLASS_SHORT = {0: "Normal", 1: "Supra", 2: "Ventricular", 3: "Fusion", 4: "Unknown"}

RISK_MAP = {
    0: ("🟢 Low Risk",    "risk-low"),
    1: ("🟡 Medium Risk", "risk-medium"),
    2: ("🔴 High Risk",   "risk-high"),
    3: ("🟡 Medium Risk", "risk-medium"),
    4: ("🟡 Medium Risk", "risk-medium"),
}

CLINICAL_DATA = {
    0: {
        "condition":      "Normal Sinus Rhythm",
        "symptoms":       "No symptoms expected. Patient is likely asymptomatic.",
        "interpretation": "This heartbeat follows a normal sinus rhythm. Waveform morphology is consistent with healthy electrical conduction originating from the sinoatrial node. P wave, QRS complex, and T wave are within expected ranges.",
        "action":         "No immediate action required. Continue routine cardiac monitoring as per standard health checkups.",
    },
    1: {
        "condition":      "Supraventricular Ectopic Beat",
        "symptoms":       "Palpitations, fluttering sensation in chest, occasional dizziness or lightheadedness.",
        "interpretation": "This heartbeat originates above the ventricles — likely from the atria or AV node — outside the normal sinoatrial conduction pathway. Isolated occurrences are often benign, but frequent episodes may indicate atrial fibrillation or flutter.",
        "action":         "Monitor frequency. If recurrent, refer for Holter monitoring and cardiology consultation. Avoid caffeine and stimulants.",
    },
    2: {
        "condition":      "Ventricular Ectopic Beat",
        "symptoms":       "Skipped heartbeat sensation, chest discomfort, shortness of breath, fatigue, or near-syncope.",
        "interpretation": "This beat originates from within the ventricles rather than through the normal conduction system. Wide QRS morphology is characteristic of ventricular ectopy. Frequent occurrences may indicate structural heart disease or electrolyte imbalance.",
        "action":         "Urgent cardiology review recommended. Investigate for coronary artery disease, electrolyte abnormalities, or cardiomyopathy. Avoid strenuous activity until assessed.",
    },
    3: {
        "condition":      "Fusion Beat",
        "symptoms":       "May be asymptomatic or present with mild palpitations and irregular heartbeat sensation.",
        "interpretation": "This beat exhibits characteristics of both a normal sinus beat and a ventricular ectopic beat occurring simultaneously. Fusion beats indicate competing pacemaker activity and are relatively rare.",
        "action":         "Clinical correlation required. Refer for extended ECG monitoring. Fusion beats in isolation are often benign but may signal ventricular tachycardia risk if frequent.",
    },
    4: {
        "condition":      "Unclassified Beat",
        "symptoms":       "Variable — may include palpitations, chest pain, or no symptoms depending on underlying cause.",
        "interpretation": "This heartbeat does not conform to standard arrhythmia morphology patterns. The atypical waveform may result from signal noise, acquisition artifact, or a rare arrhythmia not well-represented in training data.",
        "action":         "Manual review by a qualified cardiologist strongly recommended. Do not make clinical decisions based on this classification alone.",
    },
}

# ─────────────────────────────────────────────
# Load Resources
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("ecg_cnn_model.h5")

@st.cache_data
def load_data():
    return pd.read_csv("mitbih_test.csv", header=None)

def predict(signal, model):
    arr = np.array(signal).reshape(1, 187, 1)
    probs = model.predict(arr, verbose=0)[0]
    pred_class = int(np.argmax(probs))
    confidence = min(float(probs[pred_class]) * 100, 99.5)
    return pred_class, confidence, probs

try:
    cnn_model = load_model()
except Exception:
    st.error("Model file `ecg_cnn_model.h5` not found in project root.")
    st.stop()

try:
    df = load_data()
except Exception:
    st.error("Test data `mitbih_test.csv` not found in project root.")
    st.stop()

# ─────────────────────────────────────────────
# Sidebar (Ref 1 & 5)
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"<div style='font-size:15px; font-weight:700; color:{T['text']}; padding:4px 0 2px 0;'>"
        "ECG Arrhythmia Classifier</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div style='font-size:12px; color:{T['muted']}; margin-bottom:16px;'>"
        "MIT-BIH · 1D CNN · v1.0</div>",
        unsafe_allow_html=True
    )

    # ── Theme Toggle ──
    toggle_label = "☀️  Light Mode" if D else "🌙  Dark Mode"
    toggle_hint  = f"<div style='font-size:11px; color:{T['muted']}; margin-bottom:10px;'>Currently: {'🌙 Dark' if D else '☀️ Light'}</div>"
    st.markdown(toggle_hint, unsafe_allow_html=True)
    if st.button(toggle_label, use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

    st.markdown("<div style='margin: 8px 0;'>---</div>", unsafe_allow_html=True)

    # ── Class Filter (Moved Upward - Ref 1) ──
    st.markdown("<div class='sidebar-label'>Class Filter</div>", unsafe_allow_html=True)
    class_options = ["Any"] + [CLASS_NAMES[i] for i in range(5)]
    selected_class_label = st.selectbox("Class", class_options, label_visibility="collapsed")
    selected_class = None if selected_class_label == "Any" else class_options.index(selected_class_label) - 1

    st.markdown("<div style='margin: 8px 0;'>---</div>", unsafe_allow_html=True)

    # ── Model Information ──
    st.markdown("<div class='sidebar-label'>Model Information</div>", unsafe_allow_html=True)
    for k, v in {
        "Model":    "1D CNN",
        "Accuracy": "98.15%",
        "Macro F1": "0.91",
        "Dataset":  "MIT-BIH",
        "Samples":  "109,446",
        "SMOTE":    "Applied",
        "Classes":  "5",
    }.items():
        st.markdown(
            f"<div class='info-row'>"
            f"<span class='info-key'>{k}</span>"
            f"<span class='info-val'>{v}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

# ─────────────────────────────────────────────
# Build Filtered DataFrame
# ─────────────────────────────────────────────
if selected_class is not None:
    current_df = df[df.iloc[:, 187] == selected_class].reset_index(drop=True)
else:
    current_df = df.reset_index(drop=True)

if len(current_df) == 0:
    st.warning("No samples found for selected class.")
    st.stop()

# ─────────────────────────────────────────────
# Hero Header
# ─────────────────────────────────────────────
st.markdown(f"""
<div style='padding: 24px 0 8px 0;'>
    <div class='hero-title'>🫀 ECG-Based Cardiac Arrhythmia Detection System</div>
    <div style='margin-top:10px;'>
        <span class='hero-badge'>MIT-BIH Dataset</span>
        <span class='hero-badge'>1D CNN + SMOTE</span>
        <span class='hero-badge'>98.15% Test Accuracy</span>
        <span class='hero-badge'>109,446 Samples</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Sample Slider
# ─────────────────────────────────────────────
max_idx     = max(0, len(current_df) - 1)
default_idx = min(st.session_state.get("selected_idx", 0), max_idx)
selected_idx = st.slider(
    f"Sample index  ·  {len(current_df):,} samples available",
    0, max_idx, default_idx
)

row        = current_df.iloc[selected_idx]
signal     = row.iloc[:187].values.astype(float)
true_label = int(row.iloc[187]) if len(row) > 187 else None

pred_class, confidence, probs = predict(signal, cnn_model)
sig_color         = CLASS_COLOR[pred_class]
sig_fill          = CLASS_FILL[pred_class]
risk_label, risk_css = RISK_MAP[pred_class]
cd                = CLINICAL_DATA[pred_class]

# ─────────────────────────────────────────────
# ECG Waveform
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("<div class='section-header'>ECG Signal Waveform</div>", unsafe_allow_html=True)

fig_ecg = go.Figure()
fig_ecg.add_trace(go.Scatter(
    x=list(range(187)),
    y=signal,
    mode='lines',
    line=dict(color=sig_color, width=2),
    fill='tozeroy',
    fillcolor=sig_fill,
    hovertemplate="Step: %{x}<br>Amplitude: %{y:.4f}<extra></extra>"
))
fig_ecg.update_layout(
    paper_bgcolor=T["plot_bg"],
    plot_bgcolor=T["plot_bg"],
    height=210,
    margin=dict(l=0, r=0, t=8, b=0),
    xaxis=dict(
        title="Time Step (0–186)",
        color=T["muted"],
        gridcolor=T["grid"],
        title_font=dict(size=11),
        tickfont=dict(color=T["muted"])
    ),
    yaxis=dict(
        title="Amplitude",
        color=T["muted"],
        gridcolor=T["grid"],
        title_font=dict(size=11),
        tickfont=dict(color=T["muted"])
    ),
    showlegend=False,
    font=dict(family='Inter', color=T["muted"])
)
st.plotly_chart(fig_ecg, use_container_width=True)

# ─────────────────────────────────────────────
# Signal Features
# ─────────────────────────────────────────────
st.markdown("<div class='section-header'>ECG Signal Features</div>", unsafe_allow_html=True)

peak_amp  = float(np.max(np.abs(signal)))
mean_amp  = float(np.mean(signal))
variance  = float(np.var(signal))
sig_range = float(np.max(signal) - np.min(signal))
energy    = float(np.sum(signal ** 2))

f1, f2, f3, f4, f5 = st.columns(5)
for col, lbl, val, fmt in [
    (f1, "Peak Amplitude",  peak_amp,  ".4f"),
    (f2, "Mean Amplitude",  mean_amp,  ".4f"),
    (f3, "Signal Variance", variance,  ".4f"),
    (f4, "Signal Range",    sig_range, ".4f"),
    (f5, "Signal Energy",   energy,    ".2f"),
]:
    col.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>{lbl}</div>
        <div class='metric-value' style='font-size:17px;'>{val:{fmt}}</div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Prediction Cards
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("<div class='section-header'>Prediction</div>", unsafe_allow_html=True)

p1, p2, p3 = st.columns(3)
with p1:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>Predicted Class</div>
        <div class='metric-value' style='font-size:16px; color:{sig_color};'>{CLASS_NAMES[pred_class]}</div>
        <div class='metric-sub'>Class {pred_class}</div>
    </div>""", unsafe_allow_html=True)

with p2:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>Confidence</div>
        <div class='metric-value'>{confidence:.1f}%</div>
        <div class='metric-sub'>Model certainty</div>
    </div>""", unsafe_allow_html=True)

with p3:
    if true_label is not None:
        match      = "Correct" if true_label == pred_class else "Incorrect"
        match_col  = "#22C55E" if true_label == pred_class else "#EF4444"
        match_icon = "✓" if true_label == pred_class else "✗"
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>True Label</div>
            <div class='metric-value' style='font-size:16px; color:{CLASS_COLOR[true_label]};'>{CLASS_NAMES[true_label]}</div>
            <div class='metric-sub' style='color:{match_col};'>{match_icon} {match}</div>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Clinical Analysis
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("<div class='section-header'>Clinical Analysis</div>", unsafe_allow_html=True)

ca1, ca2 = st.columns(2)

with ca1:
    st.markdown(f"""<div class='clinical-box'>
        <div class='clinical-title'>Predicted Condition</div>
        <div class='clinical-text' style='font-size:15px; font-weight:600; color:{T["text"]};'>{cd["condition"]}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""<div class='clinical-box'>
        <div class='clinical-title'>Possible Symptoms</div>
        <div class='clinical-text'>{cd["symptoms"]}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"<div class='{risk_css}'>{risk_label}</div>", unsafe_allow_html=True)

with ca2:
    st.markdown(f"""<div class='clinical-box'>
        <div class='clinical-title'>Clinical Interpretation</div>
        <div class='clinical-text'>{cd["interpretation"]}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""<div class='clinical-box'>
        <div class='clinical-title'>Recommended Action</div>
        <div class='clinical-text'>{cd["action"]}</div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Class Probabilities
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("<div class='section-header'>Class Probabilities</div>", unsafe_allow_html=True)

fig_bar = go.Figure(go.Bar(
    x=[CLASS_SHORT[i] for i in range(5)],
    y=[p * 100 for p in probs],
    marker_color=[
        CLASS_COLOR[i] if i == pred_class else T["bar_inactive"]
        for i in range(5)
    ],
    marker_line_width=0,
    text=[f"{p*100:.1f}%" for p in probs],
    textposition='outside',
    textfont=dict(color=T["muted"], size=11),
    hovertemplate="%{x}: %{y:.2f}%<extra></extra>"
))
fig_bar.update_layout(
    paper_bgcolor=T["plot_bg"],
    plot_bgcolor=T["plot_bg"],
    height=200,
    margin=dict(l=0, r=0, t=28, b=0),
    yaxis=dict(
        title="Probability (%)",
        color=T["muted"],
        gridcolor=T["grid"],
        range=[0, max(probs) * 138],
        title_font=dict(size=11),
        tickfont=dict(color=T["muted"])
    ),
    xaxis=dict(
        color=T["muted"],
        tickfont=dict(color=T["muted"])
    ),
    font=dict(family='Inter', color=T["muted"]),
    showlegend=False
)
st.plotly_chart(fig_bar, use_container_width=True)

# ─────────────────────────────────────────────
# Explainability
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("<div class='section-header'>Signal Explainability</div>", unsafe_allow_html=True)

chars = []
chars.append("High waveform variation detected" if variance > 0.05 else "Low waveform variation — stable signal")
if peak_amp > 0.8:
    chars.append("Elevated peak amplitude — strong QRS complex present")
elif peak_amp < 0.2:
    chars.append("Low peak amplitude — weak QRS signal")
else:
    chars.append("Normal peak amplitude range observed")
if sig_range > 1.0:
    chars.append("Wide signal range — high morphological spread across beat")
if energy > 5.0:
    chars.append("High signal energy — significant electrical activity detected")
if pred_class != 0:
    chars.append("Abnormal waveform morphology detected")
    chars.append(f"Pattern characteristics consistent with {CLASS_NAMES[pred_class]}")
else:
    chars.append("Waveform morphology consistent with normal sinus rhythm")
if confidence < 80:
    chars.append("Low model confidence — borderline case, manual review recommended")
elif confidence > 95:
    chars.append("High model confidence — strong pattern match with predicted class")

items_html = "".join([
    f"<div class='explain-item'>&#10003;&nbsp;&nbsp;{c}</div>"
    for c in chars
])
st.markdown(f"""<div class='clinical-box'>
    <div class='clinical-title'>Top Signal Characteristics</div>
    {items_html}
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Model Comparison
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("<div class='section-header'>Model Comparison</div>", unsafe_allow_html=True)

model_df = pd.DataFrame({
    "Model":    ["Random Forest", "LSTM", "1D CNN"],
    "Accuracy": ["98.03%", "88.34%", "98.15%"],
    "Macro F1": ["0.90",   "—",      "0.91"],
    "Role":     ["Classical ML baseline", "Sequence model", "Selected — best performance"],
})
st.dataframe(model_df, use_container_width=True, hide_index=True)
st.markdown(
    f"<div style='font-size:12px; color:{T['muted']}; margin-top:6px;'>"
    "1D CNN selected as final model. LSTM underperformed because ECG classification depends on "
    "local waveform morphology, not long-range temporal dependencies.</div>",
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────
# Live Confusion Matrix & Performance Metrics (Ref 3 & 4)
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div class='section-header'>Live Confusion Matrix & Model Performance — 1D CNN</div>",
    unsafe_allow_html=True
)

try:
    X_test = df.iloc[:, :187].values
    y_true = df.iloc[:, 187].values.astype(int)
    X_test = X_test.reshape(-1, 187, 1)

    y_pred_probs = cnn_model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    accuracy = accuracy_score(y_true, y_pred)

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Model Accuracy", f"{accuracy * 100:.2f}%")
    with c2:
        st.metric("Test Samples", f"{len(y_true):,}")

    # Centering container for the compact confusion matrix
    cm_left, cm_center, cm_right = st.columns([1, 2, 1])
    
    with cm_center:
        cm = confusion_matrix(y_true, y_pred)
        
        # Reduced size by ~25% using tight layout parameters (Ref 3)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        
        # Explicit facecolors to avoid box borders bleeding in dark mode
        fig.patch.set_facecolor(T["bg"])
        ax.set_facecolor(T["surface"])

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=list(CLASS_SHORT.values()),
            yticklabels=list(CLASS_SHORT.values()),
            ax=ax,
            cbar=False, # Removed structural colorbar clutter to make it compact
            annot_kws={"size": 10}
        )

        ax.set_xlabel("Predicted Class", color=T["text"], labelpad=8, fontsize=11)
        ax.set_ylabel("Actual Class", color=T["text"], labelpad=8, fontsize=11)
        ax.set_title("CNN Confusion Matrix", color=T["text"], pad=10, fontsize=12, fontweight='bold')
        
        ax.tick_params(colors=T["muted"], labelsize=9)
        plt.tight_layout()
        st.pyplot(fig)

    # Classification Report aligned directly underneath (Ref 4)
    st.markdown(
        "<div class='section-header' style='margin-top: 12px; margin-bottom: 12px;'>Classification Report</div>",
        unsafe_allow_html=True
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=list(CLASS_NAMES.values()),
        output_dict=True
    )
    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )

except Exception as e:
    st.error(f"Unable to generate confusion matrix: {e}")

# ─────────────────────────────────────────────
# Disclaimer (Ref 2)
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""<div class='disclaimer'>
    <div style='font-size: 15px; font-weight: 700; color: {T["text"]}; margin-bottom: 6px;'>⚠️ Clinical Disclaimer</div>
    This tool is for educational and research purposes only and is not a substitute for
    professional medical diagnosis. Always consult a qualified cardiologist for clinical decisions.
    Model predictions may be incorrect.
</div>""", unsafe_allow_html=True)