# ============================================================
#  TelecoAfrica Intelligence Platform
#  Customer Churn Analytics & Prediction Engine
#  Author  : ANOH AMON FRANCKLIN HEMERSON
#  School  : INSSEDS — Institut Supérieur de Statistique,
#             d'Économie et de Développement Social
#  Sector  : Télécommunications — Analyse Prédictive du Churn
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False

try:
    from fpdf import FPDF
    FPDF_OK = True
except ImportError:
    FPDF_OK = False

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="TelecoAfrica Intelligence Platform",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens (Orange Telecom corporate) ─────────────────
ORANGE  = "#FF6200"
ORANGE2 = "#FF8C00"
DARK    = "#1A1A1A"
DARK2   = "#242424"
DARK3   = "#2E2E2E"
LIGHT   = "#F5F5F5"
GRAY    = "#8A8A8A"
WHITE   = "#FFFFFF"
GREEN   = "#00C48C"
RED     = "#E53935"
BLUE    = "#1976D2"

_PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#E0E0E0", size=12),
    title=dict(font=dict(size=14, color=WHITE)),
    xaxis=dict(gridcolor="#333", linecolor="#444", tickcolor="#555"),
    yaxis=dict(gridcolor="#333", linecolor="#444", tickcolor="#555"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#CCC")),
    margin=dict(l=10, r=10, t=44, b=10),
)

def plotly_layout(**overrides):
    """Retourne un dict de layout fusionnant _PLOTLY_BASE et les overrides.
    Evite l'erreur 'multiple values for keyword argument' quand une clé
    (ex: margin, xaxis, legend...) est présente à la fois dans _PLOTLY_BASE
    et dans les arguments de update_layout()."""
    return {**_PLOTLY_BASE, **overrides}

# ── CSS ───────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

*, html, body {{ font-family: 'DM Sans', sans-serif; }}

.stApp {{ background: {DARK}; color: #E0E0E0; }}
section[data-testid="stSidebar"] {{
    background: {DARK2} !important;
    border-right: 1px solid #333 !important;
}}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding: 0 2rem 2rem 2rem !important; max-width: 100% !important; }}

/* ─── Topbar ─────────────────────────────────────────────── */
.topbar {{
    background: {DARK2};
    border-bottom: 3px solid {ORANGE};
    padding: 0.9rem 2.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: -1rem -2rem 2rem -2rem;
}}
.topbar-logo {{ font-size: 1.5rem; font-weight: 700; color: {ORANGE}; letter-spacing: -0.5px; }}
.topbar-logo span {{ color: {WHITE}; }}
.topbar-tagline {{
    font-size: 0.74rem; color: {GRAY};
    border-left: 2px solid #444; padding-left: 1rem; line-height: 1.5;
}}
.topbar-right {{ text-align: right; font-size: 0.72rem; color: {GRAY}; line-height: 1.7; }}
.topbar-right b {{ color: {ORANGE}; font-weight: 600; }}
.topbar-school {{
    background: {ORANGE}; color: {WHITE}; font-size: 0.62rem;
    font-weight: 700; letter-spacing: 1.2px; text-transform: uppercase;
    padding: 0.18rem 0.7rem; border-radius: 4px; display: inline-block; margin-top: 0.2rem;
}}

/* ─── Section titles ─────────────────────────────────────── */
.section-title {{
    font-size: 0.68rem; font-weight: 700; letter-spacing: 2px;
    text-transform: uppercase; color: {ORANGE}; margin-bottom: 1rem;
    display: flex; align-items: center; gap: 0.5rem;
}}
.section-title::after {{
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, #333, transparent);
}}

/* ─── KPI Cards ──────────────────────────────────────────── */
.kpi-card {{
    background: {DARK2}; border: 1px solid #333; border-top: 3px solid {ORANGE};
    border-radius: 8px; padding: 1.2rem 1.4rem; position: relative; overflow: hidden;
}}
.kpi-card::before {{
    content: ''; position: absolute; top: 0; right: 0;
    width: 60px; height: 60px;
    background: radial-gradient(circle at top right, rgba(255,98,0,0.08), transparent);
}}
.kpi-icon {{ font-size: 1.1rem; margin-bottom: 0.5rem; opacity: 0.8; }}
.kpi-value {{ font-size: 1.9rem; font-weight: 700; color: {WHITE}; line-height: 1; }}
.kpi-label {{ font-size: 0.72rem; color: {GRAY}; text-transform: uppercase; letter-spacing: 0.8px; margin-top: 0.35rem; }}
.kpi-delta {{ font-size: 0.71rem; margin-top: 0.4rem; }}
.kpi-delta.up {{ color: {RED}; }}
.kpi-delta.down {{ color: {GREEN}; }}

/* ─── Panels ─────────────────────────────────────────────── */
.panel {{ background: {DARK2}; border: 1px solid #333; border-radius: 10px; padding: 1.5rem; margin-bottom: 1.2rem; }}

/* ─── Forms ──────────────────────────────────────────────── */
.stSelectbox label, .stSlider label, .stNumberInput label, .stRadio label {{
    color: #CCC !important; font-size: 0.82rem !important; font-weight: 500 !important;
}}

/* ─── Button ─────────────────────────────────────────────── */
.stButton > button {{
    background: {ORANGE} !important; color: {WHITE} !important;
    border: none !important; border-radius: 7px !important;
    font-weight: 700 !important; font-size: 0.95rem !important;
    padding: 0.8rem 2rem !important; width: 100% !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 20px rgba(255,98,0,0.25) !important;
}}
.stButton > button:hover {{
    background: {ORANGE2} !important; transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(255,98,0,0.4) !important;
}}

/* ─── Verdict ────────────────────────────────────────────── */
.verdict-churn {{
    background: linear-gradient(135deg, rgba(229,57,53,0.12), rgba(229,57,53,0.04));
    border: 1px solid rgba(229,57,53,0.35); border-left: 4px solid {RED};
    border-radius: 10px; padding: 1.8rem 2rem; text-align: center;
}}
.verdict-safe {{
    background: linear-gradient(135deg, rgba(0,196,140,0.12), rgba(0,196,140,0.04));
    border: 1px solid rgba(0,196,140,0.35); border-left: 4px solid {GREEN};
    border-radius: 10px; padding: 1.8rem 2rem; text-align: center;
}}
.verdict-pct {{ font-size: 3rem; font-weight: 800; line-height: 1; }}
.verdict-label {{ font-size: 1.3rem; font-weight: 700; color: {WHITE}; margin-bottom: 0.3rem; }}
.verdict-sub {{ font-size: 0.82rem; color: {GRAY}; margin-bottom: 1rem; }}

/* ─── Risk bar ───────────────────────────────────────────── */
.risk-bar-wrap {{
    background: #333; border-radius: 50px; height: 10px; overflow: hidden; margin: 0.5rem 0;
}}
.risk-bar-fill {{ height: 100%; border-radius: 50px; background: linear-gradient(90deg, {ORANGE}, {RED}); }}
.risk-bar-safe {{ height: 100%; border-radius: 50px; background: linear-gradient(90deg, #00966D, {GREEN}); }}
.risk-labels {{ display: flex; justify-content: space-between; font-size: 0.67rem; color: #555; margin-top: 0.2rem; }}

/* ─── Insights ───────────────────────────────────────────── */
.insight-row {{
    display: flex; align-items: flex-start; gap: 0.75rem;
    padding: 0.65rem 0; border-bottom: 1px solid #2E2E2E; font-size: 0.81rem;
}}
.insight-row:last-child {{ border-bottom: none; }}
.insight-dot {{ width: 8px; height: 8px; border-radius: 50%; margin-top: 4px; flex-shrink: 0; }}
.insight-text {{ color: #CCC; line-height: 1.55; }}
.insight-text b {{ color: {WHITE}; }}

/* ─── Data rows ──────────────────────────────────────────── */
.data-row {{
    display: flex; justify-content: space-between; padding: 0.5rem 0;
    border-bottom: 1px solid #2A2A2A; font-size: 0.82rem;
}}
.data-row:last-child {{ border-bottom: none; }}
.data-key {{ color: {GRAY}; }}
.data-val {{ color: {WHITE}; font-weight: 500; font-family: 'DM Mono', monospace; }}

/* ─── Tabs ───────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    background: {DARK2} !important; border-bottom: 2px solid #333 !important;
    gap: 0 !important; padding: 0 !important;
}}
.stTabs [data-baseweb="tab"] {{
    color: {GRAY} !important; font-weight: 500 !important; font-size: 0.88rem !important;
    padding: 0.8rem 1.5rem !important; border-bottom: 3px solid transparent !important;
    border-radius: 0 !important; margin-bottom: -2px !important;
}}
.stTabs [aria-selected="true"] {{
    color: {ORANGE} !important; border-bottom: 3px solid {ORANGE} !important;
    background: rgba(255,98,0,0.05) !important;
}}

/* ─── Sidebar ────────────────────────────────────────────── */
.sb-logo {{
    text-align: center; padding: 1.4rem 0 1rem 0;
    border-bottom: 1px solid #333; margin-bottom: 1rem;
}}
.sb-metric {{
    background: {DARK3}; border: 1px solid #383838; border-radius: 7px;
    padding: 0.7rem 1rem; margin-bottom: 0.45rem;
    display: flex; justify-content: space-between; align-items: center;
}}
.sb-metric .lbl {{ font-size: 0.75rem; color: #BBBBBB; font-weight:500; }}
.sb-metric .val {{ font-size: 0.92rem; font-weight: 700; color: {ORANGE}; font-family: 'DM Mono'; }}
.sb-section {{
    font-size: 0.7rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase;
    color: #AAAAAA; margin: 1.1rem 0 0.55rem 0;
}}

/* ─── Banners ────────────────────────────────────────────── */
.banner {{ border-radius: 7px; padding: 0.85rem 1.1rem; font-size: 0.81rem; line-height: 1.7; margin-top: 0.8rem; }}
.banner-red {{ background: rgba(229,57,53,0.1); border-left: 3px solid {RED}; color: #EF9A9A; }}
.banner-green {{ background: rgba(0,196,140,0.1); border-left: 3px solid {GREEN}; color: #80CBC4; }}
.banner b {{ color: {WHITE}; }}

/* ─── Footer ─────────────────────────────────────────────── */
.footer {{
    border-top: 1px solid #2A2A2A; margin-top: 3rem; padding-top: 1.2rem;
    display: flex; justify-content: space-between; align-items: center;
    font-size: 0.71rem; color: #444;
}}
.footer b {{ color: {ORANGE}; }}

/* ─── Expanders — toujours visibles ─────────────────────── */
div[data-testid="stExpander"] {{
    background: {DARK2} !important;
    border: 1px solid #3A3A3A !important;
    border-radius: 8px !important;
    margin-bottom: 0.6rem !important;
}}
div[data-testid="stExpander"] summary {{
    background: {DARK2} !important;
    color: {WHITE} !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    padding: 0.75rem 1rem !important;
    border-radius: 8px !important;
    list-style: none !important;
    cursor: pointer !important;
    border-bottom: 1px solid #333 !important;
}}
div[data-testid="stExpander"] summary:hover {{
    background: #2A2A2A !important;
    color: {ORANGE} !important;
}}
div[data-testid="stExpander"] summary svg {{
    fill: {ORANGE} !important;
    stroke: {ORANGE} !important;
}}
div[data-testid="stExpander"] > div:last-child {{
    padding: 1rem !important;
    background: {DARK2} !important;
    border-radius: 0 0 8px 8px !important;
}}
/* Forcer le texte des labels à l'intérieur des expanders */
div[data-testid="stExpander"] label,
div[data-testid="stExpander"] .stSelectbox label,
div[data-testid="stExpander"] .stSlider label,
div[data-testid="stExpander"] p {{
    color: #DDDDDD !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
}}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# CHARGEMENT ARTEFACTS & DONNÉES
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

@st.cache_resource
def load_artifacts():
    with open(BASE_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(BASE_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(BASE_DIR / "feature_cols.json", "r") as f:
        feature_cols = json.load(f)
    return model, scaler, feature_cols

@st.cache_data
def load_data():
    df = pd.read_csv(BASE_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df["ChurnBin"] = (df["Churn"] == "Yes").astype(int)
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=[0, 12, 24, 48, 72],
        labels=["0–12 mois", "13–24 mois", "25–48 mois", "49–72 mois"]
    )
    return df

# ── Chargement sécurisé ──────────────────────────────────────
try:
    model, scaler, feature_cols = load_artifacts()
except FileNotFoundError as e:
    st.markdown(f"""
    <div style="background:rgba(229,57,53,0.1); border:1px solid rgba(229,57,53,0.4);
                border-left:4px solid #E53935; border-radius:10px;
                padding:1.8rem 2rem; margin:2rem 0; text-align:center">
      <div style="font-size:2.5rem; margin-bottom:0.8rem">⚠️</div>
      <div style="font-size:1.1rem; font-weight:700; color:#EF9A9A; margin-bottom:0.5rem">
        Fichiers modèle introuvables
      </div>
      <div style="font-size:0.85rem; color:#888; line-height:1.8">
        Les fichiers <code style="color:#FF6200">model.pkl</code>,
        <code style="color:#FF6200">scaler.pkl</code> et
        <code style="color:#FF6200">feature_cols.json</code>
        doivent être dans le même répertoire que <code>app.py</code>.<br>
        <b style="color:#EF9A9A">Erreur :</b> {e}
      </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

try:
    df = load_data()
except FileNotFoundError as e:
    st.markdown(f"""
    <div style="background:rgba(229,57,53,0.1); border:1px solid rgba(229,57,53,0.4);
                border-left:4px solid #E53935; border-radius:10px;
                padding:1.8rem 2rem; margin:2rem 0; text-align:center">
      <div style="font-size:2.5rem; margin-bottom:0.8rem">📂</div>
      <div style="font-size:1.1rem; font-weight:700; color:#EF9A9A; margin-bottom:0.5rem">
        Dataset introuvable
      </div>
      <div style="font-size:0.85rem; color:#888; line-height:1.8">
        Le fichier <code style="color:#FF6200">WA_Fn-UseC_-Telco-Customer-Churn.csv</code>
        est manquant.<br>
        <b style="color:#EF9A9A">Erreur :</b> {e}
      </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ── Initialisation session_state (historique prédictions) ────
if "historique" not in st.session_state:
    st.session_state.historique = []

# Stats globales
total            = len(df)
churned          = int(df["ChurnBin"].sum())
churn_rate       = churned / total * 100
avg_mc_churn     = df[df["Churn"] == "Yes"]["MonthlyCharges"].mean()
avg_tenure_churn = df[df["Churn"] == "Yes"]["tenure"].mean()


# ─────────────────────────────────────────────────────────────
# GÉNÉRATION RAPPORT PDF
# ─────────────────────────────────────────────────────────────
def generate_pdf_report(inp, pct, pred, shap_names=None, shap_values=None):
    """Génère un rapport PDF complet pour un client analysé."""
    from datetime import datetime
    import io

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── Entête ──────────────────────────────────────────────────
    pdf.set_fill_color(26, 26, 26)       # DARK #1A1A1A
    pdf.rect(0, 0, 210, 40, style="F")

    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(255, 98, 0)       # ORANGE
    pdf.set_xy(10, 8)
    pdf.cell(0, 8, "TelecoAfrica Intelligence Platform", ln=True)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(200, 200, 200)
    pdf.set_xy(10, 18)
    pdf.cell(0, 6, "Prediction du Churn Client - Secteur Telecommunications", ln=True)

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(150, 150, 150)
    pdf.set_xy(10, 26)
    pdf.cell(0, 5,
        f"ANOH AMON FRANCKLIN HEMERSON  |  Master Data Science  |  INSEEDS  |  "
        f"Supervise par : MR Akposso Didier Martial",
        ln=True
    )

    pdf.set_xy(10, 33)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 5,
        f"Rapport genere le {datetime.now().strftime('%d/%m/%Y a %H:%M:%S')}",
        ln=True
    )

    pdf.set_y(45)

    # ── Verdict principal ────────────────────────────────────────
    if pred == 1:
        pdf.set_fill_color(60, 20, 20)
        verdict_txt  = "CLIENT A RISQUE DE DEPART"
        verdict_sub  = f"Probabilite de churn : {pct}%"
        score_color  = (229, 57, 53)
    else:
        pdf.set_fill_color(15, 40, 30)
        verdict_txt  = "CLIENT STABLE ET FIDELE"
        verdict_sub  = f"Probabilite de fidelite : {round(100 - pct, 1)}%"
        score_color  = (0, 196, 140)

    pdf.set_fill_color(35, 35, 35)
    pdf.rect(10, pdf.get_y(), 190, 28, style="F")

    pdf.set_xy(10, pdf.get_y() + 4)
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(*score_color)
    pdf.cell(0, 8, verdict_txt, ln=True, align="C")

    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(*score_color)
    pdf.cell(0, 10, f"{pct}%", ln=True, align="C")

    pdf.set_y(pdf.get_y() + 6)

    # ── Section : Profil client ──────────────────────────────────
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(255, 98, 0)
    pdf.cell(0, 8, "PROFIL DU CLIENT ANALYSE", ln=True)
    pdf.set_draw_color(255, 98, 0)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.set_y(pdf.get_y() + 3)

    fields = [
        ("Genre",               inp.get("gender", "-")),
        ("Senior (>=65 ans)",   inp.get("senior", "-")),
        ("Anciennete",          f"{inp.get('tenure', 0)} mois"),
        ("Charges mensuelles",  f"${inp.get('MonthlyCharges', 0):.2f}"),
        ("Type de contrat",     inp.get("contract", "-")),
        ("Mode de paiement",    inp.get("payment", "-")),
        ("Service Internet",    inp.get("internet", "-")),
        ("Facturation demat.",  inp.get("paperless", "-")),
        ("Partenaire",          inp.get("partner", "-")),
        ("Personnes a charge",  inp.get("dependents", "-")),
    ]

    pdf.set_font("Helvetica", "", 10)
    for i, (label, value) in enumerate(fields):
        if i % 2 == 0:
            pdf.set_fill_color(38, 38, 38)
        else:
            pdf.set_fill_color(30, 30, 30)
        pdf.set_fill_color(38, 38, 38) if i % 2 == 0 else pdf.set_fill_color(30, 30, 30)
        y_pos = pdf.get_y()
        pdf.rect(10, y_pos, 190, 7, style="F")
        pdf.set_text_color(150, 150, 150)
        pdf.set_xy(13, y_pos + 1)
        pdf.cell(80, 5, label, ln=False)
        pdf.set_text_color(220, 220, 220)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 5, str(value), ln=True)
        pdf.set_font("Helvetica", "", 10)

    pdf.set_y(pdf.get_y() + 5)

    # ── Section : Facteurs SHAP ──────────────────────────────────
    if shap_names and shap_values:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(255, 98, 0)
        pdf.cell(0, 8, "FACTEURS CLES DE LA DECISION (IA EXPLICABLE)", ln=True)
        pdf.set_draw_color(255, 98, 0)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.set_y(pdf.get_y() + 3)

        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(130, 130, 130)
        pdf.cell(0, 5,
            "Valeurs SHAP : positif = augmente le risque de churn | negatif = protege la fidelite",
            ln=True
        )
        pdf.set_y(pdf.get_y() + 2)

        for i, (name, val) in enumerate(zip(reversed(shap_names), reversed(shap_values))):
            is_risk = val > 0
            y_pos = pdf.get_y()
            pdf.set_fill_color(38, 38, 38)
            pdf.rect(10, y_pos, 190, 7, style="F")

            # Barre de progression proportionnelle
            max_abs = max(abs(v) for v in shap_values) or 1
            bar_w = int(abs(val) / max_abs * 60)
            if is_risk:
                pdf.set_fill_color(229, 57, 53)
            else:
                pdf.set_fill_color(0, 196, 140)
            pdf.rect(120, y_pos + 1.5, bar_w, 4, style="F")

            pdf.set_text_color(200, 200, 200)
            pdf.set_font("Helvetica", "", 9)
            pdf.set_xy(13, y_pos + 1)
            pdf.cell(105, 5, name[:45], ln=False)

            color = (229, 57, 53) if is_risk else (0, 196, 140)
            pdf.set_text_color(*color)
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_xy(183, y_pos + 1)
            pdf.cell(0, 5, f"{val:+.3f}", ln=True)
            pdf.set_font("Helvetica", "", 9)

        pdf.set_y(pdf.get_y() + 5)

    # ── Section : Recommandations ────────────────────────────────
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(255, 98, 0)
    pdf.cell(0, 8, "RECOMMANDATIONS BUSINESS", ln=True)
    pdf.set_draw_color(255, 98, 0)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.set_y(pdf.get_y() + 3)

    if pred == 1:
        recommandations = [
            "Contacter le client dans les 48h via son canal prefere",
            "Proposer une migration vers contrat 1 an avec remise de 15%",
            "Si charges > 80$ : audit tarifaire + bundle personnalise",
            "Si anciennete < 6 mois : programme d'onboarding dedie",
        ]
    else:
        recommandations = [
            "Proposer des services complementaires (streaming, securite)",
            "Encourager le parrainage avec programme de recompenses",
            "Suggerer le passage au contrat 2 ans si encore mensuel",
        ]

    pdf.set_font("Helvetica", "", 10)
    for rec in recommandations:
        pdf.set_text_color(200, 200, 200)
        pdf.set_xy(13, pdf.get_y())
        pdf.cell(5, 6, "->", ln=False)
        pdf.set_text_color(220, 220, 220)
        pdf.cell(0, 6, rec, ln=True)

    # ── Pied de page ─────────────────────────────────────────────
    pdf.set_y(-20)
    pdf.set_draw_color(255, 98, 0)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.set_y(pdf.get_y() + 2)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5,
        "TelecoAfrica Intelligence Platform  |  INSEEDS  |  Dataset IBM Telco Customer Churn (Kaggle)  |  Logistic Regression - Accuracy 82.2%",
        align="C"
    )

    # Retourner les bytes du PDF
    return bytes(pdf.output())


# ─────────────────────────────────────────────────────────────
# TOPBAR
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="topbar">
  <div style="display:flex; align-items:center; gap:1.4rem">
    <div style="font-size:2.2rem; line-height:1">📡</div>
    <div>
      <div style="
        font-size:1.45rem; font-weight:800; line-height:1.2;
        background: linear-gradient(90deg, {ORANGE}, {ORANGE2});
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        letter-spacing:-0.3px;
      ">
        Prédiction du Churn Client
      </div>
      <div style="
        font-size:1.05rem; font-weight:600; color:{WHITE};
        letter-spacing:0.2px; margin-top:0.15rem;
      ">
        dans le secteur des Télécommunications
      </div>
    </div>
  </div>
  <div class="topbar-right">
    <b style="font-size:0.85rem; color:{WHITE}; letter-spacing:0.3px">ANOH AMON FRANCKLIN HEMERSON</b><br>
    <span style="font-size:0.75rem; color:#BBB">🎓 Master Data Science · Customer Churn Prediction Project</span><br>
    <span class="topbar-school">INSEEDS</span>
    <span style="font-size:0.7rem; color:#999; margin-left:0.4rem">· Supervisé par : <b style="color:{ORANGE}">MR Akposso Didier Martial</b></span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div class="sb-logo">
      <div style="font-size:2.2rem; margin-bottom:0.6rem">📊</div>
      <div style="
        background: linear-gradient(135deg, #242424, #1e1e1e);
        border: 1px solid {ORANGE};
        border-radius: 10px;
        padding: 1rem 1.1rem;
        margin: 0 0.2rem;
        text-align: center;
      ">
        <div style="font-weight:800; font-size:0.95rem; color:{WHITE}; letter-spacing:0.4px; line-height:1.5">
          ANOH AMON<br>FRANCKLIN HEMERSON
        </div>
        <div style="margin: 0.55rem 0 0.3rem 0; height:1px; background:linear-gradient(90deg, transparent, {ORANGE}, transparent)"></div>
        <div style="font-size:0.78rem; color:#CCC; font-weight:500; margin-top:0.45rem">
          🎓 Master Data Science
        </div>
        <div style="margin-top:0.4rem">
          <span style="background:{ORANGE}; color:{WHITE}; font-size:0.65rem; font-weight:700;
                       letter-spacing:1.2px; text-transform:uppercase; padding:0.22rem 0.8rem;
                       border-radius:4px; display:inline-block">
            INSEEDS
          </span>
        </div>
        <div style="font-size:0.74rem; color:#AAA; margin-top:0.55rem; line-height:1.6">
          Supervisé par :<br>
          <b style="color:{ORANGE}; font-size:0.78rem">MR Akposso Didier Martial</b>
        </div>
        <div style="font-size:0.68rem; color:#666; margin-top:0.45rem; font-style:italic">
          Customer Churn Prediction Project
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section">Performance du Modèle</div>', unsafe_allow_html=True)
    model_metrics = [
        ("Accuracy",      "82.2%"),
        ("Algorithme",    "Logistic Regression"),
        ("Features",      "45"),
        ("Observations",  "7 043"),
        ("Split Train/Test", "80 / 20 %"),
        ("Taux de churn", f"{churn_rate:.1f}%"),
    ]
    for lbl, val in model_metrics:
        st.markdown(
            f'<div class="sb-metric"><span class="lbl">{lbl}</span><span class="val">{val}</span></div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="sb-section">À propos</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:#1E1E1E; border:1px solid #383838; border-radius:8px;
                padding:1rem 1.1rem; margin:0.2rem 0 0.8rem 0">
      <div style="font-size:0.82rem; color:#CCCCCC; line-height:1.9">
        Application de prédiction du churn client dans le secteur télécom.
        Basée sur le dataset
        <b style="color:{WHITE}">IBM Telco Customer Churn</b>.
      </div>
      <div style="margin-top:0.75rem; display:flex; flex-direction:column; gap:0.4rem">
        <div style="display:flex; justify-content:space-between; align-items:center;
                    padding:0.35rem 0; border-bottom:1px solid #2A2A2A">
          <span style="font-size:0.75rem; color:#888; font-weight:600">🏫 École</span>
          <span style="font-size:0.78rem; color:{WHITE}; font-weight:700">INSEEDS</span>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center;
                    padding:0.35rem 0; border-bottom:1px solid #2A2A2A">
          <span style="font-size:0.75rem; color:#888; font-weight:600">📦 Dataset</span>
          <span style="font-size:0.78rem; color:{WHITE}; font-weight:700">IBM / Kaggle</span>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center;
                    padding:0.35rem 0">
          <span style="font-size:0.75rem; color:#888; font-weight:600">🎯 Cibles</span>
          <span style="font-size:0.75rem; color:{ORANGE}; font-weight:700">Orange CI · MTN · Moov</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="border-top:1px solid #2E2E2E; margin-top:0.5rem; padding-top:0.75rem;
                text-align:center; line-height:2">
      <div style="font-size:0.74rem; color:#888; font-weight:600">
        TelecoAfrica Intelligence Platform
      </div>
      <div style="font-size:0.7rem; color:#666">
        © 2025 · <span style="color:{ORANGE}; font-weight:700">INSEEDS</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# KPI STRIP
# ─────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
kpi_data = [
    ("📊", f"{total:,}",         "Clients analysés",          "",   ""),
    ("⚠️", f"{churn_rate:.1f}%", "Taux de churn global",      "up", "⬆ Signal d'attrition critique"),
    ("💰", f"${avg_mc_churn:.1f}","Charges moy. churners",    "up", "⬆ +21% vs clients fidèles"),
    ("📅", f"{avg_tenure_churn:.0f} mois","Ancienneté moy. churners","down","⬇ Clients récents à risque"),
    ("📄", "88.6%",              "Churners sous contrat M/M", "up", "⬆ Contrat court = risque majeur"),
]
for col, (icon, val, lbl, ddir, dtxt) in zip([k1,k2,k3,k4,k5], kpi_data):
    dcls  = f'class="kpi-delta {ddir}"' if ddir else 'class="kpi-delta"'
    dhtml = f'<div {dcls}>{dtxt}</div>' if dtxt else ""
    col.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-icon">{icon}</div>
      <div class="kpi-value">{val}</div>
      <div class="kpi-label">{lbl}</div>
      {dhtml}
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab_pred, tab_eda, tab_insights, tab_model = st.tabs([
    "🎯  Prédiction Client",
    "📊  Analyse Exploratoire",
    "💡  Insights Business",
    "🔬  Modèle & Architecture",
])


# ═══════════════════════════════════════════════════════════════
#  TAB 1 — PRÉDICTION
# ═══════════════════════════════════════════════════════════════
with tab_pred:
    st.markdown("<br>", unsafe_allow_html=True)
    col_form, col_res = st.columns([1.05, 1], gap="large")

    # ── Formulaire ─────────────────────────────────────────────
    with col_form:
        st.markdown('<div class="section-title">📝 Profil du client à analyser</div>', unsafe_allow_html=True)

        with st.expander("👤 Informations générales", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                gender     = st.selectbox("Genre", ["Male", "Female"])
                senior     = st.selectbox("Senior (≥65 ans)", ["Non", "Oui"])
            with c2:
                partner    = st.selectbox("Partenaire", ["Yes", "No"])
                dependents = st.selectbox("Personnes à charge", ["No", "Yes"])
            with c3:
                st.markdown(f"""
                <div style='background:#1E1E1E; border:1px solid #FF6200; border-radius:7px;
                            padding:0.9rem; font-size:0.78rem; color:#CCCCCC; line-height:1.7; margin-top:1.5rem'>
                  🔍 <b style='color:#FF6200'>Astuce</b><br>
                  Les seniors et les clients sans partenaire présentent un risque de churn plus élevé (41.7% vs 23.6%).
                </div>""", unsafe_allow_html=True)

        with st.expander("📄 Contrat & Facturation", expanded=True):
            c4, c5 = st.columns(2)
            with c4:
                tenure = st.slider(
                    "Ancienneté (mois)", 0, 72, 12,
                    help="Clients 0–12 mois : 47.7% de churn"
                )
                monthly_charges = st.slider(
                    "Charges mensuelles ($)", 15.0, 120.0, 65.0, 0.5,
                    help="Churners paient en moyenne $74.4 vs $61.3"
                )
            with c5:
                contract = st.selectbox(
                    "Type de contrat",
                    ["Month-to-month", "One year", "Two year"],
                    help="M/M = 42.7% churn | 2 ans = 2.8% churn"
                )
                paperless = st.selectbox("Facturation dématérialisée", ["Yes", "No"])
                payment   = st.selectbox("Mode de paiement", [
                    "Electronic check", "Mailed check",
                    "Bank transfer (automatic)", "Credit card (automatic)"
                ])

        with st.expander("📱 Services souscrits", expanded=False):
            s1, s2, s3 = st.columns(3)
            with s1:
                phone    = st.selectbox("Téléphonie", ["Yes", "No"])
                multi    = st.selectbox("Lignes multiples", ["No", "Yes", "No phone service"])
                internet = st.selectbox("Internet", ["Fiber optic", "DSL", "No"])
            with s2:
                online_sec = st.selectbox("Sécurité en ligne", ["No", "Yes", "No internet service"])
                online_bk  = st.selectbox("Sauvegarde cloud", ["No", "Yes", "No internet service"])
                device     = st.selectbox("Protection appareil", ["No", "Yes", "No internet service"])
            with s3:
                tech_sup  = st.selectbox("Support technique", ["No", "Yes", "No internet service"])
                stream_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
                stream_mv = st.selectbox("Streaming Cinéma", ["No", "Yes", "No internet service"])

        predict_btn = st.button("🔍  Analyser le risque de churn", use_container_width=True)

    # ── Résultat ────────────────────────────────────────────────
    with col_res:
        st.markdown('<div class="section-title">📈 Résultat & Recommandations</div>', unsafe_allow_html=True)

        def build_and_predict(inp: dict):
            row = {c: 0 for c in feature_cols}
            row["tenure"]         = inp["tenure"]
            row["MonthlyCharges"] = inp["MonthlyCharges"]
            row["TotalCharges"]   = inp["tenure"] * inp["MonthlyCharges"]
            row["SeniorCitizen"]  = 1 if inp["senior"] == "Oui" else 0
            cats = {
                "gender": inp["gender"], "Partner": inp["partner"],
                "Dependents": inp["dependents"], "PhoneService": inp["phone"],
                "MultipleLines": inp["multi"], "InternetService": inp["internet"],
                "OnlineSecurity": inp["online_sec"], "OnlineBackup": inp["online_bk"],
                "DeviceProtection": inp["device"], "TechSupport": inp["tech_sup"],
                "StreamingTV": inp["stream_tv"], "StreamingMovies": inp["stream_mv"],
                "Contract": inp["contract"], "PaperlessBilling": inp["paperless"],
                "PaymentMethod": inp["payment"],
            }
            for pref, val in cats.items():
                k = f"{pref}_{val}"
                if k in row:
                    row[k] = 1
            X  = np.array([[row[c] for c in feature_cols]])
            Xs = scaler.transform(X)
            prob = float(model.predict_proba(Xs)[0][1])
            return prob, int(prob >= 0.5)

        if not predict_btn:
            st.markdown(f"""
            <div style='text-align:center; padding:4rem 2rem; color:#333;
                        border:1px dashed #2E2E2E; border-radius:10px'>
              <div style='font-size:3rem; margin-bottom:0.8rem; opacity:0.5'>📋</div>
              <div style='font-size:0.92rem; color:#555; font-weight:600'>Renseignez le profil client</div>
              <div style='font-size:0.78rem; color:#444; margin-top:0.4rem'>
                et cliquez sur "Analyser" pour obtenir la prédiction
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            inp = dict(
                tenure=tenure, MonthlyCharges=monthly_charges, senior=senior,
                gender=gender, partner=partner, dependents=dependents,
                phone=phone, multi=multi, internet=internet,
                online_sec=online_sec, online_bk=online_bk, device=device,
                tech_sup=tech_sup, stream_tv=stream_tv, stream_mv=stream_mv,
                contract=contract, paperless=paperless, payment=payment,
            )
            prob, pred = build_and_predict(inp)
            pct  = round(prob * 100, 1)
            spct = round((1 - prob) * 100, 1)

            if pred == 1:
                st.markdown(f"""
                <div class="verdict-churn">
                  <div style='font-size:2.5rem; margin-bottom:0.5rem'>⚠️</div>
                  <div class="verdict-label">Client à risque de départ</div>
                  <div class="verdict-sub">Ce client présente un profil à forte probabilité de résiliation</div>
                  <div class="verdict-pct" style="color:{RED}">{pct}%</div>
                  <div style="font-size:0.77rem; color:{GRAY}; margin-top:0.3rem">probabilité de churn</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="verdict-safe">
                  <div style='font-size:2.5rem; margin-bottom:0.5rem'>✅</div>
                  <div class="verdict-label">Client stable et fidèle</div>
                  <div class="verdict-sub">Ce client devrait rester abonné chez l'opérateur</div>
                  <div class="verdict-pct" style="color:{GREEN}">{spct}%</div>
                  <div style="font-size:0.77rem; color:{GRAY}; margin-top:0.3rem">probabilité de rester</div>
                </div>""", unsafe_allow_html=True)

            # Barre de risque
            bar_class = "risk-bar-fill" if pred == 1 else "risk-bar-safe"
            st.markdown(f"""
            <div style="margin-top:1.2rem">
              <div style="display:flex; justify-content:space-between; font-size:0.74rem; color:#555; margin-bottom:0.3rem">
                <span>Niveau de risque de churn</span>
                <span style="color:{'#E53935' if pred else '#00C48C'}; font-weight:700">{pct}%</span>
              </div>
              <div class="risk-bar-wrap"><div class="{bar_class}" style="width:{pct}%"></div></div>
              <div class="risk-labels"><span>🟢 Fidèle</span><span>🟡 Modéré</span><span>🔴 Churn</span></div>
            </div>""", unsafe_allow_html=True)

            # Mini gauge Plotly
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pct,
                number={"suffix": "%", "font": {"size": 26, "color": WHITE}},
                gauge={
                    "axis": {"range": [0, 100], "tickfont": {"size": 9, "color": "#555"}},
                    "bar": {"color": RED if pred else GREEN, "thickness": 0.28},
                    "bgcolor": "#222",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0,  30], "color": "rgba(0,196,140,0.12)"},
                        {"range": [30, 60], "color": "rgba(255,152,0,0.12)"},
                        {"range": [60,100], "color": "rgba(229,57,53,0.12)"},
                    ],
                },
                title={"text": "Score de risque", "font": {"size": 12, "color": GRAY}},
            ))
            fig_g.update_layout(**plotly_layout(height=190, margin=dict(l=20,r=20,t=40,b=5)))
            st.plotly_chart(fig_g, use_container_width=True, config={"displayModeBar": False})

            # Profil résumé
            st.markdown('<div class="section-title" style="margin-top:0.3rem">🧾 Profil analysé</div>',
                        unsafe_allow_html=True)
            rc = {"Month-to-month": f"<span style='color:{RED}'>● Élevé</span>",
                  "One year":       f"<span style='color:{ORANGE}'>● Modéré</span>",
                  "Two year":       f"<span style='color:{GREEN}'>● Faible</span>"}
            rt = (f"<span style='color:{GREEN}'>● Ancienneté solide</span>" if tenure >= 24
                  else f"<span style='color:{ORANGE}'>● Client récent</span>" if tenure >= 6
                  else f"<span style='color:{RED}'>● Très récent — critique</span>")
            rm = (f"<span style='color:{RED}'>● Élevées</span>" if monthly_charges >= 80
                  else f"<span style='color:{ORANGE}'>● Modérées</span>" if monthly_charges >= 50
                  else f"<span style='color:{GREEN}'>● Basses</span>")
            for k, v in [
                ("Ancienneté",         f"{tenure} mois &nbsp;·&nbsp; {rt}"),
                ("Charges mensuelles", f"${monthly_charges:.2f} &nbsp;·&nbsp; {rm}"),
                ("Contrat",            f"{contract} &nbsp;·&nbsp; {rc[contract]}"),
                ("Mode de paiement",   payment),
                ("Service Internet",   internet),
                ("Score churn",        f"<b style='color:{'#E53935' if pred else '#00C48C'}'>{pct}%</b>"),
            ]:
                st.markdown(
                    f'<div class="data-row"><span class="data-key">{k}</span><span class="data-val">{v}</span></div>',
                    unsafe_allow_html=True
                )

            # Recommandation
            st.markdown("<br>", unsafe_allow_html=True)
            if pred == 1:
                st.markdown(f"""
                <div class="banner banner-red">
                  <b>🚨 Action prioritaire recommandée</b><br>
                  · Contacter le client dans les <b>48h</b> via son canal préféré<br>
                  · Proposer une <b>migration vers contrat 1 an</b> avec remise de 15%<br>
                  · Si charges > $80 : audit tarifaire + bundle personnalisé<br>
                  · Si ancienneté < 6 mois : programme d'onboarding dédié
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="banner banner-green">
                  <b>✅ Client stable — Opportunité commerciale</b><br>
                  · Proposer des <b>services complémentaires</b> (streaming, sécurité)<br>
                  · Encourager le <b>parrainage</b> avec programme de récompenses<br>
                  · Suggérer le <b>passage au contrat 2 ans</b> si encore mensuel
                </div>""", unsafe_allow_html=True)


            # ── Explication SHAP ─────────────────────────────────────
            if SHAP_OK:
                try:
                    explainer = shap.LinearExplainer(
                        model, masker=shap.maskers.Independent(Xs)
                    )
                    shap_vals = explainer.shap_values(Xs)[0]

                    top_n    = 10
                    indices  = np.argsort(np.abs(shap_vals))[-top_n:]
                    top_names = [feature_cols[i] for i in indices]
                    top_vals  = [float(shap_vals[i]) for i in indices]
                    bar_colors = [RED if v > 0 else GREEN for v in top_vals]

                    fig_shap = go.Figure(go.Bar(
                        x=top_vals,
                        y=top_names,
                        orientation="h",
                        marker_color=bar_colors,
                        text=[f"{v:+.3f}" for v in top_vals],
                        textposition="outside",
                        textfont=dict(size=10, color=WHITE),
                    ))
                    fig_shap.update_layout(**plotly_layout(
                        height=340,
                        title_text="Explication SHAP — Pourquoi ce score ?",
                        xaxis=dict(title="Impact sur la prediction",
                                   zeroline=True, zerolinecolor="#555",
                                   gridcolor="#2E2E2E"),
                        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
                        margin=dict(l=10, r=60, t=44, b=10),
                        showlegend=False,
                    ))

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(
                        '<div class="section-title">🔬 IA Explicable — Facteurs clés de la décision</div>',
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f'''<div style="font-size:0.76rem; color:#666; margin-bottom:0.6rem; line-height:1.7">
  <span style="color:{RED}">&#9632; Barres rouges</span> = facteurs qui
  <b style="color:#EF9A9A">augmentent</b> le risque de churn &nbsp;&middot;&nbsp;
  <span style="color:{GREEN}">&#9632; Barres vertes</span> = facteurs qui
  <b style="color:#80CBC4">protègent</b> la fidélité
</div>''',
                        unsafe_allow_html=True
                    )
                    st.plotly_chart(fig_shap, use_container_width=True,
                                    config={"displayModeBar": False})
                except Exception:
                    pass
            else:
                st.markdown(
                    '<div style="font-size:0.74rem;color:#555;padding:0.5rem 0;font-style:italic">' +
                    '💡 Installez <code>shap</code> pour voir l\'explication détaillée : ' +
                    '<code>pip install shap</code></div>',
                    unsafe_allow_html=True
                )

            # ── Bouton Export PDF ────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-title">🖨️ Rapport PDF</div>', unsafe_allow_html=True)

            if FPDF_OK:
                # Récupérer les données SHAP si disponibles
                pdf_shap_names, pdf_shap_vals = None, None
                if SHAP_OK:
                    try:
                        _exp = shap.LinearExplainer(
                            model, masker=shap.maskers.Independent(Xs)
                        )
                        _sv  = _exp.shap_values(Xs)[0]
                        _idx = np.argsort(np.abs(_sv))[-10:]
                        pdf_shap_names = [feature_cols[i] for i in _idx]
                        pdf_shap_vals  = [float(_sv[i])   for i in _idx]
                    except Exception:
                        pass

                pdf_bytes = generate_pdf_report(
                    inp   = dict(
                        gender=gender, senior=senior, tenure=tenure,
                        MonthlyCharges=monthly_charges, contract=contract,
                        payment=payment, internet=internet,
                        paperless=paperless, partner=partner,
                        dependents=dependents,
                    ),
                    pct        = pct,
                    pred       = pred,
                    shap_names = pdf_shap_names,
                    shap_values= pdf_shap_vals,
                )
                st.download_button(
                    label     = "📄  Télécharger le rapport PDF",
                    data      = pdf_bytes,
                    file_name = f"rapport_churn_client.pdf",
                    mime      = "application/pdf",
                    use_container_width=True,
                )
            else:
                st.markdown(
                    '<div style="font-size:0.74rem;color:#555;padding:0.4rem 0;font-style:italic">'
                    '💡 Installez <code>fpdf2</code> pour activer l\'export PDF : '
                    '<code>pip install fpdf2</code></div>',
                    unsafe_allow_html=True
                )

            # ── Sauvegarde dans l'historique de session ─────────────
            st.session_state.historique.append({
                "Genre":             gender,
                "Senior":            senior,
                "Ancienneté (mois)": tenure,
                "Charges mens. ($)": monthly_charges,
                "Contrat":           contract,
                "Internet":          internet,
                "Paiement":          payment,
                "Score churn (%)":   pct,
                "Verdict":           "⚠️ Churn" if pred == 1 else "✅ Fidèle",
            })

    # ── Bloc historique — toujours visible dans l'onglet ─────────
    if st.session_state.historique:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">📋 Historique des analyses — Session en cours</div>',
                    unsafe_allow_html=True)

        df_hist = pd.DataFrame(st.session_state.historique)
        df_hist.index = range(1, len(df_hist) + 1)
        df_hist.index.name = "N°"

        st.dataframe(df_hist, use_container_width=True)

        col_exp, col_clr = st.columns([2, 1])
        with col_exp:
            csv_bytes = df_hist.to_csv(index=True).encode("utf-8")
            st.download_button(
                label="📥  Télécharger l'historique des analyses (CSV)",
                data=csv_bytes,
                file_name="historique_predictions_churn.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_clr:
            if st.button("🗑️  Effacer l'historique", use_container_width=True):
                st.session_state.historique = []
                st.rerun()


# ═══════════════════════════════════════════════════════════════
#  TAB 2 — ANALYSE EXPLORATOIRE (EDA)
# ═══════════════════════════════════════════════════════════════
with tab_eda:
    st.markdown("<br>", unsafe_allow_html=True)

    # Aperçu dataset
    st.markdown('<div class="section-title">🗂️ Aperçu du Dataset — 8 premières lignes sur 7 043 au total</div>', unsafe_allow_html=True)
    col_tbl, col_st = st.columns([1.6, 1])
    with col_tbl:
        st.dataframe(
            df.drop(columns=["ChurnBin","tenure_group"]).head(8),
            use_container_width=True, height=260
        )
    with col_st:
        st.markdown('<div class="section-title">📋 Statistiques générales</div>', unsafe_allow_html=True)
        for k, v in {
            "Observations": f"{len(df):,}",
            "Variables": "21",
            "Valeurs manquantes": "0",
            "Clients fidèles": f"{(df['Churn']=='No').sum():,}",
            "Clients ayant churné": f"{(df['Churn']=='Yes').sum():,}",
            "Ancienneté moyenne (mois)": f"{df['tenure'].mean():.1f} mois",
            "Charges mensuelles moyennes": f"${df['MonthlyCharges'].mean():.2f}",
            "Charges totales moyennes": f"${df['TotalCharges'].mean():.0f}",
        }.items():
            st.markdown(
                f'<div class="data-row"><span class="data-key">{k}</span><span class="data-val">{v}</span></div>',
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Graphiques Ligne 1 ──────────────────────────────────────
    st.markdown('<div class="section-title">📊 Distribution & Répartition du Churn</div>', unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)

    # Donut global
    with g1:
        vals = df["Churn"].value_counts()
        fig1 = go.Figure(go.Pie(
            labels=["Fidèles", "Churners"],
            values=[vals.get("No",0), vals.get("Yes",0)],
            hole=0.62,
            marker_colors=[GREEN, RED],
            textinfo="percent+label",
            textfont_size=11, textfont_color=WHITE,
        ))
        fig1.add_annotation(text="<b>26.5%</b><br>churn", x=0.5, y=0.5,
                            font_size=13, showarrow=False, font_color=WHITE)
        fig1.update_layout(**plotly_layout( height=270, title_text="Répartition Churn global",
                           showlegend=False))
        st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

    # Churn par contrat
    with g2:
        cbc = df.groupby("Contract")["ChurnBin"].mean().mul(100).round(1).reset_index()
        cbc.columns = ["Contrat", "Taux (%)"]
        fig2 = go.Figure(go.Bar(
            x=cbc["Contrat"], y=cbc["Taux (%)"],
            marker_color=[RED, ORANGE, GREEN],
            text=cbc["Taux (%)"].astype(str) + "%",
            textposition="outside", textfont_color=WHITE, textfont_size=11,
        ))
        fig2.update_layout(**plotly_layout( height=270,
                           title_text="Churn par type de contrat",
                           yaxis=dict(range=[0,55], gridcolor="#2E2E2E"),
                           showlegend=False))
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # Churn par internet
    with g3:
        cbi = df.groupby("InternetService")["ChurnBin"].mean().mul(100).round(1).reset_index()
        cbi.columns = ["Internet", "Taux (%)"]
        fig3 = go.Figure(go.Bar(
            x=cbi["Internet"], y=cbi["Taux (%)"],
            marker_color=[BLUE, RED, GREEN],
            text=cbi["Taux (%)"].astype(str) + "%",
            textposition="outside", textfont_color=WHITE, textfont_size=11,
        ))
        fig3.update_layout(**plotly_layout( height=270,
                           title_text="Churn par service Internet",
                           yaxis=dict(range=[0,55], gridcolor="#2E2E2E"),
                           showlegend=False))
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Graphiques Ligne 2 ──────────────────────────────────────
    st.markdown('<div class="section-title">📈 Distribution des Variables Numériques</div>', unsafe_allow_html=True)
    g4, g5 = st.columns(2)

    with g4:
        fig4 = go.Figure()
        fig4.add_trace(go.Histogram(
            x=df[df["Churn"]=="No"]["tenure"], name="Fidèles",
            marker_color=GREEN, opacity=0.65, nbinsx=30, histnorm="percent",
        ))
        fig4.add_trace(go.Histogram(
            x=df[df["Churn"]=="Yes"]["tenure"], name="Churners",
            marker_color=RED, opacity=0.65, nbinsx=30, histnorm="percent",
        ))
        fig4.update_layout(**plotly_layout( height=290,
                           title_text="Distribution de l'ancienneté (tenure)",
                           xaxis_title="Mois", yaxis_title="Fréquence (%)",
                           barmode="overlay",
                           legend=dict(orientation="h", y=1.15, x=0)))
        st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})

    with g5:
        fig5 = go.Figure()
        fig5.add_trace(go.Violin(
            y=df[df["Churn"]=="No"]["MonthlyCharges"], name="Fidèles",
            line_color=GREEN, fillcolor="rgba(0,196,140,0.15)",
            box_visible=True, meanline_visible=True,
        ))
        fig5.add_trace(go.Violin(
            y=df[df["Churn"]=="Yes"]["MonthlyCharges"], name="Churners",
            line_color=RED, fillcolor="rgba(229,57,53,0.15)",
            box_visible=True, meanline_visible=True,
        ))
        fig5.update_layout(**plotly_layout( height=290,
                           title_text="Charges mensuelles — Fidèles vs Churners",
                           yaxis_title="Charges ($)",
                           legend=dict(orientation="h", y=1.15, x=0)))
        st.plotly_chart(fig5, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Graphiques Ligne 3 ──────────────────────────────────────
    st.markdown('<div class="section-title">📉 Ancienneté, Paiement & Services</div>', unsafe_allow_html=True)
    g6, g7, g8 = st.columns(3)

    with g6:
        tg = df.groupby("tenure_group", observed=True)["ChurnBin"].mean().mul(100).round(1).reset_index()
        tg.columns = ["Tranche", "Taux (%)"]
        fig6 = go.Figure(go.Bar(
            x=tg["Tranche"], y=tg["Taux (%)"],
            marker_color=[RED, ORANGE, "#F5C518", GREEN],
            text=tg["Taux (%)"].astype(str) + "%",
            textposition="outside", textfont_color=WHITE, textfont_size=11,
        ))
        fig6.update_layout(**plotly_layout( height=270,
                           title_text="Churn par tranche d'ancienneté",
                           yaxis=dict(range=[0,60], gridcolor="#2E2E2E"),
                           showlegend=False))
        st.plotly_chart(fig6, use_container_width=True, config={"displayModeBar": False})

    with g7:
        pay = df.groupby("PaymentMethod")["ChurnBin"].mean().mul(100).round(1).reset_index()
        pay.columns = ["Méthode", "Taux (%)"]
        pay["Méthode"] = pay["Méthode"].str.replace(" (automatic)", "\n(auto)", regex=False)
        fig7 = go.Figure(go.Bar(
            y=pay["Méthode"], x=pay["Taux (%)"],
            orientation="h",
            marker_color=[RED if v > 30 else ORANGE if v > 18 else GREEN for v in pay["Taux (%)"]],
            text=pay["Taux (%)"].astype(str) + "%",
            textposition="outside", textfont_color=WHITE, textfont_size=11,
        ))
        fig7.update_layout(**plotly_layout( height=270,
                           title_text="Churn par mode de paiement",
                           xaxis=dict(range=[0,58], gridcolor="#2E2E2E"),
                           showlegend=False))
        st.plotly_chart(fig7, use_container_width=True, config={"displayModeBar": False})

    with g8:
        sample = df.sample(500, random_state=42)
        fig8 = go.Figure()
        for churn_val, color, name in [("No", GREEN, "Fidèles"), ("Yes", RED, "Churners")]:
            sub = sample[sample["Churn"] == churn_val]
            fig8.add_trace(go.Scatter(
                x=sub["tenure"], y=sub["MonthlyCharges"],
                mode="markers", name=name,
                marker=dict(color=color, size=4, opacity=0.55),
            ))
        fig8.update_layout(**plotly_layout( height=270,
                           title_text="Tenure vs Charges (échantillon 500)",
                           xaxis_title="Ancienneté (mois)",
                           yaxis_title="Charges ($)",
                           legend=dict(orientation="h", y=1.15, x=0)))
        st.plotly_chart(fig8, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<br>", unsafe_allow_html=True)

    # Impact services
    st.markdown('<div class="section-title">🔧 Impact des Services sur le Churn</div>', unsafe_allow_html=True)
    services_data = {
        "Sécurité en ligne":    (14.6, 41.8),
        "Support technique":    (15.2, 41.6),
        "Sauveg. cloud":        (21.5, 39.9),
        "Protect. appareil":    (22.5, 39.1),
    }
    fig9 = go.Figure()
    svcs  = list(services_data.keys())
    vals_yes = [services_data[s][0] for s in svcs]
    vals_no  = [services_data[s][1] for s in svcs]
    fig9.add_trace(go.Bar(name="Avec service", x=svcs, y=vals_yes,
                          marker_color=GREEN, text=[f"{v}%" for v in vals_yes],
                          textposition="outside", textfont_color=WHITE, textfont_size=11))
    fig9.add_trace(go.Bar(name="Sans service", x=svcs, y=vals_no,
                          marker_color=RED, text=[f"{v}%" for v in vals_no],
                          textposition="outside", textfont_color=WHITE, textfont_size=11))
    fig9.update_layout(**plotly_layout( height=280,
                       title_text="Taux de churn (%) selon l'activation du service",
                       barmode="group",
                       yaxis=dict(range=[0, 52], gridcolor="#2E2E2E"),
                       yaxis_title="Taux de churn (%)",
                       legend=dict(orientation="h", y=1.12, x=0)))
    st.plotly_chart(fig9, use_container_width=True, config={"displayModeBar": False})


# ═══════════════════════════════════════════════════════════════
#  TAB 3 — INSIGHTS BUSINESS
# ═══════════════════════════════════════════════════════════════
with tab_insights:
    st.markdown("<br>", unsafe_allow_html=True)

    col_risk, col_prot = st.columns(2)
    risk_factors = [
        (RED,    "Contrat mensuel (Month-to-month)",
         "42.7% de churn — le facteur le plus discriminant. Sans engagement, les clients sont libres de partir immédiatement."),
        (RED,    "Paiement par chèque électronique",
         "45.3% de churn — le mode le plus risqué. Corrélé à une relation transactionnelle peu engagée."),
        (ORANGE, "Fiber Optic — paradoxe qualité/fidélité",
         "41.9% de churn malgré l'offre premium. Suggère une insatisfaction sur la tarification ou le support."),
        (ORANGE, "Charges mensuelles élevées (> $80)",
         "Les churners paient $74.4/mois en moyenne (+21% vs fidèles). Le rapport qualité-prix est un levier de départ."),
        (ORANGE, "Ancienneté < 12 mois",
         "47.7% de churn dans la 1re année. L'onboarding est le moment le plus critique pour la fidélisation."),
    ]
    prot_factors = [
        (GREEN, "Contrat 2 ans — ancrage fort",
         "Seulement 2.8% de churn. L'engagement contractuel est le meilleur rempart contre l'attrition."),
        (GREEN, "Sécurité en ligne activée",
         "14.6% vs 41.8% sans service. Réduction du risque de 65%. Les services VA créent de la valeur perçue."),
        (GREEN, "Support technique actif",
         "15.2% vs 41.6% sans support. Un client assisté est un client rassuré et fidèle."),
        (GREEN, "Partenaire ou personnes à charge",
         "La stabilité familiale se traduit en stabilité contractuelle — churn nettement plus bas."),
        (GREEN, "Paiement automatique (CB/virement)",
         "15–17% de churn vs 45.3% pour le chèque électronique. L'automatisation réduit le friction de départ."),
    ]

    with col_risk:
        st.markdown('<div class="section-title">🔴 Facteurs de Risque</div>', unsafe_allow_html=True)
        for color, title, text in risk_factors:
            st.markdown(f"""
            <div class="insight-row">
              <div class="insight-dot" style="background:{color}"></div>
              <div class="insight-text"><b>{title}</b><br>{text}</div>
            </div>""", unsafe_allow_html=True)

    with col_prot:
        st.markdown('<div class="section-title">🟢 Facteurs Protecteurs</div>', unsafe_allow_html=True)
        for color, title, text in prot_factors:
            st.markdown(f"""
            <div class="insight-row">
              <div class="insight-dot" style="background:{color}"></div>
              <div class="insight-text"><b>{title}</b><br>{text}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Radar + Plan rétention
    radar_c, plan_c = st.columns([1.2, 1])

    with radar_c:
        st.markdown('<div class="section-title">🕸️ Profil Comparatif — Fidèles vs Churners</div>',
                    unsafe_allow_html=True)
        cats     = ["Ancienneté\n(norm.)", "Charges\nmens.", "Contrat\nlong", "Services\nactifs", "Paiement\nauto"]
        fideles  = [0.53, 0.51, 0.72, 0.62, 0.68]
        churners = [0.25, 0.63, 0.12, 0.28, 0.31]
        fig_r = go.Figure()
        for name, vals, color, fill in [
            ("Fidèles",  fideles,  GREEN, "rgba(0,196,140,0.12)"),
            ("Churners", churners, RED,   "rgba(229,57,53,0.12)"),
        ]:
            fig_r.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=cats + [cats[0]],
                fill="toself", name=name,
                line=dict(color=color, width=2),
                fillcolor=fill, opacity=0.9,
            ))
        fig_r.update_layout(
            **plotly_layout( height=310, title_text="Profil de risque normalisé",
            polar=dict(
                bgcolor="#1E1E1E",
                radialaxis=dict(visible=True, range=[0,1], gridcolor="#444",
                                tickfont=dict(color="#AAAAAA", size=9)),
                angularaxis=dict(gridcolor="#444", tickfont=dict(color="#DDDDDD", size=11)),
            ),
            legend=dict(orientation="h", y=-0.08, font=dict(color=WHITE, size=13)),
        ))
        st.plotly_chart(fig_r, use_container_width=True, config={"displayModeBar": False})

    with plan_c:
        st.markdown('<div class="section-title">💼 Plan de Rétention — 3 Piliers</div>',
                    unsafe_allow_html=True)
        for color, num, title, items in [
            (ORANGE, "01", "Convertir les contrats courts",
             ["🎁 2 mois offerts si passage au contrat annuel",
              "📊 Visualiser les économies sur 12 mois",
              "🔔 Relance automatique à M+3 et M+6"]),
            (BLUE, "02", "Réduire la pression tarifaire",
             ["💡 Audit clients > $80/mois",
              "📦 Bundling pour baisser le prix perçu",
              "🎯 Offre personnalisée selon l'usage"]),
            (GREEN, "03", "Renforcer l'engagement précoce",
             ["👋 Onboarding premium 0–3 mois",
              "📞 Appel conseiller à J+30",
              "🎓 Activation des services valeur ajoutée"]),
        ]:
            items_html = "".join(
                f"<div style='font-size:0.8rem; color:#BBBBBB; padding:0.22rem 0'>{i}</div>" for i in items
            )
            st.markdown(f"""
            <div style="background:{DARK3}; border:1px solid #383838; border-left:3px solid {color};
                        border-radius:8px; padding:0.9rem 1.1rem; margin-bottom:0.75rem">
              <div style="display:flex; gap:0.6rem; align-items:center; margin-bottom:0.35rem">
                <span style="color:{color}; font-weight:800; font-size:1rem; font-family:'DM Mono'">{num}</span>
                <span style="color:{WHITE}; font-weight:600; font-size:0.86rem">{title}</span>
              </div>
              {items_html}
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Impact financier
    st.markdown('<div class="section-title">💰 Impact Financier Estimé</div>', unsafe_allow_html=True)
    fi1, fi2, fi3, fi4 = st.columns(4)
    revenue_lost   = churned * avg_mc_churn * 12
    gain_10        = revenue_lost * 0.10
    gain_25        = revenue_lost * 0.25
    cost_campaign  = churned * 15
    for col, icon, val, lbl, sub in [
        (fi1, "💸", f"${revenue_lost/1e6:.2f}M", "Revenu annuel perdu", "si aucune action de rétention"),
        (fi2, "🎯", f"${gain_10/1e3:.0f}K",      "Gain si 10% rétention", "objectif minimal atteignable"),
        (fi3, "🚀", f"${gain_25/1e3:.0f}K",      "Gain si 25% rétention", "objectif ambitieux"),
        (fi4, "📬", f"${cost_campaign/1e3:.0f}K","Coût campagne estimé",  "à $15 / client ciblé"),
    ]:
        col.markdown(f"""
        <div class="kpi-card" style="border-top-color:{BLUE}">
          <div class="kpi-icon">{icon}</div>
          <div class="kpi-value" style="font-size:1.65rem">{val}</div>
          <div class="kpi-label">{lbl}</div>
          <div class="kpi-delta" style="color:#999">{sub}</div>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  TAB 4 — MODÈLE & ARCHITECTURE
# ═══════════════════════════════════════════════════════════════
with tab_model:
    st.markdown("<br>", unsafe_allow_html=True)

    mc1, mc2 = st.columns([1, 1.1])

    with mc1:
        st.markdown('<div class="section-title">🤖 Spécifications du Modèle</div>', unsafe_allow_html=True)
        for k, v in {
            "Algorithme":        "Logistic Regression (scikit-learn)",
            "Régularisation":    "L2 (Ridge) · C = 0.5",
            "Itérations max":    "1 000",
            "Normalisation":     "StandardScaler (z-score)",
            "Accuracy (test)":   "82.2%",
            "Précision classe 1":"69%",
            "Recall classe 1":   "60%",
            "F1-Score classe 1": "64%",
            "Split Train/Test":  "80% / 20% (seed=42)",
            "Nb de features":    "45 (après encodage one-hot)",
        }.items():
            st.markdown(
                f'<div class="data-row"><span class="data-key">{k}</span><span class="data-val">{v}</span></div>',
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">📋 Rapport de Classification</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Classe":      ["Fidèles (0)", "Churners (1)", "Moyenne"],
            "Précision":   ["86%", "69%", "77%"],
            "Recall":      ["90%", "60%", "75%"],
            "F1-Score":    ["88%", "64%", "76%"],
            "Support":     ["1 036", "373", "1 409"],
        }), use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔄 Pipeline de Traitement</div>', unsafe_allow_html=True)
        for color, step, desc in [
            (ORANGE, "1. Chargement",      "Lecture CSV · Correction TotalCharges (str→float)"),
            (BLUE,   "2. Preprocessing",   "Encodage Churn · drop customerID · pd.get_dummies()"),
            (ORANGE, "3. Feature Eng.",    "45 features · StandardScaler sur X_train uniquement"),
            (GREEN,  "4. Modélisation",    "LogisticRegression · fit · C=0.5 · max_iter=1000"),
            (GREEN,  "5. Évaluation",      "Accuracy 82.2% · Classification report complet"),
            (GRAY,   "6. Sérialisation",   "model.pkl · scaler.pkl · feature_cols.json"),
        ]:
            st.markdown(f"""
            <div style="display:flex; align-items:flex-start; gap:0.8rem; padding:0.45rem 0;
                        border-bottom:1px solid #2A2A2A; font-size:0.79rem">
              <span style="color:{color}; font-weight:700; min-width:90px; font-family:'DM Mono'">{step}</span>
              <span style="color:#888">{desc}</span>
            </div>""", unsafe_allow_html=True)

    with mc2:
        st.markdown('<div class="section-title">🏗️ Architecture du Projet</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="panel" style="font-family:'DM Mono', monospace; font-size:0.77rem; line-height:2.1; color:#999">
          <span style="color:{ORANGE}">📁</span> <span style="color:{WHITE}; font-weight:600">telecom-churn-ai/</span><br>
          <span style="color:#666">│</span><br>
          <span style="color:{ORANGE}">├──</span> <span style="color:{GREEN}">app.py</span>
              <span style="color:#AAAAAA">  ← Application Streamlit (ce fichier)</span><br>
          <span style="color:{ORANGE}">├──</span> <span style="color:#7CB9E8">main.py</span>
              <span style="color:#AAAAAA">  ← Pipeline d'entraînement & EDA</span><br>
          <span style="color:{ORANGE}">│</span><br>
          <span style="color:{ORANGE}">├──</span> <span style="color:#FFD700">model.pkl</span>
              <span style="color:#AAAAAA">  ← Logistic Regression sérialisé</span><br>
          <span style="color:{ORANGE}">├──</span> <span style="color:#FFD700">scaler.pkl</span>
              <span style="color:#AAAAAA">  ← StandardScaler sérialisé</span><br>
          <span style="color:{ORANGE}">├──</span> <span style="color:#FFD700">feature_cols.json</span>
              <span style="color:#AAAAAA">  ← 45 features encodées</span><br>
          <span style="color:{ORANGE}">│</span><br>
          <span style="color:{ORANGE}">└──</span> <span style="color:#CCC">WA_Fn-UseC_-Telco-Customer-Churn.csv</span><br>
              <span style="color:#AAAAAA">      └── Dataset IBM Kaggle · 7 043 lignes · 21 colonnes</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Matrice de confusion
        st.markdown('<div class="section-title">🎯 Matrice de Confusion (Test Set)</div>',
                    unsafe_allow_html=True)
        TP = int(373 * 0.60)
        FN = 373 - TP
        TN = int(1036 * 0.90)
        FP = 1036 - TN
        fig_cm = go.Figure(go.Heatmap(
            z=[[TN, FP], [FN, TP]],
            x=["Prédit : Fidèle", "Prédit : Churn"],
            y=["Réel : Fidèle",   "Réel : Churn"],
            colorscale=[[0, DARK3], [0.5, "#7B3F00"], [1, ORANGE]],
            text=[[TN, FP], [FN, TP]],
            texttemplate="%{text}", textfont=dict(size=16, color=WHITE),
            showscale=False,
        ))
        fig_cm.update_layout(**plotly_layout(height=230, title_text="Matrice de confusion approchée"))
        st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": False})

        st.markdown(f"""
        <div style="font-size:0.76rem; color:#666; line-height:1.8; margin-top:0.5rem">
          <span style="color:{GREEN}">■</span> <b style="color:#AAA">VP {TP}</b> — Churners détectés (rétention possible)<br>
          <span style="color:{RED}">■</span> <b style="color:#AAA">FN {FN}</b> — Churners manqués (risque métier)<br>
          <span style="color:{ORANGE}">■</span> <b style="color:#AAA">FP {FP}</b> — Fidèles sur-alertés (coût campagne inutile)<br>
          <span style="color:{BLUE}">■</span> <b style="color:#AAA">VN {TN}</b> — Fidèles bien identifiés (upsell possible)
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">📦 Dépendances</div>', unsafe_allow_html=True)
        deps = ["streamlit", "pandas", "numpy", "scikit-learn", "plotly", "pickle", "json"]
        dep_html = "&nbsp; ".join(
            f'<span style="background:#2A2A2A; color:{ORANGE}; font-family:DM Mono; font-size:0.71rem; '
            f'padding:0.2rem 0.6rem; border-radius:4px; border:1px solid #383838">{d}</span>'
            for d in deps
        )
        st.markdown(dep_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="footer">
  <div>
    <b>TelecoAfrica Intelligence Platform</b> ·
    Prédiction du Churn Client · Secteur Télécommunications
  </div>
  <div style="text-align:center; color:#777">
    Dataset IBM Telco Customer Churn (Kaggle) · Logistic Regression · Accuracy 82.2%
  </div>
  <div style="text-align:right">
    <b>ANOH AMON FRANCKLIN HEMERSON</b><br>
    <span style="color:#AAAAAA">Master Data Science · <span style="color:{ORANGE}">INSEEDS</span></span><br>
    <span style="color:#888; font-size:0.7rem">Supervisé par : <span style="color:#BBBBBB">MR Akposso Didier Martial</span></span>
  </div>
</div>
""", unsafe_allow_html=True)