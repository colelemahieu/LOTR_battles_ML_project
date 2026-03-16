"""
LOTR Battle Predictor — Streamlit UI
=====================================
Run with:  streamlit run streamlit_app.py
Requires:  lotr_battles_1000.csv in the same directory
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# HARDCODED CHARACTER DATA
# ─────────────────────────────────────────────

HEROES = {
    "Aragorn": {"strength": 9, "agility": 7},
    "Gimli":   {"strength": 8, "agility": 5},
    "Legolas": {"strength": 7, "agility": 10},
    "Frodo":   {"strength": 3, "agility": 5},
    "Boromir": {"strength": 8, "agility": 5},
}

WEAPON_BONUSES = {
    "Aragorn": {"Anduril": 9, "Sword": 7, "ElvenBow": 5, "DwarvenAxe": 4, "Sting": 4},
    "Gimli":   {"Anduril": 4, "Sword": 4, "ElvenBow": -2, "DwarvenAxe": 7, "Sting": 5},
    "Legolas": {"Anduril": 5, "Sword": 5, "ElvenBow": 8, "DwarvenAxe": -2, "Sting": 5},
    "Frodo":   {"Anduril": -1, "Sword": -1, "ElvenBow": -7, "DwarvenAxe": -2, "Sting": 4},
    "Boromir": {"Anduril": 6, "Sword": 7, "ElvenBow": 3, "DwarvenAxe": 4, "Sting": 4},
}

ENEMIES = {
    "Orc":      {"strength": 5, "agility": 4},
    "Troll":    {"strength": 10, "agility": 2},
    "Uruk-hai": {"strength": 7, "agility": 6},
}

WEAPONS = ["Anduril", "Sword", "ElvenBow", "DwarvenAxe", "Sting"]

# ─────────────────────────────────────────────
# MODEL TRAINING (cached)
# ─────────────────────────────────────────────

@st.cache_resource
def load_and_train(csv_path: str):
    df = pd.read_csv(csv_path)

    encoders = {}
    for col in ["HeroName", "HeroWeaponName", "EnemyName"]:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    df["HeroPower"]        = df["HeroStrength"] * 0.6 + df["HeroAgility"] * 0.4 + df["HeroWeaponBonus"]
    df["EnemyPower"]       = df["EnemyStrength"] * 0.6 + df["EnemyAgility"] * 0.4
    df["EnemyTotalThreat"] = df["EnemyPower"] * df["EnemyNumber"]
    df["PowerDelta"]       = df["HeroPower"] - df["EnemyTotalThreat"]

    feature_cols = [
        "HeroName_enc", "HeroStrength", "HeroAgility",
        "HeroWeaponName_enc", "HeroWeaponBonus",
        "EnemyName_enc", "EnemyStrength", "EnemyAgility",
        "EnemyNumber",
        "HeroPower", "EnemyPower", "EnemyTotalThreat", "PowerDelta",
    ]

    X = df[feature_cols]
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200, max_features="sqrt",
        min_samples_split=5, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    return model, encoders, feature_cols, acc


def safe_encode(le, value):
    if value in le.classes_:
        return le.transform([value])[0]
    return 0


def predict(model, encoders, feature_cols,
            hero_name, hero_wpn, enemy_name, enemy_number):
    h = HEROES[hero_name]
    e = ENEMIES[enemy_name]
    bonus = WEAPON_BONUSES[hero_name][hero_wpn]

    hero_power         = h["strength"] * 0.6 + h["agility"] * 0.4 + bonus
    enemy_power        = e["strength"] * 0.6 + e["agility"] * 0.4
    enemy_total_threat = enemy_power * enemy_number
    power_delta        = hero_power - enemy_total_threat

    row = pd.DataFrame([{
        "HeroName_enc":       safe_encode(encoders["HeroName"],      hero_name),
        "HeroStrength":       h["strength"],
        "HeroAgility":        h["agility"],
        "HeroWeaponName_enc": safe_encode(encoders["HeroWeaponName"], hero_wpn),
        "HeroWeaponBonus":    bonus,
        "EnemyName_enc":      safe_encode(encoders["EnemyName"],      enemy_name),
        "EnemyStrength":      e["strength"],
        "EnemyAgility":       e["agility"],
        "EnemyNumber":        enemy_number,
        "HeroPower":          hero_power,
        "EnemyPower":         enemy_power,
        "EnemyTotalThreat":   enemy_total_threat,
        "PowerDelta":         power_delta,
    }])

    prediction = model.predict(row)[0]
    proba      = model.predict_proba(row)[0]
    return prediction, proba, hero_power, enemy_total_threat, power_delta, bonus


# ─────────────────────────────────────────────
# PAGE CONFIG & STYLING
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="LOTR Battle Predictor",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=MedievalSharp&family=IM+Fell+English:ital@0;1&family=Cinzel:wght@400;700;900&display=swap');

html, body, [class*="css"] {
    background-color: #0f0b07;
    color: #d4b896;
}

.stApp {
    background: radial-gradient(ellipse at top, #1a1208 0%, #0f0b07 60%);
    min-height: 100vh;
}

h1, h2, h3 {
    font-family: 'Cinzel', serif !important;
    color: #c9922a !important;
    text-shadow: 0 0 30px rgba(201,146,42,0.4);
    letter-spacing: 0.05em;
}

.title-block {
    text-align: center;
    padding: 2rem 0 1rem;
    border-bottom: 1px solid #3a2a10;
    margin-bottom: 2rem;
}

.title-block h1 {
    font-size: 2.8rem;
    font-family: 'Cinzel', serif;
    color: #c9922a;
    text-shadow: 0 0 40px rgba(201,146,42,0.6), 0 2px 4px rgba(0,0,0,0.8);
    margin: 0;
    letter-spacing: 0.1em;
}

.title-block .subtitle {
    font-family: 'IM Fell English', serif;
    font-style: italic;
    color: #8a7055;
    font-size: 1.1rem;
    margin-top: 0.4rem;
}

.model-badge {
    display: inline-block;
    background: rgba(201,146,42,0.1);
    border: 1px solid #3a2a10;
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.75rem;
    color: #8a7055;
    font-family: monospace;
    margin-top: 0.5rem;
}

/* Selectbox & inputs */
.stSelectbox label, .stSlider label {
    font-family: 'Cinzel', serif !important;
    color: #c9922a !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.stSelectbox > div > div {
    background-color: #1a1208 !important;
    border: 1px solid #3a2a10 !important;
    color: #d4b896 !important;
    border-radius: 4px !important;
}

.stSlider > div > div > div {
    background-color: #c9922a !important;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid #3a2a10;
    margin: 1.5rem 0;
}

/* VS badge */
.vs-badge {
    text-align: center;
    font-family: 'Cinzel', serif;
    font-size: 1.6rem;
    color: #c9922a;
    padding: 1rem 0;
    text-shadow: 0 0 20px rgba(201,146,42,0.5);
    letter-spacing: 0.3em;
}

/* Stat pills */
.stat-row {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin: 0.5rem 0 1rem;
}
.stat-pill {
    background: rgba(201,146,42,0.08);
    border: 1px solid #3a2a10;
    border-radius: 4px;
    padding: 0.2rem 0.6rem;
    font-size: 0.78rem;
    color: #a08050;
    font-family: 'IM Fell English', serif;
}
.stat-pill span {
    color: #c9922a;
    font-weight: bold;
}

/* Result box */
.result-victory {
    background: linear-gradient(135deg, #0d1a0d 0%, #0f1a08 100%);
    border: 2px solid #2d5a1b;
    border-radius: 8px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 0 40px rgba(45,90,27,0.3), inset 0 0 40px rgba(45,90,27,0.05);
    margin-top: 1rem;
}
.result-defeat {
    background: linear-gradient(135deg, #1a0d0d 0%, #1a0808 100%);
    border: 2px solid #5a1b1b;
    border-radius: 8px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 0 40px rgba(90,27,27,0.3), inset 0 0 40px rgba(90,27,27,0.05);
    margin-top: 1rem;
}
.result-label {
    font-family: 'Cinzel', serif;
    font-size: 2.5rem;
    font-weight: 900;
    letter-spacing: 0.2em;
    text-shadow: 0 0 30px currentColor;
    margin-bottom: 0.5rem;
}
.result-victory .result-label { color: #4caf50; }
.result-defeat  .result-label { color: #e53935; }

.result-sub {
    font-family: 'IM Fell English', serif;
    font-style: italic;
    font-size: 1rem;
    color: #8a7055;
    margin-bottom: 1.2rem;
}

/* Confidence bars */
.conf-bar-wrap {
    margin: 0.8rem 0;
}
.conf-label {
    font-family: 'Cinzel', serif;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
    color: #8a7055;
}
.conf-bar-bg {
    background: #1a1208;
    border-radius: 3px;
    height: 10px;
    width: 100%;
    border: 1px solid #3a2a10;
    overflow: hidden;
}
.conf-bar-fill-v {
    height: 100%;
    background: linear-gradient(90deg, #2d5a1b, #4caf50);
    border-radius: 3px;
    transition: width 0.5s ease;
}
.conf-bar-fill-d {
    height: 100%;
    background: linear-gradient(90deg, #5a1b1b, #e53935);
    border-radius: 3px;
    transition: width 0.5s ease;
}
.conf-pct {
    font-family: 'Cinzel', serif;
    font-size: 0.85rem;
    color: #c9922a;
    margin-top: 0.2rem;
}

/* Power delta */
.delta-block {
    margin-top: 1rem;
    font-family: 'IM Fell English', serif;
    font-size: 0.9rem;
    color: #8a7055;
}
.delta-val { color: #c9922a; font-size: 1.1rem; }

/* Button */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #3a2000, #5a3500) !important;
    border: 1px solid #c9922a !important;
    color: #c9922a !important;
    font-family: 'Cinzel', serif !important;
    font-size: 1rem !important;
    letter-spacing: 0.15em !important;
    padding: 0.75rem !important;
    border-radius: 4px !important;
    cursor: pointer !important;
    text-transform: uppercase;
    transition: all 0.2s ease !important;
    box-shadow: 0 0 20px rgba(201,146,42,0.1) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #5a3500, #7a4800) !important;
    box-shadow: 0 0 30px rgba(201,146,42,0.3) !important;
    transform: translateY(-1px) !important;
    cursor: pointer !important;
}

/* Section headers */
.section-header {
    font-family: 'Cinzel', serif;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #5a4020;
    border-bottom: 1px solid #2a1a08;
    padding-bottom: 0.3rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────

CSV_PATH = "lotr_battles_1000.csv"

if not os.path.exists(CSV_PATH):
    st.error(f"CSV file '{CSV_PATH}' not found. Place it in the same directory as this script.")
    st.stop()

model, encoders, feature_cols, model_acc = load_and_train(CSV_PATH)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.markdown("""
<div class="title-block">
    <h1>⚔ LOTR Battle Predictor</h1>
    <div class="subtitle">Foretell the fate of Middle-earth's warriors</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# INPUTS
# ─────────────────────────────────────────────

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="section-header">The Hero</div>', unsafe_allow_html=True)
    hero_name = st.selectbox("Hero", list(HEROES.keys()), key="hero")
    weapon    = st.selectbox("Weapon", WEAPONS, key="weapon")

    h     = HEROES[hero_name]
    bonus = WEAPON_BONUSES[hero_name][weapon]
    bonus_color = "#4caf50" if bonus >= 5 else ("#c9922a" if bonus >= 0 else "#e53935")
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-pill">STR <span>{h['strength']}</span></div>
        <div class="stat-pill">AGI <span>{h['agility']}</span></div>
        <div class="stat-pill">WPN BONUS <span style="color:{bonus_color}">{bonus:+d}</span></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="vs-badge">— VS —</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-header">The Enemy</div>', unsafe_allow_html=True)
    enemy_name   = st.selectbox("Enemy", list(ENEMIES.keys()), key="enemy")
    enemy_number = st.slider("Number of Enemies", min_value=1, max_value=20, value=3, key="enum")

    e = ENEMIES[enemy_name]
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-pill">STR <span>{e['strength']}</span></div>
        <div class="stat-pill">AGI <span>{e['agility']}</span></div>
        <div class="stat-pill">COUNT <span>{enemy_number}</span></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────

if st.button("Foretell the Battle"):
    pred, proba, hero_pwr, enemy_threat, delta, wpn_bonus = predict(
        model, encoders, feature_cols,
        hero_name, weapon, enemy_name, enemy_number
    )

    victory     = pred == 1
    label       = "VICTORY" if victory else "DEFEAT"
    border_col  = "#2d5a1b" if victory else "#5a1b1b"
    bg_col      = "#0d1a0d" if victory else "#1a0d0d"
    label_col   = "#4caf50" if victory else "#e53935"
    flavour     = (
        f"{hero_name} overcomes the {'horde' if enemy_number > 5 else 'foe'}."
        if victory else
        f"The {'horde overwhelms' if enemy_number > 5 else 'enemy defeats'} {hero_name}."
    )
    delta_color = "#4caf50" if delta >= 0 else "#e53935"
    v_pct       = proba[1] * 100
    d_pct       = proba[0] * 100

    st.markdown(
        f'<div style="background:{bg_col};border:2px solid {border_col};border-radius:8px;'
        f'padding:2rem;text-align:center;box-shadow:0 0 40px rgba(0,0,0,0.4);margin-top:1rem;">',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div style="font-family:Cinzel,serif;font-size:2.5rem;font-weight:900;'
        f'letter-spacing:0.2em;color:{label_col};text-shadow:0 0 30px {label_col};'
        f'margin-bottom:0.3rem;">{label}</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div style="font-family:IM Fell English,serif;font-style:italic;'
        f'color:#8a7055;margin-bottom:1.2rem;">{flavour}</div>',
        unsafe_allow_html=True
    )
    st.caption("VICTORY PROBABILITY")
    st.progress(proba[1])
    st.markdown(
        f'<div style="font-family:Cinzel,serif;color:#c9922a;font-size:0.85rem;margin-bottom:0.8rem;">{v_pct:.1f}%</div>',
        unsafe_allow_html=True
    )
    st.caption("DEFEAT PROBABILITY")
    st.progress(proba[0])
    st.markdown(
        f'<div style="font-family:Cinzel,serif;color:#c9922a;font-size:0.85rem;margin-bottom:1rem;">{d_pct:.1f}%</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div style="font-family:IM Fell English,serif;color:#8a7055;font-size:0.95rem;">'
        f'Hero Power <span style="color:#c9922a;font-size:1.1rem;">{hero_pwr:.1f}</span>'
        f' &nbsp;&middot;&nbsp; '
        f'Enemy Threat <span style="color:#c9922a;font-size:1.1rem;">{enemy_threat:.1f}</span>'
        f' &nbsp;&middot;&nbsp; '
        f'&#916; <span style="color:{delta_color};font-size:1.1rem;">{delta:+.1f}</span>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f"""
<div style="text-align:center; margin-top: 2rem;">
    <span class="model-badge">Random Forest · {model.n_estimators} trees · {model_acc:.1%} test accuracy</span>
</div>
""", unsafe_allow_html=True)
