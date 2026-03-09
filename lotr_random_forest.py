"""
LOTR Battle Outcome Predictor — Random Forest Classifier
=========================================================
Train a Random Forest to predict battle outcomes 
between Middle-earth characters.

CSV Schema:
    HeroName, HeroStrength, HeroAgility, HeroWeaponName,
    HeroWeaponBonus, EnemyName, EnemyStrength, EnemyAgility,
    EnemyNumber, Outcome
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1.  LOAD DATA
# ─────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} battle records from '{path}'")
    print(f"Columns : {list(df.columns)}")
    print(f"Outcomes: {df['Outcome'].value_counts().to_dict()} (1=Victory, 0=Defeat)\n")
    return df


# ─────────────────────────────────────────────
# 2.  FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(df: pd.DataFrame):
    """
    Encode categoricals and create derived numeric features.
    Returns X (features), y (labels), and the encoders.
    """
    df = df.copy()
    encoders = {}

    # Label-encode character / weapon names
    for col in ["HeroName", "HeroWeaponName", "EnemyName"]:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Derived power scores
    # Hero power includes weapon bonus; enemy power is scaled by EnemyNumber
    df["HeroPower"]        = df["HeroStrength"] * 0.6 + df["HeroAgility"] * 0.4 + df["HeroWeaponBonus"]
    df["EnemyPower"]       = df["EnemyStrength"] * 0.6 + df["EnemyAgility"] * 0.4
    df["EnemyTotalThreat"] = df["EnemyPower"] * df["EnemyNumber"]   # combined enemy threat
    df["PowerDelta"]       = df["HeroPower"] - df["EnemyTotalThreat"]

    feature_cols = [
        "HeroName_enc", "HeroStrength", "HeroAgility",
        "HeroWeaponName_enc", "HeroWeaponBonus",
        "EnemyName_enc", "EnemyStrength", "EnemyAgility", "EnemyNumber",                                  
        "HeroPower", "EnemyPower", "EnemyTotalThreat", "PowerDelta",
    ]

    X = df[feature_cols]
    y = df["Outcome"]
    return X, y, encoders, feature_cols


# ─────────────────────────────────────────────
# 3.  TRAIN
# ─────────────────────────────────────────────
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Tune n_estimators via cross-validation
    print("── Tuning number of trees ────────────────────────")
    best_n, best_cv = 100, 0
    for n in [50, 100, 200, 300, 500]:
        clf = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
        cv  = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy").mean()
        print(f"   n_estimators={n:<4}  CV accuracy = {cv:.3f}")
        if cv > best_cv:
            best_cv, best_n = cv, n

    print(f"\n Best n_estimators = {best_n}  (CV accuracy = {best_cv:.3f})\n")

    model = RandomForestClassifier(
        n_estimators=best_n,
        max_features="sqrt",
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    print(f"── Test Set Results ──────────────────────────────")
    print(f"   Accuracy : {acc:.1%}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Defeat','Victory'])}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")

    return model, X_test, y_test


# ─────────────────────────────────────────────
# 4.  FEATURE IMPORTANCE
# ─────────────────────────────────────────────
def show_feature_importance(model, feature_cols):
    importance = sorted(
        zip(feature_cols, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    print("── Feature Importances (averaged across all trees) ─")
    for feat, imp in importance:
        bar = "█" * int(imp * 40)
        print(f"  {feat:<25} {imp:.4f}  {bar}")
    print()


# ─────────────────────────────────────────────
# 5.  PREDICT NEW BATTLE
# ─────────────────────────────────────────────
def predict_battle(
    model, encoders, feature_cols,
    hero_name, hero_str, hero_agi, hero_wpn, hero_bonus,
    enemy_name, enemy_str, enemy_agi, enemy_number
):
    """
    Predict the outcome of a single battle (unknown => index 0).
    The Random Forest returns a probability averaged across all trees.
    """
    def safe_encode(le, value):
        if value in le.classes_:
            return le.transform([value])[0]
        print(f"  [!] '{value}' not seen during training — using default encoding.")
        return 0

    hero_power         = hero_str  * 0.6 + hero_agi  * 0.4 + hero_bonus
    enemy_power        = enemy_str * 0.6 + enemy_agi * 0.4
    enemy_total_threat = enemy_power * enemy_number
    power_delta        = hero_power - enemy_total_threat

    row = pd.DataFrame([{
        "HeroName_enc":       safe_encode(encoders["HeroName"],      hero_name),
        "HeroStrength":       hero_str,
        "HeroAgility":        hero_agi,
        "HeroWeaponName_enc": safe_encode(encoders["HeroWeaponName"], hero_wpn),
        "HeroWeaponBonus":    hero_bonus,
        "EnemyName_enc":      safe_encode(encoders["EnemyName"],      enemy_name),
        "EnemyStrength":      enemy_str,
        "EnemyAgility":       enemy_agi,
        "EnemyNumber":        enemy_number,
        "HeroPower":          hero_power,
        "EnemyPower":         enemy_power,
        "EnemyTotalThreat":   enemy_total_threat,
        "PowerDelta":         power_delta,
    }])

    prediction  = model.predict(row)[0]
    proba       = model.predict_proba(row)[0]   # [P(defeat), P(victory)]
    outcome_str = "VICTORY" if prediction == 1 else "DEFEAT"

    print(f"\n── Battle Prediction ─────────────────────────────────────")
    print(f"  {hero_name} (Power={hero_power:.1f})  vs  {enemy_number}x {enemy_name} (Threat={enemy_total_threat:.1f})")
    print(f"  Power Delta : {power_delta:+.1f}")
    print(f"  Prediction  : {outcome_str}")
    print(f"  Confidence  : Victory {proba[1]:.1%} | Defeat {proba[0]:.1%}  (voted by {model.n_estimators} trees)")
    return prediction, proba


# ─────────────────────────────────────────────
# 6.  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    CSV_PATH = "lotr_battles_300.csv"

    df = load_data(CSV_PATH)
    X, y, encoders, feature_cols = engineer_features(df)
    model, X_test, y_test = train_model(X, y)
    show_feature_importance(model, feature_cols)

    # Example: Gimli vs 3 Trolls
    predict_battle(
        model, encoders, feature_cols,
        hero_name="Frodo",  hero_str=3,  hero_agi=5,  hero_wpn="DwarvenAxe", hero_bonus=-1,
        enemy_name="Troll", enemy_str=10, enemy_agi=2, enemy_number=1
     )


    predict_battle(
        model, encoders, feature_cols,
        hero_name="Gimli",  hero_str=8,  hero_agi=5,  hero_wpn="DwarvenAxe", hero_bonus=7,
        enemy_name="Troll", enemy_str=10, enemy_agi=2, enemy_number=3
    )

    predict_battle(
        model, encoders, feature_cols,
        hero_name="Aragorn",  hero_str=9,  hero_agi=7,  hero_wpn="Anduril", hero_bonus=9,
        enemy_name="Troll", enemy_str=10, enemy_agi=2, enemy_number=3
    )
