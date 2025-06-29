import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from sqlite3 import Connection
from openai import OpenAI
import streamlit as st

st.write("🔑 st.secrets:", st.secrets)



# ── 0) Instanciation OpenAI via st.secrets ──────────────────────────────
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ── 1) Fonctions de base de données ────────────────────────────────────

DB_PATH = "data.db"

def get_conn() -> Connection:
    """Ouvre (et crée si besoin) la base SQLite."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn

def init_db():
    """Crée la table 'tests' si elle n'existe pas encore."""
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            temperature REAL,
            frequence_cardiaque INTEGER,
            pression_arterielle INTEGER,
            verdict INTEGER,
            rapport TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_test(temp, fc, pa, verdict, rapport):
    """Insère une nouvelle ligne de test."""
    conn = get_conn()
    conn.execute(
        "INSERT INTO tests (temperature, frequence_cardiaque, pression_arterielle, verdict, rapport) VALUES (?, ?, ?, ?, ?)",
        (temp, fc, pa, verdict, rapport)
    )
    conn.commit()
    conn.close()

def load_history() -> pd.DataFrame:
    """Récupère toute la table sous forme de DataFrame, triée par date."""
    conn = get_conn()
    df = pd.read_sql("SELECT * FROM tests ORDER BY timestamp DESC", conn)
    conn.close()
    return df

# ── 2) Ton code d’IA existant ────────────────────────────────────────────

def generer_donnees(n=100, seed=42):
    rng = np.random.default_rng(seed)
    temp = rng.normal(37, 0.5, n)
    fc   = rng.integers(60, 100, n)
    pa   = rng.integers(100, 130, n)
    urgence = ((temp > 38.2) | (fc > 95)).astype(int)
    return pd.DataFrame({
        'température': temp,
        'fréquence_cardiaque': fc,
        'pression_arterielle': pa,
        'urgence': urgence
    })

def entrainer_modele(data):
    X = data.drop('urgence', axis=1)
    y = data['urgence']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modele = RandomForestClassifier()
    modele.fit(X_train, y_train)
    return modele

def predire_patient(modele, temp, fc, pa):
    patient = pd.DataFrame([[temp, fc, pa]], columns=['température','fréquence_cardiaque','pression_arterielle'])
    return int(modele.predict(patient)[0])

def generer_rapport_ia(temp, fc, pa, verdict) -> str:
    prompt = (
        f"Rapport médical : température {temp}°C, FC {fc} bpm, PA {pa} mmHg, verdict = {'urgence' if verdict else 'stable'}.\n"
        "Rédige un rapport succinct avec diagnostic et recommandations."
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        max_tokens=300
    )
    return resp.choices[0].message.content

# ── 3) Application principale ────────────────────────────────────────────

def main():
    st.title("IA Médicale – Démo")
    init_db()  # crée la table si besoin

    # Génération & entraînement
    data = generer_donnees()
    modele = entrainer_modele(data)

    st.sidebar.header("Test patient")
    temp = st.sidebar.number_input("Température (°C)", 34.0, 42.0, 37.0, 0.1)
    fc   = st.sidebar.number_input("Fréq. cardiaque (bpm)", 40, 180, 75, 1)
    pa   = st.sidebar.number_input("Pression (mmHg)", 80, 200, 120, 1)

    if st.sidebar.button("Prédire & Générer rapport"):
        verdict = predire_patient(modele, temp, fc, pa)
        rapport = generer_rapport_ia(temp, fc, pa, verdict)
        st.write("**Verdict :**", "🆘 Urgence" if verdict else "✅ Stable")
        st.write("**Rapport généré :**")
        st.markdown(rapport)

        # Log dans la base
        log_test(temp, fc, pa, verdict, rapport)

    # Onglet Historique
    st.header("📚 Historique des tests")
    df_hist = load_history()
    st.dataframe(df_hist)

if __name__ == "__main__":
    main()
