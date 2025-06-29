from flask import Flask, render_template_string, request
import os
import openai
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Configure l'API OpenAI
from openai import OpenAI
import os

client = OpenAI(api_key="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")


app = Flask(__name__)

def generer_donnees(n=100, seed=42):
    rng = np.random.default_rng(seed)
    temp = rng.normal(37, 0.5, n)
    fc   = rng.integers(60, 100, n)
    pa   = rng.integers(100, 130, n)
    urgence = ((temp > 38.2) | (fc > 95)).astype(int)
    return pd.DataFrame({
        'température'         : temp,
        'fréquence_cardiaque' : fc,
        'pression_arterielle' : pa,
        'urgence'             : urgence
    })

def entrainer_modele(data):
    X = data.drop('urgence', axis=1)
    y = data['urgence']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    modele = RandomForestClassifier()
    modele.fit(X_train, y_train)
    return modele

def predire_patient(modele, temp, fc, pa):
    patient = pd.DataFrame(
        [[temp, fc, pa]],
        columns=['température','fréquence_cardiaque','pression_arterielle']
    )
    return int(modele.predict(patient)[0])

def generer_rapport_ia(temp, fc, pa, verdict):
    """
    Appelle OpenAI pour générer un court rapport médical
    """
    prompt = (
        f"J'ai un patient avec :\n"
        f"- Température : {temp:.1f}°C\n"
        f"- Fréquence cardiaque : {fc} bpm\n"
        f"- Pression artérielle : {pa} mmHg\n"
        f"Le diagnostic IA est : {'urgence' if verdict==1 else 'stable'}.\n"
        "Rédige un rapport médical court en français, style professionnel."
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        max_tokens=350,
        temperature=0.3
    )
    return resp.choices[0].message.content.strip()

@app.route("/", methods=["GET","POST"])
def accueil():
    # 1) Génération et entraînement
    data = generer_donnees()
    modele = entrainer_modele(data)

    # 2) Stats pour affichage
    repartition = data["urgence"].value_counts().to_string()
    perf_str    = classification_report(
        data["urgence"],
        modele.predict(data.drop('urgence',axis=1)),
        target_names=["stable","urgence"]
    )

    resultat, rapport = None, None
    if request.method == "POST":
        t = float(request.form["temp"])
        c = int(request.form["fc"])
        p = int(request.form["pa"])
        verdict = predire_patient(modele, t, c, p)
        resultat = "🆘 Urgence !" if verdict==1 else "✅ Stable"
        rapport  = generer_rapport_ia(t, c, p, verdict)

    # 3) Template HTML
    html = """
    <html>
      <head><title>IA Médicale</title></head>
      <body style="font-family:Arial,sans-serif;max-width:600px;margin:auto">
        <h1>IA Médicale – Démo</h1>
        <h2>Répartition (0=stable,1=urgence)</h2>
        <pre>{{ repartition }}</pre>
        <h2>Performance</h2>
        <pre>{{ perf }}</pre>
        <h2>Test patient</h2>
        <form method="post">
          Température (°C): <input name="temp" step="0.1" required><br>
          Fréq. cardiaque:   <input name="fc" required><br>
          Pression (mmHg):   <input name="pa" required><br><br>
          <button type="submit">Prédire</button>
        </form>
        {% if resultat %}
          <h3>Verdict : {{ resultat }}</h3>
          <h3>Rapport généré par l'IA :</h3>
          <pre>{{ rapport }}</pre>
        {% endif %}
      </body>
    </html>
    """

    return render_template_string(
        html,
        repartition=repartition,
        perf=perf_str,
        resultat=resultat,
        rapport=rapport
    )

if __name__ == "__main__":
    app.run(debug=True)
