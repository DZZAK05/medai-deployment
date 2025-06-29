from flask import Flask, render_template_string, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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

@app.route("/", methods=["GET","POST"])
def accueil():
    # 1) Génération et entraînement
    data = generer_donnees()
    modele = entrainer_modele(data)

    # 2) Calculs pour l'affichage
    repartition = data["urgence"].value_counts().to_string()
    perf_str    = classification_report(
        data["urgence"],
        modele.predict(data.drop('urgence', axis=1)),
        target_names=["stable", "urgence"]
    )

    # 3) Traitement du formulaire
    resultat = None
    if request.method == "POST":
        t = float(request.form["temp"])
        c = int(request.form["fc"])
        p = int(request.form["pa"])
        pred = predire_patient(modele, t, c, p)
        resultat = "🆘 Urgence !" if pred == 1 else "✅ Stable"

    # 4) Template HTML
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
          <h3>Résultat : {{ resultat }}</h3>
        {% endif %}
      </body>
    </html>
    """

    return render_template_string(
        html,
        repartition = repartition,
        perf        = perf_str,
        resultat    = resultat
    )

if __name__ == "__main__":
    app.run(debug=True)

