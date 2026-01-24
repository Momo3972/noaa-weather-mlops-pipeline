import pandas as pd
import os
import json
# Importation sécurisée
try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
except ImportError:
    print("❌ Erreur : Evidently n'est toujours pas correctement installé.")

def run_expert_monitoring():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "raw_weather.csv")
    output_dir = os.path.join(base_dir, "monitoring")
    os.makedirs(output_dir, exist_ok=True)

    # Chargement des données
    df = pd.read_csv(data_path)
    
    # Simulation Expert : Comparaison de deux périodes
    reference = df.iloc[:100] # Données passées
    current = df.iloc[-100:]  # Données récentes
    
    # Génération du rapport
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    
    # Sauvegarde multiformat
    report.save_html(os.path.join(output_dir, "drift_report.html"))
    
    # Export JSON pour automatisation (Objectif Expert)
    metrics = report.as_dict()
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"✅ Dashboard généré dans : {output_dir}/drift_report.html")

if __name__ == "__main__":
    run_expert_monitoring()