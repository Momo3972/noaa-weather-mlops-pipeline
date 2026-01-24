import pandas as pd
import os

def ingest_weather_data():
    # Utilisation d'un dataset public stable pour éviter les erreurs d'API NOAA
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
    
    print("⏳ Récupération des données météo...")
    try:
        df = pd.read_csv(url)
        # Renommer pour correspondre à notre futur modèle
        df.columns = ['date', 'temp']
        
        # Sauvegarde locale pour le pipeline
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/raw_weather.csv', index=False)
        
        print("✅ Succès ! Données sauvegardées dans data/raw_weather.csv")
        print(df.head())
    except Exception as e:
        print(f"❌ Erreur : {e}")

if __name__ == "__main__":
    ingest_weather_data()