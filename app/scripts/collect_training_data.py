#!/usr/bin/env python3
# scripts/collect_training_data.py
import asyncio
import sys
from pathlib import Path

# Ajouter le chemin du projet
sys.path.append(str(Path(__file__).parent.parent))

from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.services.weather_service import WeatherService
from app.services.satellite_service import SatelliteService
from app.services.data_collector import DataCollector

async def collect_all_wilayas():
    """Collecter les données pour toutes les wilayas"""
    db = SessionLocal()
    
    try:
        # Initialiser les services
        weather_service = WeatherService()
        satellite_service = SatelliteService()
        collector = DataCollector(weather_service, satellite_service)
        
        # Liste des codes de wilayas (exemple, à compléter)
        wilayas_codes = [
            '16',  # Alger
            '31',  # Oran
            '19',  # Sétif
            '7',   # Biskra
            '1',   # Adrar
            '10',  # Bouira
            '44',  # Aïn Defla
            '38'   # Tissemsilt
        ]
        
        results = []
        
        for wilaya_code in wilayas_codes:
            print(f"\n{'='*60}")
            print(f"Traitement de la wilaya {wilaya_code}")
            print('='*60)
            
            try:
                result = await collector.collect_historical_data_for_wilaya(
                    wilaya_code=wilaya_code,
                    start_year=2018,
                    end_year=2023,
                    db=db
                )
                results.append(result)
                print(f"✓ Données collectées : {result['records_collected']} enregistrements")
                
            except Exception as e:
                print(f"✗ Erreur pour {wilaya_code}: {e}")
                continue
        
        # Résumé
        print(f"\n{'='*60}")
        print("RÉSUMÉ DE LA COLLECTE")
        print('='*60)
        
        total_records = sum(r['records_collected'] for r in results)
        total_saved = sum(r['records_saved'] for r in results)
        
        print(f"Wilayas traitées : {len(results)}/{len(wilayas_codes)}")
        print(f"Enregistrements collectés : {total_records}")
        print(f"Enregistrements sauvegardés : {total_saved}")
        
        # Exporter un résumé
        import json
        with open('data_collection_summary.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        print("\n✓ Collecte terminée !")
        
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(collect_all_wilayas())