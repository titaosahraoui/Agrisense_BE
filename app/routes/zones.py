from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session
from typing import List
import json
from ..database import get_db
from .. import models

router = APIRouter(prefix="/zones", tags=["Zones"])

import math
import os

# Globals to cache GeoJSON data
GEOJSON_DATA = {}
DATA_LOADED = False

def load_geojson_data():
    global GEOJSON_DATA, DATA_LOADED
    if DATA_LOADED:
        return

    try:
        if os.path.exists('gadm41_DZA_1.json'):
            with open('gadm41_DZA_1.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for feature in data.get('features', []):
                props = feature.get('properties', {})
                try:
                    if 'CC_1' in props:
                        code = int(props['CC_1'])
                        GEOJSON_DATA[code] = feature
                except ValueError:
                    continue
        else:
            print("Warning: gadm41_DZA_1.json not found")
    except Exception as e:
        print(f"Error loading GeoJSON: {e}")
    finally:
        DATA_LOADED = True

def calculate_area_from_geometry(geometry):
    """Estimate area in km2 from GeoJSON geometry on a sphere (approximate)"""
    if not geometry or 'coordinates' not in geometry:
        return None
    
    coordinates = geometry['coordinates']
    type_ = geometry['type']
    
    polygons = []
    if type_ == 'Polygon':
        polygons = [coordinates]
    elif type_ == 'MultiPolygon':
        polygons = coordinates
        
    total_area = 0.0
    
    for poly in polygons:
        # Safely handle empty polygons
        if not poly: continue
        
        # Exterior ring is usually the first element in GeoJSON Polygon
        ring = poly[0]
        if len(ring) < 3: continue
        
        # Approximate conversion relying on center latitude
        lats = [p[1] for p in ring]
        center_lat = sum(lats) / len(lats)
        
        # Constants for degree to km conversion
        # Lat: 1 deg ~= 110.574 km
        # Lon: 1 deg ~= 111.320 * cos(lat) km
        km_per_deg_lat = 110.574
        km_per_deg_lon = 111.320 * math.cos(math.radians(center_lat))
        
        area = 0.0
        for i in range(len(ring)):
            p1 = ring[i]
            p2 = ring[(i + 1) % len(ring)]
            
            x1 = p1[0] * km_per_deg_lon
            y1 = p1[1] * km_per_deg_lat
            x2 = p2[0] * km_per_deg_lon
            y2 = p2[1] * km_per_deg_lat
            
            area += (x1 * y2 - x2 * y1)
            
        total_area += abs(area) * 0.5
        
    return round(total_area, 2)

def generate_fallback_geometry(lat, lon, radius_km=50):
    """Generate a hexagonal polygon approximating the region"""
    # Approximation: 1 degree latitude ~= 111 km
    # 1 degree longitude ~= 111 * cos(lat) km
    
    if lat is None or lon is None:
        return None

    d_lat = radius_km / 111.0
    d_lon = radius_km / (111.0 * math.cos(math.radians(lat)))
    
    coordinates = []
    for i in range(7):
        angle = math.radians(60 * i)
        # simplistic conversion
        p_lon = lon + d_lon * math.cos(angle)
        p_lat = lat + d_lat * math.sin(angle)
        coordinates.append([p_lon, p_lat])
        
    return {
        "type": "Polygon",
        "coordinates": [coordinates]
    }

@router.get("/wilayas")
async def get_all_wilayas(db: Session = Depends(get_db)):
    """Get all wilayas with their information"""
    
    try:
        load_geojson_data()
        
        # Load from JSON file first (basic info fallback)
        try:
            with open('algeria_cordinates.json', 'r', encoding='utf-8') as f:
                coordinates_data = json.load(f)
        except:
            coordinates_data = {}
        
        # Get wilayas from database
        regions = db.query(models.Region).all()
        
        wilayas = []
        
        # 1. Process Database Regions
        for region in regions:
            wilaya_key = f"DZ{str(region.wilaya_code).zfill(2)}"
            coord_data = coordinates_data.get(wilaya_key, {})
            
            lat = region.centroid_lat
            lon = region.centroid_lon
            
            # Start with DB geometry/area
            geometry = region.geometry
            area_km2 = region.area_km2
            
            # Check GADM data if missing geometry
            if not geometry and region.wilaya_code in GEOJSON_DATA:
                feature = GEOJSON_DATA[region.wilaya_code]
                geometry = feature['geometry']
                # If area missing, calculate it
                if not area_km2:
                    area_km2 = calculate_area_from_geometry(geometry)
            
            # Fallback 2: Generate geometry if still missing
            if not geometry and lat and lon:
                geometry = generate_fallback_geometry(lat, lon)
                if not area_km2:
                    area_km2 = 5000.0
            
            wilayas.append({
                "wilaya_code": region.wilaya_code,
                "name": region.name,
                "lat": lat,
                "lon": lon,
                "area_km2": area_km2,
                "geometry": geometry,
                "has_data": True
            })
        
        # 2. Add wilayas from Coordinates JSON (not in DB)
        for code, data in coordinates_data.items():
            wilaya_code = int(code[2:])
            
            if not any(w["wilaya_code"] == wilaya_code for w in wilayas):
                lat = data.get("lat")
                lon = data.get("lon")
                
                geometry = None
                area_km2 = None
                
                # Check GADM
                if wilaya_code in GEOJSON_DATA:
                    feature = GEOJSON_DATA[wilaya_code]
                    geometry = feature['geometry']
                    area_km2 = calculate_area_from_geometry(geometry)
                
                # Fallback gen
                if not geometry and lat and lon:
                    geometry = generate_fallback_geometry(lat, lon)
                    area_km2 = 5000.0
                
                wilayas.append({
                    "wilaya_code": wilaya_code,
                    "name": data.get("name"),
                    "lat": lat,
                    "lon": lon,
                    "area_km2": area_km2,
                    "geometry": geometry,
                    "has_data": False
                })
        
        # Sort by wilaya code
        wilayas.sort(key=lambda x: x["wilaya_code"])
        
        return {
            "wilayas": wilayas,
            "count": len(wilayas),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        # Log error in production
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching wilayas: {str(e)}")

@router.get("/wilayas/{wilaya_code}")
async def get_wilaya_details(wilaya_code: int, db: Session = Depends(get_db)):
    """Get detailed information for a specific wilaya"""
    
    try:
        load_geojson_data()
        
        # Get from database
        region = db.query(models.Region).filter(
            models.Region.wilaya_code == wilaya_code
        ).first()
        
        # Load coordinates from JSON
        try:
            with open('algeria_cordinates.json', 'r', encoding='utf-8') as f:
                coordinates_data = json.load(f)
        except:
            coordinates_data = {}
        
        wilaya_key = f"DZ{str(wilaya_code).zfill(2)}"
        coord_data = coordinates_data.get(wilaya_key, {})
        
        if not region and not coord_data and wilaya_code not in GEOJSON_DATA:
            raise HTTPException(status_code=404, detail=f"Wilaya {wilaya_code} not found")
        
        # Determine basic info
        name = region.name if region else coord_data.get("name", f"Wilaya {wilaya_code}")
        lat = region.centroid_lat if region else coord_data.get("lat")
        lon = region.centroid_lon if region else coord_data.get("lon")
        
        area_km2 = region.area_km2 if region else None
        geometry = region.geometry if region else None
        
        # GADM Logic
        if wilaya_code in GEOJSON_DATA:
            feature = GEOJSON_DATA[wilaya_code]
            if not geometry:
                geometry = feature['geometry']
            if not area_km2:
                # Calculate from the (potentially new) geometry
                area_km2 = calculate_area_from_geometry(geometry)
                
        # Last resort fallback
        if not area_km2:
            area_km2 = 5000.0 # Approximate average
            
        if not geometry:
            geometry = generate_fallback_geometry(lat, lon)
        
        # Get statistics for this wilaya
        stats = db.query(
            func.count(models.TrainingData.id).label('record_count'),
            func.min(models.TrainingData.date).label('first_date'),
            func.max(models.TrainingData.date).label('last_date'),
            func.avg(models.TrainingData.stress_score).label('avg_stress'),
            func.avg(models.TrainingData.ndvi).label('avg_ndvi'),
            func.avg(models.TrainingData.precipitation).label('avg_precipitation')
        ).filter(
            models.TrainingData.wilaya_code == wilaya_code
        ).first()
        
        # Get latest stress level
        latest_data = db.query(models.TrainingData).filter(
            models.TrainingData.wilaya_code == wilaya_code
        ).order_by(models.TrainingData.date.desc()).first()
        
        wilaya_info = {
            "wilaya_code": wilaya_code,
            "name": name,
            "lat": lat,
            "lon": lon,
            "area_km2": area_km2,
            "geometry": geometry,
            "in_database": region is not None,
            "statistics": {
                "record_count": stats.record_count or 0,
                "first_date": stats.first_date.isoformat() if stats.first_date else None,
                "last_date": stats.last_date.isoformat() if stats.last_date else None,
                "avg_stress": float(stats.avg_stress) if stats.avg_stress else None,
                "avg_ndvi": float(stats.avg_ndvi) if stats.avg_ndvi else None,
                "avg_precipitation": float(stats.avg_precipitation) if stats.avg_precipitation else None,
                "current_stress_level": latest_data.stress_level if latest_data else None,
                "current_stress_score": float(latest_data.stress_score) if latest_data and latest_data.stress_score else None
            }
        }
        
        return wilaya_info
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching wilaya details: {str(e)}")