from typing import Dict, List, Optional
import numpy as np

class StressAnalysisService:
    def __init__(self):
        # Adjusted thresholds for Algeria climate
        self.thresholds = {
            'ndvi': {'low': 0.3, 'medium': 0.5, 'high': 0.7},
            'temperature': {'low': 20, 'medium': 25, 'high': 30},  # Adjusted for Algeria
            'precipitation': {'low': 5, 'medium': 2, 'high': 0.5},  # mm/day (Algeria is dry)
            'evapotranspiration': {'low': 3, 'medium': 5, 'high': 7},
            'humidity': {'low': 40, 'medium': 60, 'high': 75}  # Algeria typically has lower humidity
        }
    
    def calculate_stress_score(self, indicators: Dict) -> float:
        """
        Calculate water stress score (0-100)
        0 = no stress, 100 = extreme stress
        
        Adjusted weights for Algeria's climate:
        - Precipitation is most important (Algeria is dry)
        - Temperature is important (affects evaporation)
        - Humidity is less important (generally low in Algeria)
        """
        # Define weights based on available indicators
        available_indicators = [ind for ind in indicators.keys() if indicators[ind] is not None]
        
        weights = {
            'ndvi': 0.3 if 'ndvi' in available_indicators else 0,
            'temperature': 0.25 if 'temperature' in available_indicators else 0,
            'precipitation': 0.35 if 'precipitation' in available_indicators else 0,
            'evapotranspiration': 0.15 if 'evapotranspiration' in available_indicators else 0,
            'humidity': 0.05 if 'humidity' in available_indicators else 0,
            'avg_temperature': 0.2 if 'avg_temperature' in available_indicators else 0,
            'total_precipitation': 0.3 if 'total_precipitation' in available_indicators else 0,
            'avg_humidity': 0.05 if 'avg_humidity' in available_indicators else 0
        }
        
        # If we have average/total values, use those instead
        if 'avg_temperature' in available_indicators:
            weights['temperature'] = 0
        if 'total_precipitation' in available_indicators:
            weights['precipitation'] = 0
        if 'avg_humidity' in available_indicators:
            weights['humidity'] = 0
        
        scores = []
        
        for indicator, value in indicators.items():
            if indicator in weights and value is not None and weights[indicator] > 0:
                score = self._calculate_indicator_score(indicator, value)
                scores.append(score * weights[indicator])
        
        if not scores:
            return 50  # Default moderate stress if no data
        
        total_weight = sum(weights.values())
        if total_weight == 0:
            return 50
        
        total_score = sum(scores) / total_weight
        return min(max(total_score * 100, 0), 100)
    
    def _calculate_indicator_score(self, indicator: str, value: float) -> float:
        """Calculate score for individual indicator (0-1)"""
        thresholds = self.thresholds.get(indicator, {})
        
        # Handle special cases
        if indicator in ['avg_temperature', 'temperature']:
            # Higher temperature = more stress
            if value >= thresholds.get('high', 30):
                return 1.0
            elif value >= thresholds.get('medium', 25):
                return 0.7
            elif value >= thresholds.get('low', 20):
                return 0.4
            else:
                return 0.1
        
        elif indicator in ['total_precipitation', 'precipitation']:
            # For Algeria: Lower precipitation = much more stress
            if value <= thresholds.get('high', 0.5):  # Less than 0.5 mm/day
                return 1.0
            elif value <= thresholds.get('medium', 2):  # Less than 2 mm/day
                return 0.8
            elif value <= thresholds.get('low', 5):  # Less than 5 mm/day
                return 0.5
            else:
                return 0.2
        
        elif indicator in ['avg_humidity', 'humidity']:
            # Lower humidity = more stress
            if value <= thresholds.get('low', 40):
                return 1.0
            elif value <= thresholds.get('medium', 60):
                return 0.6
            elif value <= thresholds.get('high', 75):
                return 0.3
            else:
                return 0.1
        
        elif indicator == 'evapotranspiration':
            # Higher ET = more stress
            if value >= thresholds.get('high', 7):
                return 1.0
            elif value >= thresholds.get('medium', 5):
                return 0.7
            elif value >= thresholds.get('low', 3):
                return 0.4
            else:
                return 0.1
        
        elif indicator == 'ndvi':
            # Higher NDVI = less stress
            if value <= thresholds.get('low', 0.3):
                return 1.0
            elif value <= thresholds.get('medium', 0.5):
                return 0.6
            elif value <= thresholds.get('high', 0.7):
                return 0.3
            else:
                return 0.1
        
        # Default score for unknown indicators
        return 0.5
    
    def determine_stress_level(self, score: float) -> str:
        """Determine stress level from score"""
        if score < 25:
            return "low"
        elif score < 50:
            return "moderate"
        elif score < 75:
            return "high"
        else:
            return "severe"
    
    def generate_recommendations(self, indicators: Dict, stress_level: str) -> List[str]:
        """Generate recommendations based on stress level and indicators"""
        recommendations = []
        
        # Base recommendations on stress level
        if stress_level in ["high", "severe"]:
            recommendations.append("Irrigation immédiate recommandée")
            recommendations.append("Surveillance quotidienne de l'humidité du sol")
            
            # Specific recommendations based on indicators
            temp = indicators.get('temperature', indicators.get('avg_temperature', 0))
            if temp > 30:
                recommendations.append("Irrigation préférable en soirée/nuit pour réduire l'évaporation")
            
            precip = indicators.get('precipitation', indicators.get('total_precipitation', 0))
            if precip < 10:  # Less than 10mm total
                recommendations.append("Considérer des sources d'eau alternatives")
                recommendations.append("Utilisation d'eau recyclée recommandée")
        
        elif stress_level == "moderate":
            recommendations.append("Surveillance régulière de l'humidité du sol")
            recommendations.append("Planification d'irrigation basée sur les prévisions météo")
            recommendations.append("Envisager l'irrigation goutte-à-goutte pour économiser l'eau")
        
        elif stress_level == "low":
            recommendations.append("Surveillance normale recommandée")
            recommendations.append("Maintenir les pratiques d'irrigation actuelles")
        
        # Additional specific recommendations
        if indicators.get('evapotranspiration', 0) > 5:
            recommendations.append("Utiliser du paillage pour réduire l'évaporation")
            recommendations.append("Irrigation en matinée recommandée")
        
        if indicators.get('humidity', indicators.get('avg_humidity', 100)) < 40:
            recommendations.append("Humidité basse - surveiller l'évaporation")
        
        # Algeria-specific recommendations
        recommendations.append("Considérer les cultures résistantes à la sécheresse (olivier, figuier, amandier)")
        recommendations.append("Techniques de conservation de l'eau recommandées")
        
        return list(set(recommendations))  # Remove duplicates