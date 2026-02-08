from typing import Dict, List, Optional, Tuple
import numpy as np
from enum import Enum
from datetime import datetime
import pandas as pd
from dataclasses import dataclass
from typing import Literal

class RegionType(Enum):
    """Climate regions of Algeria"""
    MEDITERRANEAN = "mediterranean"  # Coastal areas
    HIGH_PLATEAUS = "high_plateaus"  # Central plateaus
    SAHARAN = "saharan"              # Desert regions
    TELL_ATLAS = "tell_atlas"        # Mountain ranges

@dataclass
class StressCategory:
    """Stress category with detailed thresholds"""
    name: str
    min_score: float
    max_score: float
    color: str
    description: str
    interventions: List[str]

class CropType(Enum):
    """Major crops in Algeria with their water requirements"""
    WHEAT = "wheat"
    BARLEY = "barley"
    OLIVES = "olives"
    DATES = "dates"
    CITRUS = "citrus"
    VEGETABLES = "vegetables"
    VINEYARDS = "vineyards"
    ALMONDS = "almonds"

class StressAnalysisService:
    """
    Enhanced Water Stress Analysis with:
    1. Multi-factor integration with non-linear relationships
    2. Crop-specific stress assessment
    3. Historical context and trends
    4. Season-aware scoring
    5. Soil moisture estimation
    6. Irrigation requirement calculation
    """
    
    def __init__(self, 
                 region_type: Optional[RegionType]  = RegionType.MEDITERRANEAN,
                 crop_type: Optional[CropType] = None):
        self.region_type = region_type
        self.crop_type = crop_type
        self.thresholds = self._initialize_advanced_thresholds()
        self.seasonal_adjustments = self._initialize_seasonal_adjustments()
        self.crop_requirements = self._get_crop_requirements()
         
    def _initialize_advanced_thresholds(self) -> Dict:
        """Initialize comprehensive thresholds with sub-categories"""
        
        thresholds = {
            'temperature': {
                'stress_free': {'min': 15, 'max': 25},
                'mild_stress': {'min': 26, 'max': 30},
                'moderate_stress': {'min': 31, 'max': 35},
                'severe_stress': {'min': 36, 'max': 40},
                'extreme_stress': {'min': 41, 'max': 50}
            },
            'precipitation': {
                'surplus': {'min': 10.0, 'max': float('inf')},
                'adequate': {'min': 5.0, 'max': 9.9},
                'mild_deficit': {'min': 3.0, 'max': 4.9},
                'moderate_deficit': {'min': 1.5, 'max': 2.9},
                'severe_deficit': {'min': 0.5, 'max': 1.4},
                'extreme_deficit': {'min': 0.0, 'max': 0.4}
            },
            'evapotranspiration': {
                'low': {'min': 0.0, 'max': 3.0},
                'moderate': {'min': 3.1, 'max': 5.0},
                'high': {'min': 5.1, 'max': 7.0},
                'very_high': {'min': 7.1, 'max': 9.0},
                'extreme': {'min': 9.1, 'max': 15.0}
            },
            'ndvi': {
                'barren': {'min': 0.0, 'max': 0.2},
                'very_poor': {'min': 0.21, 'max': 0.35},
                'poor': {'min': 0.36, 'max': 0.5},
                'fair': {'min': 0.51, 'max': 0.65},
                'good': {'min': 0.66, 'max': 0.75},
                'excellent': {'min': 0.76, 'max': 1.0}
            },
            'soil_moisture': {
                'wilting_point': {'min': 0.0, 'max': 15.0},
                'stress_zone': {'min': 15.1, 'max': 30.0},
                'available_water': {'min': 30.1, 'max': 60.0},
                'field_capacity': {'min': 60.1, 'max': 100.0}
            },
            'vapor_pressure_deficit': {
                'low': {'min': 0.0, 'max': 1.0},
                'moderate': {'min': 1.1, 'max': 2.0},
                'high': {'min': 2.1, 'max': 3.0},
                'very_high': {'min': 3.1, 'max': 5.0}
            }
        }
        
        # Region-specific adjustments
        region_multipliers = {
            RegionType.MEDITERRANEAN: 1.0,
            RegionType.HIGH_PLATEAUS: 1.15,  # More sensitive to temperature
            RegionType.SAHARAN: 1.25,        # More sensitive to precipitation
            RegionType.TELL_ATLAS: 0.95      # More resilient
        }
        
        # Apply region multipliers
        multiplier = region_multipliers.get(self.region_type, 1.0)
        for category in thresholds.values():
            for subcat in category.values():
                if 'max' in subcat and subcat['max'] != float('inf'):
                    subcat['max'] *= multiplier
        
        return thresholds
    
    def _initialize_seasonal_adjustments(self) -> Dict:
        """Seasonal adjustments based on Algeria's agricultural calendar"""
        return {
            'winter': {'month_range': (12, 2), 'weight_adjustment': 0.9},
            'spring': {'month_range': (3, 5), 'weight_adjustment': 1.1},  # Growing season
            'summer': {'month_range': (6, 8), 'weight_adjustment': 1.3},  # Critical water period
            'autumn': {'month_range': (9, 11), 'weight_adjustment': 1.0}
        }
    
    def _get_crop_requirements(self) -> Dict:
        """Crop-specific water requirements (mm/day)"""
        return {
            CropType.WHEAT: {'critical_temp': 30, 'water_requirement': 4.0, 'drought_tolerance': 'medium'},
            CropType.BARLEY: {'critical_temp': 32, 'water_requirement': 3.5, 'drought_tolerance': 'high'},
            CropType.OLIVES: {'critical_temp': 35, 'water_requirement': 2.5, 'drought_tolerance': 'very_high'},
            CropType.DATES: {'critical_temp': 40, 'water_requirement': 3.0, 'drought_tolerance': 'very_high'},
            CropType.CITRUS: {'critical_temp': 28, 'water_requirement': 5.0, 'drought_tolerance': 'low'},
            CropType.VEGETABLES: {'critical_temp': 25, 'water_requirement': 5.5, 'drought_tolerance': 'low'},
            CropType.VINEYARDS: {'critical_temp': 33, 'water_requirement': 3.0, 'drought_tolerance': 'high'},
            CropType.ALMONDS: {'critical_temp': 34, 'water_requirement': 3.5, 'drought_tolerance': 'high'}
        }
    
    def calculate_stress_score(self, 
                              indicators: Dict,
                              historical_context: Optional[Dict] = None,
                              date: Optional[datetime] = None) -> Tuple[float, Dict[str, float]]:
        """
        Calculate comprehensive water stress score (0-100) with component breakdown
        
        Enhanced algorithm features:
        1. Dynamic weights based on season and indicators
        2. Non-linear scoring functions
        3. Compound effect calculations
        4. Historical trend consideration
        5. Soil moisture estimation
        """
        
        # Get current month for seasonal adjustment
        current_month = date.month if date else datetime.now().month
        season = self._get_season(current_month)
        
        # Estimate soil moisture if not provided
        if 'soil_moisture' not in indicators:
            indicators['soil_moisture'] = self._estimate_soil_moisture(indicators, historical_context)
        
        # Calculate Vapor Pressure Deficit (VPD) if not provided
        if 'vpd' not in indicators and 'temperature' in indicators and 'humidity' in indicators:
            indicators['vpd'] = self._calculate_vpd(
                indicators.get('temperature', indicators.get('temperature_avg', 25)),
                indicators.get('humidity', indicators.get('avg_humidity', 50))
            )
        
        # Dynamic weights based on season and conditions
        weights = self._calculate_dynamic_weights(indicators, season)
        
        # Calculate individual component scores
        component_scores = {}
        total_weighted_score = 0
        total_weight = 0
        
        # 1. Hydrological Stress (Precipitation + ET balance)
        hydrological_score = self._calculate_hydrological_stress(indicators)
        component_scores['hydrological'] = hydrological_score
        total_weighted_score += hydrological_score * weights['hydrological']
        total_weight += weights['hydrological']
        
        # 2. Thermal Stress (Temperature + VPD)
        thermal_score = self._calculate_thermal_stress(indicators)
        component_scores['thermal'] = thermal_score
        total_weighted_score += thermal_score * weights['thermal']
        total_weight += weights['thermal']
        
        # 3. Vegetation Stress (NDVI with trend analysis)
        vegetation_score = self._calculate_vegetation_stress(indicators, historical_context)
        component_scores['vegetation'] = vegetation_score
        total_weighted_score += vegetation_score * weights['vegetation']
        total_weight += weights['vegetation']
        
        # 4. Soil Moisture Stress
        soil_score = self._calculate_soil_stress(indicators['soil_moisture'])
        component_scores['soil'] = soil_score
        total_weighted_score += soil_score * weights['soil']
        total_weight += weights['soil']
        
        # Base composite score
        if total_weight > 0:
            composite_score = total_weighted_score / total_weight
        else:
            composite_score = 50.0  # Neutral score
        
        # Apply compounding effects
        compound_multiplier = self._calculate_compound_effects(indicators, component_scores)
        final_score = composite_score * compound_multiplier
        
        # Apply crop-specific adjustments if crop type is specified
        if self.crop_type:
            crop_adjustment = self._calculate_crop_adjustment(indicators)
            final_score *= crop_adjustment
        
        # Apply seasonal adjustment
        seasonal_adjustment = self.seasonal_adjustments[season]['weight_adjustment']
        final_score *= seasonal_adjustment
        
        # Scale to 0-100 range
        final_score = final_score * 100
        
        # Normalize to 0-100 range
        final_score = min(max(final_score, 0), 100)
        
        # Scale components for output
        component_scores = {k: round(v * 100, 1) for k, v in component_scores.items()}
        
        return round(final_score, 2), component_scores
    
    def _calculate_dynamic_weights(self, indicators: Dict, season: str) -> Dict[str, float]:
        """Calculate dynamic weights based on current conditions and season"""
        
        # Base weights
        weights = {
            'hydrological': 0.35,  # Precipitation, ET, water balance
            'thermal': 0.25,       # Temperature, VPD
            'vegetation': 0.20,    # NDVI, vegetation health
            'soil': 0.20           # Soil moisture
        }
        
        # Season-based adjustments
        if season == 'summer':
            weights['thermal'] += 0.05
            weights['hydrological'] -= 0.05
        elif season == 'winter':
            weights['thermal'] -= 0.05
            weights['hydrological'] += 0.05
        
        # Condition-based adjustments
        temp = indicators.get('temperature', indicators.get('temperature_avg', 25))
        precip = indicators.get('precipitation', indicators.get('total_precipitation', 0))
        
        if temp > 35:  # Extreme heat
            weights['thermal'] += 0.05
            weights['vegetation'] += 0.05
        
        if precip < 1.0:  # Severe drought
            weights['hydrological'] += 0.10
            weights['soil'] += 0.05
        
        # Normalize weights to sum to 1.0
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    def _calculate_hydrological_stress(self, indicators: Dict) -> float:
        """Calculate hydrological stress from precipitation and ET"""
        
        precip = indicators.get('precipitation', indicators.get('total_precipitation', 0))
        et = indicators.get('evapotranspiration', 0)
        humidity = indicators.get('humidity', indicators.get('avg_humidity', 50))
        
        # Water deficit calculation
        water_deficit = max(et - precip, 0)
        
        # Score based on water deficit
        if water_deficit > 8.0:
            score = 1.0  # Extreme stress
        elif water_deficit > 5.0:
            score = 0.85
        elif water_deficit > 3.0:
            score = 0.65
        elif water_deficit > 1.0:
            score = 0.40
        elif water_deficit > 0.0:
            score = 0.20
        else:
            score = 0.05  # Water surplus
        
        # Consider humidity for evaporation potential
        if humidity < 30:
            score = min(score * 1.2, 1.0)  # Increase stress in dry air
        
        return score
    
    def _calculate_thermal_stress(self, indicators: Dict) -> float:
        """Calculate thermal stress from temperature and VPD"""
        
        temp = indicators.get('temperature', indicators.get('temperature_avg', 25))
        vpd = indicators.get('vpd', self._calculate_vpd(temp, 
            indicators.get('humidity', indicators.get('avg_humidity', 50))))
        
        # Temperature-based stress (non-linear)
        if temp > 40:
            temp_score = 1.0
        elif temp > 35:
            temp_score = 0.9
        elif temp > 30:
            temp_score = 0.7
        elif temp > 25:
            temp_score = 0.4
        elif temp > 20:
            temp_score = 0.2
        else:
            temp_score = 0.1
        
        # VPD-based stress (atmospheric dryness)
        if vpd > 3.0:
            vpd_score = 0.9
        elif vpd > 2.0:
            vpd_score = 0.7
        elif vpd > 1.0:
            vpd_score = 0.4
        else:
            vpd_score = 0.1
        
        # Combined thermal stress (weighted)
        thermal_score = (temp_score * 0.7) + (vpd_score * 0.3)
        
        return thermal_score
    
    def _calculate_vegetation_stress(self, indicators: Dict, 
                                    historical_context: Optional[Dict]) -> float:
        """Calculate vegetation stress with trend analysis"""
        
        ndvi = indicators.get('ndvi', 0.5)
        ndwi = indicators.get('ndwi', None)
        
        # Base NDVI score
        if ndvi <= 0.2:
            base_score = 1.0
        elif ndvi <= 0.35:
            base_score = 0.85
        elif ndvi <= 0.5:
            base_score = 0.65
        elif ndvi <= 0.65:
            base_score = 0.40
        elif ndvi <= 0.75:
            base_score = 0.20
        else:
            base_score = 0.10
        
        # Consider NDWI if available (water content in vegetation)
        if ndwi is not None:
            if ndwi < -0.2:
                base_score = min(base_score * 1.3, 1.0)  # Increase stress
        
        # Trend analysis if historical data available
        if historical_context and 'ndvi_trend' in historical_context:
            trend = historical_context['ndvi_trend']
            if trend < -0.1:  # Declining trend
                base_score = min(base_score * 1.2, 1.0)
            elif trend > 0.1:  # Improving trend
                base_score *= 0.9
        
        return base_score
    
    def _calculate_soil_stress(self, soil_moisture: float) -> float:
        """Calculate stress based on soil moisture"""
        
        if soil_moisture < 15.0:
            return 1.0  # Wilting point
        elif soil_moisture < 30.0:
            return 0.85  # Stress zone
        elif soil_moisture < 50.0:
            return 0.60  # Moderate available water
        elif soil_moisture < 70.0:
            return 0.30  # Good available water
        elif soil_moisture < 85.0:
            return 0.15  # Near field capacity
        else:
            return 0.05  # Field capacity/saturated
    
    def _calculate_compound_effects(self, indicators: Dict, 
                                   component_scores: Dict[str, float]) -> float:
        """Calculate compound stress multiplier when multiple factors are extreme"""
        
        multiplier = 1.0
        
        # Check for dangerous combinations
        conditions = []
        
        # 1. Heat + Drought
        temp = indicators.get('temperature', indicators.get('temperature_avg', 25))
        precip = indicators.get('precipitation', indicators.get('total_precipitation', 0))
        if temp > 35 and precip < 1.0:
            conditions.append('heat_drought')
            multiplier *= 1.25
        
        # 2. Low Soil Moisture + High ET
        soil_moisture = indicators.get('soil_moisture', 50.0)
        et = indicators.get('evapotranspiration', 0)
        if soil_moisture < 30.0 and et > 5.0:
            conditions.append('dry_soil_high_et')
            multiplier *= 1.15
        
        # 3. Poor Vegetation + Low Humidity
        ndvi = indicators.get('ndvi', 0.5)
        humidity = indicators.get('humidity', indicators.get('avg_humidity', 50))
        if ndvi < 0.35 and humidity < 40:
            conditions.append('poor_veg_dry_air')
            multiplier *= 1.10
        
        # 4. High VPD + High Temperature
        vpd = indicators.get('vpd', 0)
        if vpd > 2.5 and temp > 30:
            conditions.append('high_vpd_heat')
            multiplier *= 1.12
        
        # Cap maximum multiplier
        return min(multiplier, 1.5)
    
    def _calculate_crop_adjustment(self, indicators: Dict) -> float:
        """Adjust stress score based on crop sensitivity"""
        
        if not self.crop_type or self.crop_type not in self.crop_requirements:
            return 1.0
        
        crop_info = self.crop_requirements[self.crop_type]
        temp = indicators.get('temperature', indicators.get('temperature_avg', 25))
        precip = indicators.get('precipitation', indicators.get('total_precipitation', 0))
        
        adjustment = 1.0
        
        # Temperature sensitivity
        critical_temp = crop_info['critical_temp']
        if temp > critical_temp:
            # Non-linear penalty for exceeding critical temperature
            excess = temp - critical_temp
            adjustment *= (1 + (excess * 0.05))
        
        # Water requirement comparison
        water_req = crop_info['water_requirement']
        if precip < water_req * 0.5:  # Less than 50% of requirement
            adjustment *= 1.2
        elif precip < water_req * 0.8:  # Less than 80% of requirement
            adjustment *= 1.1
        
        # Drought tolerance consideration
        drought_tolerance = crop_info['drought_tolerance']
        if drought_tolerance == 'very_high':
            adjustment *= 0.9
        elif drought_tolerance == 'high':
            adjustment *= 0.95
        elif drought_tolerance == 'low':
            adjustment *= 1.1
        
        return adjustment
    
    def _estimate_soil_moisture(self, indicators: Dict, 
                               historical_context: Optional[Dict]) -> float:
        """Estimate soil moisture from available indicators"""
        
        precip = indicators.get('precipitation', indicators.get('total_precipitation', 0))
        et = indicators.get('evapotranspiration', 3.0)  # Default ET
        temp = indicators.get('temperature', indicators.get('temperature_avg', 25))
        
        # Simple water balance model
        water_input = precip * 0.8  # 80% infiltration rate
        water_loss = et * 0.6       # 60% of ET from soil
        
        # Base soil moisture
        if historical_context and 'previous_soil_moisture' in historical_context:
            base_moisture = historical_context['previous_soil_moisture']
        else:
            base_moisture = 50.0  # Default starting point
        
        # Update soil moisture
        soil_moisture = base_moisture + water_input - water_loss
        
        # Temperature effect (evaporation increases with temperature)
        if temp > 30:
            soil_moisture -= (temp - 30) * 0.5
        
        # Bound between 0 and 100
        return min(max(soil_moisture, 0.0), 100.0)
    
    def _calculate_vpd(self, temperature: float, humidity: float) -> float:
        """Calculate Vapor Pressure Deficit (kPa)"""
        # Saturation vapor pressure (Tetens formula)
        es = 0.6108 * np.exp(17.27 * temperature / (temperature + 237.3))
        
        # Actual vapor pressure
        ea = es * (humidity / 100.0)
        
        # Vapor Pressure Deficit
        vpd = es - ea
        
        return round(vpd, 2)
    
    def _get_season(self, month: int) -> str:
        """Get season name from month"""
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'autumn'
    
    def calculate_irrigation_requirement(self, indicators: Dict, 
                                        crop_area_ha: float = 1.0) -> Dict:
        """
        Calculate irrigation requirements based on water deficit
        
        Returns irrigation depth (mm) and volume (m³)
        """
        
        score, components = self.calculate_stress_score(indicators)
        
        # Get crop water requirement
        crop_req = 4.0  # Default mm/day
        if self.crop_type and self.crop_type in self.crop_requirements:
            crop_req = self.crop_requirements[self.crop_type]['water_requirement']
        
        # Current water availability from precipitation
        precip = indicators.get('precipitation', indicators.get('total_precipitation', 0))
        
        # Effective precipitation (considering runoff and infiltration)
        effective_precip = precip * 0.7
        
        # Water deficit
        water_deficit = max(crop_req - effective_precip, 0)
        
        # Adjust based on soil moisture
        soil_moisture = indicators.get('soil_moisture', 
                                     self._estimate_soil_moisture(indicators, None))
        
        # Soil moisture contribution
        soil_contribution = max(min(soil_moisture / 100.0, 0.3), 0) * crop_req
        
        # Final irrigation requirement
        irrigation_depth = max(water_deficit - soil_contribution, 0)
        
        # Convert to volume
        irrigation_volume = irrigation_depth * 10 * crop_area_ha  # mm to m³/ha
        
        return {
            'irrigation_depth_mm': round(irrigation_depth, 1),
            'irrigation_volume_m3': round(irrigation_volume, 1),
            'water_deficit_mm': round(water_deficit, 1),
            'crop_requirement_mm': crop_req,
            'effective_precipitation_mm': round(effective_precip, 1),
            'soil_moisture_contribution_mm': round(soil_contribution, 1)
        }
    
    def get_stress_analysis(self, indicators: Dict, 
                           historical_context: Optional[Dict] = None,
                           date: Optional[datetime] = None) -> Dict:
        """
        Comprehensive stress analysis with multiple metrics
        """
        
        # Calculate stress score and components
        stress_score, component_scores = self.calculate_stress_score(
            indicators, historical_context, date
        )
        
        # Determine stress level
        stress_level = self.determine_stress_level(stress_score)
        
        # Get recommendations
        recommendations = self.generate_recommendations(indicators, stress_level)
        
        # Calculate irrigation requirements if crop specified
        irrigation_info = {}
        if self.crop_type:
            irrigation_info = self.calculate_irrigation_requirement(indicators)
        
        # Calculate water deficit
        water_deficit = self._calculate_water_deficit(indicators)
        
        # Get compound stress analysis
        compound_analysis = self._analyze_compound_stress(indicators)
        
        return {
            'stress_score': stress_score,
            'stress_level': stress_level,
            'component_scores': component_scores,
            'water_deficit_mm': water_deficit,
            'compound_stress_factors': compound_analysis,
            'irrigation_requirements': irrigation_info,
            'recommendations': recommendations,
            'region_type': self.region_type.value,
            'crop_type': self.crop_type.value if self.crop_type else None,
            'risk_assessment': self._assess_risk_level(stress_score, indicators),
            'time_sensitivity': self._assess_time_sensitivity(stress_score, indicators)
        }
    
    def determine_stress_level(self, score: float) -> str:
        """
        Enhanced stress level determination with confidence intervals
        """
        stress_categories = [
            StressCategory('optimal', 0, 20, 'green', 'Conditions optimales', []),
            StressCategory('watch', 20.1, 40, 'yellow', 'Surveillance recommandée', 
                         ['Surveillance régulière', 'Planification préventive']),
            StressCategory('moderate', 40.1, 60, 'orange', 'Stress modéré',
                         ['Irrigation planifiée', 'Réduction des pertes d\'eau']),
            StressCategory('high', 60.1, 80, 'red', 'Stress élevé',
                         ['Irrigation immédiate', 'Mesures d\'urgence']),
            StressCategory('severe', 80.1, 100, 'darkred', 'Stress sévère',
                         ['Irrigation intensive', 'Protection des cultures', 
                          'Mesures exceptionnelles'])
        ]
        
        for category in stress_categories:
            if category.min_score <= score <= category.max_score:
                return category.name
        
        return 'unknown'
    
    def generate_recommendations(self, indicators: Dict, stress_level: str) -> List[str]:
        """Generate crop and region-specific recommendations"""
        
        recommendations = []
        
        # Base recommendations by stress level
        if stress_level in ['high', 'severe']:
            recommendations.extend([
                "Irrigation immédiate nécessaire",
                "Surveillance horaire recommandée",
                "Activation du plan d'urgence hydrique",
                "Contacter le service agricole régional"
            ])
        elif stress_level == 'moderate':
            recommendations.extend([
                "Planification d'irrigation dans les 48h",
                "Surveillance bi-quotidienne",
                "Optimisation des systèmes d'irrigation",
                "Paillage pour réduire l'évaporation"
            ])
        elif stress_level == 'watch':
            recommendations.extend([
                "Surveillance quotidienne",
                "Préparation des systèmes d'irrigation",
                "Évaluation des réserves d'eau",
                "Consultation des prévisions météo"
            ])
        
        # Crop-specific recommendations
        if self.crop_type:
            if self.crop_type == CropType.WHEAT or self.crop_type == CropType.BARLEY:
                recommendations.append("Prioriser l'irrigation au stade de remplissage des grains")
            elif self.crop_type == CropType.OLIVES:
                recommendations.append("Irrigation déficitaire contrôlée acceptable pour l'olivier")
            elif self.crop_type == CropType.CITRUS:
                recommendations.append("Irrigation régulière essentielle pour les agrumes")
                recommendations.append("Éviter le stress hydrique pendant la floraison")
        
        # Region-specific recommendations
        if self.region_type == RegionType.SAHARAN:
            recommendations.extend([
                "Utiliser des systèmes d'irrigation localisée (goutte-à-goutte)",
                "Protéger les cultures du vent desséchant",
                "Utiliser des brise-vent végétaux"
            ])
        elif self.region_type == RegionType.HIGH_PLATEAUS:
            recommendations.extend([
                "Irrigation en matinée pour éviter le gel nocturne",
                "Choix de variétés résistantes au froid nocturne"
            ])
        
        # Condition-specific recommendations
        temp = indicators.get('temperature', indicators.get('temperature_avg', 25))
        if temp > 35:
            recommendations.append("Irrigation nocturne pour réduire l'évaporation")
        
        if indicators.get('evapotranspiration', 0) > 6.0:
            recommendations.append("Utiliser des anti-transpirants si disponibles")
        
        # General water conservation
        recommendations.extend([
            "Recyclage des eaux usées traitées",
            "Collecte des eaux de pluie",
            "Pratiques agricoles de conservation de l'eau"
        ])
        
        return list(set(recommendations))[:10]  # Limit to top 10 recommendations
    
    def _calculate_water_deficit(self, indicators: Dict) -> float:
        """Calculate daily water deficit"""
        precip = indicators.get('precipitation', indicators.get('total_precipitation', 0))
        et = indicators.get('evapotranspiration', 3.0)
        
        # Consider crop requirement if specified
        if self.crop_type and self.crop_type in self.crop_requirements:
            crop_req = self.crop_requirements[self.crop_type]['water_requirement']
            return max(crop_req - precip, 0)
        
        return max(et - precip, 0)
    
    def _analyze_compound_stress(self, indicators: Dict) -> List[str]:
        """Identify compound stress factors"""
        factors = []
        
        temp = indicators.get('temperature', indicators.get('temperature_avg', 25))
        precip = indicators.get('precipitation', indicators.get('total_precipitation', 0))
        
        # Heat + Drought
        if temp > 35 and precip < 1.0:
            factors.append("Canicule + Sécheresse sévère")
        
        # High ET + Low Humidity
        et = indicators.get('evapotranspiration', 0)
        humidity = indicators.get('humidity', indicators.get('avg_humidity', 50))
        if et > 6.0 and humidity < 40:
            factors.append("Évapotranspiration élevée + Air sec")
        
        # Poor Vegetation + High VPD
        ndvi = indicators.get('ndvi', 0.5)
        vpd = indicators.get('vpd', self._calculate_vpd(temp, humidity))
        if ndvi < 0.4 and vpd > 2.0:
            factors.append("Végétation stressée + Déficit de pression de vapeur")
        
        return factors
    
    def _assess_risk_level(self, stress_score: float, indicators: Dict) -> Dict:
        """Assess risk level and potential impacts"""
        
        risk_levels = {
            'low': (0, 40),
            'medium': (40, 60),
            'high': (60, 80),
            'critical': (80, 100)
        }
        
        for level, (min_score, max_score) in risk_levels.items():
            if min_score <= stress_score < max_score:
                break
        
        # Potential impacts based on risk level
        impacts = {
            'low': ["Aucun impact significatif"],
            'medium': ["Réduction modérée du rendement", "Qualité des récoltes affectée"],
            'high': ["Rendement sérieusement réduit", "Perte partielle possible", 
                    "Stress permanent sur les plantes"],
            'critical': ["Rendement catastrophique", "Perte totale possible", 
                        "Mort des plantes sensibles"]
        }
        
        return {
            'level': level,
            'score_range': f"{min_score}-{max_score}",
            'potential_impacts': impacts[level],
            'recommended_action_time': {
                'low': '1-2 semaines',
                'medium': '3-7 jours',
                'high': '24-48 heures',
                'critical': 'Immédiat'
            }[level]
        }
    
    def _assess_time_sensitivity(self, stress_score: float, indicators: Dict) -> Dict:
        """Assess time sensitivity of required actions"""
        
        if stress_score < 40:
            sensitivity = 'low'
            time_window = '1-2 semaines'
        elif stress_score < 60:
            sensitivity = 'moderate'
            time_window = '3-7 jours'
        elif stress_score < 80:
            sensitivity = 'high'
            time_window = '24-48 heures'
        else:
            sensitivity = 'critical'
            time_window = 'immédiat (0-24 heures)'
        
        return {
            'sensitivity_level': sensitivity,
            'response_time_window': time_window,
            'urgency_factor': round(stress_score / 20, 1)  # 1-5 scale
        }