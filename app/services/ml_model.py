import numpy as np
import pandas as pd
import pickle
import json
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import joblib
from pathlib import Path

# Import des librairies ML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

class WaterStressModel:
    def __init__(self, model_type: str = 'regression'):
        """
        Initialiser le modèle de prédiction du stress hydrique
        
        model_type: 'regression' pour prédire le score (0-100)
                   'classification' pour prédire le niveau (low/medium/high/severe)
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder() if model_type == 'classification' else None
        self.feature_names = None
        self.model_info = {}
        
        # Configuration des modèles
        self.model_configs = {
            'regression': {
                'rf': RandomForestRegressor(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                ),
                'xgb': XGBRegressor(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42,
                    n_jobs=-1
                ),
                'gbr': GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                )
            },
            'classification': {
                'rf': RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced'
                ),
                'xgb': XGBClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42,
                    n_jobs=-1
                ),
                'gbr': GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                )
            }
        }
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
              test_size: float = 0.2, model_name: str = 'xgb'):
        """
        Entraîner le modèle
        
        Parameters:
        -----------
        X : DataFrame
            Features d'entraînement
        y : Series
            Target variable
        test_size : float
            Proportion pour le test set
        model_name : str
            'rf' (Random Forest), 'xgb' (XGBoost), 'gbr' (Gradient Boosting)
        """
        print(f"Entraînement d'un modèle {self.model_type} avec {model_name}...")
        
        # Encoder les labels pour la classification
        if self.model_type == 'classification':
            y_encoded = self.label_encoder.fit_transform(y)
            print(f"Classes encodées: {dict(zip(self.label_encoder.classes_, 
                                               range(len(self.label_encoder.classes_))))}")
        else:
            y_encoded = y.values
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42
        )
        
        # Sauvegarder les noms de features
        self.feature_names = X.columns.tolist()
        
        # Normalisation des features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Sélectionner le modèle
        self.model = self.model_configs[self.model_type][model_name]
        
        # Entraînement
        print(f"Entraînement sur {len(X_train)} échantillons...")
        self.model.fit(X_train_scaled, y_train)
        
        # Évaluation
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        print(f"Score train: {train_score:.3f}")
        print(f"Score test: {test_score:.3f}")
        
        # Prédictions détaillées
        y_pred = self.model.predict(X_test_scaled)
        
        # Métriques selon le type de modèle
        if self.model_type == 'regression':
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            print(f"\nMétriques de régression:")
            print(f"MAE: {mae:.2f}")
            print(f"RMSE: {rmse:.2f}")
            print(f"R²: {r2:.3f}")
            
            self.model_info['metrics'] = {
                'train_score': float(train_score),
                'test_score': float(test_score),
                'mae': float(mae),
                'rmse': float(rmse),
                'r2': float(r2)
            }
            
        else:  # Classification
            y_pred_labels = self.label_encoder.inverse_transform(y_pred)
            y_test_labels = self.label_encoder.inverse_transform(y_test)
            
            accuracy = accuracy_score(y_test_labels, y_pred_labels)
            precision = precision_score(y_test_labels, y_pred_labels, average='weighted')
            recall = recall_score(y_test_labels, y_pred_labels, average='weighted')
            f1 = f1_score(y_test_labels, y_pred_labels, average='weighted')
            
            print(f"\nMétriques de classification:")
            print(f"Accuracy: {accuracy:.3f}")
            print(f"Precision: {precision:.3f}")
            print(f"Recall: {recall:.3f}")
            print(f"F1-Score: {f1:.3f}")
            
            print("\nRapport de classification:")
            print(classification_report(y_test_labels, y_pred_labels))
            
            self.model_info['metrics'] = {
                'train_score': float(train_score),
                'test_score': float(test_score),
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1)
            }
        
        # Importance des features
        self._calculate_feature_importance(X_train.columns)
        
        # Sauvegarder les infos du modèle
        self.model_info.update({
            'model_type': self.model_type,
            'model_name': model_name,
            'n_features': len(self.feature_names),
            'n_samples': len(X),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'training_date': datetime.now().isoformat(),
            'feature_names': self.feature_names
        })
        
        return {
            'model': self.model,
            'metrics': self.model_info['metrics'],
            'feature_importance': self.model_info.get('feature_importance', {})
        }
    
    def _calculate_feature_importance(self, feature_names):
        """Calculer l'importance des features"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            
            # Créer un DataFrame d'importance
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Sauvegarder les 10 plus importantes
            top_features = importance_df.head(10).to_dict('records')
            
            self.model_info['feature_importance'] = {
                'top_features': top_features,
                'importance_df': importance_df.to_dict()
            }
            
            print("\nTop 10 des features les plus importantes:")
            for i, row in enumerate(top_features[:10], 1):
                print(f"{i:2d}. {row['feature']:30s} : {row['importance']:.4f}")
    
    def predict(self, X: pd.DataFrame, return_proba: bool = False):
        """
        Faire des prédictions avec le modèle entraîné
        
        Parameters:
        -----------
        X : DataFrame
            Features pour la prédiction
        return_proba : bool
            Pour la classification, retourner les probabilités
        """
        if self.model is None:
            raise ValueError("Le modèle n'est pas encore entraîné")
        
        # Normaliser les features
        X_scaled = self.scaler.transform(X)
        
        # Prédictions
        if self.model_type == 'regression':
            predictions = self.model.predict(X_scaled)
            return predictions
        else:
            if return_proba and hasattr(self.model, 'predict_proba'):
                probas = self.model.predict_proba(X_scaled)
                return probas
            else:
                predictions_encoded = self.model.predict(X_scaled)
                predictions = self.label_encoder.inverse_transform(predictions_encoded)
                return predictions
    
    def save_model(self, filepath: str = "models/water_stress_model.pkl"):
        """Sauvegarder le modèle entraîné"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'model_info': self.model_info,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, filepath)
        print(f"Modèle sauvegardé dans {filepath}")
    
    def load_model(self, filepath: str = "models/water_stress_model.pkl"):
        """Charger un modèle sauvegardé"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.model_info = model_data['model_info']
        self.model_type = model_data['model_type']
        
        print(f"Modèle chargé depuis {filepath}")
        print(f"Type: {self.model_type}, Entraîné le: {self.model_info.get('training_date', 'N/A')}")
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5):
        """
        Validation croisée du modèle
        """
        if self.model is None:
            raise ValueError("Le modèle doit être entraîné d'abord")
        
        # Encoder y si classification
        if self.model_type == 'classification':
            y_encoded = self.label_encoder.transform(y)
        else:
            y_encoded = y
        
        X_scaled = self.scaler.transform(X)
        
        # Scores de validation croisée
        cv_scores = cross_val_score(self.model, X_scaled, y_encoded, cv=cv, n_jobs=-1)
        
        print(f"Scores de validation croisée ({cv}-fold):")
        print(f"Moyenne: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        return cv_scores