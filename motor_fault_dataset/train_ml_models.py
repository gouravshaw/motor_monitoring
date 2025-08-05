#!/usr/bin/env python3
"""
ML Training Script for Motor Fault Detection
Generated for: Gourav Shaw (T0436800)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare_data():
    """Load and prepare the synthetic dataset for ML training"""
    
    print("📊 Loading synthetic dataset...")
    df = pd.read_csv('motor_fault_dataset.csv')
    
    print(f"✅ Loaded {len(df):,} records")
    print(f"📊 Date range: {df['created_at'].min()} to {df['created_at'].max()}")
    
    # Feature selection for ML
    feature_columns = [
        'temperature', 'humidity', 'pressure',
        'vibration', 'vibration_x', 'vibration_y', 'vibration_z',
        'current', 'voltage', 'power_consumption',
        'motor_speed', 'motor_efficiency',
        'bearing_wear_level', 'thermal_stress_level', 'electrical_stress_level'
    ]
    
    # Target variable
    target_column = 'fault_type'
    
    # Filter operating data only (exclude stopped motor data for fault detection)
    operating_data = df[df['motor_speed'] > 0].copy()
    
    print(f"📊 Operating data: {len(operating_data):,} records")
    
    X = operating_data[feature_columns]
    y = operating_data[target_column]
    
    return X, y, operating_data

def train_anomaly_detection():
    """Train anomaly detection model"""
    
    print("\n🤖 Training Anomaly Detection Model...")
    
    X, y, data = load_and_prepare_data()
    
    # Use only normal operation data for anomaly detection training
    normal_data = X[y == 'normal_operation']
    
    print(f"📊 Normal operation samples: {len(normal_data):,}")
    
    # Scale features
    scaler = StandardScaler()
    X_normal_scaled = scaler.fit_transform(normal_data)
    
    # Train Isolation Forest
    isolation_forest = IsolationForest(
        contamination=0.1,  # Expect 10% anomalies
        random_state=42,
        n_estimators=100
    )
    
    isolation_forest.fit(X_normal_scaled)
    
    # Test on full dataset
    X_all_scaled = scaler.transform(X)
    anomaly_predictions = isolation_forest.predict(X_all_scaled)
    
    # -1 = anomaly, 1 = normal
    anomaly_score = (anomaly_predictions == -1).sum() / len(anomaly_predictions)
    print(f"📊 Detected {anomaly_score*100:.1f}% anomalies")
    
    # Save models
    joblib.dump(isolation_forest, 'anomaly_detection_model.pkl')
    joblib.dump(scaler, 'feature_scaler.pkl')
    
    print("✅ Anomaly detection model saved")
    
    return isolation_forest, scaler

def train_fault_classification():
    """Train fault classification model"""
    
    print("\n🎯 Training Fault Classification Model...")
    
    X, y, data = load_and_prepare_data()
    
    # Encode fault types
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print("📊 Fault type distribution:")
    fault_counts = pd.Series(y).value_counts()
    for fault, count in fault_counts.items():
        print(f"   {fault}: {count:,} ({count/len(y)*100:.1f}%)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    
    rf_classifier.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = rf_classifier.predict(X_test_scaled)
    
    print("\n📊 Classification Results:")
    print(classification_report(y_test, y_pred, 
                              target_names=label_encoder.classes_))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_classifier.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🔍 Top 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Save models
    joblib.dump(rf_classifier, 'fault_classification_model.pkl')
    joblib.dump(label_encoder, 'fault_label_encoder.pkl')
    joblib.dump(scaler, 'classification_scaler.pkl')
    
    print("✅ Classification model saved")
    
    return rf_classifier, label_encoder, scaler

if __name__ == "__main__":
    print("🚀 Starting ML Model Training...")
    print("=" * 50)
    
    # Train both models
    anomaly_model, anomaly_scaler = train_anomaly_detection()
    classification_model, label_encoder, classification_scaler = train_fault_classification()
    
    print("\n✅ ML Training Complete!")
    print("🎯 Models ready for deployment and validation")
    print("📁 Files created:")
    print("   • anomaly_detection_model.pkl")
    print("   • fault_classification_model.pkl")
    print("   • feature_scaler.pkl")
    print("   • classification_scaler.pkl") 
    print("   • fault_label_encoder.pkl")
