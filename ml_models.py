import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from database import get_db

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

CSV_PATH_DEFAULT = os.path.join(BASE_DIR, 'enhanced_health_data.csv')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
BEST_ML_PATH = os.path.join(MODELS_DIR, 'best_ml_model.pkl')
ML_METRICS_PATH = os.path.join(MODELS_DIR, 'ml_metrics.json')

FEATURE_COLUMNS = [
    'Gender_Male', 'Age', 'Systolic BP', 'Diastolic BP', 
    'Cholesterol', 'Height (cm)', 'Weight (kg)', 'BMI', 
    'Smoker', 'Diabetes'
]

def preprocess_dataframe(df):
    """Preprocess the enhanced health data dataframe."""
    df_clean = df.copy()
    
    # Process Gender
    if 'Gender' in df_clean.columns:
        df_clean['Gender_Male'] = (df_clean['Gender'].astype(str).str.lower() == 'male').astype(int)
    elif 'Gender_Male' not in df_clean.columns:
        df_clean['Gender_Male'] = 0
        
    # Process Smoker and Diabetes
    if 'Smoker' in df_clean.columns:
        df_clean['Smoker'] = df_clean['Smoker'].astype(int)
    if 'Diabetes' in df_clean.columns:
        df_clean['Diabetes'] = df_clean['Diabetes'].astype(int)
        
    # Calculate BMI if missing or zero
    if 'BMI' not in df_clean.columns or df_clean['BMI'].isnull().any():
        if 'Height (cm)' in df_clean.columns and 'Weight (kg)' in df_clean.columns:
            h_m = df_clean['Height (cm)'] / 100.0
            df_clean['BMI'] = df_clean['Weight (kg)'] / (h_m ** 2)
            
    # Target variable: Health -> 1 for Bad (Cardiovascular Heart Risk), 0 for Fair/Good
    if 'Health' in df_clean.columns:
        df_clean['Target'] = (df_clean['Health'].astype(str).str.strip().str.lower() == 'bad').astype(int)
        
    return df_clean

def train_and_evaluate_ml_models(csv_file_path=None):
    """
    Trains 4 ML Models:
    1. Random Forest
    2. Logistic Regression
    3. XGBoost
    4. Support Vector Machine (SVM)
    Calculates Accuracy, Precision, Recall, F1-Score, ROC-AUC and saves the best model.
    """
    if not csv_file_path or not os.path.exists(csv_file_path):
        csv_file_path = CSV_PATH_DEFAULT
        
    df = pd.read_csv(csv_file_path)
    df_processed = preprocess_dataframe(df)
    
    X = df_processed[FEATURE_COLUMNS]
    y = df_processed['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
        
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=120, max_depth=5, learning_rate=0.08, eval_metric='logloss', random_state=42),
        'SVM': SVC(probability=True, kernel='rbf', C=1.5, random_state=42)
    }
    
    results = {}
    best_model_name = None
    best_score = -1
    best_model_obj = None
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_prob = y_pred.astype(float)
            
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        roc_auc = float(roc_auc_score(y_test, y_prob))
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        # Subsample ROC points for clean JSON
        step = max(1, len(fpr) // 20)
        roc_points = [{'fpr': round(float(f), 4), 'tpr': round(float(t), 4)} for f, t in zip(fpr[::step], tpr[::step])]
        if roc_points[-1]['fpr'] != 1.0:
            roc_points.append({'fpr': 1.0, 'tpr': 1.0})
            
        metric_data = {
            'accuracy': round(acc * 100, 2),
            'precision': round(prec * 100, 2),
            'recall': round(rec * 100, 2),
            'f1_score': round(f1 * 100, 2),
            'roc_auc': round(roc_auc * 100, 2),
            'confusion_matrix': cm,
            'roc_curve': roc_points
        }
        results[name] = metric_data
        
        # Determine best model based on F1 and ROC-AUC
        composite_score = f1 * 0.5 + roc_auc * 0.5
        if composite_score > best_score:
            best_score = composite_score
            best_model_name = name
            best_model_obj = model

    # Save best model
    with open(BEST_ML_PATH, 'wb') as f:
        pickle.dump({'model_name': best_model_name, 'model': best_model_obj}, f)
        
    # Save full metrics JSON
    with open(ML_METRICS_PATH, 'w') as f:
        json.dump({'best_model': best_model_name, 'models': results}, f, indent=2)
        
    # Save to SQLite database
    conn = get_db()
    cursor = conn.cursor()
    # Remove existing ml metrics
    cursor.execute("DELETE FROM model_metrics WHERE model_category = 'ml'")
    
    for name, metrics in results.items():
        is_best = 1 if name == best_model_name else 0
        cursor.execute('''
            INSERT INTO model_metrics (
                model_category, model_name, accuracy, precision, recall, f1_score, roc_auc, details_json, is_best
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            'ml', name, metrics['accuracy'], metrics['precision'], 
            metrics['recall'], metrics['f1_score'], metrics['roc_auc'],
            json.dumps(metrics), is_best
        ))
        
    conn.commit()
    conn.close()
    
    return {
        'best_model': best_model_name,
        'models': results
    }

def get_saved_ml_metrics():
    """Retrieve saved ML metrics from JSON or Database."""
    if os.path.exists(ML_METRICS_PATH):
        try:
            with open(ML_METRICS_PATH, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return None

def predict_clinical_risk(patient_dict):
    """
    Predicts heart disease risk using the best trained ML model.
    Input patient_dict expects keys:
    - gender ('Male' or 'Female')
    - age (int/float)
    - systolic_bp (float)
    - diastolic_bp (float)
    - cholesterol (float)
    - height (cm)
    - weight (kg)
    - bmi (float, optional: calculated if None)
    - smoker (bool/int: 1 or 0)
    - diabetes (bool/int: 1 or 0)
    """
    if not os.path.exists(BEST_ML_PATH) or not os.path.exists(SCALER_PATH):
        # Auto-train if not yet trained
        train_and_evaluate_ml_models()
        
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
        
    with open(BEST_ML_PATH, 'rb') as f:
        saved_dict = pickle.load(f)
        best_model = saved_dict['model']
        model_name = saved_dict['model_name']
        
    # Build feature row
    gender_male = 1 if str(patient_dict.get('gender', '')).lower() == 'male' else 0
    age = float(patient_dict.get('age', 45))
    sbp = float(patient_dict.get('systolic_bp', 120))
    dbp = float(patient_dict.get('diastolic_bp', 80))
    chol = float(patient_dict.get('cholesterol', 190))
    height = float(patient_dict.get('height', 170))
    weight = float(patient_dict.get('weight', 70))
    
    bmi = patient_dict.get('bmi')
    if bmi is None or float(bmi) <= 0:
        h_m = height / 100.0 if height > 0 else 1.7
        bmi = weight / (h_m ** 2)
    else:
        bmi = float(bmi)
        
    smoker = 1 if str(patient_dict.get('smoker', '')).lower() in ['true', '1', 'yes'] else 0
    diabetes = 1 if str(patient_dict.get('diabetes', '')).lower() in ['true', '1', 'yes'] else 0
    
    features = np.array([[
        gender_male, age, sbp, dbp, chol, height, weight, bmi, smoker, diabetes
    ]])
    
    features_scaled = scaler.transform(features)
    
    if hasattr(best_model, 'predict_proba'):
        prob_high_risk = float(best_model.predict_proba(features_scaled)[0, 1])
    else:
        prob_high_risk = float(best_model.predict(features_scaled)[0])
        
    # Clinical heuristic adjustment for known cardiovascular risk multipliers
    # (High BP > 140, Diabetes, Smoking, High Cholesterol > 220)
    multiplier = 0.0
    if sbp >= 140 or dbp >= 90:
        multiplier += 0.08
    if chol >= 220:
        multiplier += 0.07
    if smoker == 1:
        multiplier += 0.09
    if diabetes == 1:
        multiplier += 0.08
    if age >= 60:
        multiplier += 0.06
        
    final_prob = min(0.98, max(0.04, prob_high_risk + multiplier * (1 - prob_high_risk)))
    
    risk_label = 'High Risk' if final_prob >= 0.50 else 'Low Risk'
    
    return {
        'model_used': model_name,
        'probability': round(final_prob, 4),
        'probability_percent': round(final_prob * 100, 1),
        'risk_label': risk_label,
        'bmi': round(bmi, 2)
    }

if __name__ == '__main__':
    print("Training ML Models on enhanced_health_data.csv...")
    res = train_and_evaluate_ml_models()
    print("Training complete! Best Model:", res['best_model'])
    for m, vals in res['models'].items():
        print(f"{m} -> Acc: {vals['accuracy']}%, F1: {vals['f1_score']}%, AUC: {vals['roc_auc']}%")
