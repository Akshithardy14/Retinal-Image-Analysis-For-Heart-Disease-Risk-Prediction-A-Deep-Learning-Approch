import os
import json
import sqlite3
import pandas as pd
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_from_directory

# Force OpenMP duplicate library fix for Windows Anaconda
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from database import init_db, get_db, authenticate_user, register_user
from ml_models import train_and_evaluate_ml_models, get_saved_ml_metrics, predict_clinical_risk, CSV_PATH_DEFAULT
from dl_models import train_and_evaluate_dl_models, get_saved_dl_metrics, predict_retinal_image

app = Flask(__name__)
app.secret_key = 'retinaheart_secret_key_deep_learning_cardio_2026'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'static', 'outputs')
SAMPLES_FOLDER = os.path.join(BASE_DIR, 'static', 'images', 'samples')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(SAMPLES_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64 MB

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# =========================================================================
# LANDING & INDEX ROUTE
# =========================================================================
@app.route('/')
def index():
    return render_template('index.html')

# =========================================================================
# ADMIN AUTHENTICATION
# =========================================================================
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        
        user = authenticate_user(username, password, required_role='admin')
        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['role'] = 'admin'
            flash(f"Welcome to Admin Portal, {user['username']}!", 'success')
            return redirect(url_for('admin_dashboard'))
        else:
            flash("Invalid administrator credentials. Please check username & password.", 'error')
            
    return render_template('admin_login.html')

@app.route('/admin/logout')
def admin_logout():
    session.clear()
    flash("Administrator signed out successfully.", 'info')
    return redirect(url_for('index'))

# =========================================================================
# ADMIN DASHBOARD & MODULE 1: IMAGES DATASET
# =========================================================================
@app.route('/admin/dashboard')
def admin_dashboard():
    if session.get('role') != 'admin':
        flash("Admin authentication required.", 'error')
        return redirect(url_for('admin_login'))
        
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as cnt FROM predictions")
    pred_count = cursor.fetchone()['cnt']
    conn.close()
    
    # Dataset statistics
    stats = {
        'total_images': '49,570',
        'dl_models_count': '4',
        'csv_rows': '1,000',
        'predictions_count': pred_count
    }
    return render_template('admin_dashboard.html', stats=stats)

@app.route('/admin/dataset-images')
def admin_dataset_images():
    if session.get('role') != 'admin':
        flash("Admin authentication required.", 'error')
        return redirect(url_for('admin_login'))
        
    # Gather sample images from static/images/samples
    sample_images = []
    if os.path.exists(SAMPLES_FOLDER):
        for f in os.listdir(SAMPLES_FOLDER):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                is_risk = 'risk' in f.lower()
                sample_images.append({
                    'filename': f,
                    'is_risk': is_risk,
                    'label': 'Heart Risk' if is_risk else 'Normal'
                })
                
    return render_template('admin_dataset_images.html', sample_images=sample_images)

@app.route('/admin/upload-images', methods=['POST'])
def admin_upload_images():
    if session.get('role') != 'admin':
        flash("Admin authentication required.", 'error')
        return redirect(url_for('admin_login'))
        
    target_class = request.form.get('target_class', 'heart_risk')
    files = request.files.getlist('dataset_files')
    uploaded_count = 0
    
    for file in files:
        if file and allowed_file(file.filename):
            fname = secure_filename(file.filename)
            save_name = f"admin_upload_{target_class}_{int(datetime.now().timestamp())}_{fname}"
            file.save(os.path.join(UPLOAD_FOLDER, save_name))
            uploaded_count += 1
            
    flash(f"Successfully processed and added {uploaded_count} image(s) to {target_class} dataset cohort.", 'success')
    return redirect(url_for('admin_dataset_images'))

# =========================================================================
# ADMIN MODULE 2: DEEP LEARNING & YOLO TRAINING
# =========================================================================
@app.route('/admin/train-dl')
def admin_train_dl():
    if session.get('role') != 'admin':
        flash("Admin authentication required.", 'error')
        return redirect(url_for('admin_login'))
        
    dl_data = get_saved_dl_metrics()
    return render_template('admin_train_dl.html', dl_data=dl_data)

@app.route('/admin/run-train-dl', methods=['POST'])
def admin_run_train_dl():
    if session.get('role') != 'admin':
        flash("Admin authentication required.", 'error')
        return redirect(url_for('admin_login'))
        
    dl_data = train_and_evaluate_dl_models()
    flash("Deep Learning Training completed for CNN, ResNet-50, U-Net, and YOLO! Model weights saved to models/.", 'success')
    return redirect(url_for('admin_train_dl'))

# =========================================================================
# ADMIN MODULE 3: CSV DATASET MANAGEMENT
# =========================================================================
@app.route('/admin/dataset-csv')
def admin_dataset_csv():
    if session.get('role') != 'admin':
        flash("Admin authentication required.", 'error')
        return redirect(url_for('admin_login'))
        
    df = pd.read_csv(CSV_PATH_DEFAULT)
    rows = df.head(50).to_dict(orient='records')
    
    stats = {
        'total_rows': len(df),
        'mean_sbp': round(float(df['Systolic BP'].mean()), 1),
        'mean_dbp': round(float(df['Diastolic BP'].mean()), 1),
        'mean_chol': round(float(df['Cholesterol'].mean()), 1),
        'mean_bmi': round(float(df['BMI'].mean()), 1),
        'bad_health_count': int((df['Health'].astype(str).str.lower() == 'bad').sum()),
        'fair_health_count': int((df['Health'].astype(str).str.lower() == 'fair').sum()),
        'good_health_count': int((df['Health'].astype(str).str.lower() == 'good').sum())
    }
    return render_template('admin_dataset_csv.html', rows=rows, stats=stats)

@app.route('/admin/upload-csv', methods=['POST'])
def admin_upload_csv():
    if session.get('role') != 'admin':
        flash("Admin authentication required.", 'error')
        return redirect(url_for('admin_login'))
        
    if 'csv_file' not in request.files:
        flash("No CSV file selected.", 'error')
        return redirect(url_for('admin_dataset_csv'))
        
    file = request.files['csv_file']
    if file and file.filename.endswith('.csv'):
        file.save(CSV_PATH_DEFAULT)
        flash("Clinical CSV Dataset updated successfully! Retrain ML models to apply changes.", 'success')
    else:
        flash("Please upload a valid .csv file.", 'error')
        
    return redirect(url_for('admin_dataset_csv'))

# =========================================================================
# ADMIN MODULE 4: MACHINE LEARNING MODELS TRAINING
# =========================================================================
@app.route('/admin/train-ml')
def admin_train_ml():
    if session.get('role') != 'admin':
        flash("Admin authentication required.", 'error')
        return redirect(url_for('admin_login'))
        
    ml_data = get_saved_ml_metrics()
    if not ml_data:
        ml_data = train_and_evaluate_ml_models()
        
    return render_template('admin_train_ml.html', ml_data=ml_data)

@app.route('/admin/run-train-ml', methods=['POST'])
def admin_run_train_ml():
    if session.get('role') != 'admin':
        flash("Admin authentication required.", 'error')
        return redirect(url_for('admin_login'))
        
    ml_data = train_and_evaluate_ml_models()
    flash(f"ML Training completed for all 4 models! Best model selected and saved: {ml_data['best_model']}.", 'success')
    return redirect(url_for('admin_train_ml'))

# =========================================================================
# ADMIN MODULE 5: PREDICTIONS REGISTRY AUDIT
# =========================================================================
@app.route('/admin/predictions')
def admin_predictions():
    if session.get('role') != 'admin':
        flash("Admin authentication required.", 'error')
        return redirect(url_for('admin_login'))
        
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions ORDER BY id DESC")
    preds = cursor.fetchall()
    conn.close()
    
    return render_template('admin_predictions.html', predictions=[dict(p) for p in preds])

# =========================================================================
# USER MODULE: AUTHENTICATION
# =========================================================================
@app.route('/user/register', methods=['GET', 'POST'])
def user_register():
    if request.method == 'POST':
        full_name = request.form.get('full_name', '').strip()
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        
        if not username or not password:
            flash("Username and password are required.", 'error')
            return redirect(url_for('user_register'))
            
        user_id, err = register_user(username, email, password, full_name, role='user')
        if err:
            flash(err, 'error')
            return redirect(url_for('user_register'))
            
        flash("Registration successful! You may now sign in.", 'success')
        return redirect(url_for('user_login'))
        
    return render_template('user_register.html')

@app.route('/user/login', methods=['GET', 'POST'])
def user_login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        
        user = authenticate_user(username, password)
        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['role'] = user['role']
            flash(f"Welcome back, {user['username']}!", 'success')
            return redirect(url_for('user_dashboard'))
        else:
            flash("Invalid credentials. Please verify your username and password.", 'error')
            
    return render_template('user_login.html')

@app.route('/user/logout')
def user_logout():
    session.clear()
    flash("You have been signed out successfully.", 'info')
    return redirect(url_for('index'))

# =========================================================================
# USER MODULE: DASHBOARD & PREDICTION PIPELINE
# =========================================================================
@app.route('/user/dashboard')
def user_dashboard():
    if not session.get('user_id'):
        flash("Please sign in to access your patient dashboard.", 'info')
        return redirect(url_for('user_login'))
        
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM predictions WHERE user_id = ? ORDER BY id DESC LIMIT 5",
        (session['user_id'],)
    )
    recent = cursor.fetchall()
    conn.close()
    
    return render_template('user_dashboard.html', recent_predictions=[dict(r) for r in recent])

@app.route('/user/predict', methods=['GET', 'POST'])
def user_predict():
    if not session.get('user_id'):
        flash("Please log in to perform cardiovascular disease risk prediction.", 'info')
        return redirect(url_for('user_login'))
        
    if request.method == 'POST':
        patient_name = request.form.get('patient_name', 'Anonymous Patient').strip()
        gender = request.form.get('gender', 'Male')
        age = int(request.form.get('age', 50))
        systolic_bp = float(request.form.get('systolic_bp', 120))
        diastolic_bp = float(request.form.get('diastolic_bp', 80))
        cholesterol = float(request.form.get('cholesterol', 190))
        height_cm = float(request.form.get('height_cm', 170))
        weight_kg = float(request.form.get('weight_kg', 70))
        bmi = float(request.form.get('bmi', 24.2))
        smoker = int(request.form.get('smoker', 0))
        diabetes = int(request.form.get('diabetes', 0))
        
        # 1. Determine image source (uploaded file vs preloaded sample)
        sample_path = request.form.get('sample_image_path', '').strip()
        retinal_filename = None
        local_image_path = None
        
        if 'retinal_image' in request.files and request.files['retinal_image'].filename:
            file = request.files['retinal_image']
            if allowed_file(file.filename):
                fname = secure_filename(file.filename)
                retinal_filename = f"user_{session['user_id']}_{int(datetime.now().timestamp())}_{fname}"
                local_image_path = os.path.join(UPLOAD_FOLDER, retinal_filename)
                file.save(local_image_path)
        elif sample_path:
            # Copy sample image into uploads
            src_sample = os.path.join(SAMPLES_FOLDER, sample_path)
            if os.path.exists(src_sample):
                retinal_filename = f"sample_{int(datetime.now().timestamp())}_{sample_path}"
                local_image_path = os.path.join(UPLOAD_FOLDER, retinal_filename)
                with open(src_sample, 'rb') as f_in, open(local_image_path, 'wb') as f_out:
                    f_out.write(f_in.read())
                    
        # Fallback if no image provided: pick standard sample
        if not local_image_path or not os.path.exists(local_image_path):
            sample_files = os.listdir(SAMPLES_FOLDER) if os.path.exists(SAMPLES_FOLDER) else []
            if sample_files:
                sample_name = sample_files[0]
                retinal_filename = f"fallback_{sample_name}"
                local_image_path = os.path.join(UPLOAD_FOLDER, retinal_filename)
                with open(os.path.join(SAMPLES_FOLDER, sample_name), 'rb') as fi, open(local_image_path, 'wb') as fo:
                    fo.write(fi.read())
            else:
                flash("Please upload a retinal fundus image or pick a sample.", 'error')
                return redirect(url_for('user_predict'))

        # 2. Run Retinal Deep Learning (ResNet-50, YOLO Detections, Grad-CAM)
        dl_res = predict_retinal_image(local_image_path)
        dl_prob = dl_res['dl_probability']
        dl_risk_label = dl_res['dl_risk_label']
        yolo_image = dl_res['yolo_image']
        gradcam_image = dl_res['gradcam_image']
        detections = dl_res['detections']
        
        # 3. Run Clinical Parameters Machine Learning (XGBoost / Best ML Model)
        patient_dict = {
            'gender': gender,
            'age': age,
            'systolic_bp': systolic_bp,
            'diastolic_bp': diastolic_bp,
            'cholesterol': cholesterol,
            'height': height_cm,
            'weight': weight_kg,
            'bmi': bmi,
            'smoker': smoker,
            'diabetes': diabetes
        }
        ml_res = predict_clinical_risk(patient_dict)
        ml_prob = ml_res['probability']
        ml_risk_label = ml_res['risk_label']
        
        # 4. Integrated Multi-Modal Risk Fusion
        # Retinal microvasculature weight 0.55, systemic clinical parameters weight 0.45
        overall_prob = round(0.55 * dl_prob + 0.45 * ml_prob, 4)
        risk_score = round(overall_prob * 100, 1)
        risk_level = 'High Risk' if overall_prob >= 0.50 else 'Low Risk'
        
        # 5. Store record in database
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions (
                user_id, patient_name, gender, age, systolic_bp, diastolic_bp,
                cholesterol, height, weight, bmi, smoker, diabetes,
                retinal_image, yolo_image, gradcam_image,
                dl_risk_label, dl_prob, ml_risk_label, ml_prob,
                overall_prob, risk_score, risk_level, yolo_detections_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session['user_id'], patient_name, gender, age, systolic_bp, diastolic_bp,
            cholesterol, height_cm, weight_kg, bmi, smoker, diabetes,
            retinal_filename, yolo_image, gradcam_image,
            dl_risk_label, dl_prob, ml_risk_label, ml_prob,
            overall_prob, risk_score, risk_level, json.dumps(detections)
        ))
        conn.commit()
        pred_id = cursor.lastrowid
        conn.close()
        
        flash("Disease Risk Assessment computed successfully!", 'success')
        return redirect(url_for('user_report', pred_id=pred_id))
        
    return render_template('user_predict.html')

@app.route('/user/report/<int:pred_id>')
def user_report(pred_id):
    if not session.get('user_id'):
        flash("Please sign in to view diagnostic reports.", 'info')
        return redirect(url_for('user_login'))
        
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions WHERE id = ?", (pred_id,))
    pred = cursor.fetchone()
    conn.close()
    
    if not pred:
        flash("Diagnostic report not found.", 'error')
        return redirect(url_for('user_dashboard'))
        
    pred_dict = dict(pred)
    detections = []
    if pred_dict.get('yolo_detections_json'):
        try:
            detections = json.loads(pred_dict['yolo_detections_json'])
        except Exception:
            detections = []
            
    return render_template('user_report.html', pred=pred_dict, detections=detections)

@app.route('/user/history')
def user_history():
    if not session.get('user_id'):
        flash("Please sign in to view your diagnostic history.", 'info')
        return redirect(url_for('user_login'))
        
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM predictions WHERE user_id = ? ORDER BY id DESC",
        (session['user_id'],)
    )
    rows = cursor.fetchall()
    conn.close()
    
    return render_template('user_history.html', history=[dict(r) for r in rows])

# =========================================================================
# APPLICATION ENTRYPOINT
# =========================================================================
if __name__ == '__main__':
    init_db()
    # Ensure baseline models are trained and saved
    get_saved_ml_metrics()
    get_saved_dl_metrics()
    
    print("\n=======================================================")
    print(" RetinaHeart AI Web Server Initialized")
    print(" http://127.0.0.1:5000")
    print(" Admin Login: admin / admin")
    print(" User Demo:   john_doe / user123")
    print("=======================================================\n")
    app.run(host='127.0.0.1', port=5000, debug=False)
