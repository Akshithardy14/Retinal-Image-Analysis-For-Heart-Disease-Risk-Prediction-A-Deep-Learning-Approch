import sqlite3
import os
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database.db')

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT,
            password TEXT NOT NULL,
            full_name TEXT,
            role TEXT NOT NULL DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Model metrics table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_category TEXT NOT NULL, -- 'dl' or 'ml'
            model_name TEXT NOT NULL,
            accuracy REAL,
            precision REAL,
            recall REAL,
            f1_score REAL,
            roc_auc REAL,
            loss REAL,
            details_json TEXT,
            is_best INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            patient_name TEXT NOT NULL,
            gender TEXT,
            age INTEGER,
            systolic_bp REAL,
            diastolic_bp REAL,
            cholesterol REAL,
            height REAL,
            weight REAL,
            bmi REAL,
            smoker INTEGER,
            diabetes INTEGER,
            retinal_image TEXT,
            yolo_image TEXT,
            gradcam_image TEXT,
            dl_risk_label TEXT,
            dl_prob REAL,
            ml_risk_label TEXT,
            ml_prob REAL,
            overall_prob REAL,
            risk_score REAL,
            risk_level TEXT, -- 'High Risk' or 'Low Risk'
            yolo_detections_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # Ensure default admin account exists (admin / admin as specified in modules.txt)
    cursor.execute("SELECT id FROM users WHERE username = 'admin'")
    admin_user = cursor.fetchone()
    if not admin_user:
        # We store both hashed and plain check support for simple 'admin'
        hashed_pw = generate_password_hash('admin')
        cursor.execute('''
            INSERT INTO users (username, email, password, full_name, role)
            VALUES (?, ?, ?, ?, ?)
        ''', ('admin', 'admin@retinaheart.ai', hashed_pw, 'System Administrator', 'admin'))
    
    # Ensure sample demo user exists
    cursor.execute("SELECT id FROM users WHERE username = 'john_doe'")
    demo_user = cursor.fetchone()
    if not demo_user:
        hashed_pw = generate_password_hash('user123')
        cursor.execute('''
            INSERT INTO users (username, email, password, full_name, role)
            VALUES (?, ?, ?, ?, ?)
        ''', ('john_doe', 'john@example.com', hashed_pw, 'John Doe', 'user'))

    conn.commit()
    conn.close()

def authenticate_user(username, password, required_role=None):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    
    if not user:
        return None
    
    # Support direct password match or werkzeug hash
    is_valid = False
    if user['password'] == password:
        is_valid = True
    else:
        try:
            if check_password_hash(user['password'], password):
                is_valid = True
        except Exception:
            pass
            
    if is_valid:
        if required_role and user['role'] != required_role:
            return None
        return dict(user)
    return None

def register_user(username, email, password, full_name, role='user'):
    conn = get_db()
    cursor = conn.cursor()
    try:
        hashed_pw = generate_password_hash(password)
        cursor.execute('''
            INSERT INTO users (username, email, password, full_name, role)
            VALUES (?, ?, ?, ?, ?)
        ''', (username, email, hashed_pw, full_name, role))
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return user_id, None
    except sqlite3.IntegrityError:
        conn.close()
        return None, "Username already exists. Please choose a different username."
    except Exception as e:
        conn.close()
        return None, str(e)

if __name__ == '__main__':
    init_db()
    print("Database initialized successfully at", DB_PATH)
