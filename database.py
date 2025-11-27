import sqlite3
import json

def init_db():
    """Initializes the SQLite database with all necessary tables."""
    try:
        with sqlite3.connect('users.db') as conn:
            c = conn.cursor()
            # Create users table if not exists
            c.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    password TEXT NOT NULL
                )
            ''')
            # Create user_inputs table to store structured prediction inputs
            c.execute('''
                CREATE TABLE IF NOT EXISTS user_inputs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT,
                    age INTEGER,
                    gender INTEGER,
                    height REAL,
                    weight REAL,
                    blood_pressure_systolic INTEGER,
                    blood_pressure_diastolic INTEGER,
                    cholesterol REAL,
                    glucose REAL,
                    smoking INTEGER,
                    activity INTEGER,
                    history INTEGER,
                    bmi REAL,
                    risk_score REAL,
                    risk_level TEXT,
                    prediction_text TEXT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (username) REFERENCES users(username)
                )
            ''')
            # Create image_detections table to store image prediction inputs
            c.execute('''
                CREATE TABLE IF NOT EXISTS image_detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT,
                    image_path TEXT,
                    prediction_text TEXT,
                    confidence REAL,
                    analysis_details TEXT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (username) REFERENCES users(username)
                )
            ''')
            conn.commit()
    except sqlite3.Error as e:
        print(f"An error occurred while initializing the database: {e}")

def add_user(username, password):
    """Adds a new user to the database."""
    try:
        with sqlite3.connect('users.db') as conn:
            c = conn.cursor()
            c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
            conn.commit()
    except sqlite3.Error as e:
        print(f"An error occurred while adding a user: {e}")

def get_user(username):
    """Fetches a user from the database by username."""
    try:
        with sqlite3.connect('users.db') as conn:
            c = conn.cursor()
            c.execute('SELECT * FROM users WHERE username = ?', (username,))
            user = c.fetchone()
            return user
    except sqlite3.Error as e:
        print(f"An error occurred while fetching the user: {e}")
        return None

def save_structured_input(username, age, gender, height, weight, systolic, diastolic, cholesterol, glucose, smoking, activity, history, bmi, risk_score, risk_level, prediction_text):
    """Saves structured risk assessment data to the user_inputs table."""
    try:
        with sqlite3.connect('users.db') as conn:
            c = conn.cursor()
            c.execute('''
                INSERT INTO user_inputs (username, age, gender, height, weight, blood_pressure_systolic,
                                         blood_pressure_diastolic, cholesterol, glucose, smoking, activity,
                                         history, bmi, risk_score, risk_level, prediction_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (username, age, gender, height, weight, systolic, diastolic, cholesterol, glucose, smoking, activity, history, bmi, risk_score, risk_level, prediction_text))
            conn.commit()
    except sqlite3.Error as e:
        print(f"An error occurred while saving user input: {e}")

def save_image_detection(username, image_path, prediction_text, confidence, analysis_details):
    """Saves image detection data to the image_detections table."""
    try:
        with sqlite3.connect('users.db') as conn:
            c = conn.cursor()
            c.execute('''
                INSERT INTO image_detections (username, image_path, prediction_text, confidence, analysis_details)
                VALUES (?, ?, ?, ?, ?)
            ''', (username, image_path, prediction_text, confidence, json.dumps(analysis_details)))
            conn.commit()
    except sqlite3.Error as e:
        print(f"An error occurred while saving image detection: {e}")

def get_user_data(username):
    """Retrieves all structured inputs and image detections for a specific user."""
    try:
        with sqlite3.connect('users.db') as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            c.execute('SELECT * FROM user_inputs WHERE username = ? ORDER BY id ASC', (username,))
            user_inputs = c.fetchall()
            
            c.execute('SELECT * FROM image_detections WHERE username = ? ORDER BY id ASC', (username,))
            image_detections = c.fetchall()
            
            return user_inputs, image_detections
    except sqlite3.Error as e:
        print(f"An error occurred while fetching user data: {e}")
        return [], []

def get_all_user_predictions(username):
    """Retrieves and combines all predictions for a specific user from both tables."""
    try:
        with sqlite3.connect('users.db') as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            # Fetch structured inputs
            c.execute('SELECT * FROM user_inputs WHERE username = ?', (username,))
            user_inputs = [dict(row) for row in c.fetchall()]
            
            # Fetch image detections
            c.execute('SELECT * FROM image_detections WHERE username = ?', (username,))
            image_detections = [dict(row) for row in c.fetchall()]
            
            # Combine the two lists and standardize the fields
            combined_predictions = []
            for item in user_inputs:
                combined_predictions.append({
                    'id': item['id'],
                    'type': 'Structured',
                    'username': item['username'],
                    'prediction_text': item['prediction_text'],
                    'score': item['risk_score'],
                    'timestamp': item['timestamp']
                })
            
            for item in image_detections:
                combined_predictions.append({
                    'id': item['id'],
                    'type': 'Image',
                    'username': item['username'],
                    'prediction_text': item['prediction_text'],
                    'score': round(item['confidence'] * 100, 2) if item['confidence'] else 0, # Normalize confidence to 100
                    'timestamp': item['timestamp']
                })
                
            # Sort the combined list by timestamp in descending order
            combined_predictions.sort(key=lambda x: x['timestamp'], reverse=True)
            
            return combined_predictions
    except sqlite3.Error as e:
        print(f"An error occurred while fetching and combining predictions: {e}")
        return []

def get_all_image_detections():
    """Retrieves all image detections from the database."""
    try:
        with sqlite3.connect('users.db') as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute('SELECT * FROM image_detections ORDER BY id DESC')
            detections = c.fetchall()
            return detections
    except sqlite3.Error as e:
        print(f"An error occurred while fetching image detections: {e}")
        return []
        
def get_detection_by_id(detection_id):
    """Retrieves a single image detection by its ID."""
    try:
        with sqlite3.connect('users.db') as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute('SELECT * FROM image_detections WHERE id = ?', (detection_id,))
            detection = c.fetchone()
            if detection and detection['analysis_details']:
                detection = dict(detection)
                detection['analysis_details'] = json.loads(detection['analysis_details'])
            return detection
    except sqlite3.Error as e:
        print(f"An error occurred while fetching detection details: {e}")
        return None

def get_all_user_inputs():
    """Retrieves all user_inputs from the database."""
    try:
        with sqlite3.connect('users.db') as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute('SELECT * FROM user_inputs ORDER BY id DESC')
            predections = c.fetchall()
            return predections
    except sqlite3.Error as e:
        print(f"An error occurred while fetching user_inputs: {e}")
        return []
        
def get_predection_by_id(predection_id):
    """Retrieves a single user_inputs by its ID."""
    try:
        with sqlite3.connect('users.db') as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute('SELECT * FROM user_inputs WHERE id = ?', (predection_id,))
            predection = c.fetchone()
            return predection
    except sqlite3.Error as e:
        print(f"An error occurred while fetching predection details: {e}")
        return None

