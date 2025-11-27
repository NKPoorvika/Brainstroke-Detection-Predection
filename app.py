import os
import numpy as np
import pandas as pd
import cv2
import json
import sqlite3
from flask import Flask, request, render_template, redirect, session, url_for, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import pickle
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from database import init_db, add_user, get_user, save_structured_input, save_image_detection, get_user_data, get_all_image_detections, get_detection_by_id, get_all_user_inputs, get_predection_by_id, get_all_user_predictions
from datetime import timedelta
from collections import Counter
import warnings
import calendar
from datetime import datetime

# Suppress warnings
warnings.filterwarnings("ignore")

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a secure key
app.permanent_session_lifetime = timedelta(hours=4)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Load structured model
try:
    with open('model.pickle', 'rb') as f:
        model_structured = pickle.load(f)
except FileNotFoundError:
    print("Structured model 'model.pickle' not found. Structured prediction will not work.")
    model_structured = None

# Image model setup
IMG_SIZE = 224
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
for layer in base_model.layers:
    layer.trainable = False

x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)
model_image = Model(inputs=base_model.input, outputs=output)

try:
    if os.path.exists('vgg_unfrozen.h5'):
        model_image.load_weights('vgg_unfrozen.h5')
    else:
        raise FileNotFoundError("Weights file 'vgg_unfrozen.h5' not found.")
except FileNotFoundError as e:
    print(e)
    model_image = None
    
# Init DB
init_db()

# Basic Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/detection')
def detection():
    return render_template('detection.html')

@app.route('/bmi')
def bmi():
    return render_template('bmi.html')

@app.route('/counsel')
def counsel():
    return render_template('counsel.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')    

@app.route('/brain-health')
def brain_health():
    return render_template('brain_health.html')

@app.route('/medications')
def medications():
    return render_template('medications.html')

@app.route('/lifestyle')
def lifestyle():
    return render_template('lifestyle.html')

@app.route('/life-after-stroke')
def life_after_stroke():
    return render_template('life_after_stroke.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/analytics')
def analytics():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    return render_template('analytics.html')

# Placeholder for search route
@app.route('/search')
def search():
    # You can implement actual search logic here
    return "Search page - Not yet implemented"

# Placeholder for donate route
@app.route('/donate')
def donate():
    # You can implement actual donate page logic here
    return "Donate page - Not yet implemented"


@app.route('/api/user_analytics_data')
def user_analytics_data():
    """
    New endpoint to provide analytics data specific to the logged-in user.
    It combines data from structured inputs and image detections.
    """
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    combined_data = get_all_user_predictions(session['username'])
    
    # Process data for the distribution chart
    normal_count = sum(1 for item in combined_data if 'no' in item['prediction_text'].lower() or 'not' in item['prediction_text'].lower())
    stroke_count = len(combined_data) - normal_count

    # Process data for the line chart (score over time)
    score_over_time = []
    # Sort the data chronologically for the line chart
    combined_data.sort(key=lambda x: x['timestamp'])
    
    for item in combined_data:
        score_over_time.append({
            'date': item['timestamp'].split(' ')[0],
            'score': item['score']
        })

    # Mock data for performance charts (as the model metrics are not dynamic)
    accuracy = 94.7
    precision = 92.3
    recall = 95.1
    f1_score = 93.7
    
    stats_data = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
    }

    return jsonify({
        'stats': stats_data,
        'stroke_count': stroke_count,
        'normal_count': normal_count,
        'score_over_time': score_over_time
    })

def get_all_years_with_data():
    """
    Helper function to get all years for which data exists in either table.
    """
    try:
        with sqlite3.connect('users.db') as conn:
            c = conn.cursor()
            
            # Get years from user_inputs table
            c.execute("SELECT DISTINCT strftime('%Y', timestamp) FROM user_inputs WHERE timestamp IS NOT NULL")
            user_years = [row[0] for row in c.fetchall()]
            
            # Get years from image_detections table
            c.execute("SELECT DISTINCT strftime('%Y', timestamp) FROM image_detections WHERE timestamp IS NOT NULL")
            image_years = [row[0] for row in c.fetchall()]
            
            # Combine and sort the unique years
            all_years = sorted(list(set(user_years + image_years)), reverse=True)
            return all_years
    except sqlite3.Error as e:
        print(f"An error occurred while fetching years: {e}")
        return []

@app.route('/api/monthly_analytics_data')
def monthly_analytics_data():
    """
    New endpoint to provide data aggregated by month for a given year.
    It combines data from structured inputs and image detections.
    """
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    # Get the year from the request, default to current year
    current_year = datetime.now().year
    year = request.args.get('year', default=str(current_year), type=str)
    
    combined_data = get_all_user_predictions(session['username'])
    
    # Initialize a data structure for all months
    monthly_data = {
        month_name: {
            "Patient have stroke risk": 0,
            "Patient does NOT have stroke risk": 0,
            "Brain Stroke Detected": 0,
            "No Stroke Detected": 0,
            "Total_Scores": 0.0,
            "Count_Scores": 0,
        }
        for month_name in calendar.month_abbr[1:]
    }
    
    # Filter and aggregate data for the selected year
    for prediction in combined_data:
        try:
            timestamp = datetime.strptime(prediction['timestamp'], '%Y-%m-%d %H:%M:%S')
            if str(timestamp.year) == year:
                month_name = calendar.month_abbr[timestamp.month]
                
                # Aggregate counts for the bars
                if prediction['type'] == 'Structured':
                    if prediction['prediction_text'] == '⚠️ Patient has stroke risk':
                        monthly_data[month_name]["Patient have stroke risk"] += 1
                    elif prediction['prediction_text'] == '✅ Patient does NOT have stroke risk':
                        monthly_data[month_name]["Patient does NOT have stroke risk"] += 1
                elif prediction['type'] == 'Image':
                    if prediction['prediction_text'] == '🧠 Brain Stroke Detected':
                        monthly_data[month_name]["Brain Stroke Detected"] += 1
                    elif prediction['prediction_text'] == '✅ No Stroke Detected':
                        monthly_data[month_name]["No Stroke Detected"] += 1

                # Aggregate scores for the trend line
                if prediction['score'] is not None:
                    monthly_data[month_name]["Total_Scores"] += prediction['score']
                    monthly_data[month_name]["Count_Scores"] += 1
                    
        except ValueError:
            # Handle malformed timestamps if any exist
            continue

    # Prepare the final list of data for the chart
    final_chart_data = []
    for month_name, data in monthly_data.items():
        average_score = (data["Total_Scores"] / data["Count_Scores"]) if data["Count_Scores"] > 0 else 0
        
        final_chart_data.append({
            "month": month_name,
            "Patient have stroke risk": data["Patient have stroke risk"],
            "Patient does NOT have stroke risk": data["Patient does NOT have stroke risk"],
            "Brain Stroke Detected": data["Brain Stroke Detected"],
            "No Stroke Detected": data["No Stroke Detected"],
            "Average": round(average_score, 2),
        })

    # Get all years for the dropdown
    all_years = get_all_years_with_data()
    
    return jsonify({
        'data': final_chart_data,
        'years': all_years
    })

@app.route('/api/combined_predictions')
def combined_predictions():
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    page = request.args.get('page', 1, type=int)
    per_page = 10
    
    all_predictions = get_all_user_predictions(session['username'])
    total = len(all_predictions)
    pages = (total + per_page - 1) // per_page
    
    start = (page - 1) * per_page
    end = start + per_page
    
    paginated_predictions = all_predictions[start:end]

    return jsonify({
        'predictions': paginated_predictions,
        'total': total,
        'pages': pages,
        'page': page
    })

@app.route('/dataset')
def dataset():
    if 'username' not in session:
        return redirect(url_for('login'))
        
    all_detections = get_all_image_detections()
    
    # Pagination logic
    page = request.args.get('page', 1, type=int)
    per_page = 12
    total = len(all_detections)
    pages = total // per_page + (1 if total % per_page > 0 else 0)
    
    start = (page - 1) * per_page
    end = start + per_page
    paginated_detections = all_detections[start:end]

    dataset_stats = {
        'total_images': total,
        'stroke_cases': sum(1 for det in all_detections if 'brain' in det['prediction_text'].lower()),
        'normal_cases': sum(1 for det in all_detections if 'no stroke' in det['prediction_text'].lower())
    }
    
    return render_template('dataset.html', 
        detections=paginated_detections, 
        stats=dataset_stats,
        page=page,
        pages=pages,
        per_page=per_page
    )

@app.route('/browse')
def browse():
    if 'username' not in session:
        return redirect(url_for('login'))
        
    all_predections = get_all_user_inputs()

    # Pagination logic
    page = request.args.get('page', 1, type=int)
    per_page = 6
    total = len(all_predections)
    pages = total // per_page + (1 if total % per_page > 0 else 0)
    
    start = (page - 1) * per_page
    end = start + per_page
    paginated_predections = all_predections[start:end]
    
    return render_template('browse.html', 
        predections=paginated_predections, 
        page=page,
        pages=pages,
        per_page=per_page
    )

    # For browse page, we'll just display a list of all structured predictions
    return render_template('browse.html', predections=all_predections)

@app.route('/api/get_detection_details/<int:detection_id>')
def get_detection_details(detection_id):
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    detection = get_detection_by_id(detection_id)
    if detection:
        # Check if analysis_details is a JSON string and parse it
        if isinstance(detection['analysis_details'], str):
            detection['analysis_details'] = json.loads(detection['analysis_details'])
        return jsonify(dict(detection))
    
    return jsonify({'error': 'Detection not found'}), 404

# NEW ROUTE for image viewer
@app.route('/image-viewer/<int:detection_id>')
def image_viewer(detection_id):
    if 'username' not in session:
        return redirect(url_for('login'))

    detection = get_detection_by_id(detection_id)
    if not detection:
        return "Image not found", 404
        
    # === CORRECTED CODE ===
    # Here, we ensure the image path is formatted correctly before
    # passing it to the template, guaranteeing a valid URL.
    detection['image_path'] = url_for('static', filename=detection['image_path'].replace('\\', '/'))
    # ======================

    return render_template('image_viewer.html', detection=detection)


@app.route('/api/get_structured_prediction_details/<int:prediction_id>')
def get_structured_prediction_details(prediction_id):
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    predection = get_predection_by_id(prediction_id)
    if predection:
        # Convert the Row object to a dictionary for JSON serialization
        predection_dict = dict(predection)
        return jsonify(predection_dict)
    
    return jsonify({'error': 'Prediction not found'}), 404


@app.route('/api/analytics_summary')
def analytics_summary():
    if 'username' not in session:
        return jsonify({'error': 'Authentication required'}), 401

    user_inputs, image_detections = get_user_data(session['username'])
    
    total_predictions = len(user_inputs) + len(image_detections)
    stroke_count = sum(1 for item in user_inputs if 'risk' in item['prediction_text'].lower()) + sum(1 for item in image_detections if 'stroke' in item['prediction_text'].lower())
    normal_count = total_predictions - stroke_count
    
    all_risk_scores = [inp['risk_score'] for inp in user_inputs if inp['risk_score'] is not None]
    average_risk_score = sum(all_risk_scores) / len(all_risk_scores) if all_risk_scores else 0
    
    return jsonify({
        'total_predictions': total_predictions,
        'stroke_count': stroke_count,
        'normal_count': normal_count,
        'average_risk_score': average_risk_score
    })

@app.route('/footer')
def footer():
    return render_template('footer.html')

# Auth Routes
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        add_user(username, password)
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = get_user(username)
        if user and check_password_hash(user[2], password):
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('home'))

# Prediction Routes
@app.route('/predict-image', methods=['POST'])
def predict_image():
    """Endpoint for image prediction."""
    if 'username' not in session:
        return jsonify({'error': 'Authentication required', 'login_url': url_for('login')}), 401

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    return handle_image_prediction(file, session['username'])

@app.route('/predict-structured', methods=['POST'])
def predict_structured():
    """Endpoint for structured prediction."""
    print("Received request at /predict-structured")
    if 'username' not in session:
        print("User not authenticated.")
        return jsonify({'error': 'Authentication required', 'login_url': url_for('login')}), 401

    # Get JSON payload from the request body
    data = request.get_json()
    if not data:
        print("No JSON payload received.")
        return jsonify({'error': 'No JSON payload provided'}), 400
        
    print(f"Received data: {data}")
    return handle_structured_prediction(data, session['username'])

def handle_image_prediction(file, username):
    """Processes image, makes prediction, and saves to DB."""
    try:
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        img = cv2.imread(filepath)
        if img is None:
            return jsonify({'error': 'Invalid image file'}), 400
            
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        prediction_score = model_image.predict(img)[0][0]
        prediction_text = "🧠 Brain Stroke Detected" if prediction_score > 0.5 else "✅ No Stroke Detected"
        confidence = prediction_score if prediction_score > 0.5 else 1 - prediction_score
        
        # Mock analysis details for the report page
        analysis_details = [
            {'title': 'Left Middle Cerebral Artery (MCA) Territory', 'content': 'Affected area shows significant hypodensity consistent with acute ischemic stroke. Approximately 35% of the left MCA territory is involved.'},
            {'title': 'Right Posterior Cerebral Artery (PCA) Territory', 'content': 'Small focal area of restricted diffusion noted in the right occipital lobe, suggesting early ischemic changes in the PCA territory.'}
        ]
        
        # This is the original logic that was correct
        # It stores the full path relative to the app's root
        save_image_detection(username, filepath, prediction_text, float(confidence), analysis_details)
        
        return jsonify({
            'prediction_text': prediction_text,
            'img_path': url_for('static', filename=f'uploads/{filename}'),
            'confidence': f'{confidence * 100:.2f}%',
            'analysis_details': analysis_details
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def handle_structured_prediction(data, username):
    """Processes structured data, makes prediction, and saves to DB."""
    try:
        # Get values directly from the data dictionary (JSON payload)
        age = int(data['age'])
        gender = int(data['gender'])
        height = float(data['height'])
        weight = float(data['weight'])
        systolic = int(data['systolic'])
        diastolic = int(data['diastolic'])
        cholesterol = float(data['cholesterol'])
        glucose = float(data['glucose'])
        smoking = int(data['smoking'])
        activity = int(data['activity'])
        history = int(data['history'])

        bmi = round(weight / ((height / 100) ** 2), 1)

        # Calculate a mock risk score and level
        risk_score = 0
        if age >= 75: risk_score += 30
        elif age >= 65: risk_score += 20
        elif age >= 55: risk_score += 10
        elif age >= 45: risk_score += 5

        if systolic >= 140 or diastolic >= 90: risk_score += 25
        elif systolic >= 130 or diastolic >= 80: risk_score += 15
        elif systolic >= 120: risk_score += 5

        if cholesterol >= 240: risk_score += 20
        elif cholesterol >= 200: risk_score += 10

        if glucose >= 126: risk_score += 15
        elif glucose >= 100: risk_score += 5

        if smoking == 2: risk_score += 20
        elif smoking == 1: risk_score += 10

        if activity == 0: risk_score += 15
        elif activity == 1: risk_score += 5

        if history == 1: risk_score += 10
        if gender == 1: risk_score += 5

        if risk_score >= 70:
            risk_level = "High Risk"
            prediction_text = '⚠️ Patient has stroke risk'
        elif risk_score >= 40:
            risk_level = "Moderate Risk"
            prediction_text = '⚠️ Patient has stroke risk'
        else:
            risk_level = "Low Risk"
            prediction_text = '✅ Patient does NOT have stroke risk'

        # Save the structured data to the database
        save_structured_input(username, age, gender, height, weight, systolic, diastolic, cholesterol, glucose, smoking, activity, history, bmi, risk_score, risk_level, prediction_text)
        
        # Map integer codes back to text for the JSON response
        gender_text = 'Male' if gender == 1 else 'Female'
        smoking_text = {0: 'Never smoked', 1: 'Former smoker', 2: 'Current smoker'}.get(smoking, 'Unknown')
        activity_text = {0: 'Sedentary', 1: 'Light', 2: 'Moderate', 3: 'Active'}.get(activity, 'Unknown')
        history_text = 'Yes' if history == 1 else 'No'

        return jsonify({
            'risk_level': risk_level,
            'risk_score': risk_score,
            'prediction_text': prediction_text,
            'age': age,
            'gender': gender_text,
            'height': height,
            'weight': weight,
            'bloodPressure': f'{systolic}/{diastolic}',
            'cholesterol': cholesterol,
            'glucose': glucose,
            'smoking': smoking_text,
            'activity': activity_text,
            'history': history_text
        })

    except Exception as e:
        print(f"Error in handle_structured_prediction: {e}")
        return jsonify({'error': f'Invalid data provided: {str(e)}'}), 400

# Favicon and static files
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                             'favicon.ico', mimetype='image/vnd.microsoft.icon')

# New route to get all image detections
@app.route('/api/image_detections')
def api_image_detections():
    """
    API endpoint to retrieve all image detections from the database.
    """
    detections = get_all_image_detections()
    detections_list = []
    for det in detections:
        # Convert sqlite3.Row object to dictionary
        det_dict = dict(det)
        if det_dict['analysis_details']:
            det_dict['analysis_details'] = json.loads(det_dict['analysis_details'])
        detections_list.append(det_dict)
    return jsonify(detections_list)


if __name__ == '__main__':
    app.run(debug=True)