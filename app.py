from flask import Flask, jsonify, send_from_directory, Response, request
from flask_cors import CORS
import cv2
import threading
import time
import os
import logging
import firebase_admin
from firebase_admin import credentials, db
from attention import RealTimeAttentionAnalyzer
from datetime import datetime
import json
import numpy as np
import requests

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:4200"}})

# Initialize Firebase Admin SDK with retry logic
max_retries = 3
for attempt in range(max_retries):
    try:
        cred = credentials.Certificate('key.json')
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://beekideeapp-default-rtdb.firebaseio.com/'
        })
        logger.info("Firebase initialized successfully")
        test_ref = db.reference('test')
        test_ref.set({'test_key': 'test_value', 'timestamp': datetime.now().isoformat()})
        logger.info("Test write to Firebase successful")
        break
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error during Firebase initialization attempt {attempt + 1}: {str(e)}")
        if attempt == max_retries - 1:
            raise
        time.sleep(2)  # Wait before retrying
    except Exception as e:
        logger.error(f"Failed to initialize Firebase on attempt {attempt + 1}: {str(e)}")
        if attempt == max_retries - 1:
            raise
        time.sleep(2)

# Global variables
analyzer = None
analysis_thread = None
is_running = False
output_folder = None
current_session = {}

# Define metric groups
METRIC_GROUPS = {
    'eyes': ['eye_attention', 'gaze_score', 'blink_rate', 'ear_value'],
    'face': ['face_attention', 'emotion'],
    'noise': ['noise_attention'],
    'posture': ['posture', 'engagement_status'],
    'all': ['posture', 'eye_attention', 'face_attention', 'noise_attention', 'overall', 'emotion', 'gaze_score', 'blink_rate', 'ear_value', 'engagement_status']
}

def serialize_data(data):
    """Convert data dictionary to a JSON-serializable format."""
    serialized = {}
    for key, value in data.items():
        if isinstance(value, (list, tuple)):
            serialized[key] = [
                float(v) if isinstance(v, (np.floating, np.integer)) else
                v.tolist() if isinstance(v, np.ndarray) else
                v if v is not None else None
                for v in value
            ]
        elif isinstance(value, (np.floating, np.integer)):
            serialized[key] = float(value)
        elif isinstance(value, np.ndarray):
            serialized[key] = value.tolist()
        else:
            serialized[key] = value if value is not None else None
    return serialized

@app.route('/start', methods=['POST'])
def start_analysis():
    global analyzer, is_running, output_folder, current_session
    if is_running:
        logger.warning("Analysis already running")
        return jsonify({'status': 'error', 'message': 'Analysis already running'}), 400
    
    data = request.get_json()
    student_id = data.get('student_id') if data else request.args.get('student_id')
    if not student_id:
        logger.warning("Student ID is required")
        return jsonify({'status': 'error', 'message': 'Student ID is required'}), 400
    
    try:
        logger.info(f"Starting new analysis for student: {student_id}")
        analyzer = RealTimeAttentionAnalyzer(student_id=student_id)
        output_folder = analyzer.output_folder
        is_running = True
        analyzer.is_tracking = True
        
        session_id = db.reference('students').child(student_id).child('sessions').push().key
        analyzer.session_id = session_id
        current_session = {
            'student_id': student_id,
            'session_id': session_id,
            'start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Initialize session data in Firebase
        session_ref = db.reference('students').child(student_id).child('sessions').child(session_id)
        try:
            session_ref.set({
                'timestamp': current_session['start_time'],
                'duration': 0,
                'data': {}
            })
            logger.info(f"Initialized Firebase session for student: {student_id}, session: {session_id}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to initialize Firebase session: {str(e)}")
            return jsonify({'status': 'error', 'message': f"Failed to initialize Firebase session: {str(e)}"}), 500
        
        def run_analyzer():
            global is_running
            try:
                logger.debug("Starting analyzer.run()")
                analyzer.run()
                logger.debug("Analyzer.run() completed")
            except Exception as e:
                logger.error(f"Error in analyzer.run(): {str(e)}")
            finally:
                logger.info("Cleaning up analysis")
                is_running = False
                analyzer.noise_detector.is_recording = False
                if analyzer.noise_thread:
                    analyzer.noise_thread.join(timeout=1.0)
                analyzer.save_results()
                logger.info("Analysis stopped")
        
        global analysis_thread
        analysis_thread = threading.Thread(target=run_analyzer, daemon=True)
        analysis_thread.start()
        logger.info(f"Analysis started, output folder: {output_folder}, session_id: {session_id}")
        return jsonify({
            'status': 'success',
            'message': 'Analysis started',
            'output_folder': output_folder,
            'session_id': session_id
        })
    except Exception as e:
        logger.error(f"Error starting analysis: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/stop', methods=['POST'])
def stop_analysis():
    global analyzer, is_running, current_session
    logger.debug("Entering stop_analysis endpoint")
    logger.debug(f"Current session: {current_session}")
    
    if not is_running or not analyzer:
        logger.warning("No analysis running")
        return jsonify({'status': 'error', 'message': 'No analysis running'}), 400
    
    try:
        logger.info("Stopping analysis")
        analyzer.is_tracking = False
        is_running = False
        if analysis_thread:
            analysis_thread.join(timeout=2.0)
        
        if not current_session:
            logger.error("No current session data available")
            return jsonify({'status': 'error', 'message': 'No current session data'}), 400
        
        # Log analyzer data state
        logger.debug(f"Analyzer data: {analyzer.data}")
        logger.debug(f"Timestamp length: {len(analyzer.data['timestamp'])}")
        
        if not analyzer.data['timestamp']:
            logger.error("No timestamp data available in analyzer")
            # Attempt to collect one final frame
            logger.debug("Attempting to process one final frame")
            if hasattr(analyzer, 'cap') and analyzer.cap.isOpened():
                ret, frame = analyzer.cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    analyzer.process_frame(frame)
                    logger.debug(f"After final frame, timestamp length: {len(analyzer.data['timestamp'])}")
        
        if not analyzer.data['timestamp']:
            logger.error("Still no data collected after final attempt")
            return jsonify({'status': 'error', 'message': 'No data collected during analysis'}), 400
        
        student_id = current_session['student_id']
        session_id = current_session['session_id']
        session_duration = time.time() - analyzer.start_time
        session_ref = db.reference('students').child(student_id).child('sessions').child(session_id)
        
        session_data = {
            'timestamp': current_session['start_time'],
            'duration': session_duration,
            'data': serialize_data(analyzer.data)
        }
        
        logger.debug(f"Serialized data to save: {json.dumps(session_data, indent=2)}")
        
        # Retry Firebase write up to 3 times
        for attempt in range(3):
            try:
                session_ref.set(session_data)
                logger.info(f"Final data saved to Firebase for student: {student_id}, session: {session_id}")
                saved_data = session_ref.get()
                if saved_data:
                    logger.info("Data verified in Firebase")
                    break
                else:
                    logger.warning(f"Data saved but not retrievable on attempt {attempt + 1}")
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to save final data to Firebase on attempt {attempt + 1}: {str(e)}")
                if attempt == 2:
                    return jsonify({'status': 'error', 'message': f"Failed to save final data to Firebase after 3 attempts: {str(e)}"}), 500
                time.sleep(1)  # Wait before retrying
        
        analyzer.save_results()
        logger.info("Analysis stopped successfully")
        return jsonify({
            'status': 'success',
            'message': 'Analysis stopped and data saved',
            'session_id': session_id
        })
    except Exception as e:
        logger.error(f"Error stopping analysis: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

def get_data(metrics=None):
    global analyzer
    if not analyzer or not is_running:
        logger.warning("No analysis running for data request")
        return jsonify({'status': 'error', 'message': 'No analysis running'}), 400
    try:
        if not analyzer.data['timestamp']:
            logger.warning("No data available yet")
            return jsonify({'status': 'error', 'message': 'No data available yet'}), 400
        
        if metrics is None:
            metrics = METRIC_GROUPS['all']
        else:
            valid_metrics = METRIC_GROUPS['all']
            invalid_metrics = [m for m in metrics if m not in valid_metrics]
            if invalid_metrics:
                logger.warning(f"Invalid metrics requested: {invalid_metrics}")
                return jsonify({'status': 'error', 'message': f"Invalid metrics: {invalid_metrics}"}), 400
        
        data_array = [
            {'timestamp': analyzer.data['timestamp'][i], **{key: analyzer.data[key][i] for key in metrics}}
            for i in range(len(analyzer.data['timestamp']))
        ]
        logger.debug(f"Returning {len(data_array)} data points with metrics: {metrics}")
        return jsonify({'status': 'success', 'data': data_array})
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/data', methods=['GET'])
def get_data_combined():
    metrics_param = request.args.get('metrics', '')
    if not metrics_param:
        return get_data(metrics=METRIC_GROUPS['all'])
    metrics = [m.strip() for m in metrics_param.split(',')]
    return get_data(metrics=metrics)

@app.route('/data/eyes', methods=['GET'])
def get_eyes_data():
    return get_data(metrics=METRIC_GROUPS['eyes'])

@app.route('/data/face', methods=['GET'])
def get_face_data():
    return get_data(metrics=METRIC_GROUPS['face'])

@app.route('/data/noise', methods=['GET'])
def get_noise_data():
    return get_data(metrics=METRIC_GROUPS['noise'])

@app.route('/data/posture', methods=['GET'])
def get_posture_data():
    return get_data(metrics=METRIC_GROUPS['posture'])

@app.route('/data/all', methods=['GET'])
def get_all_data():
    return get_data(metrics=METRIC_GROUPS['all'])

@app.route('/reports/<path:filename>', methods=['GET'])
def get_report(filename):
    global output_folder
    if not output_folder or not os.path.exists(output_folder):
        logger.warning("No reports available")
        return jsonify({'status': 'error', 'message': 'No reports available'}), 404
    try:
        file_path = os.path.join(output_folder, filename)
        if os.path.exists(file_path):
            logger.info(f"Serving report: {file_path}")
            return send_from_directory(output_folder, filename)
        else:
            logger.warning(f"Report file not found: {file_path}")
            return jsonify({'status': 'error', 'message': 'File not found'}), 404
    except Exception as e:
        logger.error(f"Error serving report: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/video_feed')
def video_feed():
    def generate():
        global analyzer, is_running
        logger.debug("Starting video feed")
        while is_running and analyzer:
            try:
                ret, frame = analyzer.cap.read() if hasattr(analyzer, 'cap') and analyzer.cap.isOpened() else (False, None)
                if not ret:
                    logger.warning("Failed to read frame from webcam")
                    time.sleep(0.1)
                    continue
                frame = cv2.flip(frame, 1)
                frame = analyzer.process_frame(frame)
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    logger.warning("Failed to encode frame")
                    continue
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(0.03)
            except Exception as e:
                logger.error(f"Error in video feed: {str(e)}")
                time.sleep(0.1)
                continue
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    logger.info("Starting Flask server")
    app.run(host='0.0.0.0', port=5000, debug=True)