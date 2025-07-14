import firebase_admin
from firebase_admin import credentials, db
import os
import time
from flask import Flask, render_template, jsonify, request, Response
import threading
import cv2
from real_time_analysis import RealTimeAttentionAnalyzer

# Initialize Flask app
app = Flask(__name__)

# Firebase Admin initialization
cred = credentials.Certificate("key.json")  # Replace with the actual path
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://beekideeapp-default-rtdb.firebaseio.com/'  # Replace with your Firebase Realtime Database URL
})

# Initialize the real-time attention analyzer with student_id and session_id
analyzer = None
is_tracking = False

def generate_frames():
    global is_tracking, analyzer
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)  # Set camera FPS to ensure smoother capture

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)  # Flip the frame horizontally
        if analyzer and is_tracking:
            frame = analyzer.process_frame(frame)  # Process the frame using the real-time analyzer

        # Encode the frame to be sent to the frontend
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])  # Reduce JPEG quality for faster streaming
        frame = buffer.tobytes()

        # Yield the frame in a specific format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()

@app.route('/')
def index():
    # Show the video feed and controls for starting/stopping the tracking
    return render_template('index.html')

@app.route('/start_tracking', methods=['POST'])
def start_tracking():
    global is_tracking, analyzer

    # Get student_id and session_id from the POST request body
    student_id = request.json.get("student_id")
    session_id = request.json.get("session_id")
    
    if not student_id or not session_id:
        return jsonify({"status": "Student ID and Session ID are required!"}), 400

    if not is_tracking:
        is_tracking = True
        analyzer = RealTimeAttentionAnalyzer(student_id=student_id, session_id=session_id)
        analyzer.is_tracking = True
        
        # Start a separate thread for the real-time analyzer
        tracking_thread = threading.Thread(target=analyzer.run)
        tracking_thread.daemon = True
        tracking_thread.start()

        return jsonify({'status': 'Tracking started!', 'student_id': student_id, 'session_id': session_id})
    else:
        return jsonify({'status': 'Tracking is already running!', 'student_id': student_id, 'session_id': session_id})

@app.route('/pause_tracking', methods=['POST'])
def pause_tracking():
    global is_tracking, analyzer
    
    # Get student_id from the POST request body
    student_id = request.json.get("student_id")
    
    if not student_id:
        return jsonify({"status": "Student ID is required!"}), 400
    
    if is_tracking:
        is_tracking = False
        if analyzer:
            analyzer.is_tracking = False
        return jsonify({'status': 'Tracking paused!', 'student_id': student_id})
    else:
        return jsonify({'status': 'Tracking is already paused!', 'student_id': student_id})

@app.route('/stop_tracking', methods=['POST'])
def stop_tracking():
    global is_tracking, analyzer

    # Get student_id from the POST request body
    student_id = request.json.get("student_id")
    
    if not student_id:
        return jsonify({"status": "Student ID is required!"}), 400

    if is_tracking:
        is_tracking = False
        if analyzer:
            analyzer.is_tracking = False
            analyzer.stop()  # Call stop to clean up resources
        analyzer = None
        return jsonify({'status': 'Tracking stopped!', 'student_id': student_id})
    else:
        return jsonify({'status': 'Tracking is not running!', 'student_id': student_id})

@app.route('/get_attention_data/<student_id>', methods=['GET'])
def get_attention_data(student_id):
    # Fetch data specific to the student from Firebase
    if analyzer and analyzer.data['timestamp']:
        latest_data = analyzer.data
        return jsonify({
            'student_id': student_id,
            'data': latest_data
        })
    else:
        return jsonify({'student_id': student_id, 'message': 'No data available'})

@app.route('/video_feed')
def video_feed():
    # Stream the video feed to the frontend
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, threaded=True)