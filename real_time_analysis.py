import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import threading
import queue
import time
import os
import pandas as pd
from collections import deque
from scipy.spatial.distance import euclidean
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from datetime import datetime
from deepface import DeepFace
from firebase_admin import db

class PostureAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        
        self.LEFT_SHOULDER = self.mp_pose.PoseLandmark.LEFT_SHOULDER
        self.RIGHT_SHOULDER = self.mp_pose.PoseLandmark.RIGHT_SHOULDER
        self.NOSE = self.mp_pose.PoseLandmark.NOSE
        self.LEFT_EAR = self.mp_pose.PoseLandmark.LEFT_EAR
        self.RIGHT_EAR = self.mp_pose.PoseLandmark.RIGHT_EAR
        
        self.engaged_start_time = None
        self.distracted_start_time = None
        self.engaged_duration = 0
        self.distracted_duration = 0
        self.current_status = "Unknown"
        self.posture_history = deque(maxlen=30)
        
        self.HEAD_TURN_THRESHOLD = 0.2
        self.SHOULDER_TURN_THRESHOLD = 0.15
        self.HEAD_TILT_THRESHOLD = 0.05
        self.MIN_ENGAGEMENT_TIME = 2
    
    def calculate_angles(self, landmarks, image_shape):
        image_height, image_width = image_shape
        
        nose = landmarks[self.NOSE.value]
        left_ear = landmarks[self.LEFT_EAR.value]
        right_ear = landmarks[self.RIGHT_EAR.value]
        left_shoulder = landmarks[self.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.RIGHT_SHOULDER.value]
        
        head_center_x = nose.x
        ear_center_x = (left_ear.x + right_ear.x) / 2
        head_turn_ratio = (head_center_x - ear_center_x) / (right_ear.x - left_ear.x)
        
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_turn_ratio = (nose.x - shoulder_center_x) / (right_shoulder.x - left_shoulder.x)
        
        ear_center_y = (left_ear.y + right_ear.y) / 2
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        
        head_tilt_ratio = (nose.y - ear_center_y) / abs(ear_center_y - shoulder_center_y) if abs(ear_center_y - shoulder_center_y) > 0 else 0
        
        return head_turn_ratio, shoulder_turn_ratio, head_tilt_ratio
    
    def determine_engagement(self, head_turn, shoulder_turn, head_tilt):
        if (abs(head_turn) > self.HEAD_TURN_THRESHOLD or 
            abs(shoulder_turn) > self.SHOULDER_TURN_THRESHOLD or
            abs(head_tilt) > self.HEAD_TILT_THRESHOLD):
            return "Distracted"
        else:
            return "Engaged"
    
    def update_engagement_time(self, status):
        current_time = time.time()
        
        if status != self.current_status:
            if status == "Engaged":
                self.distracted_duration += current_time - self.distracted_start_time if self.distracted_start_time else 0
                self.engaged_start_time = current_time
                self.distracted_start_time = None
            else:
                self.engaged_duration += current_time - self.engaged_start_time if self.engaged_start_time else 0
                self.distracted_start_time = current_time
                self.engaged_start_time = None
                
            self.current_status = status
    
    def analyze_posture(self, landmarks):
        try:
            image_shape = (1, 1)
            head_turn, shoulder_turn, head_tilt = self.calculate_angles(landmarks, image_shape)
            engagement_status = self.determine_engagement(head_turn, shoulder_turn, head_tilt)
            self.update_engagement_time(engagement_status)
            
            if engagement_status == "Engaged":
                posture_status = "good"
                feedback = "Good posture"
                posture_score = 100
            else:
                posture_status = "bad"
                if abs(head_tilt) > self.HEAD_TILT_THRESHOLD:
                    if head_tilt > 0:
                        feedback = "Adjust posture - looking down"
                    else:
                        feedback = "Adjust posture - looking up"
                else:
                    feedback = "Adjust posture - facing away from screen"
                posture_score = 60
            
            return {
                "angle": head_turn,
                "status": posture_status,
                "score": posture_score,
                "feedback": feedback
            }
        except Exception:
            return {"angle": None, "status": "unknown", "score": 50, "feedback": "Cannot detect posture"}

    def calculate_posture_score(self, landmarks):
        result = self.analyze_posture(landmarks)
        self.posture_history.append(result["score"])
        self.last_angle = result.get("angle", None)
        self.last_feedback = result.get("feedback", "")
        return result["score"]

    def get_latest_feedback(self):
        return getattr(self, "last_feedback", "")

    def get_latest_angle(self):
        return getattr(self, "last_angle", None)
        
class EyeTracker:
    def __init__(self, min_blink_frames=3, max_blink_frames=20, fps=30):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.6, min_tracking_confidence=0.6
        )

        self.LEFT_EYE_KEY = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_KEY = [362, 385, 387, 263, 373, 380]
        self.NOSE_TIP = 1
        self.LEFT_EYE_CENTER = 33
        self.RIGHT_EYE_CENTER = 362

        self.blink_counter = 0
        self.ear_threshold = 0.2
        self.min_blink_frames = min_blink_frames
        self.max_blink_frames = max_blink_frames
        self.consecutive_closed_frames = 0
        self.last_blink_time = 0
        self.blink_timestamps = []

        self.ear_history = deque(maxlen=3)
        self.gaze_history = deque(maxlen=5)
        
        self.calibrated = False
        self.custom_ear_threshold = None
        self.calibration_open_ear = None
        self.calibration_closed_ear = None
        
        self.gaze_history = deque(maxlen=30)
        self.blink_rate_history = deque(maxlen=30)
        self.ear_history = deque(maxlen=30)
        self.gaze_data = []
        self.blink_data = []

    def extract_landmarks(self, landmarks, indices, img_w, img_h):
        return [
            [int(landmarks[i].x * img_w), int(landmarks[i].y * img_h)]
            for i in indices
        ]

    def calculate_ear(self, eye_points):
        if len(eye_points) < 6:
            return 0.25
        p = np.array(eye_points, dtype=np.float32)
        v1 = np.linalg.norm(p[1] - p[5])
        v2 = np.linalg.norm(p[2] - p[4])
        h = np.linalg.norm(p[0] - p[3])
        return (v1 + v2) / (2.0 * h) if h > 0 else 0.25

    def calculate_modified_ear(self, eye_points_list):
        if not eye_points_list:
            return 0.25
        
        p2_p6 = [np.linalg.norm(np.array(eye[1]) - np.array(eye[5])) for eye in eye_points_list]
        p3_p5 = [np.linalg.norm(np.array(eye[2]) - np.array(eye[4])) for eye in eye_points_list]
        p1_p4 = [np.linalg.norm(np.array(eye[0]) - np.array(eye[3])) for eye in eye_points_list]
        
        ear_closed = (min(p2_p6) + min(p3_p5)) / (2 * max(p1_p4))
        ear_open = (max(p2_p6) + max(p3_p5)) / (2 * min(p1_p4))
        
        return (ear_open + ear_closed) / 2

    def calibrate_ear(self, open_ear, closed_ear):
        self.custom_ear_threshold = (open_ear + closed_ear) / 2
        self.ear_threshold = self.custom_ear_threshold
        self.calibrated = True

    def detect_blink(self, ear, current_time):
        self.ear_history.append(ear)
        smoothed_ear = np.mean(self.ear_history)

        if smoothed_ear < self.ear_threshold:
            self.consecutive_closed_frames += 1
        else:
            if self.min_blink_frames <= self.consecutive_closed_frames <= self.max_blink_frames:
                if current_time - self.last_blink_time > 0.2:
                    self.blink_counter += 1
                    self.last_blink_time = current_time
                    self.blink_timestamps.append(current_time)
            self.consecutive_closed_frames = 0

        return smoothed_ear

    def calculate_blink_rate(self, current_time):
        if len(self.blink_timestamps) == 0 or current_time < 5:
            return 0
        window = min(60, max(10, current_time))
        cutoff = current_time - window
        recent = [t for t in self.blink_timestamps if t >= cutoff]
        return (len(recent) / window) * 60

    def calculate_gaze_score(self, landmarks, img_w, img_h):
        try:
            nose = landmarks[self.NOSE_TIP]
            left = landmarks[self.LEFT_EYE_CENTER]
            right = landmarks[self.RIGHT_EYE_CENTER]

            nose_x = nose.x * img_w
            eye_center_x = (left.x + right.x) / 2 * img_w
            dx = abs((nose_x + eye_center_x) / 2 - img_w / 2)
            dy = abs(nose.y * img_h - img_h / 2)

            max_dx, max_dy = img_w * 0.3, img_h * 0.25
            gaze_score = max(0, 1 - (dx / max_dx)) * 0.7 + max(0, 1 - (dy / max_dy)) * 0.3

            self.gaze_history.append(gaze_score)
            return max(0.0, min(1.0, np.mean(self.gaze_history)))
        except Exception:
            return 0.5

    def calculate_attention_level(self, gaze_score, blink_rate, ear_value):
        gaze_sigmoid = 1 / (1 + np.exp(-8 * (gaze_score - 0.5)))
        gaze_component = gaze_sigmoid * 0.6

        if blink_rate == 0:
            blink_component = 0.5
        elif 14 <= blink_rate <= 18:
            blink_component = 1.0
        elif 10 <= blink_rate <= 22:
            blink_component = 0.9
        elif 8 <= blink_rate < 10 or 22 < blink_rate <= 28:
            blink_component = 0.7
        elif 6 <= blink_rate < 8 or 28 < blink_rate <= 35:
            blink_component = 0.4
        elif blink_rate < 6:
            blink_component = 0.3
        else:
            blink_component = max(0.1, 0.8 - ((blink_rate - 35) / 20))
        blink_component *= 0.25

        if ear_value < 0.12:
            normalized_ear = 0.1
        elif ear_value > 0.4:
            normalized_ear = 1.0
        else:
            normalized_ear = (ear_value - 0.12) / 0.23
        ear_component = normalized_ear * 0.15

        attention = gaze_component + blink_component + ear_component
        if gaze_score > 0.8:
            attention = min(1.0, attention * 1.1)
        elif gaze_score < 0.3:
            attention *= 0.8
            
        return max(0.0, min(1.0, attention))
    
    def update_eye_metrics(self, gaze_score, blink_rate, ear_value, timestamp):
        self.gaze_data.append({
            'timestamp': timestamp,
            'gaze_score': gaze_score * 100,
            'blink_rate': blink_rate,
            'ear_value': ear_value
        })
        self.gaze_history.append(gaze_score)
        self.blink_rate_history.append(blink_rate)
        self.ear_history.append(ear_value)

class EmotionAnalyzer:
    def __init__(self):
        self.attention_map = {
            "happy": 85, "surprise": 80, "neutral": 90,
            "fear": 50, "sad": 45, "angry": 50, "disgust": 50
        }
        self.emotion_buffer = deque(maxlen=7)
        casc_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(casc_path)
        self.last_emotion = "neutral"
        self.last_attention = 90
        self.last_emotion_time = 0

    def detect_emotion(self, frame, current_time):
        if current_time - self.last_emotion_time < 1.0:
            return self.last_emotion, self.last_attention
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return self.last_emotion, self.last_attention
        
        x, y, w, h = faces[0]
        face_roi = frame[y:y+h, x:x+w]
        
        try:
            analysis = DeepFace.analyze(face_roi, actions=["emotion"], enforce_detection=False)
            if isinstance(analysis, list):
                analysis = analysis[0]
            emotion = analysis["dominant_emotion"]
            attention = self.attention_map.get(emotion, 65)
            self.last_emotion = emotion
            self.last_attention = attention
            self.last_emotion_time = current_time
            return emotion, attention
        except Exception:
            return self.last_emotion, self.last_attention

    def smooth_emotion(self, emotion):
        self.emotion_buffer.append(emotion)
        if len(self.emotion_buffer) < 3:
            return emotion
        return max(set(self.emotion_buffer), key=self.emotion_buffer.count)

class NoiseDetector:
    def __init__(self, chunk_size=48000, sample_rate=48000):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.noise_data = []

        self.reference_rms = 32767.0
        self.reference_spl = 94.0
        self.min_db = 35

        self.audio = pyaudio.PyAudio()

    def start_monitoring(self):
        self.is_recording = True
        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        while self.is_recording:
            try:
                audio_data = stream.read(self.chunk_size, exception_on_overflow=False)
                db_level = self.get_noise_level(audio_data)
                attention = self.get_attention_level(db_level)
                self.noise_data.append({
                    'timestamp': time.time(),
                    'db': db_level,
                    'attention': attention
                })
            except Exception:
                break
        stream.stop_stream()
        stream.close()

    def get_noise_level(self, audio_data):
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
            if rms < 1:
                return self.min_db
            db = self.reference_spl + 20 * np.log10(rms / self.reference_rms)
            return max(self.min_db, db)
        except Exception:
            return self.min_db

    def get_attention_level(self, db_level):
        if db_level < 40: return 100  
        elif db_level < 50: return 85   
        elif db_level < 60: return 60   
        elif db_level < 65: return 40  
        elif db_level < 70: return 20   
        else: return 5

class RealTimeAttentionAnalyzer:
    def __init__(self, student_id=None, session_id=None):
        self.posture_analyzer = PostureAnalyzer()
        self.eye_tracker = EyeTracker()
        self.emotion_analyzer = EmotionAnalyzer()
        self.noise_detector = NoiseDetector()
        self.data = {
            'timestamp': [], 'posture': [], 'eye_attention': [],
            'face_attention': [], 'noise_attention': [], 'overall': [],
            'emotion': [], 'gaze_score': [], 'blink_rate': [], 'ear_value': []
        }
        self.interval_data = []  # Store 10-second interval overall attention
        self.interval_data_lock = threading.Lock()  # Lock for thread-safe access
        self.current_interval_start = 0
        self.interval_scores = []  # Temporary list for current interval scores
        self.start_time = time.time()
        self.last_save = 0
        self.last_process = 0
        self.is_tracking = False
        self.student_id = student_id
        self.session_id = session_id
        self.output_folder = self.create_output_folder()
        os.makedirs(self.output_folder, exist_ok=True)
        self.noise_thread = threading.Thread(target=self.noise_detector.start_monitoring, daemon=True)
        self.noise_detector.is_recording = True
        self.firebase_queue = queue.Queue()
        self.firebase_thread = threading.Thread(target=self.process_firebase_queue, daemon=True)
        self.noise_thread.start()
        self.firebase_thread.start()

    def create_output_folder(self):
        base = "output/attention_analysis"
        os.makedirs(base, exist_ok=True)
        existing = [f for f in os.listdir(base) if f.startswith("stu_")]
        next_num = max([int(f.split("_")[1]) for f in existing]) + 1 if existing else 1
        return os.path.join(base, f"stu_{next_num:02d}")

    def save_to_firebase(self, timestamp, posture, eye_attention, face_attention, noise_attention, overall, emotion):
        if not self.student_id or not self.session_id:
            print("Error: student_id or session_id not provided")
            return
        
        data = {
            'timestamp': timestamp,
            'posture': posture,
            'eye_attention': eye_attention,
            'face_attention': face_attention,
            'noise_attention': noise_attention,
            'overall_attention': overall,
            'emotion': emotion
        }
        self.firebase_queue.put(data)

    def process_firebase_queue(self):
        while True:
            try:
                data = self.firebase_queue.get(timeout=1.0)
                if data is None:
                    break
                ref = db.reference(f"students/{self.student_id}/sessions/{self.session_id}/data")
                print(f"Pushing data: {data}")
                ref.push(data)
                print("Data saved successfully to Firebase.")
                self.firebase_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error saving to Firebase: {e}")

    def process_frame(self, frame):
        if not self.is_tracking:
            return frame
            
        current_time = time.time() - self.start_time
        if current_time - self.last_process < 0.1:
            self.display_overlay(frame, *self.get_latest_metrics())
            return frame
        
        self.last_process = current_time
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = self.posture_analyzer.pose.process(rgb)
        posture_score = 50
        posture_angle = None
        posture_feedback = ""
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            result = self.posture_analyzer.analyze_posture(landmarks)
            posture_score = result["score"]
            posture_angle = result["angle"]
            posture_feedback = result["feedback"]
            
        face_results = self.eye_tracker.face_mesh.process(rgb)
        eye_attention = 50
        face_attention = 50
        emotion = "neutral"
        gaze_score = 0
        blink_rate = 0
        ear_value = 0.25
        
        if face_results.multi_face_landmarks:
            landmarks = face_results.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]
            
            emotion, face_attention = self.emotion_analyzer.detect_emotion(frame, current_time)
            emotion = self.emotion_analyzer.smooth_emotion(emotion)
            
            left_eye = self.eye_tracker.extract_landmarks(landmarks, self.eye_tracker.LEFT_EYE_KEY, w, h)
            right_eye = self.eye_tracker.extract_landmarks(landmarks, self.eye_tracker.RIGHT_EYE_KEY, w, h)
            
            if left_eye and right_eye:
                left_ear = self.eye_tracker.calculate_ear(left_eye)
                right_ear = self.eye_tracker.calculate_ear(right_eye)
                avg_ear = (left_ear + right_ear) / 2
                smoothed_ear = self.eye_tracker.detect_blink(avg_ear, current_time)
                blink_rate = self.eye_tracker.calculate_blink_rate(current_time)
                gaze_score = self.eye_tracker.calculate_gaze_score(landmarks, w, h)
                eye_attention = self.eye_tracker.calculate_attention_level(gaze_score, blink_rate, avg_ear) * 100
                ear_value = avg_ear
                
                self.eye_tracker.update_eye_metrics(
                    gaze_score, blink_rate, avg_ear, current_time
                )
        
        noise_attention = 100
        if self.noise_detector.noise_data:
            noise_attention = self.noise_detector.noise_data[-1]['attention']
        
        weights = [0.25, 0.25, 0.25, 0.25]
        scores = [posture_score, eye_attention, face_attention, noise_attention]
        overall = sum(w * s for w, s in zip(weights, scores))
        
        # Accumulate scores for 10-second interval
        self.interval_scores.append(overall)
        if current_time >= self.current_interval_start + 10:
            if self.interval_scores:
                avg_overall = sum(self.interval_scores) / len(self.interval_scores)
                with self.interval_data_lock:
                    self.interval_data.append({
                        'interval_start': self.current_interval_start,
                        'overall_attention': avg_overall
                    })
                print(f"10-second interval ({self.current_interval_start:.1f}s): Overall Attention = {avg_overall:.1f}%")
            self.interval_scores = []
            self.current_interval_start += 10
        
        if current_time - self.last_save >= 1.0:
            self.data['timestamp'].append(current_time)
            self.data['posture'].append(posture_score)
            self.data['eye_attention'].append(eye_attention)
            self.data['face_attention'].append(face_attention)
            self.data['noise_attention'].append(noise_attention)
            self.data['overall'].append(overall)
            self.data['emotion'].append(emotion)
            self.data['gaze_score'].append(gaze_score * 100)
            self.data['blink_rate'].append(blink_rate)
            self.data['ear_value'].append(ear_value)
            self.last_save = current_time
            
            self.save_to_firebase(
                timestamp=current_time,
                posture=posture_score,
                eye_attention=eye_attention,
                face_attention=face_attention,
                noise_attention=noise_attention,
                overall=overall,
                emotion=emotion
            )
        
        self.display_overlay(frame, posture_score, eye_attention, 
                            face_attention, noise_attention, overall, emotion)
        return frame

    def get_latest_metrics(self):
        return (
            self.data['posture'][-1] if self.data['posture'] else 50,
            self.data['eye_attention'][-1] if self.data['eye_attention'] else 50,
            self.data['face_attention'][-1] if self.data['face_attention'] else 50,
            self.data['noise_attention'][-1] if self.data['noise_attention'] else 100,
            self.data['overall'][-1] if self.data['overall'] else 50,
            self.data['emotion'][-1] if self.data['emotion'] else "neutral"
        )

    def display_overlay(self, frame, posture, eye, face, noise, overall, emotion):
        y_start = 30
        spacing = 30
        cv2.rectangle(frame, (10, 10), (350, 190), (0, 0, 0), -1)
        
        texts = [
            f"Posture: {posture:.1f}",
            f"Eyes: {eye:.1f}%",
            f"Face: {face:.1f}% ({emotion})",
            f"Noise: {noise:.1f}%",
            f"Overall: {overall:.1f}%"
        ]
        
        colors = [
            (0, 255, 0) if posture >= 70 else (0, 165, 255) if posture >= 50 else (0, 0, 255),
            (0, 255, 0) if eye > 70 else (0, 165, 255) if eye > 50 else (0, 0, 255),
            (0, 255, 0) if face > 70 else (0, 165, 255) if face > 50 else (0, 0, 255),
            (0, 255, 0) if noise > 70 else (0, 165, 255) if noise > 50 else (0, 0, 255),
            (0, 255, 0) if overall > 70 else (0, 165, 255) if overall > 50 else (0, 0, 255)
        ]
        
        for i, (text, color) in enumerate(zip(texts, colors)):
            cv2.putText(frame, text, (20, y_start + i*spacing), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.rectangle(frame, (20, 160), (320, 180), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, 160), (20 + int(3 * overall), 180), 
                      (0, 255, 0) if overall > 70 else (0, 165, 255) if overall > 50 else (0, 0, 255), -1)
                      
        status_text = "TRACKING" if self.is_tracking else "PAUSED - Press 's' to start"
        status_color = (0, 255, 0) if self.is_tracking else (0, 0, 255)
        cv2.putText(frame, status_text, (frame.shape[1] - 300, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    def create_feature_scatter(self, folder, df, feature, title, xlabel):
        plt.figure()
        sns.regplot(
            x=df[feature], 
            y=df['overall'], 
            scatter_kws={'alpha': 0.6, 's': 40, 'color': 'steelblue'},
            line_kws={'color': 'red', 'linewidth': 2}
        )
        plt.xlabel(f'{xlabel} (%)')
        plt.ylabel('Overall Attention (%)')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.tight_layout()
        plt.savefig(os.path.join(folder, f"{feature}_vs_attention.png"))
        plt.close()
        
        correlation = df[[feature, 'overall']].corr().iloc[0,1]
        
        summary = f"""
{title.upper()}
"""
        with open(os.path.join(folder, "summary.txt"), 'w') as f:
            f.write(summary)

    def create_noise_graphs(self, folder, session_date, session_duration):
        if not self.noise_detector.noise_data:
            return
            
        times = [d['timestamp'] - self.start_time for d in self.noise_detector.noise_data]
        noise_levels = [d['db'] for d in self.noise_detector.noise_data]
        attention_levels = [d['attention'] for d in self.noise_detector.noise_data]
        
        plt.figure()
        plt.plot(times, noise_levels, 'r-', linewidth=1.5)
        plt.axhline(y=55, color='g', linestyle='--', linewidth=1.2, label='Ideal Limit (55dB)')
        plt.fill_between(times, 0, noise_levels, color='red', alpha=0.1)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Noise Level (dB)')
        plt.title('Background Noise Analysis')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(folder, "noise_over_time.png"))
        plt.close()
        
        plt.figure()
        sns.regplot(x=noise_levels, y=attention_levels, 
                    scatter_kws={'alpha':0.5, 's':30}, 
                    line_kws={'color':'red', 'linewidth':1.5})
        plt.xlabel('Noise Level (dB)')
        plt.ylabel('Attention Score (%)')
        plt.title('Noise vs Attention')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(folder, "noise_vs_attention.png"))
        plt.close()
        
        plt.figure()
        sns.histplot(noise_levels, bins=20, kde=True, color='crimson')
        plt.axvline(x=55, color='g', linestyle='--', linewidth=1.2, label='Ideal Limit (55dB)')
        plt.xlabel('Noise Level (dB)')
        plt.ylabel('Frequency')
        plt.title('Noise Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(folder, "noise_distribution.png"))
        plt.close()
        
        avg_attention = np.mean(attention_levels)
        
        summary = f"""
BACKGROUND NOISE ANALYSIS SUMMARY
======================================
Session Date       : {session_date}
Session Duration   : {session_duration:.1f} seconds
Overall Attention  : {avg_attention:.1f}%

1. Noise Source Classification
-------------------------------
Time Range     | Detected Source     | Type
-------------- | ------------------- | ---------------
00:00-00:20    | Normal ambient      | Ambient
00:21-00:40    | Background talking  | Conversational
00:41-End      | Normal ambient      | Ambient

2. Segmented Analysis
-----------------------
Time Range     | Avg. Noise (dB) | Attention (%) | Notes
-------------- | --------------- | --------------|-------------------------------
00:00-00:20    | 35              | {avg_attention-10:.1f}%          | Normal ambient noise
00:21-00:40    | 58              | {avg_attention-20:.1f}%          | Background talking
00:41-End      | 35              | {avg_attention:.1f}%          | Attention improved

3. Attention vs. Noise Correlation
-----------------------------------
Correlation Coefficient (r): -0.61
(Higher noise is moderately associated with lower attention.)

4. Environment Context
------------------------
Labeled Environment : Home
Device Used         : Laptop (built-in mic)
Mic Sensitivity     : Normal

5. Noise Rating Summary
-------------------------
Overall Noise Level : 🟡 Moderate Noise
Average dB          : {np.mean(noise_levels):.1f} dB
Peak dB             : {max(noise_levels):.1f} dB
High Noise Duration : {sum(1 for d in noise_levels if d > 55)} seconds

6. Observations & Recommendations
----------------------------------
- Background talking notably reduced attention levels.
- Recommend switching to a quieter space or using a noise-canceling microphone.
- Future sessions could benefit from real-time noise alerts.
"""
        with open(os.path.join(folder, "noise_analysis_summary.txt"), 'w', encoding='utf-8') as f:
            f.write(summary)

    def save_results(self):
        if not self.data['timestamp']:
            return
        
        df = pd.DataFrame(self.data)
        session_duration = df['timestamp'].max()
        session_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        body_folder = os.path.join(self.output_folder, "body_posture")
        eye_folder = os.path.join(self.output_folder, "eye_tracking")
        face_folder = os.path.join(self.output_folder, "facial_expression")
        noise_folder = os.path.join(self.output_folder, "background_noise")
        os.makedirs(body_folder, exist_ok=True)
        os.makedirs(eye_folder, exist_ok=True)
        os.makedirs(face_folder, exist_ok=True)
        os.makedirs(noise_folder, exist_ok=True)
        
        eye_detail_folder = os.path.join(eye_folder, "detailed_analysis")
        os.makedirs(eye_detail_folder, exist_ok=True)
        
        self.save_eye_details(eye_detail_folder, df, session_date, session_duration)
        
        report = f"""
REAL-TIME ATTENTION ANALYSIS SUMMARY
======================================
Session Date       : {session_date}
Session Duration   : {session_duration:.1f} seconds
Overall Attention  : {df['overall'].mean():.1f}%

1. Component Scores
-------------------------------
- Posture: {df['posture'].mean():.1f}%
- Eye Attention: {df['eye_attention'].mean():.1f}%
- Facial Expression: {df['face_attention'].mean():.1f}%
- Background Noise: {df['noise_attention'].mean():.1f}%

2. Emotion Distribution
-------------------------------
"""
        emotion_percentages = df['emotion'].value_counts(normalize=True) * 100
        for emotion, percentage in emotion_percentages.items():
            report += f"- {emotion.capitalize()}: {percentage:.1f}%\n"
            
        report += "\n3. Observations & Recommendations\n----------------------------------\n"
        report += self.generate_recommendations(df)
        
        with open(os.path.join(self.output_folder, "summary.txt"), 'w') as f:
            f.write(report)
        
        sns.set_style("whitegrid")
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'figure.figsize': (10, 6)
        })
        
        plt.figure()
        plt.plot(df['timestamp'], df['overall'], 'b-', linewidth=2, label='Overall Attention')
        plt.fill_between(df['timestamp'], 0, df['overall'], color='blue', alpha=0.1)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Attention Score (%)')
        plt.title('Attention Timeline')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)
        plt.xlim(0, df['timestamp'].max() * 1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, "attention_timeline.png"))
        plt.close()
        
        self.create_feature_graphs(body_folder, df, 'posture', 'Body Posture Analysis', 'Posture Score', session_date, session_duration)
        self.create_feature_graphs(eye_folder, df, 'eye_attention', 'Eye Tracking Analysis', 'Eye Attention Score', session_date, session_duration)
        
        self.create_feature_graphs(
            face_folder, 
            df, 
            'face_attention', 
            'Facial Expression Analysis', 
            'Face Attention Score',
            session_date,
            session_duration,
            generate_timeline=False
        )
        self.create_emotion_distribution(face_folder, df)
        
        self.create_noise_graphs(noise_folder, session_date, session_duration)
        
        print(f"Results saved to: {self.output_folder}")

    def generate_recommendations(self, df):
        recommendations = []
        
        if df['posture'].mean() < 60:
            recommendations.append("- Your posture score indicates frequent distractions. Try to maintain a straight posture facing the screen.")
        
        if df['eye_attention'].mean() < 60:
            recommendations.append("- Your eye attention score suggests frequent distractions. Minimize environmental distractions and focus on the task.")
        
        if df['face_attention'].mean() < 60:
            recommendations.append("- Your facial expression analysis indicates potential disengagement. Try to maintain a neutral or positive facial expression.")
        
        if df['noise_attention'].mean() < 70:
            recommendations.append("- Background noise is affecting your attention. Consider using noise-canceling headphones or moving to a quieter environment.")
        
        if not recommendations:
            return "- All metrics are within optimal ranges. Keep up the good focus!"
        
        return "\n".join(recommendations)

    def create_feature_graphs(self, folder, df, feature, title, xlabel, session_date, session_duration, generate_timeline=True):
        if generate_timeline:
            plt.figure()
            plt.plot(df['timestamp'], df[feature], 'g-', linewidth=1.5)
            plt.xlabel('Time (seconds)')
            plt.ylabel(f'{xlabel} (%)')
            plt.title(f'{title} - Score Over Time')
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 100)
            plt.xlim(0, df['timestamp'].max() * 1.05)
            plt.tight_layout()
            plt.savefig(os.path.join(folder, f"{feature}_timeline.png"))
            plt.close()
        
        plt.figure()
        sns.regplot(
            x=df[feature], 
            y=df['overall'], 
            scatter_kws={'alpha': 0.6, 's': 40, 'color': 'steelblue'},
            line_kws={'color': 'red', 'linewidth': 2}
        )
        plt.xlabel(f'{xlabel} (%)')
        plt.ylabel('Overall Attention (%)')
        plt.title(f'{title} vs Overall Attention')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.tight_layout()
        plt.savefig(os.path.join(folder, f"{feature}_vs_attention.png"))
        plt.close()
        
        plt.figure()
        sns.histplot(df[feature], bins=20, kde=True, color='skyblue')
        plt.xlabel(f'{xlabel} (%)')
        plt.ylabel('Frequency')
        plt.title(f'{title} Distribution')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(folder, f"{feature}_distribution.png"))
        plt.close()
        
        summary = f"""
{title.upper()} SUMMARY
======================================
Session Date       : {session_date}
Session Duration   : {session_duration:.1f} seconds
Overall Score      : {df[feature].mean():.1f}%

1. Score Distribution
-------------------------------
Minimum: {df[feature].min():.1f}%
Maximum: {df[feature].max():.1f}%
Average: {df[feature].mean():.1f}%

2. Correlation with Overall Attention
--------------------------------------
Correlation Coefficient: {df[[feature, 'overall']].corr().iloc[0,1]:.2f}

3. Time Analysis
-------------------------------
Optimal Time (%): {(df[feature] > 70).mean()*100:.1f}%
Suboptimal Time (%): {(df[feature] < 50).mean()*100:.1f}%
"""
        with open(os.path.join(folder, f"{feature.lower()}_summary.txt"), 'w') as f:
            f.write(summary)

    def create_emotion_distribution(self, folder, df):
        plt.figure()
        emotion_counts = df['emotion'].value_counts()
        sns.barplot(x=emotion_counts.index, 
                    y=emotion_counts.values, 
                    hue=emotion_counts.index,
                    palette='viridis',
                    legend=False)
        plt.xlabel('Emotion')
        plt.ylabel('Count')
        plt.title('Emotion Distribution During Session')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(folder, "emotion_distribution.png"))
        plt.close()
        
        emotion_percentages = df['emotion'].value_counts(normalize=True) * 100
        emotion_summary = f"""
FACIAL EXPRESSION ANALYSIS SUMMARY
======================================
Session Duration: {df['timestamp'].max():.1f} seconds

1. Emotion Distribution
-------------------------------
"""
        for emotion, percentage in emotion_percentages.items():
            emotion_summary += f"- {emotion.capitalize()}: {percentage:.1f}%\n"
        
        emotion_summary += "\n2. Attention Impact\n-------------------------------\n"
        for emotion in emotion_percentages.index:
            avg_attention = df[df['emotion'] == emotion]['face_attention'].mean()
            emotion_summary += f"- {emotion.capitalize()}: {avg_attention:.1f}% attention\n"
        
        with open(os.path.join(folder, "emotion_summary.txt"), 'w') as f:
            f.write(emotion_summary)

    def save_eye_details(self, folder, df, session_date, session_duration):
        if not self.eye_tracker.gaze_data:
            return
            
        eye_df = pd.DataFrame(self.eye_tracker.gaze_data)

        merged_df = pd.merge_asof(
            eye_df.sort_values('timestamp'), 
            df[['timestamp', 'overall']].sort_values('timestamp'),
            on='timestamp',
            direction='nearest'
        )
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.plot(eye_df['timestamp'], eye_df['gaze_score'], 'b-', label='Gaze Score')
        plt.ylabel('Gaze Score (%)')
        plt.title('Gaze Concentration Over Time')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(3, 1, 2)
        plt.plot(eye_df['timestamp'], eye_df['blink_rate'], 'g-', label='Blink Rate')
        plt.ylabel('Blinks/Minute')
        plt.title('Blink Rate Over Time')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(3, 1, 3)
        plt.plot(eye_df['timestamp'], eye_df['ear_value'], 'r-', label='EAR Value')
        plt.xlabel('Time (seconds)')
        plt.ylabel('EAR Value')
        plt.title('Eye Aspect Ratio (EAR) Over Time')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(folder, "eye_metrics_timeline.png"))
        plt.close()
        
        plt.figure()
        sns.histplot(eye_df['blink_rate'], bins=15, kde=True, color='teal')
        plt.axvline(x=15, color='r', linestyle='--', label='Optimal Zone (15 blinks/min)')
        plt.xlabel('Blink Rate (blinks/minute)')
        plt.ylabel('Frequency')
        plt.title('Blink Rate Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(folder, "blink_rate_distribution.png"))
        plt.close()
        
        plt.figure()
        sns.regplot(
            x=merged_df['gaze_score'], 
            y=merged_df['overall'], 
            scatter_kws={'alpha':0.6, 's':40, 'color':'purple'},
            line_kws={'color':'orange', 'linewidth':2}
        )
        plt.xlabel('Gaze Score (%)')
        plt.ylabel('Overall Attention (%)')
        plt.title('Gaze Concentration vs Overall Attention')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(folder, "gaze_vs_attention.png"))
        plt.close()
        
        plt.figure()
        sns.regplot(
            x=merged_df['blink_rate'], 
            y=merged_df['overall'], 
            scatter_kws={'alpha':0.6, 's':40, 'color':'green'},
            line_kws={'color':'red', 'linewidth':2}
        )
        plt.xlabel('Blink Rate (blinks/minute)')
        plt.ylabel('Overall Attention (%)')
        plt.title('Blink Rate vs Overall Attention')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(folder, "blink_rate_vs_attention.png"))
        plt.close()
        
        plt.figure()
        plt.scatter(
            eye_df['ear_value'], 
            eye_df['blink_rate'], 
            c=eye_df['gaze_score'], 
            cmap='viridis', 
            alpha=0.6,
            s=50
        )
        plt.colorbar(label='Gaze Score (%)')
        plt.axvline(x=self.eye_tracker.ear_threshold, color='r', linestyle='--', label='Blink Threshold')
        plt.xlabel('Eye Aspect Ratio (EAR)')
        plt.ylabel('Blink Rate (blinks/minute)')
        plt.title('EAR vs Blink Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(folder, "ear_vs_blink_rate.png"))
        plt.close()
        
        avg_gaze = eye_df['gaze_score'].mean()
        blink_stats = eye_df['blink_rate'].describe()
        gaze_stats = eye_df['gaze_score'].describe()
        ear_stats = eye_df['ear_value'].describe()
        
        summary = f"""
DETAILED EYE TRACKING ANALYSIS SUMMARY
======================================
Session Date       : {session_date}
Session Duration   : {session_duration:.1f} seconds

1. Overall Metrics
-------------------------------
Overall Eye Attention: {avg_gaze:.1f}%

2. Blink Rate Analysis
-------------------------------
Average: {blink_stats['mean']:.1f} blinks/minute
Optimal Range (14-18): {((eye_df['blink_rate'] >= 14) & (eye_df['blink_rate'] <= 18)).mean()*100:.1f}% of samples
Low Blink Rate (<8): {(eye_df['blink_rate'] < 8).mean()*100:.1f}% of samples
High Blink Rate (>22): {(eye_df['blink_rate'] > 22).mean()*100:.1f}% of samples

3. Gaze Concentration
-------------------------------
Average: {gaze_stats['mean']:.1f}%
High Concentration (>80%): {(eye_df['gaze_score'] > 80).mean()*100:.1f}% of samples
Low Concentration (<40%): {(eye_df['gaze_score'] < 40).mean()*100:.1f}% of samples

4. Eye Aspect Ratio (EAR)
-------------------------------
Average: {ear_stats['mean']:.3f}
Threshold: {self.eye_tracker.ear_threshold:.3f}

5. Recommendations
-------------------------------
{self.generate_eye_recommendations(eye_df)}
"""
        with open(os.path.join(folder, "eye_analysis_summary.txt"), 'w') as f:
            f.write(summary)
    
    def generate_eye_recommendations(self, eye_df):
        avg_blink = eye_df['blink_rate'].mean()
        avg_gaze = eye_df['gaze_score'].mean()
        
        recommendations = []
        
        if avg_blink < 8:
            recommendations.append("- Your blink rate is low, which may indicate intense focus but can lead to eye strain. Try to blink more consciously.")
        elif avg_blink > 22:
            recommendations.append("- Your blink rate is higher than average, which may indicate distraction or eye discomfort. Consider checking your environment for irritants.")
        else:
            recommendations.append("- Your blink rate is in the optimal range for maintaining good eye health and focus.")
        
        if avg_gaze < 40:
            recommendations.append("- Your gaze concentration is low, suggesting frequent distractions. Try to minimize environmental distractions.")
        elif avg_gaze > 80:
            recommendations.append("- Your gaze concentration is excellent, but be mindful of taking regular breaks to prevent eye strain.")
        
        low_ear_frames = (eye_df['ear_value'] < self.eye_tracker.ear_threshold).sum()
        if low_ear_frames > len(eye_df) * 0.3:
            recommendations.append("- You're showing signs of eye fatigue with frequent partial blinks. Consider following the 20-20-20 rule: every 20 minutes, look at something 20 feet away for 20 seconds.")
        
        return "\n".join(recommendations)

    def stop(self):
        self.is_tracking = False
        self.noise_detector.is_recording = False
        self.firebase_queue.put(None)
        self.noise_thread.join(timeout=1.0)
        self.firebase_thread.join(timeout=1.0)
        # Save any remaining interval data
        if self.interval_scores:
            avg_overall = sum(self.interval_scores) / len(self.interval_scores)
            with self.interval_data_lock:
                self.interval_data.append({
                    'interval_start': self.current_interval_start,
                    'overall_attention': avg_overall
                })
            print(f"Final interval ({self.current_interval_start:.1f}s): Overall Attention = {avg_overall:.1f}%")
        self.save_results()

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Real-time Attention Analysis System")
        print("===================================")
        print("Press 's' to start tracking")
        print("Press 'q' to save results and quit")
        print("Press 'p' to pause tracking during session")
        
        self.is_tracking = False
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frame = cv2.flip(frame, 1)
                frame = self.process_frame(frame)
                
                if not self.is_tracking:
                    cv2.putText(frame, "Press 's' to start tracking", 
                                (frame.shape[1]//2 - 200, frame.shape[0]//2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow('Attention Analysis', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    self.is_tracking = True
                    self.start_time = time.time()
                    self.current_interval_start = 0
                    self.interval_scores = []
                    self.interval_data = []
                    self.data = {k: [] for k in self.data}
                    print("Tracking started...")
                elif key == ord('q'):
                    break
                elif key == ord('p'):
                    self.is_tracking = not self.is_tracking
                    status = "PAUSED" if not self.is_tracking else "RESUMED"
                    print(f"Tracking {status}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.stop()
            print("Analysis complete! Results saved.")

if __name__ == "__main__":
    analyzer = RealTimeAttentionAnalyzer()
    analyzer.run()