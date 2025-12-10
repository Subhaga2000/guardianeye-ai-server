import cv2
import torch
from flask import Flask, request
import mysql.connector
import threading
from ultralytics import YOLO
import time
import os
import numpy as np
from tensorflow.keras.models import load_model 

app = Flask(__name__)
print("Loading AI models...")

# --- Load models ---
model_violence = YOLO('models/best_violence.pt') 
print("Loaded Violence (YOLO) model: best_violence.pt")
model_gunshot = YOLO('models/gunshot_best_finetune.pt') 
print("Loaded Gunshot (YOLO) model: gunshot_best_finetune.pt")

try:
    model_robbery = load_model('models/shoplifting_model.h5')
    print("Loaded Robbery (CNN) model: shoplifting_model.h5")
except Exception as e:
    print(f"--- !!! WARNING: Robbery (CNN) model FAILED to load: {e} !!! ---")
    model_robbery = None

try:
    device = torch.device('cpu')
    model_fighting = torch.jit.load('models/fighting_model.pt', map_location=device)
    model_fighting.eval()
    print("Loaded Fighting (RNN/TorchScript) model: fighting_model.pt (Forced to CPU)")
except Exception as e:
    print(f"--- !!! ERROR: Fighting (RNN) model FAILED to load: {e} !!! ---")
    model_fighting = None

print("\nAll models loaded. AI Server is running...")

# --- Database Connections ---
db_config_cctv = {
    'user': 'root',
    'password': '',
    'host': '127.0.0.1',
    'database': 'guardianeye_cctv' 
}
db_config_dashboard = {
    'user': 'root',
    'password': '',
    'host': '127.0.0.1',
    'database': 'guardianeye' 
}

# --- Helper: Crime Priority ---
def get_crime_priority(crime_name):
    priorities = {
        "Gunshot": 3,
        "Robbery": 2,
        "Fighting": 1,
        "Violence Detected": 0,
        "Unknown": -1
    }
    return priorities.get(crime_name, 0)

# --- Preprocessing ---
def preprocess_frame_for_keras(frame):
    try:
        img_size = (224, 224) 
        img = cv2.resize(frame, img_size)
        img = img / 255.0  
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        print(f"preprocess_frame_for_keras error: {e}")
        return None

def preprocess_frame_for_pytorch(frame):
    try:
        img_size = (224, 224) 
        img = cv2.resize(frame, img_size)
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return torch.from_numpy(img).float()
    except Exception as e:
        print(f"preprocess_frame_for_pytorch error: {e}")
        return None

# --- Classify Crime ---
def classify_crime(frame):
    # 1. Check Gunshot (Priority 1)
    try:
        # Keep at 0.60 to balance sensitivity vs false positives
        gunshot_results = model_gunshot(frame, conf=0.60, verbose=False)
        for r in gunshot_results:
            if len(r.boxes.cls) > 0:
                return "Gunshot" 
    except Exception as e:
        print(f"classify_crime (gunshot) error: {e}")

    # 2. Check Robbery (Priority 2)
    try:
        if model_robbery is not None:
            input_keras = preprocess_frame_for_keras(frame)
            if input_keras is not None:
                prediction = model_robbery.predict(input_keras, verbose=0)
                if prediction[0][0] > 0.5: 
                    return "Robbery"
    except Exception as e:
        print(f"classify_crime (robbery) error: {e}")

    # 3. Check Fighting (Priority 3)
    try:
        if model_fighting is not None:
            input_tensor = preprocess_frame_for_pytorch(frame)
            if input_tensor is not None:
                with torch.no_grad():
                    prediction = model_fighting(input_tensor)
                predicted_index = torch.argmax(prediction, dim=1).item()
                if predicted_index == 1:
                    return "Fighting"
    except Exception as e:
        print(f"classify_crime (fighting) error: {e}")

    return "Violence Detected"

# --- Process Camera Feed ---
def process_camera_feed(cctv_id, stream_url):
    print(f"Starting to process feed from source: {stream_url} for CCTV ID: {cctv_id}")
    if str(stream_url).isdigit():
        camera_source = int(stream_url)
    else:
        camera_source = stream_url
    
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print(f"Error: Cannot open camera source '{camera_source}'.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = 20.0

    is_recording = False
    video_writer = None
    incident_id_being_recorded = None
    recording_start_time = 0
    VIDEO_DURATION = 15 
    
    current_recorded_label = "Unknown"

    dashboard_path = "C:/xampp new/htdocs/GuardianEye/"
    
    image_capture_interval = 1.5 
    last_image_time = 0
    images_taken = 0
    MAX_IMAGES = 8

    # --- NEW: Glitch Filter Variables ---
    violent_frame_streak = 0 
    REQUIRED_STREAK = 5  # Must see violence for 5 frames in a row (approx 0.25s)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(1)
            continue
            
        status_text = "Status: Non-Violence"
        status_color = (0, 255, 0)
        is_violent_frame = False
        
        # 1. VIOLENCE DETECTION
        try:
            violence_results = model_violence(frame, verbose=False)
            for r in violence_results:
                if len(getattr(r, "boxes", [])) > 0:
                    for conf in r.boxes.conf:
                        # CHANGED: Raised slightly to 0.55 to reduce weak false alarms
                        if float(conf) > 0.55:
                            is_violent_frame = True
                            break
                if is_violent_frame:
                    break
        except Exception as e:
            print(f"violence detection error: {e}")

        # 2. STREAK CHECK (The Glitch Filter)
        if is_violent_frame:
            violent_frame_streak += 1
        else:
            violent_frame_streak = 0 # Reset if violence stops even for 1 frame

        # Only consider it real if streak is met OR we are already recording
        # 
        is_sustained_violence = (violent_frame_streak >= REQUIRED_STREAK)

        # 3. RECORDING LOGIC
        if is_sustained_violence and not is_recording:
            # --- START RECORDING ---
            is_recording = True
            recording_start_time = time.time()
            
            current_recorded_label = classify_crime(frame)
            
            status_text = f"VIOLENCE DETECTED - CRIME: {current_recorded_label.upper()}"
            status_color = (0, 0, 255)
            print(f"VIOLENCE DETECTED (Streak {violent_frame_streak})! Starting recording for {current_recorded_label}...")
            
            incident_id_being_recorded = handle_crime_detection_START(frame.copy(), current_recorded_label, cctv_id)
            
            if incident_id_being_recorded:
                incident_folder_name = f"incident_{incident_id_being_recorded}"
                incident_folder_path = os.path.join(dashboard_path, "assets", "incident", incident_folder_name)
                os.makedirs(incident_folder_path, exist_ok=True)
                
                video_file_name = f"crime_video_{incident_id_being_recorded}.mp4"
                video_save_path = os.path.join(incident_folder_path, video_file_name)
                
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                video_writer = cv2.VideoWriter(video_save_path, fourcc, fps, (frame_width, frame_height))
                if not video_writer.isOpened():
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(video_save_path, fourcc, fps, (frame_width, frame_height))

                images_taken = 0
                last_image_time = 0
            
        elif is_recording:
            # --- RECORDING IN PROGRESS ---
            
            # Check for upgrades
            new_detection = classify_crime(frame)
            current_priority = get_crime_priority(current_recorded_label)
            new_priority = get_crime_priority(new_detection)
            
            if new_priority > current_priority:
                print(f"CRIME ESCALATED: {current_recorded_label} -> {new_detection}")
                current_recorded_label = new_detection
                threading.Thread(target=update_incident_crime_type, args=(incident_id_being_recorded, new_detection)).start()

            status_text = f"VIOLENCE DETECTED - CRIME: {current_recorded_label.upper()}"
            status_color = (0, 0, 255)
            
            if video_writer is not None and video_writer.isOpened():
                video_writer.write(frame)
            
            if images_taken < MAX_IMAGES and (time.time() - last_image_time > image_capture_interval):
                if incident_id_being_recorded:
                    incident_folder_name = f"incident_{incident_id_being_recorded}"
                    threading.Thread(
                        target=save_extra_image,
                        args=(frame.copy(), incident_id_being_recorded, images_taken + 1, incident_folder_name)
                    ).start()
                    images_taken += 1
                    last_image_time = time.time()

            if time.time() - recording_start_time > VIDEO_DURATION:
                print(f"Finished recording for incident {incident_id_being_recorded}.")
                is_recording = False
                violent_frame_streak = 0 # Reset streak so it doesn't restart immediately
                
                if video_writer is not None:
                    video_writer.release()
                video_writer = None
                
                if incident_id_being_recorded:
                    handle_crime_detection_END_VIDEO(
                        incident_id_being_recorded,
                        f"incident_{incident_id_being_recorded}",
                        f"crime_video_{incident_id_being_recorded}.mp4"
                    )

                incident_id_being_recorded = None
                current_recorded_label = "Unknown"
                threading.Thread(target=update_cctv_status, args=(cctv_id, "Active")).start()

        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2, cv2.LINE_AA)
        cv2.imshow(f'Monitoring CCTV ID: {cctv_id}', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    if is_recording and video_writer:
        video_writer.release()
    threading.Thread(target=update_cctv_status, args=(cctv_id, "Active")).start()


# --- Database Functions ---

def update_incident_crime_type(incident_id, new_crime_name):
    if not incident_id: return
    try:
        db_dash = mysql.connector.connect(**db_config_dashboard)
        cursor_dash = db_dash.cursor(dictionary=True)
        
        cursor_dash.execute("SELECT crime_type_id FROM crime_types WHERE crime_name = %s", (new_crime_name,))
        crime_type_data = cursor_dash.fetchone()
        crime_type_id = crime_type_data['crime_type_id'] if crime_type_data else 1
        
        sql_update = "UPDATE incidents SET crime_type_id = %s WHERE incident_id = %s"
        cursor_dash.execute(sql_update, (crime_type_id, incident_id))
        db_dash.commit()
        
        msg_append = f" [Updated to {new_crime_name}]"
        sql_notif = "UPDATE notifications SET message = CONCAT(message, %s), status = 'Unread' WHERE incident_id = %s"
        cursor_dash.execute(sql_notif, (msg_append, incident_id))
        db_dash.commit()
        
        cursor_dash.close()
        db_dash.close()
        print(f"DB UPDATED: Incident {incident_id} is now {new_crime_name}")
    except Exception as e:
        print(f"update_incident_crime_type error: {e}")

def update_cctv_status(cctv_id, status):
    try:
        db_cctv = mysql.connector.connect(**db_config_cctv)
        cursor_cctv = db_cctv.cursor()
        cursor_cctv.execute("UPDATE Cameras SET status = %s WHERE camera_id = %s", (status, cctv_id))
        db_cctv.commit()
        cursor_cctv.close()
        db_cctv.close()
    except Exception as e:
        print(f"CCTV status error: {e}")

def save_extra_image(frame, incident_id, img_index, folder_name):
    try:
        dashboard_path = "C:/xampp new/htdocs/GuardianEye/"
        incident_folder_path = os.path.join(dashboard_path, "assets", "incident", folder_name)
        os.makedirs(incident_folder_path, exist_ok=True)
        
        snapshot_name = f"snapshot_{img_index}.jpg"
        snapshot_save_path = os.path.join(incident_folder_path, snapshot_name)
        cv2.imwrite(snapshot_save_path, frame)
        
        web_path = f"assets/incident/{folder_name}/{snapshot_name}"
        
        db_dash = mysql.connector.connect(**db_config_dashboard)
        cursor_dash = db_dash.cursor()
        sql_media = "INSERT INTO crime_media (incident_id, media_type, file_path) VALUES (%s, %s, %s)"
        val_media = (incident_id, 'image', web_path)
        cursor_dash.execute(sql_media, val_media)
        db_dash.commit()
        cursor_dash.close()
        db_dash.close()
    except Exception as e:
        print(f"Image save error: {e}")

def handle_crime_detection_START(frame, crime_type, cctv_id):
    new_incident_id = None
    try:
        db_cctv = mysql.connector.connect(**db_config_cctv)
        cursor_cctv = db_cctv.cursor(dictionary=True)
        cursor_cctv.execute("SELECT location_name, latitude, longitude FROM Cameras WHERE camera_id = %s", (cctv_id,))
        cctv_data = cursor_cctv.fetchone()
        cursor_cctv.close()
        db_cctv.close()
        if not cctv_data:
            return None
        location, lat, lon = cctv_data['location_name'], cctv_data['latitude'], cctv_data['longitude']

        db_dash = mysql.connector.connect(**db_config_dashboard)
        cursor_dash = db_dash.cursor(dictionary=True)
        cursor_dash.execute("SELECT crime_type_id FROM crime_types WHERE crime_name = %s", (crime_type,))
        crime_type_data = cursor_dash.fetchone()
        crime_type_id = crime_type_data['crime_type_id'] if crime_type_data else 1
        
        sql = "INSERT INTO incidents (crime_type_id, location, latitude, longitude, incident_time, status) VALUES (%s, %s, %s, %s, NOW(), 'Active')"
        val = (crime_type_id, location, lat, lon)
        cursor_dash.execute(sql, val)
        db_dash.commit()
        new_incident_id = cursor_dash.lastrowid
        print(f"New incident created with ID: {new_incident_id}")

        sql_notif = "INSERT INTO notifications (incident_id, notifications_message, status, created_at) VALUES (%s, %s, 'Unread', NOW())"
        notif_text = f"New {crime_type} incident detected at {location}."
        cursor_dash.execute(sql_notif, (new_incident_id, notif_text))
        db_dash.commit()
        
        cursor_dash.close()
        db_dash.close()
        threading.Thread(target=update_cctv_status, args=(cctv_id, "Violence")).start()
        return new_incident_id
    except Exception as e:
        print(f"DB START error: {e}")
        return None

def handle_crime_detection_END_VIDEO(incident_id, folder_name, video_name):
    try:
        db_dash = mysql.connector.connect(**db_config_dashboard)
        cursor_dash = db_dash.cursor()
        web_path = f"assets/incident/{folder_name}/{video_name}"
        sql_media = "INSERT INTO crime_media (incident_id, media_type, file_path) VALUES (%s, %s, %s)"
        val_media = (incident_id, 'video', web_path)
        cursor_dash.execute(sql_media, val_media)
        db_dash.commit()
        cursor_dash.close()
        db_dash.close()
        print(f"Video path saved for incident {incident_id} -> {web_path}")
    except Exception as e:
        print(f"DB END VIDEO error: {e}")

@app.route('/start-monitoring', methods=['GET'])
def start_monitoring():
    cctv_id = request.args.get('cctv_id')
    stream_url = request.args.get('stream_url')
    if not cctv_id or not stream_url:
        return "Error", 400
    monitor_thread = threading.Thread(target=process_camera_feed, args=(cctv_id, stream_url), daemon=True)
    monitor_thread.start()
    return f"Started monitoring CCTV ID {cctv_id}"

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)