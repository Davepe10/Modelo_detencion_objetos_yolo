from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO
import cv2
import os
import threading
from collections import Counter
from utils.tts import speak, message_queue, process_queue
import logging
import queue
import time

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# -------- Configuración Global ----------
MODEL_FOLDER = 'models'
DEFAULT_MODEL = 'modelo2.pt'
models = [f for f in os.listdir(MODEL_FOLDER) if f.endswith('.pt')]
selected_model = DEFAULT_MODEL
selected_camera = 0
camera_on = False
cap = None
frame_rate = 15
previous_counts = {}
model = None
tts_enabled = True

# -------- Utilidades ----------
def load_model(model_name):
    logging.info(f"Cargando modelo: {model_name}")
    return YOLO(os.path.join(MODEL_FOLDER, model_name))

def get_available_cameras(max_test=5):
    available = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:
            available.append((i, f"Cámara {i+1}"))
        cap.release()
    return available or [(0, "Cámara 1")]

def notify(text):
    if tts_enabled:
        message_queue.put(text)

def detect_objects():
    global selected_model, selected_camera, camera_on, previous_counts, cap, model
    cap = cv2.VideoCapture(selected_camera)
    model = load_model(selected_model)

    while camera_on and cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        boxes = results.boxes
        names = model.names
        counts = Counter()

        for box in boxes:
            cls = int(box.cls[0])
            counts[names[cls]] += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{names[cls]} {box.conf[0]:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if counts != previous_counts:
            text = "Veo " + ", ".join([f"{v} {k}" for k, v in counts.items()])
            notify(text)
            previous_counts = counts

        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

        elapsed_time = time.time() - start_time
        sleep_time = 1 / frame_rate - elapsed_time
        if sleep_time > 0:
            time.sleep(sleep_time)

    if cap:
        cap.release()


@app.route('/')
def index():
    cameras = get_available_cameras()
    return render_template('index.html',
                           models=models,
                           default_model=DEFAULT_MODEL,
                           cameras=cameras,
                           selected_camera=selected_camera)

@app.route('/video_feed')
def video_feed():
    global selected_model, selected_camera, camera_on
    selected_model = request.args.get('model', DEFAULT_MODEL)
    selected_camera = int(request.args.get('camera', 0))
    camera_on = True
    threading.Thread(target=detect_objects, daemon=True).start()
    return Response(detect_objects(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/change_model', methods=['POST'])
def change_model():
    global selected_model, model
    new_model = request.json['model']
    if new_model in models:
        selected_model = new_model
        notify(f"Modelo cambiado a {selected_model}")
        return jsonify({"message": f"Modelo cambiado a {selected_model}"})
    else:
        return jsonify({"message": "Modelo no encontrado"}), 404

@app.route('/toggle_camera', methods=['POST'])
def toggle_camera():
    global camera_on, cap
    camera_on = not camera_on
    if not camera_on and cap:
        cap.release()
        cap = None
    notify("Cámara encendida" if camera_on else "Cámara apagada")
    return jsonify({"status": "on" if camera_on else "off"})

@app.route('/change_camera', methods=['POST'])
def change_camera():
    global selected_camera, camera_on
    new_camera = int(request.json['camera_index'])
    if new_camera in [cam[0] for cam in get_available_cameras()]:
        selected_camera = new_camera
        if cap:
            cap.release()
        camera_on = False
        notify(f"Cambiando a cámara {selected_camera}")
        return jsonify({"message": f"Cámara cambiada a {selected_camera}"})
    else:
        return jsonify({"message": "Cámara no encontrada"}), 404

@app.route('/test_modelo1')
def test_modelo1():
    return render_template('test_modelo1.html')

@app.route('/test_modelo2')
def test_modelo2():
    return render_template('test_modelo2.html')

@app.route('/guia')
def guia():
    return render_template('guia.html')


if __name__ == "__main__":
    threading.Thread(target=process_queue, daemon=True).start()
    model = load_model(DEFAULT_MODEL)
    app.run(debug=True)
