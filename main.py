from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from ultralytics import YOLO
import cv2
import joblib
import pandas as pd
import threading
import time
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")


yolo_model = YOLO("yolov8n.pt")
congestion_model = joblib.load("congestion_model_new.pkl")

cap = cv2.VideoCapture("footages/footage2.mp4")

vehicle_classes = ["car", "truck", "bus", "motorcycle"]

frame_counts = []
latest_frame = None

current_data = {
    "vehicle_count": 0,
    "avg_count": 0,
    "congestion": "LOW",
    "hour": datetime.now().hour
}


def run_detection():
    global current_data, latest_frame

    while True:
        ret, frame = cap.read()

        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        height, width, _ = frame.shape

        
        left_frame = frame[:, :width//2]

        results = yolo_model(left_frame)

        count = 0
        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                class_name = yolo_model.names[class_id]
                if class_name in vehicle_classes:
                    count += 1

        frame_counts.append(count)
        if len(frame_counts) > 20:
            frame_counts.pop(0)

        avg_count = int(sum(frame_counts) / len(frame_counts))
        hour = datetime.now().hour

        feature_df = pd.DataFrame(
            [[count, avg_count, hour]],
            columns=["vehicle_count", "avg_count", "hour"]
        )

        prediction = congestion_model.predict(feature_df)[0]

        current_data = {
            "vehicle_count": count,
            "avg_count": avg_count,
            "congestion": prediction,
            "hour": hour
        }


        annotated_left = results[0].plot()


        frame[:, :width//2] = annotated_left


        cv2.line(frame, (width//2, 0), (width//2, height), (255,255,255), 2)

       
        if prediction == "HIGH":
            color = (0, 0, 255)
        elif prediction == "MEDIUM":
            color = (0, 255, 255)
        else:
            color = (0, 255, 0)

        cv2.putText(frame, f"Vehicles (Left Lane): {avg_count}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2)

        cv2.putText(frame, f"Congestion: {prediction}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    3)

        latest_frame = frame
        time.sleep(0.03)

threading.Thread(target=run_detection, daemon=True).start()


def generate_frames():
    global latest_frame
    while True:
        if latest_frame is not None:
            _, buffer = cv2.imencode('.jpg', latest_frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        time.sleep(0.03)



@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/stats")
def get_stats():
    return current_data

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(),
                             media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/shap")
def get_shap():
    try:
        feature_df = pd.DataFrame(
            [[current_data["vehicle_count"],
              current_data["avg_count"],
              current_data["hour"]]],
            columns=["vehicle_count", "avg_count", "hour"]
        )

        explainer = shap.TreeExplainer(congestion_model)
        shap_values = explainer.shap_values(feature_df)

        predicted_class = congestion_model.predict(feature_df)[0]
        class_index = list(congestion_model.classes_).index(predicted_class)

        
        if isinstance(shap_values, list):
            values = shap_values[class_index][0]
        else:
            if len(shap_values.shape) == 3:
                values = shap_values[0][:, class_index]
            elif len(shap_values.shape) == 2:
                values = shap_values[:, class_index]
            else:
                values = shap_values[0]

        feature_names = ["vehicle_count", "avg_count", "hour"]

     
        feature_impact = list(zip(feature_names, values))

    
        feature_impact.sort(key=lambda x: abs(x[1]), reverse=True)

    

        report = f"Traffic congestion is currently classified as {predicted_class}. "

        main_feature, main_value = feature_impact[0]

        if main_feature == "avg_count":
            report += "The primary contributing factor is the high average vehicle density on the road. "
        elif main_feature == "vehicle_count":
            report += "The number of vehicles currently present is the main driver of congestion. "
        elif main_feature == "hour":
            report += "The time of day is playing a significant role in traffic conditions. "

   
        for feature, value in feature_impact[1:]:

            if abs(value) < 0.05:
                continue

            if feature == "avg_count":
                if value > 0:
                    report += "Vehicle density is adding additional pressure to traffic flow. "
                else:
                    report += "Vehicle density is helping ease congestion slightly. "

            elif feature == "vehicle_count":
                if value > 0:
                    report += "The number of vehicles is further increasing congestion levels. "
                else:
                    report += "The vehicle count is not significantly contributing to congestion. "

            elif feature == "hour":
                if value > 0:
                    report += "The current time of day is contributing to heavier traffic conditions. "
                else:
                    report += "The current time of day is not a major contributor to congestion. "

        report += "<br> Overall, the system analysis indicates that traffic flow patterns are consistent with the predicted congestion level."

        return {
            "prediction": predicted_class,
            "report": report
        }

    except Exception as e:
        return {"error": str(e)}
    
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)