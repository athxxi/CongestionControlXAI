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
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
from enum import Enum
from typing import Dict, List, Tuple
import json


app = FastAPI()

from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

yolo_model = YOLO("yolov8n.pt")
congestion_model = joblib.load("CongestionModel001.pkl")

cap = cv2.VideoCapture("footages/footage2.mp4")

vehicle_classes = ["car", "truck", "bus", "motorcycle"]

frame_counts = []
latest_frame = None

current_data = {
    "vehicle_count": 0,
    "congestion": "LOW",
    "hour": datetime.now().hour
}

class CongestionLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

class TrafficExplanationEngine:
    """Advanced NLP explanation engine for traffic congestion analysis"""
    
    def __init__(self):
        
        self.thresholds = {
            "vehicle_count": {"LOW": 10, "MEDIUM": 25, "HIGH": 40},
            "avg_count": {"LOW": 8, "MEDIUM": 20, "HIGH": 35},
            "hour_peaks": {
                "morning_peak": [7, 8, 9],
                "evening_peak": [17, 18, 19],
                "lunch_peak": [12, 13, 14],
                "night_off": [22, 23, 0, 1, 2, 3, 4, 5]
            }
        }
        
        
        self.phrases = {
            "severity": {
                "LOW": ["minimal", "light", "smooth-flowing", "uncongested"],
                "MEDIUM": ["moderate", "building", "noticeable", "increasing"],
                "HIGH": ["severe", "heavy", "critical", "gridlocked"]
            },
            "trends": {
                "increasing": ["rising", "building up", "increasing steadily", "growing"],
                "decreasing": ["dissipating", "reducing", "easing", "clearing"],
                "stable": ["stable", "consistent", "steady", "maintaining"]
            },
            "impacts": {
                "time_loss": {
                    "LOW": "minimal delays of 0-5 minutes",
                    "MEDIUM": "moderate delays of 10-20 minutes",
                    "HIGH": "significant delays of 30+ minutes"
                },
                "fuel_efficiency": {
                    "LOW": "optimal fuel consumption",
                    "MEDIUM": "reduced fuel efficiency by 15-20%",
                    "HIGH": "poor fuel efficiency with 30-40% increase"
                },
                "safety": {
                    "LOW": "low risk environment",
                    "MEDIUM": "elevated risk due to stop-and-go traffic",
                    "HIGH": "high risk with frequent braking and congestion"
                }
            }
        }
        
        
        self.history = []
        self.max_history = 10
        
    def add_to_history(self, data: Dict):
        """Store historical congestion data for trend analysis"""
        self.history.append({
            "timestamp": datetime.now(),
            "congestion": data["congestion"],
            "vehicle_count": data["vehicle_count"],
            "avg_count": data["avg_count"]
        })
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def analyze_trend(self) -> str:
        """Analyze traffic trend based on historical data"""
        if len(self.history) < 3:
            return "stable"
        
        recent = self.history[-3:]
        congestion_values = [h["congestion"] for h in recent]
        
        
        value_map = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
        numeric_trend = [value_map[v] for v in congestion_values]
        
        if all(numeric_trend[i] <= numeric_trend[i+1] for i in range(len(numeric_trend)-1)):
            if numeric_trend[-1] > numeric_trend[0]:
                return "increasing"
            else:
                return "stable"
        elif all(numeric_trend[i] >= numeric_trend[i+1] for i in range(len(numeric_trend)-1)):
            if numeric_trend[-1] < numeric_trend[0]:
                return "decreasing"
            else:
                return "stable"
        return "stable"
    
    def get_time_context(self, hour: int) -> Dict:
        """Get contextual information about the current time"""
        if hour in self.thresholds["hour_peaks"]["morning_peak"]:
            return {"period": "morning rush hour", "typical": "high", "reason": "commuters heading to work"}
        elif hour in self.thresholds["hour_peaks"]["evening_peak"]:
            return {"period": "evening rush hour", "typical": "high", "reason": "commuters returning home"}
        elif hour in self.thresholds["hour_peaks"]["lunch_peak"]:
            return {"period": "lunch hour", "typical": "medium", "reason": "lunch traffic and deliveries"}
        elif hour in self.thresholds["hour_peaks"]["night_off"]:
            return {"period": "late night/early morning", "typical": "low", "reason": "off-peak hours with minimal traffic"}
        else:
            return {"period": "regular hours", "typical": "medium", "reason": "normal daytime traffic patterns"}
    
    def calculate_severity_score(self, feature_impacts: List[Tuple[str, float]], 
                                 current_data: Dict) -> float:
        """Calculate a severity score (0-100) for congestion"""
        weights = {"avg_count": 0.5, "vehicle_count": 0.3, "hour": 0.2}
        base_score = 0
        
        
        if current_data["congestion"] == "LOW":
            base_score = 20
        elif current_data["congestion"] == "MEDIUM":
            base_score = 60
        else:
            base_score = 85
        
        
        adjustment = 0
        for feature, impact in feature_impacts:
            if feature == "avg_count" and impact > 0:
                adjustment += min(15, abs(impact) * 10)
            elif feature == "vehicle_count" and impact > 0:
                adjustment += min(10, abs(impact) * 5)
        
        return min(100, base_score + adjustment)
    
    def generate_detailed_report(self, shap_values, feature_df: pd.DataFrame, 
                                 predicted_class: str, current_data: Dict) -> Dict:
        """Generate comprehensive NLP report with industrial-grade explanations"""
        
        
        feature_names = ["vehicle_count", "avg_count", "hour"]
        class_index = list(congestion_model.classes_).index(predicted_class)
        
        if isinstance(shap_values, list):
            values = shap_values[class_index][0]
        elif len(shap_values.shape) == 3:
            values = shap_values[0][:, class_index]
        elif len(shap_values.shape) == 2:
            values = shap_values[:, class_index]
        else:
            values = shap_values[0]
        
        feature_impacts = list(zip(feature_names, values))
        feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        
        
        self.add_to_history(current_data)
        
        
        trend = self.analyze_trend()
        
        
        time_context = self.get_time_context(current_data["hour"])
        
        
        severity_score = self.calculate_severity_score(feature_impacts, current_data)
        
        
        severity_adj = self.phrases["severity"][predicted_class][0]
        trend_desc = self.phrases["trends"][trend][0]
        
        summary = f"""
        <div style="font-family: Arial, sans-serif; line-height: 1.6;">
            <h3>🚦 Traffic Congestion Analysis Report</h3>
            <p><strong>Severity Level:</strong> <span style="color: {'#2ecc71' if predicted_class == 'LOW' else '#f39c12' if predicted_class == 'MEDIUM' else '#e74c3c'}; font-weight: bold;">{predicted_class}</span> 
            (Severity Score: {severity_score:.1f}/100)</p>
            
            <p><strong>Executive Summary:</strong> The traffic condition is currently <strong>{severity_adj}</strong> with a <strong>{trend_desc}</strong> trend. 
            During this {time_context['period']}, {time_context['reason'].lower()}. 
            {self._generate_trend_sentence(trend, predicted_class)}</p>
        """
        
        
        summary += "<div style='margin-top: 20px;'><h4>📊 Key Contributing Factors:</h4><ul>"
        
        for feature, impact in feature_impacts[:3]:
            if abs(impact) > 0.05:
                summary += f"<li><strong>{self._format_feature_name(feature)}:</strong> {self._generate_factor_explanation(feature, impact, current_data)}</li>"
        
        summary += "</ul></div>"
        
        
        summary += f"""
        <div style='margin-top: 20px; background-color: rgba(0, 255, 204, 0.05); padding: 15px; border-radius: 5px;'>
            <h4>📈 Comparative Analysis:</h4>
            <p>• Current vehicle count: <strong>{current_data['vehicle_count']}</strong> vehicles 
            (vs. typical {self._get_typical_value('vehicle_count', time_context['typical'])} for this time)</p>
            <p>• Average density: <strong>{current_data['avg_count']}</strong> vehicles 
            (vs. typical {self._get_typical_value('avg_count', time_context['typical'])})</p>
            <p>• Time factor: {time_context['period']} - {time_context['reason'].lower()}</p>
        </div>
        """
        
        
        summary += f"""
        <div style='margin-top: 20px;'>
            <h4>⚠️ Expected Impacts:</h4>
            <ul>
                <li><strong>Time Loss:</strong> {self.phrases['impacts']['time_loss'][predicted_class]}</li>
                <li><strong>Fuel Efficiency:</strong> {self.phrases['impacts']['fuel_efficiency'][predicted_class]}</li>
                <li><strong>Safety Level:</strong> {self.phrases['impacts']['safety'][predicted_class]}</li>
            </ul>
        </div>
        """
        
        
        summary += f"""
        <div style='margin-top: 20px; background-color: rgba(0, 255, 204, 0.05); padding: 15px; border-radius: 5px;'>
            <h4>💡 Recommendations:</h4>
            <ul>
                {self._generate_recommendations(predicted_class, trend, time_context)}
            </ul>
        </div>
        """
        
        summary += "</div>"
        
        return {
            "prediction": predicted_class,
            "report": summary,
            "severity_score": severity_score,
            "trend": trend,
            "time_context": time_context["period"],
            "key_factors": [self._format_feature_name(f) for f, _ in feature_impacts[:3]]
        }
    
    def _generate_trend_sentence(self, trend: str, congestion: str) -> str:
        """Generate trend description"""
        if trend == "increasing":
            return f"Traffic congestion is showing signs of {self.phrases['trends']['increasing'][0]}, suggesting conditions may worsen if the trend continues."
        elif trend == "decreasing":
            return f"Traffic is showing signs of {self.phrases['trends']['decreasing'][0]}, indicating improving conditions ahead."
        else:
            return "Traffic patterns appear to be relatively stable at this time."
    
    def _format_feature_name(self, feature: str) -> str:
        """Format feature names for display"""
        names = {
            "vehicle_count": "Current Vehicle Count",
            "avg_count": "Average Vehicle Density",
            "hour": "Time of Day"
        }
        return names.get(feature, feature)
    
    def _generate_factor_explanation(self, feature: str, impact: float, data: Dict) -> str:
        """Generate human-readable explanation for each factor"""
        impact_direction = "positive" if impact > 0 else "negative"
        impact_strength = "strongly" if abs(impact) > 0.3 else "moderately" if abs(impact) > 0.1 else "slightly"
        
        if feature == "avg_count":
            if data["avg_count"] > 20:
                return f"High average density ({data['avg_count']} vehicles) is {impact_strength} contributing to congestion"
            elif data["avg_count"] < 10:
                return f"Low average density ({data['avg_count']} vehicles) is {impact_strength} helping maintain smooth flow"
            else:
                return f"Moderate density levels ({data['avg_count']} vehicles) have a {impact_strength} {impact_direction} influence on traffic"
        
        elif feature == "vehicle_count":
            if data["vehicle_count"] > 25:
                return f"The current high volume ({data['vehicle_count']} vehicles) is creating {impact_strength} increased congestion"
            elif data["vehicle_count"] < 15:
                return f"Current vehicle volume ({data['vehicle_count']}) is {impact_strength} below typical congestion thresholds"
            else:
                return f"Vehicle volume ({data['vehicle_count']}) shows a {impact_strength} {impact_direction} correlation with current congestion"
        
        elif feature == "hour":
            hour = data["hour"]
            if 7 <= hour <= 9 or 17 <= hour <= 19:
                return f"Rush hour timing ({hour}:00) is {impact_strength} amplifying congestion levels"
            elif 22 <= hour <= 5:
                return f"Late night timing ({hour}:00) is {impact_strength} reducing typical congestion"
            else:
                return f"The current time ({hour}:00) has a {impact_strength} {impact_direction} influence on traffic patterns"
        
        return f"This factor has a {impact_strength} {impact_direction} impact on congestion"
    
    def _get_typical_value(self, metric: str, typical_level: str) -> str:
        """Get typical value range based on congestion level"""
        if metric == "vehicle_count":
            if typical_level == "high":
                return "30-50 vehicles"
            elif typical_level == "medium":
                return "15-30 vehicles"
            else:
                return "0-15 vehicles"
        else:
            if typical_level == "high":
                return "25-40 vehicles"
            elif typical_level == "medium":
                return "12-25 vehicles"
            else:
                return "0-12 vehicles"
    
    def _generate_recommendations(self, congestion: str, trend: str, time_context: Dict) -> str:
        """Generate actionable recommendations"""
        recommendations = []
        
        if congestion == "HIGH":
            recommendations.append("🚗 Consider alternative routes or postpone non-essential travel")
            recommendations.append("⏰ Allow 30+ minutes additional travel time for critical trips")
            recommendations.append("🔄 Use public transportation or ridesharing if available")
            if trend == "increasing":
                recommendations.append("📱 Monitor traffic updates - conditions expected to worsen")
        elif congestion == "MEDIUM":
            recommendations.append("🕐 Allow 10-20 minutes buffer for your journey")
            recommendations.append("🗺️ Check real-time navigation for optimal routing")
            if time_context["period"] in ["morning rush hour", "evening rush hour"]:
                recommendations.append("⏰ Consider shifting travel by 30-60 minutes to avoid peak")
        else:  
            recommendations.append("✅ Good time for travel - minimal delays expected")
            recommendations.append("🚀 Enjoy smooth traffic conditions")
            if trend == "decreasing":
                recommendations.append("📈 Conditions are improving further")
        
        return "\n".join([f"<li>{rec}</li>" for rec in recommendations[:3]])



explanation_engine = TrafficExplanationEngine()

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
            [[count, hour]],
            columns=["vehicle_count", "hour"]
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
    """Enhanced SHAP explanation with industrial-grade NLP report"""
    try:
        feature_df = pd.DataFrame(
            [[current_data["vehicle_count"],
            current_data["hour"]]],
            columns=["vehicle_count", "hour"]
        )

        explainer = shap.TreeExplainer(congestion_model)
        shap_values = explainer.shap_values(feature_df)
        predicted_class = congestion_model.predict(feature_df)[0]
        
        
        detailed_report = explanation_engine.generate_detailed_report(
            shap_values, feature_df, predicted_class, current_data
        )
        
        return detailed_report

    except Exception as e:
        return {"error": str(e), "prediction": "ERROR", "report": f"Analysis failed: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)