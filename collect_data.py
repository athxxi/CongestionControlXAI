from ultralytics import YOLO
import cv2
import csv
import os
from datetime import datetime


model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("footages/footage2.mp4")

vehicle_classes = ["car", "truck", "bus", "motorcycle"]


if not os.path.exists("TrafficData001.csv"):
    with open("TrafficData001.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["vehicle_count", "hour", "congestion"])

frame_counts = []


while True:
    ret, frame = cap.read()

    if not ret:
        break

    height, width = frame.shape[:2]
    left_half = frame[:, :width//2]

    results = model(left_half)

    count = 0
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            if class_name in vehicle_classes:
                count += 1

    frame_counts.append(count)
    if len(frame_counts) > 20:
        frame_counts.pop(0)

    avg_count = int(sum(frame_counts) / len(frame_counts))
    hour = datetime.now().hour


    if avg_count > 11:
        congestion = "HIGH"
    elif avg_count > 7:
        congestion = "MEDIUM"
    else:
        congestion = "LOW"

    print(f"Vehicles: {avg_count} | Label: {congestion}")

    
    with open("TrafficData001.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([count, hour, congestion])

    annotated = results[0].plot()
    cv2.putText(annotated, f"Vehicles: {avg_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2)

    cv2.imshow("Data Collection", annotated)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()