from ultralytics import YOLO
import cv2
import csv
import os
from datetime import datetime

# Initialize YOLO model
model = YOLO("yolov8n.pt")

# Open video file
cap = cv2.VideoCapture("footages/footage2.mp4")

# Vehicle classes to detect
vehicle_classes = ["car", "truck", "bus", "motorcycle"]

# Create CSV file with headers if it doesn't exist
if not os.path.exists("traffic_data_main.csv"):
    with open("traffic_data_main.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["vehicle_count", "delta", "hour", "congestion"])

# Store previous frame count to calculate delta
previous_count = 0
frame_counts = []

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Use only left half of the frame (oncoming traffic)
    height, width = frame.shape[:2]
    left_half = frame[:, :width//2]
    
    # Run YOLO inference
    results = model(left_half)
    
    # Count vehicles in left half
    count = 0
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            if class_name in vehicle_classes:
                count += 1
    
    # Calculate delta (change in vehicle count from previous frame)
    delta = count - previous_count
    
    # Store for rolling history (optional, for smoothing)
    frame_counts.append(count)
    if len(frame_counts) > 20:
        frame_counts.pop(0)
    
    # Get current hour
    hour = datetime.now().hour
    
    # Determine congestion level based on delta and count
    # Delta positive = increasing traffic (more congestion)
    # Delta negative = decreasing traffic (less congestion)
    
    if count > 11:
        congestion = "HIGH"
    elif count > 7:
        # If traffic is increasing rapidly, upgrade to HIGH
        if delta > 4:
            congestion = "HIGH"
        else:
            congestion = "MEDIUM"
    else:
        # If traffic is decreasing, it might be LOW even with moderate count
        if delta < -2 and count < 10:
            congestion = "LOW"
        elif count < 5:
            congestion = "LOW"
        else:
            congestion = "MEDIUM"
    
    print(f"Count: {count} | Delta: {delta:+d} | Hour: {hour} | Congestion: {congestion}")
    
    # Save to CSV
    with open("traffic_data_main.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([count, delta, hour, congestion])
    
    # Update previous count for next frame
    previous_count = count
    
    # Display annotated frame
    annotated = results[0].plot()
    cv2.putText(annotated, f"Vehicles: {count} | Delta: {delta:+d}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2)
    cv2.putText(annotated, f"Congestion: {congestion}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2)
    
    cv2.imshow("Data Collection", annotated)
    
    # Press ESC to exit
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("\nData collection complete! CSV file saved.")