import cv2
import time
from ultralytics import YOLO

# Use a better model
model = YOLO('yolov8s.pt')  

# Open the webcam
webcamera = cv2.VideoCapture(0)
frame_width = int(webcamera.get(3))
frame_height = int(webcamera.get(4))
fps = 16

# Define the video writer
out = cv2.VideoWriter(
    'output.avi',
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (frame_width, frame_height)
)

while True:
    success, frame = webcamera.read()
    if not success:
        break

    # Run object tracking
    results = model.track(frame, conf=0.25, persist=True, imgsz=640)
    boxes = results[0].boxes
    total_detected = len(boxes)

    # Draw thinner boxes and labels manually
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = f"{model.names[cls_id]} {conf:.2f}"

        # Thin bounding box (thickness=1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Show total count
    cv2.putText(frame, f"Total: {total_detected}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show and save
    cv2.imshow("YOLOv8 Live Tracking", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
webcamera.release()
out.release()
cv2.destroyAllWindows()
