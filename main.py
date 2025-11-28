import cv2
from ultralytics import YOLO
import torch
import numpy as np

# ================= SETTINGS =================
rtsp_IP   = "bore.pub"
rtsp_PORT = 1945
MODEL    = "model/yolov12n.pt"
# ============================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = YOLO(MODEL).to(device)

rtsp_url = f"rtsp://{rtsp_IP}:{rtsp_PORT}"

cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)      # 1 frame buffer
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10)  # Fast timeout
cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10)

if not cap.isOpened():
    print("ERROR: UDP stream is not open!")
    print("Check: FFmpeg command is working? IP:Port is ok?")
    exit()

print("UDP Stream connected! YOLO start...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame is not coming... waiting...")
        continue

    # YOLO inference (super fast)
    results = model(frame, imgsz=640, device=device, half=True, verbose=False)[0]

    # Draw boxes
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        label = f"{results.names[cls]} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("AI CCTV", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
