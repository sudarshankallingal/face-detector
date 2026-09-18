from flask import Flask, Response
import cv2
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO('yolov8n.pt')
camera = cv2.VideoCapture(0)  # change to 1 if it grabs the wrong camera

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Run detection, filter for class 0 = 'person'
        results = model(frame, classes=[0], verbose=False)

        person_count = 0
        for result in results:
            boxes = result.boxes
            person_count = len(boxes)
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Show count on frame
        cv2.putText(frame, f'People: {person_count}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return '<html><body><h1>Live Person Counter</h1><img src="/video_feed"></body></html>'

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
