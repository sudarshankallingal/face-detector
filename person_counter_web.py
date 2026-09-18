from flask import Flask, Response, redirect, url_for
import cv2
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO('yolov8n.pt')
camera = cv2.VideoCapture(0)  # change to 1 if it grabs the wrong camera

# Vertical line position (50% of frame width)
LINE_X_RATIO = 0.5

# Tracking state
track_last_x = {}       # track_id -> last known x-center
left_to_right_count = 0
right_to_left_count = 0

def generate_frames():
    global left_to_right_count, right_to_left_count

    while True:
        success, frame = camera.read()
        if not success:
            break

        h, w = frame.shape[:2]
        line_x = int(w * LINE_X_RATIO)

        # Draw the vertical counting line
        cv2.line(frame, (line_x, 0), (line_x, h), (255, 0, 0), 2)

        results = model.track(frame, classes=[0], persist=True, verbose=False)

        for result in results:
            boxes = result.boxes
            if boxes.id is None:
                continue

            for box, track_id in zip(boxes, boxes.id):
                track_id = int(track_id)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2  # center x-coordinate of this person

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'ID {track_id}', (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if track_id in track_last_x:
                    prev_x = track_last_x[track_id]
                    if prev_x < line_x <= cx:
                        left_to_right_count += 1
                    elif prev_x > line_x >= cx:
                        right_to_left_count += 1

                track_last_x[track_id] = cx

        cv2.putText(frame, f'Left->Right: {left_to_right_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f'Right->Left: {right_to_left_count}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return '''
    <html>
    <head>
        <title>Footfall Counter</title>
        <style>
            html, body {
                margin: 0;
                padding: 0;
                height: 100%;
                overflow: hidden;
                background: #111;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                font-family: sans-serif;
            }
            img {
                max-width: 100vw;
                max-height: 85vh;
            }
            button {
                margin-top: 10px;
                padding: 10px 20px;
                font-size: 16px;
                cursor: pointer;
                background: #e74c3c;
                color: white;
                border: none;
                border-radius: 5px;
            }
        </style>
    </head>
    <body>
        <img src="/video_feed">
        <form action="/reset" method="post">
            <button type="submit">Reset Count</button>
        </form>
    </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/reset', methods=['POST'])
def reset():
    global left_to_right_count, right_to_left_count, track_last_x
    left_to_right_count = 0
    right_to_left_count = 0
    track_last_x = {}
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
