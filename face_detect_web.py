from flask import Flask, Response
import cv2
from face_detect import resolve_model_path, load_detector, detect_faces, draw_faces

app = Flask(__name__)
detector = load_detector(resolve_model_path())
camera = cv2.VideoCapture(0)  # change to 1 if it grabs the wrong camera

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        faces = detect_faces(detector, frame)
        draw_faces(frame, faces)

        cv2.putText(
            frame, f"faces: {len(faces)}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2,
        )

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return '<html><body><h1>Live Face Detection</h1><img src="/video_feed"></body></html>'

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
