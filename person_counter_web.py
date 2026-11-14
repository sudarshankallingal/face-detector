from flask import Flask, Response, redirect, url_for
import threading
import time
import cv2
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO('yolov8n.pt')
CAMERA_INDEX = 0  # change to 1 if it grabs the wrong camera
# Capture as compressed MJPG at a moderate size: uncompressed full-resolution frames
# only manage a few fps over USB 2.
CAPTURE_WIDTH, CAPTURE_HEIGHT, CAPTURE_FPS = 1280, 720, 30
JPEG_QUALITY = 80
REOPEN_AFTER = 5       # failed reads in a row before the camera is reopened
REOPEN_WAIT_S = 2      # pause between reopen attempts
failed_reads = 0


def open_camera():
    cam = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    cam.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always process the newest frame
    if cam.isOpened():
        print(f"camera: /dev/video{CAMERA_INDEX} at {int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
              f"{int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))} {cam.get(cv2.CAP_PROP_FPS):.0f} fps", flush=True)
    return cam


camera = open_camera()


def read_frame():
    """Next frame, or None. When the camera stops giving frames (e.g. the USB camera dropped
    and came back), close it and open it again instead of ending the feed for good."""
    global camera, failed_reads
    success, frame = camera.read()
    if success:
        failed_reads = 0
        return frame
    failed_reads += 1
    if failed_reads >= REOPEN_AFTER:
        print(f"camera: {failed_reads} failed reads, reopening /dev/video{CAMERA_INDEX}", flush=True)
        camera.release()
        time.sleep(REOPEN_WAIT_S)
        camera = open_camera()
        failed_reads = 0
    time.sleep(0.2)
    return None

# Vertical line position (50% of frame width)
LINE_X_RATIO = 0.5

# Tracking state
track_last_x = {}       # track_id -> last known x-center
left_to_right_count = 0
right_to_left_count = 0

# The camera is read and YOLO runs in ONE background thread; every viewer streams the latest
# annotated frame from it. (Previously each open page ran its own read + tracking loop, which
# split the frames between viewers and fed one shared tracker twice, double-counting crossings.)
latest_jpeg = None
latest_seq = 0
latest_cond = threading.Condition()


def process_frames():
    global left_to_right_count, right_to_left_count, latest_jpeg, latest_seq

    while True:
        frame = read_frame()
        if frame is None:
            continue  # camera is being reopened; viewers keep the last frame

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

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ret:
            continue
        with latest_cond:
            latest_jpeg = buffer.tobytes()
            latest_seq += 1
            latest_cond.notify_all()


def generate_frames():
    """Stream each new annotated frame to one viewer."""
    seen = 0
    while True:
        with latest_cond:
            latest_cond.wait_for(lambda: latest_seq != seen, timeout=5)
            if latest_seq == seen or latest_jpeg is None:
                continue
            seen, frame_bytes = latest_seq, latest_jpeg
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    # DDLC branding: logo, colours and fonts match the DDLC HMI in ~/generic (static/hmi.css).
    return '''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#1A1A1D">
<title>Footfall Counter | DDLC</title>
<link rel="icon" type="image/svg+xml" href="/static/favicon.svg">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@800&family=Montserrat:wght@700&family=Inter:wght@400;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg-dark: #1A1A1D;
    --panel-bg: #1E1E22;
    --content-bg: #252529;
    --border: #2A2A2E;
    --primary: #008100;
    --primary-tint: color-mix(in srgb, var(--primary) 55%, white);
    --text-muted: #A1A1AA;
    --red: #EF4444;
    --font-heading: "Barlow Condensed", -apple-system, "Segoe UI", Roboto, sans-serif;
    --font-subheading: "Montserrat", -apple-system, "Segoe UI", Roboto, sans-serif;
    --font-body: "Inter", -apple-system, "Segoe UI", Roboto, sans-serif;
  }
  html, body { height: 100%; }
  body {
    margin: 0;
    display: flex;
    flex-direction: column;
    background: var(--bg-dark);
    color: #fff;
    font-family: var(--font-body);
  }
  header {
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
    padding: 14px 24px;
    background: var(--panel-bg);
    border-bottom: 1px solid var(--border);
  }
  .header-left { display: flex; align-items: center; gap: 14px; min-width: 0; }
  .logo-mark { height: 26px; width: auto; flex-shrink: 0; }
  .brand h1 {
    margin: 0;
    font-family: var(--font-heading);
    font-weight: 800;
    text-transform: uppercase;
    font-size: 22px;
    letter-spacing: 0.3px;
  }
  .brand p {
    margin: 2px 0 0;
    font-family: var(--font-subheading);
    font-weight: 700;
    font-size: 12px;
    color: var(--text-muted);
  }
  .brand .ddlc { color: var(--primary-tint); }
  .live {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: var(--font-subheading);
    font-weight: 700;
    font-size: 12px;
    color: var(--primary-tint);
    text-transform: uppercase;
  }
  .live::before {
    content: "";
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--primary);
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--primary) 30%, transparent);
  }
  main {
    flex: 1;
    min-height: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 14px;
    padding: 16px;
  }
  .feed {
    max-width: 100%;
    max-height: calc(100vh - 190px);
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--content-bg);
  }
  button {
    padding: 10px 22px;
    font-family: var(--font-subheading);
    font-weight: 700;
    font-size: 14px;
    cursor: pointer;
    background: transparent;
    color: var(--red);
    border: 1px solid var(--red);
    border-radius: 8px;
  }
  button:hover { background: var(--red); color: #fff; }
  footer {
    flex-shrink: 0;
    text-align: center;
    color: var(--text-muted);
    font-size: 11px;
    padding: 10px 24px;
  }
  @media (max-width: 600px) {
    header { padding: 12px 16px; }
    .brand h1 { font-size: 18px; }
    .brand p, .live { display: none; }
  }
</style>
</head>
<body>
  <header>
    <div class="header-left">
      <img class="logo-mark" src="/static/ddlc-logo-white.svg" alt="DDLC">
      <div class="brand">
        <h1>Footfall Counter</h1>
        <p><span class="ddlc">DDLC.AI</span> &middot; People counting with YOLO tracking</p>
      </div>
    </div>
    <span class="live">Live</span>
  </header>
  <main>
    <img class="feed" src="/video_feed" alt="Live camera feed with people counts">
    <form action="/reset" method="post"
          onsubmit="return confirm('Reset both footfall counts to zero?');">
      <button type="submit">Reset Count</button>
    </form>
  </main>
  <footer>&copy; DDLC.AI &middot; Footfall Counter</footer>
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
    threading.Thread(target=process_frames, daemon=True, name="process_frames").start()
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
