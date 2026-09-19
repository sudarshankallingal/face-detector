# Face Detector — Jetson Setup

Real-time face detection (OpenCV YuNet) running on the Jetson (`tegra-ubuntu`), viewable
in a browser from any machine on the WireGuard VPN.

## Prerequisites on the Jetson

- Python 3 with OpenCV installed (`python3 -c "import cv2; print(cv2.__version__)"`)
- A working camera at `/dev/video0` (check with `ls /dev/video*`)
- Flask (`pip3 install flask` if not already present)
- WireGuard VPN active (`ip a` should show a `wg0` interface with an address like `10.0.0.3`)

**Note on OpenCV version:** this Jetson runs OpenCV 4.5.4, which is too old to load the
current `2023mar` YuNet model from opencv_zoo (it throws a
`Layer with requested id=-1 not found` error). Use the older `2022mar` model instead
(see download step below), saved under the filename `face_detection_yunet_2023mar.onnx`
since that's the name `face_detect.py` looks for by default.

## Setup steps

```bash
git clone https://github.com/sudarshankallingal/face-detector.git
cd face-detector

# Download the YuNet model (2022mar build — compatible with OpenCV 4.5.4)
curl -fsSL -o face_detection_yunet_2023mar.onnx \
  "https://github.com/opencv/opencv_zoo/raw/91fb0290f50896f38a0ab1e558b74b16bc009428/models/face_detection_yunet/face_detection_yunet_2022mar.onnx"
```

## Running the browser-viewable stream

```bash
python3 face_detect_web.py
```

This starts a Flask server on port 5001, bound to all network interfaces (`0.0.0.0`),
so it's reachable from:

- The Jetson's local wifi IP (e.g. `http://192.168.0.158:5001`)
- Any machine on the WireGuard VPN, using the Jetson's VPN IP (e.g. `http://10.0.0.3:5001`)

Find the Jetson's current IPs with `ip a` (look for `wlan0` for local wifi, `wg0` for VPN).

Open the URL in a browser on another machine (e.g. the t14s at `10.0.0.2`) to see the
live annotated video feed — no local display or `cv2.imshow` window needed on the Jetson.

## Notes

- OpenCV on this Jetson has no CUDA/cuDNN support, so detection runs on CPU only.
- `Ctrl+C` in the terminal running `face_detect_web.py` stops the server.
