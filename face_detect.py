#!/usr/bin/env python3
"""Face detection with OpenCV's YuNet DNN face detector (cv2.FaceDetectorYN).

OpenCV 5.x removed the old Haar-cascade CascadeClassifier bindings/data, so
this uses the modern DNN-based detector instead (more accurate: handles
profile/tilted faces and low light much better than Haar ever did).

Requires the YuNet ONNX model file. Download it once with:
    curl -fsSL -o face_detection_yunet_2023mar.onnx \\
      https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

Usage:
    python face_detect.py                          # live webcam
    python face_detect.py photo.jpg                 # single image, shows result
    python face_detect.py photo.jpg -o out.jpg       # single image, saves result
    python face_detect.py -m path/to/model.onnx ...  # custom model location
"""

import argparse
import os
import sys

import cv2

DEFAULT_MODEL_CANDIDATES = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_detection_yunet_2023mar.onnx"),
    os.path.expanduser("~/venv-face-model/face_detection_yunet_2023mar.onnx"),
]


def resolve_model_path(explicit_path=None):
    if explicit_path:
        if not os.path.isfile(explicit_path):
            sys.exit(f"Model file not found: {explicit_path}")
        return explicit_path

    for candidate in DEFAULT_MODEL_CANDIDATES:
        if os.path.isfile(candidate):
            return candidate

    sys.exit(
        "Could not find the YuNet model file. Download it with:\n"
        "  curl -fsSL -o face_detection_yunet_2023mar.onnx \\\n"
        "    https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
        "face_detection_yunet_2023mar.onnx\n"
        "then re-run, or pass -m /path/to/model.onnx"
    )


def load_detector(model_path, input_size=(320, 320)):
    detector = cv2.FaceDetectorYN_create(
        model_path,
        "",                 # no separate config file needed for ONNX models
        input_size,
        score_threshold=0.7,
        nms_threshold=0.3,
        top_k=5000,
    )
    return detector


def detect_faces(detector, frame):
    """Return an (N, 15) array: x, y, w, h, 5 landmark pairs, confidence."""
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    return faces if faces is not None else []


def draw_faces(frame, faces):
    for i, face in enumerate(faces, start=1):
        x, y, w, h = face[:4].astype(int)
        score = face[-1]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame, f"face {i} ({score:.2f})", (x, max(y - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )
        # 5 landmarks: right eye, left eye, nose tip, right mouth, left mouth
        for lx, ly in face[4:14].reshape(5, 2).astype(int):
            cv2.circle(frame, (lx, ly), 2, (0, 0, 255), -1)
    return frame


def run_image(detector, path, out_path=None):
    frame = cv2.imread(path)
    if frame is None:
        sys.exit(f"Could not read image: {path}")

    faces = detect_faces(detector, frame)
    print(f"Found {len(faces)} face(s) in {path}")
    draw_faces(frame, faces)

    if out_path:
        cv2.imwrite(out_path, frame)
        print(f"Saved annotated image to {out_path}")
    else:
        cv2.imshow("Face detection - press any key to close", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_webcam(detector, camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {camera_index}")

    print("Press 'q' to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            faces = detect_faces(detector, frame)
            draw_faces(frame, faces)
            cv2.putText(
                frame, f"faces: {len(faces)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2,
            )

            cv2.imshow("Face detection - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Detect faces in an image or webcam feed.")
    parser.add_argument("image", nargs="?", help="path to an image; omit to use the webcam")
    parser.add_argument("-o", "--output", help="save the annotated image here instead of displaying it")
    parser.add_argument("-c", "--camera", type=int, default=0, help="camera index (default: 0)")
    parser.add_argument("-m", "--model", help="path to the YuNet .onnx model file")
    args = parser.parse_args()

    detector = load_detector(resolve_model_path(args.model))
    if args.image:
        run_image(detector, args.image, args.output)
    else:
        run_webcam(detector, args.camera)


if __name__ == "__main__":
    main()
