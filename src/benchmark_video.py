
import argparse
import time
import statistics

import cv2
import numpy as np
import onnxruntime as ort

from . import models as modelz
from .run_realtime import detect_faces, preprocess_faces, classify_batch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to a short video file")
    ap.add_argument("--providers", nargs="*", default=None, help="ONNX Runtime EP list (e.g., CUDAExecutionProvider)")
    ap.add_argument("--repeat", type=int, default=3, help="Number of passes over the video")
    args = ap.parse_args()

    face_net = modelz.load_face_detector()
    sess = modelz.load_ferplus_session(providers=args.providers)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {args.video}")

    times = []
    for _ in range(args.repeat):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        t0 = time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            boxes = detect_faces(face_net, frame, 0.5)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            crops = preprocess_faces(gray, boxes)
            classify_batch(sess, crops)
        times.append(time.time() - t0)

    cap.release()
    print(f"Runs: {len(times)} | secs each: {[round(t,3) for t in times]} | median secs: {statistics.median(times):.3f}")
    print("Providers used:", sess.get_providers())

if __name__ == "__main__":
    main()
