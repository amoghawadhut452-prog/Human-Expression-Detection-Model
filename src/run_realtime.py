import cv2
import time
import csv
import os
from src.emotion_model import EmotionModel

def main(source=0, show=True, log_fps="fps.csv", use_gpu=False):
    cap = cv2.VideoCapture(source)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    emotion_model = EmotionModel(use_gpu=use_gpu)

    fps_list = []
    prev_time = time.time()

    if log_fps and not os.path.exists(os.path.dirname(log_fps)):
        if os.path.dirname(log_fps) != "":
            os.makedirs(os.path.dirname(log_fps))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50,50))

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]

            # Predict emotion
            emotion, confidence = emotion_model.predict(face_img)

            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            label = f"{emotion} ({confidence*100:.1f}%)"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        # Calculate FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time
        fps_list.append(fps)

        if show:
            cv2.imshow("Real-Time Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if log_fps:
        with open(log_fps, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Frame","FPS"])
            for i, fps_val in enumerate(fps_list):
                writer.writerow([i, fps_val])

    print(f"Average FPS: {sum(fps_list)/len(fps_list):.2f}")


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="Video source (0 for webcam or file path)")
    parser.add_argument("--show", type=int, default=1, help="Show video window")
    parser.add_argument("--log_fps", type=str, default="fps.csv", help="CSV file to log FPS")
    parser.add_argument("--use_gpu", type=int, default=0, help="Use GPU if available (1=yes, 0=no)")
    args = parser.parse_args()

    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    main(source=source, show=bool(args.show), log_fps=args.log_fps, use_gpu=bool(args.use_gpu))
