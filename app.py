import cv2
import numpy as np
import time

# Load models
face_net = cv2.dnn.readNet(
    "models/res10_300x300_ssd_iter_140000.caffemodel",
    "models/deploy.prototxt"
)

age_net = cv2.dnn.readNet(
    "models/age_net.caffemodel",
    "models/age_deploy.prototxt"
)

# Age ranges
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
               '(25-32)', '(38-43)', '(48-53)', '(60-100)']

cap = cv2.VideoCapture(0)

print("Starting Age Detection... Press Q to quit")

start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # Face detection blob
    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300),
        (104, 177, 123)
    )

    face_net.setInput(blob)
    detections = face_net.forward()

    # FPS calculation
    fps = 1/(time.time() - start_time)
    start_time = time.time()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Higher confidence = fewer false detections
        if confidence > 0.85:
            box = detections[0, 0, i, 3:7] * \
                  np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype(int)

            # Add margin for better crop
            margin = 15
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(w, x2 + margin)
            y2 = min(h, y2 + margin)

            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            # Age prediction blob
            blob = cv2.dnn.blobFromImage(
                face, 1.0, (227, 227),
                (78.426, 87.769, 114.895),
                swapRB=False
            )

            age_net.setInput(blob)
            preds = age_net.forward()
            age = AGE_BUCKETS[preds[0].argmax()]

            # Draw results
            cv2.rectangle(frame, (x1, y1),
                          (x2, y2), (0,255,0), 2)

            cv2.putText(frame, f"Age: {age}",
                        (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0,255,0), 2)

    # Show FPS
    cv2.putText(frame, f"FPS: {int(fps)}",
                (20,30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,(0,255,0),2)

    cv2.imshow("Age Detection System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
