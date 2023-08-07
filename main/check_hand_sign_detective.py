from __future__ import annotations
import sys

import cv2

sys.path.append(".")
from hand_sign_recognition.lib import HandDetector, HandSignClassifier

from main.train import HandSign

cap = cv2.VideoCapture(0)
detector = HandDetector()
classifier = HandSignClassifier(len(HandSign), pretrained_model_path="sample_weight.pth")

while True:
    ret, frame = cap.read()
    handpoints = detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if handpoints is not None:
        handpoints.draw(frame)
        xmin, ymin, xmax, ymax = handpoints.bbox()
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 255), 1)
        hand_sign_label = classifier.predict(handpoints)

        text = HandSign(hand_sign_label).name
        cv2.putText(frame, text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 0), fontScale=1.0, thickness=3)

    cv2.imshow("img", frame)
    key = cv2.waitKey(1)
