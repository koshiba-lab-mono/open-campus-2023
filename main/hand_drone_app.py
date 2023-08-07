from __future__ import annotations
import sys
from threading import Thread
from time import time

import cv2
from djitellopy import Tello

sys.path.append(".")
from hand_sign_recognition.lib import HandDetector, HandSignClassifier
from main.train import HandSign


#
def check_same_sign_continuity(enable_interval: float):
    """
    enable_interval秒だけ継続して同じ入力を受けているか確認するクロージャ.
    使い方の例はtests/test_check_same_sign_continuity.py参照.
    """

    last_sign: HandSign | int | None = None
    first_sign_time: float | None = None

    def _is_same_sign_continuous(sign: HandSign):
        nonlocal last_sign, first_sign_time
        if last_sign != sign:
            last_sign = sign
            first_sign_time = time()
            return False, enable_interval

        interval_sign_time = time() - first_sign_time

        if interval_sign_time >= enable_interval:
            return True, 0.0

        return False, enable_interval - interval_sign_time

    return _is_same_sign_continuous


def control_tello():
    global hand_sign_label, is_continous_sign
    while True:
        print(f"バッテリー残量:{tello.get_battery()}")

        if not is_continous_sign:  # 0.5秒継続して同じサインをしていない場合
            continue

        if tello.is_flying:
            if hand_sign_label == HandSign.INDEX_FINGER:
                tello.move_up(30)
            elif hand_sign_label == HandSign.BAD:
                tello.move_down(30)
            elif hand_sign_label == HandSign.GUN:
                tello.move_forward(30)
            elif hand_sign_label == HandSign.PAPER:
                tello.move_back(30)
            elif hand_sign_label == HandSign.FINGER_HEART:
                tello.rotate_counter_clockwise(30)
            elif hand_sign_label == HandSign.OK:
                tello.rotate_clockwise(30)
            elif hand_sign_label == HandSign.PEACE and tello.is_flying:
                tello.land()
        else:
            if hand_sign_label == HandSign.INDEX_FINGER:
                tello.takeoff()


if __name__ == "__main__":
    # グローバル変数 (別スレッドで読み取る用)
    hand_sign_label = None
    is_continous_sign = False

    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    classifier = HandSignClassifier(len(HandSign), pretrained_model_path="sample_weight.pth")

    tello = Tello()
    tello.connect()
    tello.streamon()
    read_img = tello.get_frame_read()

    is_same_sign_continuous = check_same_sign_continuity(0.5)
    key = None

    thread = Thread(target=control_tello)
    thread.start()

    while True:
        ret, frame = cap.read()
        drone_img = read_img.frame
        handpoints = detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if handpoints is not None:
            handpoints.draw(frame)
            xmin, ymin, xmax, ymax = handpoints.bbox()
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 255), 1)
            hand_sign_label = classifier.predict(handpoints)

            is_continous_sign, continous_time = is_same_sign_continuous(hand_sign_label)

            color = (0, 0, 0) if not is_continous_sign else (0, 0, 255)

            text = HandSign(hand_sign_label).name
            cv2.putText(frame, text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, color=color, fontScale=1.0, thickness=3)

        cv2.imshow("img", frame)
        cv2.imshow("drone", cv2.cvtColor(drone_img, cv2.COLOR_BGR2RGB))
        key = cv2.waitKey(1)
