import os
from enum import IntEnum
from sklearn.model_selection import train_test_split
from hand_sign_recognition.lib import (
    HandSignDataLoggerGUI,
    NdJsonLabeledHandPointsStore,
)


store = NdJsonLabeledHandPointsStore(os.path.join("sample.ndjson"))


class HandSign(IntEnum):
    ROCK = 0
    GUN = 1
    FINGER_HEART = 2
    PAPER = 3
    OK = 4
    PEACE = 5
    BAD = 6
    INDEX_FINGER = 7


def get_data():
    HandSignDataLoggerGUI(store).start()


def train():
    from hand_sign_recognition.lib import HandSignTrainer, HandSignClassifier, NdJsonLabeledHandPointsStore

    # モデル
    classifier = HandSignClassifier(output_size=len(HandSign))

    # モデル学習オブジェクト
    trainer = HandSignTrainer(store, classifier)
    trainer.train(os.path.join(os.path.dirname(__file__), "sample_weight.pth"))


if __name__ == "__main__":
    train()
