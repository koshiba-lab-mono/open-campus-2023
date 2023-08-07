from __future__ import annotations
import sys
import unittest
from time import time

sys.path.append(".")
from hand_drone_app import check_same_sign_continuity


class TestCheckSameSignContinuity(unittest.TestCase):
    def test_continuity(self):
        expected_countinuity_sec = 3.0
        check_same_input_continuity = check_same_sign_continuity(expected_countinuity_sec)
        start_sec = time()

        while True:
            loop_sec = time() - start_sec

            fake_input = 0
            is_same_input_continuity, left_time = check_same_input_continuity(fake_input)

            if is_same_input_continuity:
                actual_countinuity_sec = loop_sec
                break

        self.assertAlmostEqual(expected_countinuity_sec, actual_countinuity_sec, 2)

    def test_continuity2(self):
        input_change_sec = 1.0

        arg_sec = 2.0
        check_same_input_continuity = check_same_sign_continuity(arg_sec)
        start_sec = time()

        while True:
            loop_sec = time() - start_sec
            fake_input = 0 if loop_sec <= input_change_sec else 1

            is_same_input_continuity, left_time = check_same_input_continuity(fake_input)

            if is_same_input_continuity:
                actual_countinuity_sec = loop_sec
                break

        expected_countinuity_sec = input_change_sec + arg_sec
        self.assertAlmostEqual(expected_countinuity_sec, actual_countinuity_sec, 2)


if __name__ == "__main__":
    unittest.main()
