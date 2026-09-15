import math
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from pynput.keyboard import Controller, Key


@dataclass(frozen=True)
class Config:
    camera_index: int = 0
    max_hands: int = 1
    detection_confidence: float = 0.7
    tracking_confidence: float = 0.7

    min_distance: int = 25
    max_distance: int = 250
    volume_threshold: float = 5.0
    smoothing_factor: float = 0.25

    bar_left: int = 50
    bar_right: int = 85
    bar_top: int = 150
    bar_bottom: int = 400

    window_title: str = "Hand Volume Controller"
    quit_key: str = "q"


class HandTracker:
    def __init__(self, config: Config):
        self.config = config
        self.mp_hands = mp.solutions.hands
        self.drawer = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.max_hands,
            min_detection_confidence=config.detection_confidence,
            min_tracking_confidence=config.tracking_confidence,
        )

    def process(self, frame):
        height, width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if not results.multi_hand_landmarks:
            return None

        hand = results.multi_hand_landmarks[0]

        self.drawer.draw_landmarks(
            frame,
            hand,
            self.mp_hands.HAND_CONNECTIONS,
        )

        return [
            (int(landmark.x * width), int(landmark.y * height))
            for landmark in hand.landmark
        ]

    def close(self):
        self.hands.close()


class VolumeController:
    def __init__(self, config: Config):
        self.config = config
        self.keyboard = Controller()
        self.volume = 50.0

    def distance_to_volume(self, distance):
        return float(
            np.interp(
                distance,
                [self.config.min_distance, self.config.max_distance],
                [0, 100],
            )
        )

    def update(self, target_volume):
        delta = target_volume - self.volume

        if delta > self.config.volume_threshold:
            self.keyboard.tap(Key.media_volume_up)
        elif delta < -self.config.volume_threshold:
            self.keyboard.tap(Key.media_volume_down)
        else:
            return

        self.volume = target_volume

    @property
    def current_volume(self):
        return self.volume


class GestureRenderer:
    def __init__(self, config: Config):
        self.config = config

    def draw_hand_gesture(self, frame, thumb, index, distance):
        center = (
            (thumb[0] + index[0]) // 2,
            (thumb[1] + index[1]) // 2,
        )

        color = (255, 0, 255)

        cv2.circle(frame, thumb, 15, color, cv2.FILLED)
        cv2.circle(frame, index, 15, color, cv2.FILLED)
        cv2.line(frame, thumb, index, color, 3)
        cv2.circle(frame, center, 15, color, cv2.FILLED)

        if distance < 50:
            cv2.circle(frame, center, 15, (0, 255, 0), cv2.FILLED)

    def draw_volume(self, frame, volume):
        bar_position = int(
            np.interp(
                volume,
                [0, 100],
                [self.config.bar_bottom, self.config.bar_top],
            )
        )

        cv2.rectangle(
            frame,
            (self.config.bar_left, self.config.bar_top),
            (self.config.bar_right, self.config.bar_bottom),
            (0, 255, 0),
            3,
        )

        cv2.rectangle(
            frame,
            (self.config.bar_left, bar_position),
            (self.config.bar_right, self.config.bar_bottom),
            (0, 255, 0),
            cv2.FILLED,
        )

        cv2.putText(
            frame,
            f"{int(volume)}%",
            (40, 450),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 255, 0),
            3,
        )


class HandVolumeApplication:
    def __init__(self):
        self.config = Config()
        self.tracker = HandTracker(self.config)
        self.volume_controller = VolumeController(self.config)
        self.renderer = GestureRenderer(self.config)
        self.camera = cv2.VideoCapture(self.config.camera_index)
        self.smoothed_volume = self.volume_controller.current_volume

    def run(self):
        if not self.camera.isOpened():
            raise RuntimeError("Could not open the camera.")

        try:
            while True:
                success, frame = self.camera.read()

                if not success:
                    break

                frame = cv2.flip(frame, 1)
                landmarks = self.tracker.process(frame)

                if landmarks is not None:
                    self.process_gesture(frame, landmarks)

                self.renderer.draw_volume(
                    frame,
                    self.volume_controller.current_volume,
                )

                cv2.imshow(self.config.window_title, frame)

                if cv2.waitKey(1) & 0xFF == ord(self.config.quit_key):
                    break

        finally:
            self.close()

    def process_gesture(self, frame, landmarks):
        thumb = landmarks[4]
        index = landmarks[8]

        distance = math.hypot(
            index[0] - thumb[0],
            index[1] - thumb[1],
        )

        target_volume = self.volume_controller.distance_to_volume(distance)

        self.smoothed_volume += (
            target_volume - self.smoothed_volume
        ) * self.config.smoothing_factor

        self.volume_controller.update(self.smoothed_volume)

        self.renderer.draw_hand_gesture(
            frame,
            thumb,
            index,
            distance,
        )

    def close(self):
        self.camera.release()
        self.tracker.close()
        cv2.destroyAllWindows()


def main():
    application = HandVolumeApplication()
    application.run()


if __name__ == "__main__":
    main()
