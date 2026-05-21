"""Record gesture keypoint sequences from your webcam.

Usage:
    python collect_data.py

Output:
    dataset/<class>/<sample_idx>/sequence.npy   shape (30, 126)
"""
import os
import cv2
import numpy as np
import mediapipe as mp

CLASSES = [
    "hello", "thank you", "yes", "no", "please",
    "sorry", "help", "water", "food", "love",
]
SEQ_LENGTH = 30
NUM_SAMPLES = 30
DATA_PATH = "dataset"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
)


def extract_keypoints(results) -> np.ndarray:
    out = np.zeros(126, dtype=np.float32)
    if not results.multi_hand_landmarks:
        return out
    for i, hand_lms in enumerate(results.multi_hand_landmarks[:2]):
        arr = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_lms.landmark], dtype=np.float32
        ).flatten()
        out[i * 63 : (i + 1) * 63] = arr
    return out


def main() -> None:
    cap = cv2.VideoCapture(0)
    for cls in CLASSES:
        for sample_idx in range(NUM_SAMPLES):
            print(f"Get ready: {cls} — sample {sample_idx + 1}/{NUM_SAMPLES}")
            input("Press Enter to record...")
            sequence = []
            for _ in range(SEQ_LENGTH):
                ok, frame = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                sequence.append(extract_keypoints(results))
                cv2.imshow("Recording", frame)
                cv2.waitKey(1)
            out_dir = os.path.join(DATA_PATH, cls, str(sample_idx))
            os.makedirs(out_dir, exist_ok=True)
            np.save(os.path.join(out_dir, "sequence.npy"), np.array(sequence))
    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()