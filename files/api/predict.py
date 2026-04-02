import tensorflow as tf
import os, sys
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

EXPECTED_LANDMARKS = 21
EXPECTED_FEATURES = EXPECTED_LANDMARKS * 3  # x, y, z

# Path to the hand landmarker task model (mediapipe 0.10+ Tasks API)
_HAND_LANDMARKER_MODEL = os.path.join(
    os.path.dirname(__file__), "..", "model", "hand_landmarker.task"
)

def _create_detector():
    if not os.path.exists(_HAND_LANDMARKER_MODEL):
        print(f"Warning: hand_landmarker.task not found at {_HAND_LANDMARKER_MODEL}. Hand detection disabled.")
        return None
    base_options = mp_python.BaseOptions(model_asset_path=_HAND_LANDMARKER_MODEL)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)

_detector = _create_detector()


def get_marks(image_path):
    """
    Processes an image file to extract hand landmarks for the first detected hand.

    Returns:
        np.ndarray of shape (1, 63) with flattened x, y, z landmark coords,
        or None if no hand is detected.
    """
    if _detector is None:
        print("Hand detector not available.")
        return None

    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image file at {image_path}")
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    result = _detector.detect(mp_image)

    if not result.hand_landmarks:
        return None

    hand = result.hand_landmarks[0]

    if len(hand) != EXPECTED_LANDMARKS:
        print(f"Warning: Detected {len(hand)} landmarks, expected {EXPECTED_LANDMARKS}.")

    landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in hand])
    flattened = landmarks_array.flatten()
    return flattened.reshape(1, EXPECTED_FEATURES)


def _predict(landmark_data, loaded_model):
    """
    Predicts the ASL letter from landmark data using the loaded Keras model.

    Returns:
        str: Predicted ASL letter (A-Z), or empty string on failure.
    """
    if landmark_data is None:
        print("Prediction skipped: No landmark data provided.")
        return ""

    if landmark_data.shape != (1, EXPECTED_FEATURES):
        print(f"Prediction skipped: Incorrect shape {landmark_data.shape}, expected (1, {EXPECTED_FEATURES}).")
        return ""

    try:
        prediction = loaded_model.predict(landmark_data)

        if prediction is None or prediction.shape[0] < 1:
            print("Prediction failed: Model returned invalid output.")
            return ""

        index = np.argmax(prediction[0])
        return chr(index + 65)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return ""


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)

    image_file_path = sys.argv[1]
    model_path_local = os.path.join(os.curdir, "files", "model", "asl_model.keras")

    if not os.path.exists(model_path_local):
        print(f"Error: Model file not found at {model_path_local}")
        sys.exit(1)

    try:
        loaded_model_local = tf.keras.models.load_model(model_path_local, compile=False)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    landmarks = get_marks(image_file_path)
    if landmarks is not None:
        prediction_result = _predict(landmarks, loaded_model_local)
        print(f"Prediction for {image_file_path}: {prediction_result}")
    else:
        print(f"Could not get landmarks from {image_file_path}. No prediction made.")
