#!/usr/bin/env python3
"""Download MediaPipe Hand Landmarker task model into files/model/."""
from __future__ import annotations

import os
import urllib.request

# Official MediaPipe hand_landmarker float16 bundle (Tasks API)
URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dest_dir = os.path.join(root, "files", "model")
    dest = os.path.join(dest_dir, "hand_landmarker.task")
    os.makedirs(dest_dir, exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        print(f"Already present: {dest}")
        return
    print(f"Downloading to {dest} …")
    urllib.request.urlretrieve(URL, dest)
    print("Done.")


if __name__ == "__main__":
    main()
