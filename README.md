# Roots In Sign

[![License: MPL 2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)

Bidirectional **sign language ↔ text** web app: **Sign-to-Text** (ASL fingerspelling from the webcam) and **Spoken-to-Signed** (speech or typed English → animated ASL poses, rendered as MP4 with optional neural rendering).

---

## Features

| Area | What it does |
|------|----------------|
| **Sign-to-Text** | MediaPipe hand landmarks → TensorFlow/Keras classifier (letters A–Z). |
| **Spoken-to-Signed** | `faster-whisper` transcription → gloss → `.pose` lookup → concatenated sequence → **MP4** (AnimateDiff + ControlNet on GPU when available, else pix2pix + optional RealESRGAN). |
| **Accounts** | Register / sign in with Flask-Login; data stored in **SQLite** by default (no MySQL/XAMPP). |

---

## Requirements

- **Python 3.12** (recommended; TensorFlow wheels are reliable here).
- **FFmpeg** on `PATH` (Whisper + video encoding).
- **GPU (NVIDIA)** optional but strongly recommended for spoken-to-signed video (PyTorch + diffusers). CPU will fall back to lighter paths where possible.

---

## Quick start

```bash
git clone https://github.com/greene80501/Roots_In_Sign.git
cd Roots_In_Sign

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

pip install --upgrade pip
# Optional: CUDA PyTorch first — see https://pytorch.org/get-started/locally/
pip install -r requirements.txt

copy .env.example .env   # Windows; use cp on Unix — then edit SECRET_KEY
python scripts/download_hand_landmarker.py

set FLASK_APP=app.py
flask db upgrade
flask run --host=127.0.0.1 --port=5000
```

Open **http://127.0.0.1:5000** — use `/signup`, then **`/app`** for the tools.

**Windows:** you can use `.\start.ps1` from the repo root (sets common env vars and runs Flask).

### Environment variables

| Variable | Purpose |
|----------|---------|
| `SECRET_KEY` | Flask session signing (**required** for any shared deployment). |
| `DB_PATH` | Optional path to SQLite file (default: `sign2speak.db` in project root). |
| `FLASK_DEBUG` | `1` / `true` for dev only. |
| `PORT` | Dev server port when using `python app.py` (default `5000`). |
| `HF_HUB_DISABLE_SYMLINKS` | Set to `1` on Windows if Hugging Face cache fails without symlink rights. |
| `LOG_LEVEL` | Logging level, e.g. `INFO`, `DEBUG`. |

See **`.env.example`** for a template.

### Models and downloads

| Asset | When / where |
|--------|----------------|
| `files/model/hand_landmarker.task` | Run `scripts/download_hand_landmarker.py` once. |
| `files/model/asl_model.keras` | Bundled or supplied by your training pipeline (needed for sign-to-text). |
| `files/static/asl/*.pose` | Pose lexicon for gloss lookup (the app expects lowercase gloss names, e.g. `hello.pose`). |
| `~/.sign/models/pix2pix.h5` | Auto-downloaded on first video fallback (Firebase URL in `app.py`). |
| Hugging Face caches | SD 1.5, ControlNet, AnimateDiff motion adapter — pulled on first GPU video run. |

---

## Project layout

```
Roots_In_Sign/
├── app.py                 # Flask app, pipelines, video generation
├── gloss.py               # Simple English → ASL-style gloss helper (optional CLI)
├── requirements.txt
├── .env.example
├── SECURITY.md
├── scripts/
│   └── download_hand_landmarker.py
├── files/
│   ├── api/predict.py     # Sign-to-text inference
│   ├── model/             # Keras ASL model + hand_landmarker.task
│   ├── static/asl/        # .pose lexicon
│   ├── static/videos/     # Generated outputs (gitignored except .gitkeep)
│   └── templates/         # HTML + React (Babel) app UI
├── spoken-to-signed-translation/   # Vendored pipeline helpers (gloss_to_pose, etc.)
└── migrations/            # Flask-Migrate
```

---

## API overview

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/signup`, `/api/signin`, `/api/logout`, `/api/check_auth` | POST/GET | Authentication. |
| `/upload_frame`, `/predict` | POST | Sign-to-text frame upload / prediction. |
| `/process` | POST | Audio → SSE stream (transcript + video URL). |
| `/process_text` | POST | Text → same SSE shape. |
| `/`, `/app`, `/signin`, `/signup` | GET | Pages. |

---

## Troubleshooting

| Issue | What to try |
|-------|-------------|
| TensorFlow “GPU not available” on Windows | TensorFlow ≥2.11 does not use CUDA on native Windows; sign-to-text still runs on CPU. For GPU everywhere, use WSL2 or adjust your TF build. |
| `hand_landmarker.task` missing | Run `python scripts/download_hand_landmarker.py`. |
| Hugging Face symlink error on Windows | Set `HF_HUB_DISABLE_SYMLINKS=1` (see `start.ps1`). |
| Spoken-to-signed video OOM | Use a shorter sentence, or rely on pix2pix fallback after AnimateDiff fails; close other GPU apps. |

---

## Production notes

- Use a real `SECRET_KEY` and **disable** `FLASK_DEBUG`.
- Run behind **HTTPS** with a production WSGI server (e.g. **Gunicorn** on Linux, **Waitress** on Windows).
- Spoken-to-signed video is **heavy**; cap request size/timeouts at your reverse proxy and consider a job queue for long clips.
- Some NVIDIA driver + cuDNN combinations print `cudnnGetLibConfig` warnings; the app may disable cuDNN paths for stability — expect lower throughput on affected systems.

Details: **`SECURITY.md`**.

---

## Contributing

1. Fork and create a branch.
2. Keep changes focused; match existing style.
3. Open a PR with a clear description.

---

## Acknowledgments

- [sign-language-processing](https://github.com/sign-language-processing) — `pose-format`, `pose-to-video`, ControlNet weights.
- [MediaPipe](https://ai.google.dev/edge/mediapipe), [OpenAI Whisper](https://github.com/openai/whisper) / [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [diffusers](https://github.com/huggingface/diffusers).
- [spoken-to-signed-translation](https://github.com/ZurichNLP/spoken-to-signed-translation) (vendored subset for gloss → pose).

---

## License

[Mozilla Public License 2.0](LICENSE).
