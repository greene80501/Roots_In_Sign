# Sign to Speak Backend

A Flask-based backend that powers the Sign to Speak application. It provides American Sign Language (ASL) sign-to-text recognition, spoken-to-signed GIF generation, user authentication, and database-backed session management.

## Features

- **Sign-to-Text Recognition** – Accepts uploaded image frames and predicts ASL glosses using a TensorFlow model.
- **Spoken-to-Signed Translation** – Generates animated poses/GIFs for spoken phrases via Faster Whisper transcription and pose concatenation utilities.
- **User Accounts** – Supports registration, authentication, and session management with Flask-Login and a MySQL database.
- **Integrated Web UI** – Serves both JSON APIs and rendered HTML pages from the `files/templates` directory, enabling a full-stack experience.

## Project Structure

```
app.py                        # Flask entry point and route definitions
files/
├── api/                     # API helpers (prediction, temporary storage)
├── model/                   # Trained TensorFlow ASL model assets
├── static/                  # Front-end assets (JS, CSS, videos, pose files)
└── templates/               # Jinja2 HTML templates for the web UI
migrations/                   # Flask-Migrate database revision history
temp/                         # Temporary audio/pose storage for spoken-to-signed feature
spoken-to-signed-translation/ # Spoken-to-signed helper modules
```

## Prerequisites

- Python 3.10+
- MySQL 8.x (or compatible) with access credentials for the application
- `ffmpeg` available on the system `PATH` (required by Faster Whisper)
- CUDA-capable GPU (optional) for faster inference

## Installation

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd PROD_ASL_WEB
   ```
2. **Create and activate a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # On Windows: .venv\Scripts\activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   > If a `requirements.txt` file is not present, install the packages imported in `app.py` manually (Flask, Flask-Login, Flask-Migrate, Flask-SQLAlchemy, mysql-connector-python, tensorflow, faster-whisper, pose-format, Pillow, etc.).
4. **Download model assets**
   - Place the trained Keras model (`asl_model.keras`) inside `files/model/`.
   - Ensure ASL pose files (`*.pose`) exist in `files/static/asl/` for the spoken-to-signed workflow.

## Configuration

Set the following environment variables (or edit the defaults in `app.py`):

| Variable      | Description                       | Default                |
|---------------|-----------------------------------|------------------------|
| `SECRET_KEY`  | Flask session secret              | `a_very_insecure...`   |
| `DB_USER`     | MySQL username                    | `sign2speak_user`      |
| `DB_PASS`     | MySQL password                    | `Sv227199`             |
| `DB_HOST`     | MySQL hostname                    | `localhost`            |
| `DB_PORT`     | MySQL port                        | `3306`                 |
| `DB_NAME`     | MySQL database name               | `sign2speak_db`        |

## Database Setup

Initialize and migrate the database using Flask-Migrate:

```bash
flask db init       # Run once to create the migrations/ folder
flask db migrate -m "Initial migration"
flask db upgrade
```

## Running the Application

Set the `FLASK_APP` environment variable and start the development server:

```bash
export FLASK_APP=app.py           # PowerShell: $env:FLASK_APP="app.py"
flask run --host=0.0.0.0 --port=5000
```

For production deployments, run the app under a WSGI server such as Gunicorn or uWSGI and configure HTTPS, logging, and process supervision as needed.

## API Highlights

- `POST /upload_frame` – Stores a base64-encoded PNG frame and returns a token for prediction.
- `POST /predict` – Uses a stored frame token to perform ASL classification.
- `POST /process` – Streams transcription events and pose animation GIF URLs for spoken phrases.
- `GET /`, `/signin`, `/signup`, `/app` – Render the primary web interface and authentication pages.

Refer to `app.py` for the full list of routes, request payloads, and response schemas.

## Troubleshooting

- Verify the ASL model and pose files are present if prediction or GIF generation fails.
- Ensure `ffmpeg` is installed and discoverable by Faster Whisper.
- Check the console logs for warnings about missing directories (`files/static/asl`, `files/static/videos`) and create them manually if necessary.
- When running in production, set a strong `SECRET_KEY` and provide secure database credentials via environment variables.

## License

This project is distributed under the terms of the [LICENSE](LICENSE) file included in the repository.

