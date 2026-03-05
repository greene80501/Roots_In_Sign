# Roots In Sign (Sign2Speak)

[![License: MPL 2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)

A **bidirectional sign language translation platform** that bridges the communication gap between the deaf/hard-of-hearing community and hearing individuals. This web application enables **Sign-to-Text** recognition (ASL fingerspelling) and **Spoken-to-Signed** translation (converting speech/text to animated ASL gestures).

---

## Features

### 1. Sign-to-Text Translation
- **Real-time ASL Fingerspelling Recognition**: Uses your camera to capture hand gestures
- **AI-Powered Prediction**: Deep learning model (TensorFlow/Keras) trained to recognize ASL letters A-Z
- **MediaPipe Integration**: Accurate hand landmark detection for robust gesture recognition
- **Organized Data Collection**: Successfully recognized signs are saved for continuous model improvement

### 2. Spoken-to-Signed Translation
- **Speech-to-Text**: Fast and accurate transcription using OpenAI's Whisper model
- **ASL Gloss Generation**: Converts English text to ASL gloss notation
- **Animated Sign Output**: Generates animated GIFs of ASL signs using a pose library of **2,000+ signs**
- **Word-by-Word Pose Concatenation**: Smoothly chains individual sign poses into coherent sequences

### 3. User Management
- **Secure Authentication**: User registration and login system with password hashing
- **Session Management**: Persistent login sessions with Flask-Login
- **MySQL Database**: Robust backend for storing user accounts and application data

### 4. Modern Web Interface
- **Responsive Design**: Built with Tailwind CSS for mobile and desktop compatibility
- **Dark/Light Mode**: Automatic theme switching for comfortable viewing
- **Real-time Feedback**: Server-sent events (SSE) for streaming translation progress

---

## Architecture and Technology Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   Browser    │  │    Camera    │  │   Microphone Input   │  │
│  │  (HTML/JS)   │  │   (WebRTC)   │  │     (Web Audio)      │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
└─────────┼─────────────────┼─────────────────────┼──────────────┘
          │                 │                     │
          └─────────────────┴─────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────────────┐
│                      APPLICATION LAYER                          │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                    Flask Web Server                        │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐   │ │
│  │  │  Auth API   │  │  Sign2Text  │  │  Spoken2Signed  │   │ │
│  │  │   (/api)    │  │  (/predict) │  │    (/process)   │   │ │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘   │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
          │                    │                     │
┌─────────▼────────┐  ┌────────▼────────┐  ┌───────▼────────┐
│   DATA LAYER     │  │   AI/ML MODELS  │  │  POSE LIBRARY  │
│                  │  │                  │  │                │
│  ┌────────────┐  │  │  ┌────────────┐  │  │  ┌──────────┐  │
│  │   MySQL    │  │  │  │ TensorFlow │  │  │  │ 2,000+   │  │
│  │  (XAMPP)   │  │  │  │   (ASL)    │  │  │  │ .pose    │  │
│  │            │  │  │  │  Model     │  │  │  │  files   │  │
│  └────────────┘  │  │  └────────────┘  │  │  └──────────┘  │
│                  │  │                  │  │                │
│  ┌────────────┐  │  │  ┌────────────┐  │  │  ┌──────────┐  │
│  │ Flask-SQL  │  │  │  │   Whisper  │  │  │  │pose-format│  │
│  │  Alchemy   │  │  │  │  (faster)  │  │  │  │ Visualizer│  │
│  └────────────┘  │  │  └────────────┘  │  │  └──────────┘  │
└──────────────────┘  └──────────────────┘  └────────────────┘
```

### Core Technologies

| Component | Technology |
|-----------|------------|
| **Backend** | Python 3.x, Flask |
| **Database** | MySQL (via XAMPP), SQLAlchemy ORM |
| **ML/AI** | TensorFlow, Keras, MediaPipe, OpenAI Whisper |
| **Pose Processing** | pose-format library, PoseVisualizer |
| **Frontend** | HTML5, Tailwind CSS, Vanilla JavaScript |
| **Authentication** | Flask-Login, Werkzeug Security |

---

## Prerequisites

Before setting up the project, ensure you have the following installed:

### Required Software
- **Python 3.8+** - [Download here](https://www.python.org/downloads/)
- **XAMPP** (includes MySQL and Apache) - [Download here](https://www.apachefriends.org/)
- **Git** - [Download here](https://git-scm.com/downloads)
- **FFmpeg** (required for Whisper audio processing) - [Download here](https://ffmpeg.org/download.html)

### System Requirements
- **OS**: Windows 10/11, macOS, or Linux
- **RAM**: Minimum 8GB (16GB recommended for ML models)
- **Storage**: ~2GB free space (for models and pose library)
- **Camera**: Webcam for Sign-to-Text feature
- **Microphone**: For Spoken-to-Signed feature

---

## Setup Instructions

### Step 1: Clone the Repository

```bash
git clone https://github.com/greene80501/Roots_In_Sign.git
cd Roots_In_Sign
```

### Step 2: Set Up the Database (XAMPP)

1. **Start XAMPP Control Panel**
   - Launch XAMPP and start **Apache** and **MySQL** services

2. **Create the Database**
   - Open phpMyAdmin: `http://localhost/phpmyadmin`
   - Create a new database called `sign2speak_db`
   - Create a MySQL user (or use root) with appropriate privileges

3. **Configure Database Credentials**
   
   You can either:
   - **Option A**: Set environment variables (recommended for security)
     ```bash
     # Windows PowerShell
     $env:DB_USER = "your_mysql_username"
     $env:DB_PASS = "your_mysql_password"
     $env:DB_HOST = "localhost"
     $env:DB_PORT = "3306"
     $env:DB_NAME = "sign2speak_db"
     $env:SECRET_KEY = "your_secret_key_here"
     ```
   
   - **Option B**: Edit `app.py` directly (for development only)
     ```python
     db_username = 'your_mysql_username'
     db_password = 'your_mysql_password'
     ```

### Step 3: Create Python Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 4: Install Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

### Step 5: Initialize Database Migrations

```bash
# Set Flask app environment variable
# Windows PowerShell
$env:FLASK_APP = "app.py"

# Windows CMD
set FLASK_APP=app.py

# macOS/Linux
export FLASK_APP=app.py

# Initialize migrations (if not already done)
flask db init

# Create migration
flask db migrate -m "Initial migration"

# Apply migration
flask db upgrade
```

### Step 6: Verify Model and Pose Files

Ensure these directories and files exist:

```
Roots_In_Sign/
├── files/
│   ├── model/
│   │   └── asl_model.keras          # ASL prediction model
│   ├── static/
│   │   └── asl/                     # 2,000+ .pose files
│   │       ├── a.pose
│   │       ├── hello.pose
│   │       └── ...
│   └── api/
│       └── predict.py               # Prediction module
└── spoken-to-signed-translation/    # Translation module
    └── ...
```

### Step 7: Run the Application

```bash
# Development mode with debug
flask run --host=0.0.0.0 --port=5000 --debug

# Or run directly with Python
python app.py
```

The application will be available at: `http://localhost:5000`

---

## How It Works

### Sign-to-Text Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Camera    │───▶│   Capture   │───▶│  MediaPipe  │───▶│  Landmarks  │
│   Input     │    │    Frame    │    │   Hands     │    │   (21 pts)  │
└─────────────┘    └─────────────┘    └─────────────┘    └──────┬──────┘
                                                                  │
┌─────────────┐    ┌─────────────┐    ┌─────────────┐            │
│   Output    │◀───│  Predicted  │◀───│    ASL      │◀───────────┘
│   Letter    │    │   Letter    │    │   Model     │
│   (A-Z)     │    │   (A-Z)     │    │  (Keras)    │
└─────────────┘    └─────────────┘    └─────────────┘
```

1. **Frame Capture**: User shows ASL hand sign to camera
2. **Landmark Detection**: MediaPipe Hands extracts 21 hand landmarks (x, y, z coordinates)
3. **Feature Processing**: Landmarks are flattened into a 63-dimensional vector (21 x 3)
4. **Model Prediction**: TensorFlow model predicts the ASL letter (A-Z)
5. **Result Display**: The predicted letter is displayed to the user

### Spoken-to-Signed Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Audio     │───▶│   Whisper   │───▶│   English   │───▶│    ASL      │
│   Input     │    │ Transcribe  │    │    Text     │    │   Gloss     │
└─────────────┘    └─────────────┘    └─────────────┘    └──────┬──────┘
                                                                  │
┌─────────────┐    ┌─────────────┐    ┌─────────────┐            │
│   Display   │◀───│    GIF      │◀───│  Concaten.  │◀───────────┘
│   GIF/Video │    │  Preview    │    │    Poses    │
└─────────────┘    └─────────────┘    └─────────────┘
```

1. **Audio Recording**: User speaks into microphone
2. **Speech Recognition**: Whisper model transcribes audio to English text
3. **Text Processing**: English text is tokenized and converted to ASL gloss
4. **Pose Lookup**: Each word is matched to a corresponding `.pose` file
5. **Pose Concatenation**: Individual poses are smoothly chained together
6. **Visualization**: A GIF animation is generated and displayed

---

## API Endpoints

### Authentication
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/signup` | POST | Register a new user |
| `/api/signin` | POST | Login existing user |
| `/api/logout` | GET | Logout current user |
| `/api/check_auth` | GET | Check authentication status |

### Sign-to-Text
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/upload_frame` | POST | Upload a frame for processing |
| `/predict` | POST | Get prediction for uploaded frame |

### Spoken-to-Signed
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/process` | POST | Upload audio and get SSE stream with transcription/GIF |

### Pages
| Endpoint | Description |
|----------|-------------|
| `/` | Landing page |
| `/signin` | Sign in page |
| `/signup` | Sign up page |
| `/app` | Main application (requires login) |

---

## Project Structure

```
Roots_In_Sign/
├── app.py                          # Main Flask application
├── gloss.py                        # ASL gloss utility functions
├── LICENSE                         # Mozilla Public License 2.0
├── README.md                       # This file
│
├── files/
│   ├── api/
│   │   ├── predict.py              # ASL prediction module
│   │   └── tmp/                    # Temporary uploaded frames
│   │       ├── A/                  # Organized by prediction
│   │       ├── B/
│   │       └── ...
│   ├── model/
│   │   └── asl_model.keras         # Trained ASL model
│   ├── static/
│   │   ├── asl/                    # 2,000+ ASL pose files (.pose)
│   │   └── videos/                 # Generated GIF output
│   └── templates/
│       ├── index.html              # Landing page
│       ├── signin.html             # Login page
│       ├── signup.html             # Registration page
│       └── app.html                # Main application
│
├── migrations/                     # Flask-Migrate database migrations
│   ├── env.py
│   └── versions/
│
├── spoken-to-signed-translation/   # ASL translation module
│   └── spoken_to_signed/
│       ├── gloss_to_pose/
│       │   └── concatenate.py      # Pose concatenation logic
│       └── ...
│
└── temp/                           # Temporary audio/pose files
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SECRET_KEY` | Flask secret key for sessions | `a_very_insecure_default_key_for_dev_12345` |
| `DB_USER` | MySQL username | `sign2speak_user` |
| `DB_PASS` | MySQL password | *(required)* |
| `DB_HOST` | MySQL host | `localhost` |
| `DB_PORT` | MySQL port | `3306` |
| `DB_NAME` | MySQL database name | `sign2speak_db` |

### Model Configuration

Edit `app.py` to change model settings:

```python
# ASL Model (Sign-to-Text)
MODEL_NAME = "asl_model.keras"

# Whisper Model (Spoken-to-Signed)
# Options: "tiny", "base", "small", "medium", "large", "distil-small.en"
spoken_to_signed_model_size = "distil-small.en"
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Ensure virtual environment is activated and all dependencies are installed |
| `MySQL connection failed` | Check XAMPP is running and credentials are correct |
| `ASL model not found` | Verify `files/model/asl_model.keras` exists |
| `Pose files not found` | Ensure `files/static/asl/` contains .pose files |
| `Whisper model fails` | Install FFmpeg and ensure it's in system PATH |
| `Camera not working` | Allow browser camera permissions |
| `No hand detected` | Ensure good lighting and hand is clearly visible |

### Debug Mode

Enable Flask debug mode for detailed error messages:

```bash
$env:FLASK_DEBUG = "1"
flask run
```

---

## Contributing

Contributions are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the **Mozilla Public License 2.0** - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for hand tracking
- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition
- [pose-format](https://github.com/sign-language-processing/pose) for ASL pose handling
- [Tailwind CSS](https://tailwindcss.com/) for styling

---

## Contact

For questions or support, please open an issue on GitHub.

---

**Made with care for the Deaf and Hard-of-Hearing Community**
