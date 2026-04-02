# Security

If you discover a vulnerability, please **do not** open a public issue. Email the maintainers or use GitHub private vulnerability reporting for this repository, if enabled.

## Deployment reminders

- Set a strong `SECRET_KEY` via environment (never use the default in production).
- Run behind HTTPS and a production WSGI server; do not expose `FLASK_DEBUG=1` on the internet.
- The app stores user accounts in SQLite by default; protect filesystem permissions on the database file.
- GPU video generation downloads large model weights from Hugging Face on first use; use a locked-down cache directory in shared environments.
