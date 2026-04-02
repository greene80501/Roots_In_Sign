# Optional: copy .env.example to .env and set SECRET_KEY for anything beyond local dev.
$env:TF_ENABLE_ONEDNN_OPTS   = '0'
$env:TF_USE_LEGACY_KERAS     = '1'
$env:FLASK_APP                = 'app.py'
$env:TORCH_CUDNN_NOSUPPORT   = '1'
$env:HF_HUB_DISABLE_SYMLINKS = '1'

Write-Host "Starting Roots In Sign server..." -ForegroundColor Cyan
Write-Host "Open http://localhost:5000 in your browser" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop`n" -ForegroundColor Yellow

.\venv\Scripts\python -m flask run --host=0.0.0.0 --port=5000
