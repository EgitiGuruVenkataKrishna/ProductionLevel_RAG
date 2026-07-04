"""
Legal RAG System — Local Development Entry Point

Run with: python main.py
This starts a local server with both API and frontend.
For Vercel deployment, this file is not used (Vercel uses api/ endpoints directly).
"""
import sys
import os
import logging
from pathlib import Path
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the fully configured app from api.ask
from api.ask import app

# ==================== SERVE FRONTEND ====================
frontend_dir = Path(__file__).parent / "frontend"

if frontend_dir.exists():
    # Mount the frontend directory as static files
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

# ==================== RUN ====================
PORT = int(os.getenv("PORT", 8000))

if __name__ == "__main__":
    import uvicorn
    # Use api.ask:app as the entrypoint for uvicorn
    uvicorn.run("api.ask:app", host="0.0.0.0", port=PORT, reload=True)