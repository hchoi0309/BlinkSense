"""
BlinkSense FastAPI Application
Real-time drowsiness detection using FastAPI + WebSockets
"""
import logging.config
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from .config import (
    APP_NAME, APP_VERSION, APP_DESCRIPTION,
    ALLOWED_ORIGINS, STATIC_DIR, TEMPLATES_DIR,
    LOGGING_CONFIG
)

# Configure logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=APP_NAME,
    description=APP_DESCRIPTION,
    version=APP_VERSION
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    logger.info(f"Mounted static files from {STATIC_DIR}")
else:
    logger.warning(f"Static directory not found: {STATIC_DIR}")

# Set up templates
if TEMPLATES_DIR.exists():
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    logger.info(f"Configured templates from {TEMPLATES_DIR}")
else:
    logger.warning(f"Templates directory not found: {TEMPLATES_DIR}")
    templates = None

@app.get("/")
async def index(request: Request):
    """Main page for drowsiness detection"""
    if templates is None:
        return {"message": "BlinkSense API is running, but templates not found"}
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "BlinkSense"}

# Import WebSocket endpoint after app initialization
from .api.websocket import websocket_endpoint

# Register WebSocket endpoint
app.websocket("/ws/drowsiness/")(websocket_endpoint)

if __name__ == "__main__":
    import uvicorn
    from .config import HOST, PORT
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=True)