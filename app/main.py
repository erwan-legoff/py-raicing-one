import logging
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

ORIGINS = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5500",
    "null",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_origin_regex=r".*",   # enlever en prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.websocket("/ai")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info("WebSocket connection established")
    try:
        while True:
            data = await ws.receive_text()
            logger.info(f"data is {data}")
            await ws.send_text(f"Message text was: {data}")
    except Exception as e:
        logger.warning(f"WebSocket closed: {e}")
