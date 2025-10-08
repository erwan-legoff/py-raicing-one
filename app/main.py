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
            # <-- parse directement en dict Python
            payload = await ws.receive_json()

            # payload ressemble à {"front": 12.3, "left": 10000, ...}
            # log concis pour éviter d’inonder la console
            # (si tu veux tout voir: logger.info(payload))
            min_name, min_dist = min(payload.items(), key=lambda kv: kv[1])
            logger.info(f"min distance: {min_dist:.3f} ({min_name})")

            # renvoie un message utile au front
            await ws.send_json({
                "min": {"name": min_name, "distance": min_dist},
                "all": payload  # enlève en prod si trop verbeux
            })
    except Exception as e:
        logger.warning(f"WebSocket closed: {e}")
