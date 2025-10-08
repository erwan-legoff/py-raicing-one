import logging
from fastapi import FastAPI

app = FastAPI()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

@app.get("/")
async def root():
    return {"message": "Hello World"}
@app.websocket("/ai")
async def websocket_endpoint(websocket):
    logger.info("WebSocket connection established")
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        logger.info(f"data is {data}")
        await websocket.send_text(f"Message text was: {data}")