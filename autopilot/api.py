"""FastAPI server wiring the new RL backend."""
from __future__ import annotations

import logging
import os
import random
from contextlib import contextmanager
from contextvars import ContextVar

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from .env_bridge import EnvironmentBridge
from .models import ActorCritic, ModelPaths, load_model
from .reward_engine import RewardEngine
from .session_manager import SessionManager
from .trainer import Trainer

LOGGER = logging.getLogger(__name__)
CURRENT_CLIENT_ID: ContextVar[str] = ContextVar("current_client_id", default="global")


class ClientIDFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - logging glue
        record.client_id = CURRENT_CLIENT_ID.get()
        return True


class ClientIDFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - logging glue
        if not hasattr(record, "client_id"):
            record.client_id = CURRENT_CLIENT_ID.get()
        return super().format(record)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s [client=%(client_id)s]: %(message)s",
    )
    logging.getLogger().addFilter(ClientIDFilter())
    for handler in logging.getLogger().handlers:
        handler.setFormatter(
            ClientIDFormatter(
                "%(asctime)s [%(levelname)s] %(name)s [client=%(client_id)s]: %(message)s"
            )
        )
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "watchfiles"):
        logging.getLogger(name).addFilter(ClientIDFilter())


@contextmanager
def client_logging_context(client_id: str):
    token = CURRENT_CLIENT_ID.set(client_id)
    try:
        yield
    finally:
        CURRENT_CLIENT_ID.reset(token)


def create_app() -> FastAPI:
    configure_logging()

    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    model = ActorCritic()
    startup_path = os.path.join("saved_models", "autopilot_v3_slow_20251102_164846.pt")
    # load_model(model, ModelPaths(startup_path=startup_path))

    session_manager = SessionManager()
    trainer = Trainer(model)
    reward_engine = RewardEngine()
    bridge = EnvironmentBridge(model, trainer, reward_engine, session_manager)

    @app.get("/")
    async def root():  # pragma: no cover - simple ping endpoint
        return {"message": "Py-Raicing autopilot"}

    @app.websocket("/ai")
    async def websocket_endpoint(ws: WebSocket):
        await ws.accept()
        client_id = ws.query_params.get("id") or f"anon_{random.randint(1, 10**9)}"
        session_manager.get(client_id)  # ensure exists
        initial_reset_sent = False
        LOGGER.info("🌐 WebSocket connection opened")
        try:
            while True:
                if not initial_reset_sent:
                    await ws.send_json(
                        {
                            "driving_inputs": ["RESET"],
                            "reward": 0.0,
                            "tags": ["INIT"],
                            "training": bridge.training_metadata(),
                        }
                    )
                    initial_reset_sent = True
                payload = await ws.receive_json()
                with client_logging_context(client_id):
                    message = bridge.process_step(client_id, payload)
                await ws.send_json(message)
        except Exception as exc:  # pragma: no cover - network path
            LOGGER.warning("⚠️ WebSocket closed: %s", exc)
            session_manager.remove(client_id)

    return app
