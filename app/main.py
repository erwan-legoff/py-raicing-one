import logging
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

logger.info(f"🖥️  Using device: {device} on torch {torch.__version__}")

@app.get("/")
async def root():
    return {"message": "Hello World"}



# NN Module
class AutoPilot(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_layer = nn.Linear(input_size,hidden_size, device=device)
        self.non_linear_1 = nn.ReLU()
        self.hidden_layer_1 = nn.Linear(hidden_size,hidden_size, device=device)
        self.non_linear_2 = nn.ReLU()
        self.hidden_layer_2 = nn.Linear(hidden_size,hidden_size//2, device=device)
        self.non_linear_3 = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size//2,output_size, device=device)

    def forward(self, inputs):
        inputs = self.input_layer(inputs)
        inputs = self.non_linear_1(inputs)
        inputs = self.hidden_layer_1(inputs)
        inputs = self.non_linear_2(inputs)
        inputs = self.hidden_layer_2(inputs)
        inputs = self.non_linear_3(inputs)
        inputs = self.output_layer(inputs)
        return inputs
    
input_size = 13  # 7 capteurs + 3 vitesses + 3 accélérations
hidden_size = 128
output_size = 4
sensor_order = [
    "left45Ray",
    "left22Ray",
    "leftNarrowRay",
    "frontRay",
    "rightNarrowRay",
    "right22Ray",
    "right45Ray",
]

auto_pilot = AutoPilot(input_size,hidden_size,output_size)
auto_pilot.to(device)
driving_inputs = {0: "LEFT", 1: "FORWARD", 2: "RIGHT", 3:"BACKWARD"}
@app.websocket("/ai")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info("✅ WebSocket connection established")

    try:
        while True:
            payload = await ws.receive_json()
            # payload = { "sensors": {...}, "speeds": {...}, "accelerations": {...} }

            sensors = payload.get("sensors", {})
            speeds = payload.get("speeds", {})
            accels = payload.get("accelerations", {})

            # 1️⃣ Distances dans l’ordre défini
            sensor_values = [sensors.get(k, 0.0) for k in sensor_order]

            # 2️⃣ Vitesses et accélérations (x, y, z)
            speed_values = [speeds.get(axis, 0.0) for axis in ("x", "y", "z")]
            accel_values = [accels.get(axis, 0.0) for axis in ("x", "y", "z")]

            # 3️⃣ Fusion complète : [7 capteurs] + [3 vitesses] + [3 accels] = 13 features
            values = sensor_values + speed_values + accel_values

            data_input = torch.tensor([values], dtype=torch.float32, device=device)

            with torch.no_grad():
                logits = auto_pilot(data_input)
                probs = torch.sigmoid(logits)
                active = (probs > 0.66).int()[0].tolist()
                chosen = [driving_inputs[i] for i, v in enumerate(active) if v == 1]

            logger.info(f"Inputs: {[int(v) for v in values]}")

            logger.info(f"Actions: {chosen}")

            await ws.send_json({"driving_inputs": chosen})

    except Exception as e:
        logger.warning(f"⚠️ WebSocket closed: {e}")


        