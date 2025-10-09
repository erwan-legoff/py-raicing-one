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
from dataclasses import dataclass, field

@dataclass
class HistoryPoint:
    input: list = field(default_factory=list)
    output: list = field(default_factory=list)
    world: dict = field(default_factory=dict)
    result: dict = field(default_factory=dict)
    reward: float = 0.0

auto_pilot = AutoPilot(input_size,hidden_size,output_size)
auto_pilot.to(device)
driving_inputs = {0: "LEFT", 1: "FORWARD", 2: "RIGHT", 3:"BACKWARD"}
simulation_history = []
import json
from datetime import datetime
SAVE_INTERVAL = 60
def save_simulation_history(simulation_history, directory="sessions"):
    """
    Sauvegarde la simulation actuelle dans un fichier JSON horodaté.

    Args:
        simulation_history (list[HistoryPoint]): liste d’objets HistoryPoint
        directory (str): dossier de sortie (créé s’il n’existe pas)
    """
    if not simulation_history:
        logger.warning("⚠️ Aucune simulation à sauvegarder.")
        return

    # Création du dossier s’il n’existe pas
    os.makedirs(directory, exist_ok=True)

    # Génère un nom de fichier unique
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"simulation_{timestamp}.json"
    path = os.path.join(directory, filename)

    # Conversion en dicts JSON-friendly
    serializable = [hp.__dict__ for hp in simulation_history]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    logger.info(f"💾 Simulation sauvegardée dans {path}")
import threading
def periodic_auto_save(simulation_history):
    """
    Sauvegarde périodiquement la simulation toutes les SAVE_INTERVAL secondes.
    Fonction récursive via threading.Timer.
    """
    save_simulation_history(simulation_history)
    threading.Timer(SAVE_INTERVAL, periodic_auto_save, args=[simulation_history]).start()
if __name__ == "__main__":
    periodic_auto_save(simulation_history)

def compute_reward(previous: HistoryPoint, current: HistoryPoint) -> float:
        """
        Calcule la récompense (reward) entre deux états successifs de simulation.
        - Pénalise fort si la voiture tombe sous la route
        - Récompense le déplacement vers l'avant (axe Z)
        """

        prev_pos = previous.result.get("positions", {}).get("car", {})
        curr_pos = current.world.get("positions", {}).get("car", {})
        curr_speed = -current.world.get("speeds", {}).get("z", {})
        road_pos = current.world.get("positions", {}).get("road", {})
        
        
        # Vérifie que les positions sont valides
        if not curr_pos or not prev_pos:
            return 0.0

        # Punition si la voiture est tombée sous la route
        if curr_pos.get("y", 0) < road_pos.get("y", 0):
            return -100.0

        # Avancement positif sur Z (plus on va loin, mieux c’est)
        prev_z = prev_pos.get("z", 0.0)
        curr_z = curr_pos.get("z", 0.0)
        logger.info(f"prev_z: {prev_z:.3f}")
        logger.info(f"curr_z: {curr_z:.3f}")
        # Reward = distance parcourue vers l’avant * facteur de gain
        if(curr_speed < 0.2 and curr_speed >= 0):
            return -10
        if(curr_speed < 0):
            return 10 * curr_speed
        reward = curr_speed
        
        logger.info(f"Reward: {reward:.3f}")
        return reward


import random

@app.websocket("/ai")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info("✅ WebSocket connection established")

    try:
        while True:
            payload = await ws.receive_json()
            # payload = { "sensors": {...}, "speeds": {...}, "accelerations": {...} }
            if(len(simulation_history) > 0):
                simulation_history[-1].result = payload
                reward = compute_reward(simulation_history[-1], HistoryPoint(world=payload))
                simulation_history[-1].reward = reward
            new_history_point = HistoryPoint()
            new_history_point.world = payload
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
                active = (probs>=torch.rand_like(probs)).int()[0].tolist()
                chosen = [driving_inputs[i] for i, v in enumerate(active) if v == 1]

            logger.info(f"Inputs: {[round(v, 2) for v in values]}")

            
            new_history_point.input = values
            new_history_point.output = chosen
            logger.info(f"Actions: {chosen}")

            await ws.send_json({"driving_inputs": chosen})
            simulation_history.append(new_history_point)
    except Exception as e:
        save_simulation_history(simulation_history)
        logger.warning(f"⚠️ WebSocket closed: {e}")

    




        