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
simulation_history: list[HistoryPoint] = []
import json
from datetime import datetime
SAVE_INTERVAL = 120
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
        left_sensor = current.world.get("sensors",{}).get("left45Ray",  0)
        right_sensor = current.world.get("sensors",{}).get("left45Ray",  0)
        curr_z_speed = -float(current.world.get("speeds", {}).get("z", 0.0))
        curr_x_speed = -float(current.world.get("speeds", {}).get("x", 0.0))
        road_pos = current.world.get("positions", {}).get("road", {})
        
        
        # Vérifie que les positions sont valides
        if not curr_pos or not prev_pos:
            return 0.0 

        # Punition si la voiture est tombée sous la route
        if curr_pos.get("y", 0) < road_pos.get("y", 0):
            return -10 * abs(curr_x_speed)
    
        
        if(left_sensor < 0.1):
            return curr_x_speed*5
        if(right_sensor < 0.1):
            return -curr_x_speed*5

        # Avancement positif sur Z (plus on va loin, mieux c’est)
        prev_z = prev_pos.get("z", 0.0)
        curr_z = curr_pos.get("z", 0.0)
        # Reward = distance parcourue vers l’avant * facteur de gain
        if(curr_z_speed < 0.2 and curr_z_speed >= -1):
            return -10
        if(curr_z_speed < 0):
            return 10 * curr_z_speed
        reward = curr_z_speed - curr_z
        return reward
import math
import torch.optim as optim
optimizer = optim.Adam(auto_pilot.parameters(), lr=1e-4)
def get_action_log_probability(actions, probabilities):
    eps = 1e-8
    clipped_probs = torch.clamp(probabilities, eps, 1 - eps)
    log_probs = torch.log(clipped_probs) * actions 
    return log_probs.sum(dim=1)


SENSOR_MIN = 0.0
SENSOR_MAX = 50.0
SPEED_MAX = 10.0        
ACCEL_MAX = 2.0

def normalize_inputs(values: list[float]) -> torch.Tensor:
    # 7 capteurs + 3 vitesses + 3 accélérations = 13 features
    sensors = torch.tensor(values[:7])
    speeds = torch.tensor(values[7:10])
    accels = torch.tensor(values[10:13])

    # Capteurs déjà positifs
    sensors = (sensors - SENSOR_MIN) / (SENSOR_MAX - SENSOR_MIN + 1e-8)
    
    # Vitesse et accel : map [-max, +max] → [0, 1]
    speeds = (speeds + SPEED_MAX) / (2 * SPEED_MAX + 1e-8)
    accels = (accels + ACCEL_MAX) / (2 * ACCEL_MAX + 1e-8)

    # Fusionne tout
    normalized = torch.cat([sensors, speeds, accels])

    # Clamp par sécurité
    return torch.clamp(normalized, 0.0, 1.0)


def train(batch_size= 64, save_every=1000, resume: bool = True):
    
    if resume:
        loaded = load_latest_model(auto_pilot)
        if loaded:
            logger.info("🔄 Modèle existant chargé, reprise de l'entraînement.")
        else:
            logger.info("🆕 Aucun modèle trouvé — entraînement à partir de zéro.")
    inputs = torch.tensor([h.input for h in simulation_history], dtype=torch.float32, device=device)
    tries = torch.tensor([h.output for h in simulation_history], dtype=torch.float32, device=device)
    rewards = torch.tensor([h.reward for h in simulation_history], dtype=torch.float32, device=device)
    num_batches = math.ceil(len(inputs) / batch_size)
    logger.info(f"🧮 Début entraînement sur {len(inputs)} transitions ({num_batches} batchs)")
    logger.info("🖥️🖥️🖥️🖥️🖥️🖥️🖥️🖥️🖥️🖥️🖥️🖥️🖥️🖥️🖥️🖥️🖥️🖥️🖥️🖥️🖥️🖥️🖥️🖥️🖥️🖥️🖥️🖥️🖥️🖥️🖥️")
    auto_pilot.train()
    for batch_id in range(num_batches):
        start = batch_id * batch_size
        end = start + batch_size
        batch_inputs = inputs[start:end]
        batch_tries = tries[start:end]
        batch_rewards = rewards[start:end]
        # On normalise les rewards
        batch_rewards = (batch_rewards - batch_rewards.mean()) / (batch_rewards.std() + 1e-8)

        action_logits = auto_pilot(batch_inputs)
        action_probabilities = torch.sigmoid(action_logits)
        log_probabilities = get_action_log_probability(batch_tries,action_probabilities)
        loss = -(batch_rewards.view(-1, 1) * log_probabilities).mean()
        optimizer.zero_grad()
        loss.backward()
        # on évite que ça explose
        torch.nn.utils.clip_grad_norm_(auto_pilot.parameters(), max_norm=1.0)
        optimizer.step()
        # Log toutes les 50 itérations
        if batch_id % 10 == 0:
            avg_reward = batch_rewards.mean().item()
            logger.info(f"📉 Batch {batch_id+1}/{num_batches} | Loss={loss.item():.5f} | AvgReward={avg_reward:.3f}")


        # Sauvegarde périodique du modèle
        if save_every > 0 and (batch_id + 1) % save_every == 0:
            save_model(auto_pilot)
    logger.info("🖥️🖥️🖥️🖥️🖥️🖥️🖥️🖥️🖥️🖥️🖥️🖥️🖥️🖥️🖥️🖥️🖥️🖥️🖥️🖥️🖥️🖥️🖥️🖥️🖥️🖥️🖥️🖥️🖥️🖥️🖥️")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_model(auto_pilot, filename=f"autopilot_final_{timestamp}.pt")
    logger.info(f"✅ Entraînement terminé — modèle final sauvegardé avec timestamp {timestamp}.")

def load_simulation_from_file(path: str) -> list[HistoryPoint]:
    """
    Charge une simulation sauvegardée depuis un fichier JSON.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        history_points = []
        for item in data:
            hp = HistoryPoint(
                input=item.get("input", []),
                output=item.get("output", []),
                world=item.get("world", {}),
                result=item.get("result", {}),
                reward=item.get("reward", 0.0)
            )
            history_points.append(hp)
        logger.info(f"📂 Fichier chargé : {path} ({len(history_points)} transitions)")
        return history_points
    except Exception as e:
        logger.error(f"❌ Erreur de lecture du fichier {path} : {e}")
        return []

def load_latest_model(model: nn.Module, directory: str = "models") -> bool:
    """
    Recharge automatiquement le dernier modèle sauvegardé (le plus récent)
    depuis le dossier spécifié.

    Args:
        model (nn.Module): le modèle PyTorch à recharger
        directory (str): dossier contenant les fichiers .pt

    Returns:
        bool: True si un modèle a été chargé, False sinon
    """
    if not os.path.exists(directory):
        logger.warning(f"⚠️ Le dossier {directory} n'existe pas.")
        return False

    files = [f for f in os.listdir(directory) if f.endswith(".pt")]
    if not files:
        logger.warning(f"⚠️ Aucun fichier .pt trouvé dans {directory}.")
        return False

    # Trie par date de modification
    files.sort(key=lambda f: os.path.getmtime(os.path.join(directory, f)), reverse=True)
    latest_file = files[0]
    latest_path = os.path.join(directory, latest_file)

    try:
        model.load_state_dict(torch.load(latest_path, map_location=device))
        model.to(device)
        logger.info(f"📂 Modèle rechargé : {latest_file} sur {device}")
        return True
    except Exception as e:
        logger.error(f"❌ Échec du chargement de {latest_file} : {e}")
        return False

def train_from_file(directory: str = "sessions"):
    """
    Itère sur tous les fichiers d'entraînement (simulation_*.json)
    et entraîne le modèle sur chaque simulation.
    """
    if not os.path.exists(directory):
        logger.warning(f"⚠️ Dossier {directory} inexistant.")
        return

    files = [f for f in os.listdir(directory) if f.endswith(".json")]
    if not files:
        logger.warning(f"⚠️ Aucun fichier de simulation trouvé dans {directory}")
        return

    logger.info(f"🔍 {len(files)} fichiers trouvés dans {directory}.")
    total_transitions = 0

    for file in sorted(files):
        path = os.path.join(directory, file)
        simulation_data = load_simulation_from_file(path)
        if not simulation_data:
            continue

        # remplace temporairement la simulation active
        global simulation_history
        simulation_history = simulation_data

        train()
        total_transitions += len(simulation_data)

    logger.info(f"✅ Entraînement terminé sur {len(files)} fichiers ({total_transitions} transitions)")
    # sauvegarde du modèle
    torch.save(auto_pilot.state_dict(), "autopilot_trained.pt")
    logger.info("💾 Modèle sauvegardé : autopilot_trained.pt")

def save_model(model: nn.Module, directory: str = "models", filename: str | None = None):
    """
    Sauvegarde le modèle PyTorch dans le dossier spécifié.

    Args:
        model (nn.Module): le modèle à sauvegarder
        directory (str): dossier de sortie
        filename (str | None): nom de fichier (sinon horodaté)
    """
    os.makedirs(directory, exist_ok=True)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"autopilot_{timestamp}.pt"

    path = os.path.join(directory, filename)
    torch.save(model.state_dict(), path)
    # logger.info(f"💾 Modèle sauvegardé dans {path}")

@app.websocket("/ai")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info("✅ WebSocket connection established")
    prediction_seconds_before_learning = 15
    fps = 60
    predictions_before_learning : int = int(prediction_seconds_before_learning * fps)
    predictions_count: int = 0
    try:
        while True:
            if(predictions_count == 0):
                await ws.send_json({"driving_inputs": ["RESET"]})
                logger.info("🔁 Envoi d’un reset à la simulation")
            payload = await ws.receive_json()
    
            predictions_count = predictions_count + 1  
            # payload = { "sensors": {...}, "speeds": {...}, "accelerations": {...} }
            
            new_history_point, chosen = predict_actions(payload, predictions_count)
            if(predictions_count >= predictions_before_learning):
                predictions_count = 0
                train()
                save_simulation_history(simulation_history)
                simulation_history.clear()

            await ws.send_json({"driving_inputs": chosen})
            simulation_history.append(new_history_point)
    except Exception as e:
        save_simulation_history(simulation_history)
        logger.warning(f"⚠️ WebSocket closed: {e}")

def predict_actions(payload, i = 0):
    # i contient l'indice (ou compteur) de la frame actuelle depuis la boucle websocket
    frame_idx = int(i)
    if(len(simulation_history) > 0):
        simulation_history[-1].result = payload
        reward = compute_reward(simulation_history[-1], HistoryPoint(world=payload))
        # Log reward seulement toutes les 10 frames pour réduire le bruit
        if frame_idx % 10 == 0:
            logger.info(f"👌Reward: {reward:.2f}👌")
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
    normalized_input = normalize_inputs(values).unsqueeze(0).to(device)
    data_input = normalized_input

    with torch.no_grad():
        logits = auto_pilot(data_input)
        probs = torch.sigmoid(logits)
        active = (probs>=torch.rand_like(probs)).int()[0].tolist()
        chosen = [driving_inputs[i] for i, v in enumerate(active) if v == 1]

    # Ne logger que toutes les 10 frames pour éviter un trop grand volume de logs
    if frame_idx % 10 == 0:
        # Affiche la position Z de la voiture (curr_z) — déplacé depuis compute_reward
        try:
            curr_z = float(payload.get("positions", {}).get("car", {}).get("z", 0.0))
        except Exception:
            curr_z = 0.0
        try:
            curr_x_speed = float(payload.get("speeds", {}).get("x", 0.0))
        except Exception:
            curr_x_speed = 0.0
        logger.info(f"curr_z: {curr_z:.3f} | curr_x_speed: {curr_x_speed:.3f}")
        logger.info(f"Inputs: {[round(v, 2) for v in values]}")
        # log normalized inputs
        logger.info(f"Normalized: {[round(v.item(), 2) for v in normalized_input[0]]}")
            
    new_history_point.input = values
    new_history_point.output = active
    if frame_idx % 10 == 0:
        logger.info(f"Actions: {chosen}")
    return new_history_point,chosen

    




        