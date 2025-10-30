import logging
import random
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
predictions_count: int = 0
predictions_since_training: int = 0

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
        self.layer_normalization_1 = nn.LayerNorm(hidden_size)
        self.non_linear_1 = nn.ReLU()
        self.hidden_layer_1 = nn.Linear(hidden_size,hidden_size, device=device)
        self.layer_normalization_2 = nn.LayerNorm(hidden_size)
        self.non_linear_2 = nn.ReLU()
        self.hidden_layer_2 = nn.Linear(hidden_size,hidden_size//2, device=device)
        self.layer_normalization_3 = nn.LayerNorm(hidden_size // 2)
        self.non_linear_3 = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size//2,output_size, device=device)
        self.training_count = 0  # Compteur d'itérations d'entraînement

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        state["_training_count"] = self.training_count
        return state
    
    def load_state_dict(self, state_dict, strict=True):
        self.training_count = state_dict.pop("_training_count", 0)
        super().load_state_dict(state_dict, strict)

    def forward(self, inputs):
        x = self.input_layer(inputs)
        x = self.layer_normalization_1(x)
        x = self.non_linear_1(x)

        x = self.hidden_layer_1(x)
        x = self.layer_normalization_2(x)
        x = self.non_linear_2(x)

        x = self.hidden_layer_2(x)
        x = self.layer_normalization_3(x)
        x = self.non_linear_3(x)

        return self.output_layer(x)
    
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
    world_input: dict = field(default_factory=dict)
    world_result: dict = field(default_factory=dict)
    reward: float = 0.0

auto_pilot = AutoPilot(input_size,hidden_size,output_size)
auto_pilot.to(device)
STARTUP_MODEL_PATH = os.path.join("saved_models", "autopilot_v2_random_position_20251018_223123.pt")
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

def reward_emoji(value: float) -> str:
    if value >= 300:
        return "🏆"
    if value >= 50:
        return "🎉"
    if value > 0:
        return "😄"
    if value <= -500:
        return "💀"
    if value <= -100:
        return "🔥"
    if value < 0:
        return "😤"
    return "😐"

def log_reward(value: float, reason: str) -> float:
    emoji = reward_emoji(value)
    logger.info(f"{emoji} Reward ({reason}): {value:.2f}")
    return value

def log_reward_update(current: float, delta: float, reason: str) -> float:
    new_value = current + delta
    emoji = reward_emoji(delta)
    total_emoji = reward_emoji(new_value)
    logger.info(f"{emoji} Reward update ({reason}): {delta:+.2f} -> {total_emoji} total {new_value:.2f}")
    return new_value

def compute_reward(historyPoint: HistoryPoint, road_size:dict, car_size:dict, frame_before_interaction:int ,frame_idx:int) -> float:
        """
        Calcule la récompense (reward) entre deux états successifs de simulation.
        - Pénalise fort si la voiture tombe sous la route
        - Récompense le déplacement vers l'avant (axe Z)
        """

        world_result = historyPoint.world_result
        world_input = historyPoint.world_input
        
        position_result = world_result.get("positions", {}).get("car", {})
        position_input = world_input.get("positions", {}).get("car", {})

        input_left_sensor = world_input.get("sensors",{}).get("left45Ray",  0)
        input_right_sensor = world_input.get("sensors",{}).get("right45Ray",  0)

        result_left_sensor = world_result.get("sensors",{}).get("left45Ray",  0)
        result_right_sensor = world_result.get("sensors", {}).get("right45Ray", 0)
        # On inverse l’axe Z pour que avancer → reward positif
        result_z_speed = -float(world_result.get("speeds", {}).get("z", 0.0))
        curr_y_speed = float(world_result.get("speeds", {}).get("y", 0.0))
        result_x_speed = float(world_result.get("speeds", {}).get("x", 0.0))
        road_pos = world_input.get("positions", {}).get("road", {})
        # Avancement positif sur Z (plus on va loin, mieux c’est)
        input_position_z = -position_input.get("z", 0.0)
        result_position_z = -position_result.get("z", 0.0)
        result_position_y = position_result.get("y", 0.0)
        input_position_x = position_input.get("x", 0.0)
        result_position_x = position_result.get("x", 0.0)
        road_width = road_size.get("width",0)
        car_width = car_size.get("width",0)
        reward = 0
        # Vérifie que les positions sont valides
        if not position_result or not position_input:
            return log_reward(0.0, "positions invalides")
        if frame_idx <= frame_before_interaction:
            return log_reward(0.0, "début d'épisode")
        road_length = (road_size.get("depth",0) / 2) - 10 
        if(abs(result_position_z) > road_length):
            return log_reward(100, "franchissement de la ligne d'arrivée")
        # Punition si la voiture est tombée sous la route
        if result_position_y < road_pos.get("y", 0):
            return log_reward(-10 * abs(result_x_speed), "chute sous la route")

        
        # Si la voiture revient au début, c'est que la prédiction est mauvaise, on punit
        if(input_position_z > 5 and result_position_z < 1):
            return log_reward(-100, "retour au départ")
        # Road width = 5 
        # Car Width = 1
        x_max = (road_width/2) - (car_width/2)
        x_min = x_max/3
        if(x_min < abs(input_position_x)):
            logger.info("🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴")
        side_proximity_ration = min(abs(result_position_x) / x_max, 1)
        # Si on était pas dehors mais que l'action fait sortir
        # alors on punit et on ajoute une punition proportionnel à l'engouement vers x
        if(abs(result_position_x) > x_max and abs(input_position_x) < x_max):
            return log_reward(-100 - 10 * abs(result_position_x) - abs(input_position_x), "sortie de route")
        # Si on était déjà dehors alors on punit et ajoute une proportionalité à l'engouement vers x
        if(abs(result_position_x) > x_max and abs(input_position_x) > x_max):
            return log_reward(-10 - 10 * abs(result_position_x) - abs(input_position_x), "persistance hors route")
        # Si on était pas dans X et qu'on rentre dedans, alors on ajoute une punition avec une proportionnalité de l'engouement vers x
        if(abs(result_position_x)>x_min and abs(input_position_x) < x_min):
            delta = -(10 + 5 * (abs(result_position_x) - abs(input_position_x)))
            reward = log_reward_update(reward, delta, "entrée zone dangereuse")
        # Si on se dirige vers x, et que de base on était dans la zone dangereuse
        # alors on punit de plus en plus qu'on s'approche du bord
        if(abs(result_position_x) > abs(input_position_x) and abs(input_position_x) > x_min):
            delta = -55*result_z_speed * side_proximity_ration**3
            reward = log_reward_update(reward, delta, "approche du bord")
        # Si on se pars de x, et que de base on était dans la zone dangereuse
        # alors on récompense proportionnellement à la proximité
        if(abs(result_position_x) < abs(input_position_x) and abs(input_position_x) > x_min):
            delta = 20*result_z_speed * side_proximity_ration**2
            reward = log_reward_update(reward, delta, "éloignement du bord")

        # Si on sort de la zone dangereuse on a un petit bonus
        if(abs(input_position_x) > x_min and abs(result_position_x) < x_min):
            reward = log_reward_update(reward, 10, "sortie zone dangereuse")
            
        
        # Si on avance peu ou qu'on recule légèrement, on punit de 10
        if(result_z_speed < 0.2 and result_z_speed >= -1):
            reward = log_reward_update(reward, -10, "vitesse avant insuffisante")

        # On récompense par rapport à la vitesse en avant et donc on punit autant si il recule
        centric_reward = 30 * result_z_speed * (max(0,1 - abs(result_position_x) /(x_min)))
        reward = log_reward_update(reward, centric_reward, "progression axiale")
        # if(abs(input_position_x) > abs(result_position_x)):
        #     delta = 5*result_z_speed
        #     reward = log_reward_update(reward, delta, "recentrage latéral")


        return log_reward(reward, "cumul")
import math
import torch.optim as optim
optimizer = optim.Adam(auto_pilot.parameters(), lr=1e-4)
from torch.distributions import Bernoulli
def get_action_log_probability(actions, logits):
    dist = Bernoulli(logits=logits)   
    return dist.log_prob(actions).sum(dim=1, keepdim=True) 


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

def process_rewards(simulation_history: list[HistoryPoint]) -> torch.Tensor:
    """Nettoie, normalise et log les rewards pour le training."""
    raw_rewards = [h.reward for h in simulation_history]
    rewards = torch.tensor(raw_rewards, dtype=torch.float32, device=device)

    if len(rewards) == 0:
        logger.warning("⚠️ Aucun reward trouvé — impossible d'entraîner.")
        return rewards

    # 🧮 Log initial des rewards bruts
    logger.info(
        f"🎯 Rewards (raw): min={rewards.min().item():.3f}, "
        f"max={rewards.max().item():.3f}, "
        f"mean={rewards.mean().item():.3f}, "
        f"std={rewards.std().item():.3f}"
    )

    # 1️⃣ Clipping des extrêmes pour éviter explosions
    rewards = torch.clamp(rewards, -50, 50)

    # 2️⃣ Mise à l’échelle douce [-1,1] via tanh
    rewards = torch.tanh(rewards / 20.0)

    # 📈 Log après mise à l’échelle
    logger.info(
        f"📈 Rewards (tanh-scaled): min={rewards.min().item():.3f}, "
        f"max={rewards.max().item():.3f}, "
        f"mean={rewards.mean().item():.3f}, "
        f"std={rewards.std().item():.3f}"
    )

    # 3️⃣ Normalisation globale (centrage + écart-type)
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    # 📊 Log final des rewards normalisés
    logger.info(
        f"📊 Rewards (normalized): min={rewards.min().item():.3f}, "
        f"max={rewards.max().item():.3f}, "
        f"mean={rewards.mean().item():.3f}, "
        f"std={rewards.std().item():.3f}"
    )

    return rewards



def train(batch_size=16, save_every=1000, resume: bool = False):

    if resume:
        loaded = load_latest_model(auto_pilot)
        if loaded:
            logger.info("🔄 Modèle existant chargé, reprise de l'entraînement.")
        else:
            logger.info("🆕 Aucun modèle trouvé — entraînement à partir de zéro.")

    inputs = torch.stack([normalize_inputs(h.input) for h in simulation_history]).to(device)
    tries = torch.tensor([h.output for h in simulation_history], dtype=torch.float32, device=device)
    rewards = process_rewards(simulation_history)  # <---- nouvelle fonction ici

    num_batches = math.ceil(len(inputs) / batch_size)
    logger.info(f"🧮 Début entraînement sur {len(inputs)} transitions ({num_batches} batchs)")
    logger.info("🖥️" * 30)
    auto_pilot.train()

    for batch_id in range(num_batches):
        start = batch_id * batch_size
        end = start + batch_size
        batch_inputs = inputs[start:end]
        batch_tries = tries[start:end]
        batch_rewards = rewards[start:end]

        # Log intermédiaire toutes les 10 itérations
        if batch_id % 3 == 0:
            logger.info(f"🔎 Batch {batch_id+1}: Reward mean={batch_rewards.mean().item():.4f}, std={batch_rewards.std().item():.4f}")

        # Forward pass
        action_logits = auto_pilot(batch_inputs)
        log_probabilities = get_action_log_probability(batch_tries, action_logits)
        loss = -(batch_rewards.view(-1, 1) * log_probabilities).mean()

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(auto_pilot.parameters(), max_norm=1.0)
        optimizer.step()

        # Log plus détaillé toutes les 10 itérations
        if batch_id % 3 == 0:
            with torch.no_grad():
                avg_reward = batch_rewards.mean().item()
                avg_abs_reward = batch_rewards.abs().mean().item()
                avg_logprob = log_probabilities.mean().item()
                avg_grad = sum(
                    (p.grad.abs().mean().item() if p.grad is not None else 0)
                    for p in auto_pilot.parameters()
                ) / (len(list(auto_pilot.parameters())) or 1)

            logger.info(
                f"📊 Batch {batch_id+1}/{num_batches} | "
                f"Loss={loss.item():.6f} | "
                f"Reward(mean/abs)={avg_reward:.3f}/{avg_abs_reward:.3f} | "
                f"LogProb={avg_logprob:.3f} | Grad={avg_grad:.6f}"
            )

        # Sauvegarde périodique
        if save_every > 0 and (batch_id + 1) % save_every == 0:
            save_model(auto_pilot)
    auto_pilot.training_count += 1
    logger.info(f"🚀 Entraînement n°{auto_pilot.training_count} fini")
    logger.info("🖥️" * 30)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_model(auto_pilot, filename=f"autopilot_final_{timestamp}.pt")
    logger.info(f"✅ Entraînement terminé — modèle final sauvegardé ({timestamp}).")


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
                world_input=item.get("world", {}),
                world_result=item.get("result", {}),
                reward=item.get("reward", 0.0)
            )
            history_points.append(hp)
        logger.info(f"📂 Fichier chargé : {path} ({len(history_points)} transitions)")
        return history_points
    except Exception as e:
        logger.error(f"❌ Erreur de lecture du fichier {path} : {e}")
        return []

def load_model_from_path(path: str, model: nn.Module = auto_pilot) -> bool:
    """
    Charge un modèle précis depuis un chemin complet.

    Args:
        path (str): chemin vers le fichier .pt
        model (nn.Module): instance à recharger

    Returns:
        bool: True si le chargement a réussi, False sinon
    """
    if not os.path.exists(path):
        logger.warning(f"⚠️ Modèle introuvable: {path}")
        return False

    try:
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        logger.info(f"📂 Modèle chargé : {os.path.basename(path)} sur {device}")
        return True
    except Exception as e:
        logger.error(f"❌ Échec du chargement du modèle {path}: {e}")
        return False

def load_latest_model(model: nn.Module = auto_pilot, directory: str = "models") -> bool:
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

    return load_model_from_path(latest_path, model)

# # Chargement du modèle par défaut au démarrage
load_model_from_path(STARTUP_MODEL_PATH)

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
        filename = f"autopilot_v2_{timestamp}.pt"

    path = os.path.join(directory, filename)
    torch.save(model.state_dict(), path)
    # logger.info(f"💾 Modèle sauvegardé dans {path}")



async def maybe_send_positional_reset(ws: WebSocket, payload: dict, simulation_history: list[HistoryPoint]) -> bool:
    """
    Vérifie si la position précédente ET la position résultante dépassent x_max.
    Si oui, envoie un RESET via le WebSocket et retourne True.
    Retourne False sinon.
    Inclut également un reset aléatoire (1/10) quand la voiture avance peu mais est déjà loin.
    """
    try:
        if len(simulation_history) == 0:
            return False
        prev_input = simulation_history[-1].world_input or {}
        input_position_x = float(prev_input.get("positions", {}).get("car", {}).get("x", 0.0))
        result_position_x = float(payload.get("positions", {}).get("car", {}).get("x", 0.0))
        result_position_z = float(payload.get("positions", {}).get("car", {}).get("z", 0.0))
        result_z_speed = float(payload.get("speeds", {}).get("z", 0.0))

        if abs(result_z_speed) < 0.5 and abs(result_position_z) > 1 and random.randint(1, 10) == 1:
            logger.info("🎲 Reset aléatoire déclenché (faible vitesse, position avancée)")
            await ws.send_json({"driving_inputs": ["RESET"]})
            return True

        road_width = float(payload.get("roadSize", {}).get("width", 0))
        car_width = float(payload.get("carSize", {}).get("width", 0))
        # calcul de x_max identique à compute_reward
        if not (road_width and car_width):
            return False
        x_max = (road_width / 2) - (car_width / 2)

        if abs(result_position_x) > x_max+2 and abs(input_position_x) > x_max:
            logger.info("🔁 Condition bord route détectée — envoi d'un RESET au simulateur")
            await ws.send_json({"driving_inputs": ["RESET"]})
            return True
    except Exception as e:
        logger.warning(f"⚠️ Erreur lors du test de reset automatique: {e}")
    return False
@app.websocket("/ai")
async def websocket_endpoint(ws: WebSocket):
    global predictions_count, predictions_since_training
    await ws.accept()
    logger.info("✅ WebSocket connection established")
    prediction_seconds_before_learning = 120
    fps = 8
    predictions_before_learning : int = int(prediction_seconds_before_learning * fps)
    initial_reset_sent = False
    
    try:
        while True:
            if not initial_reset_sent:
                await ws.send_json({"driving_inputs": ["RESET"]})
                logger.info("🔁 Envoi d’un reset à la simulation")
                initial_reset_sent = True
            payload = await ws.receive_json()
    
            predictions_count = predictions_count + 1
            predictions_since_training = predictions_since_training + 1
            # payload = { "sensors": {...}, "speeds": {...}, "accelerations": {...} }
            
            new_history_point, chosen = predict_actions(payload, predictions_count)
            predictions_since_training = periodically_train(predictions_before_learning, predictions_since_training)
                # simulation_history.clear()
            # logger.info(chosen)
            if(chosen.__contains__("BACKWARD")):
                logger.info("❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️")
            sent = await maybe_send_positional_reset(ws, payload, simulation_history)
            if sent:
                continue
            reward_value = simulation_history[-1].reward if simulation_history else 0.0
            await ws.send_json({"driving_inputs": chosen, "reward": reward_value})
            simulation_history.append(new_history_point)
    except Exception as e:
        save_simulation_history(simulation_history)
        logger.warning(f"⚠️ WebSocket closed: {e}")

def periodically_train(predictions_before_learning: int, predictions_since_training: int) -> int:
    if(predictions_since_training >= predictions_before_learning):
        train()
        save_simulation_history(simulation_history)
        return 0
    return predictions_since_training
IDX_LEFT, IDX_FORWARD, IDX_RIGHT, IDX_BACKWARD = 0, 1, 2, 3
def predict_actions(payload, i = 0):
    frame_before_interaction = 200
    # i contient l'indice (ou compteur) de la frame actuelle depuis la boucle websocket
    frame_idx = int(i)
    frame = payload.get("frameId", 0)
    if(len(simulation_history) > 0):
        # Ceci est le résultat de la prédiction précédente
        simulation_history[-1].world_result = payload
        reward = compute_reward(simulation_history[-1], road_size= payload.get("roadSize"), car_size= payload.get("carSize"), frame_before_interaction=frame_before_interaction, frame_idx=frame)
        # Log reward seulement toutes les 10 frames pour réduire le bruit        
        if frame_idx % 1 == 0:
            logger.info(f"👌Reward: {reward:.0f}👌")
        simulation_history[-1].reward = reward
    new_history_point = HistoryPoint()
    new_history_point.world_input = payload
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
        chosen_names, action_mask = choose_actions_tensor(logits, threshold=0.2, device=device, frame=frame, frame_before_interaction=frame_before_interaction)


    # Ne logger que toutes les 10 frames pour éviter un trop grand volume de logs
    if frame_idx % 2 == 0:
        # Affiche la position Z de la voiture (curr_z) — déplacé depuis compute_reward
        
        curr_z = float(payload.get("positions", {}).get("car", {}).get("z", 0.0))
        curr_x_speed = float(payload.get("speeds", {}).get("x", 0.0))
            
        # logger.info(f"curr_z: {curr_z:.3f} | curr_x_speed: {curr_x_speed:.3f}")
        # logger.info(f"Inputs: {[round(v, 2) for v in values]}")
        # log normalized inputs            
    new_history_point.input = values
    new_history_point.output = action_mask  
    if frame_idx % 3 == 0:
        logger.info(f"Actions: {chosen_names}")
    return new_history_point,chosen_names

# Indices: 0=LEFT, 1=FORWARD, 2=RIGHT, 3=BACKWARD
IDX_LEFT, IDX_FORWARD, IDX_RIGHT, IDX_BACKWARD = 0, 1, 2, 3

def choose_actions_tensor(logits: torch.Tensor,
                          frame,
                          threshold: float = 0.4,
                          device: torch.device = torch.device("cpu"),
                          frame_before_side = 800,
                          frame_before_interaction = 200) -> tuple[list[str], list[int]]:
    """
    Pipeline propre:
    - probs = sigmoid(logits)
    - masque threshold > 0.4
    - FORWARD domine BACKWARD
    - LEFT vs RIGHT: garder le plus probable
    - sampling Bernoulli sur probs masquées
    - fallback si rien choisi: argmax(prob)
    Retour: liste des noms d'actions activées
    """
    # logits shape attendu: [1, 4] ou [4]
    if logits.dim() == 2:
        probs = torch.sigmoid(logits)[0]  # [4]
    else:
        probs = torch.sigmoid(logits)     # [4]
    probs = probs.to(device)

    # 1) threshold
    mask = probs > threshold  # torch.bool, [4]

    # FORWARD prioritaire
    if mask[IDX_FORWARD] and mask[IDX_BACKWARD]:
        mask[IDX_BACKWARD] = False

    # 3) LEFT vs RIGHT: garder le plus probable si les deux passent
    if mask[IDX_LEFT] and mask[IDX_RIGHT]:
        if probs[IDX_LEFT] >= probs[IDX_RIGHT]:
            mask[IDX_RIGHT] = False
        else:
            mask[IDX_LEFT] = False

    if frame < frame_before_side:
         mask[IDX_RIGHT] = False
         mask[IDX_LEFT] = False
         mask[IDX_BACKWARD] = False
    if frame < frame_before_interaction:
        mask[IDX_FORWARD] = False

    # 4) Appliquer le masque aux probabilités
    masked_probs = probs.clone()
    masked_probs[~mask] = 0.0

    # 5) Tirage stochastique Bernoulli indépendant
    #    Attention: il faut un tensor random du même shape [4]
    rnd = torch.rand_like(masked_probs)
    active = (masked_probs >= rnd).int()  # [4], 0/1

    # # 6) Fallback: si toutes à 0, choisir l'argmax des probs d'origine
    # if active.sum().item() == 0:
    #     best_idx = int(torch.argmax(probs).item())
    #     # Respecter nos règles: si best=BACKWARD mais FORWARD passe le seuil, on préfère FORWARD
    #     if best_idx == IDX_BACKWARD and (probs[IDX_FORWARD] > threshold):
    #         best_idx = IDX_FORWARD
    #     active[best_idx] = 1

    # 7) Mapping indices -> noms en gardant l’ordre fixe 0..3
    driving_inputs = {0: "LEFT", 1: "FORWARD", 2: "RIGHT", 3: "BACKWARD"}
    chosen_names = [driving_inputs[i] for i in range(4) if active[i].item() == 1]
    action_mask = [int(active[i].item()) for i in range(4)]  # <-- toujours longueur 4

    return chosen_names, action_mask



