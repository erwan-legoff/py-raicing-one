import logging
import random
import math
from contextlib import contextmanager
from contextvars import ContextVar
from collections import defaultdict
from datetime import datetime, timezone
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import json
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

current_client_id: ContextVar[str] = ContextVar("current_client_id", default="global")


class ClientIDFilter(logging.Filter):
    """Inject the current client identifier into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.client_id = current_client_id.get()
        return True


class ClientIDFormatter(logging.Formatter):
    """Ensure every log record carries a `client_id`, even if filters were skipped."""

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "client_id"):
            record.client_id = current_client_id.get()
        return super().format(record)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s [client=%(client_id)s]: %(message)s",
)
logging.getLogger().addFilter(ClientIDFilter())
# Replace default formatters so missing client_id becomes impossible.
for _handler in logging.getLogger().handlers:
    _handler.setFormatter(
        ClientIDFormatter("%(asctime)s [%(levelname)s] %(name)s [client=%(client_id)s]: %(message)s")
    )
# Ensure third-party loggers that don't inherit root filters still get `client_id`.
for _third_party_logger in ("uvicorn", "uvicorn.error", "uvicorn.access", "watchfiles"):
    logging.getLogger(_third_party_logger).addFilter(ClientIDFilter())

LOG_THROTTLE_FACTOR = 180
_log_counters: dict[str, int] = defaultdict(int)


def log_every(key: str, interval: int = LOG_THROTTLE_FACTOR) -> bool:
    """Return True when the log associated with `key` should be emitted."""

    count = _log_counters[key]
    should_emit = (count % interval) == 0
    _log_counters[key] = count + 1
    return should_emit


@contextmanager
def client_logging_context(client_id: str):
    """Context manager ensuring logs include the originating client identifier."""

    token = current_client_id.set(client_id)
    try:
        yield
    finally:
        current_client_id.reset(token)
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
        self.last_training_at = datetime.now(timezone.utc)

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        state["_training_count"] = self.training_count
        state["_last_training_at"] = self.last_training_at.isoformat()
        return state

    def load_state_dict(self, state_dict, strict=True):
        self.training_count = state_dict.pop("_training_count", 0)
        last_training_raw = state_dict.pop("_last_training_at", None)
        if isinstance(last_training_raw, (int, float)):
            self.last_training_at = datetime.fromtimestamp(last_training_raw, tz=timezone.utc)
        elif isinstance(last_training_raw, str):
            try:
                parsed = datetime.fromisoformat(last_training_raw)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                self.last_training_at = parsed
            except ValueError:
                self.last_training_at = datetime.now(timezone.utc)
        else:
            self.last_training_at = datetime.now(timezone.utc)
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
    
input_size = 28  # 13 features courantes + 14 mémoire/agrégats
hidden_size = 256
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
    tags: list[str] = field(default_factory=list)

auto_pilot = AutoPilot(input_size,hidden_size,output_size)
auto_pilot.to(device)
STARTUP_MODEL_PATH = os.path.join("saved_models", "autopilot_v3_slow_20251102_164846.pt")
driving_inputs = {0: "LEFT", 1: "FORWARD", 2: "RIGHT", 3:"BACKWARD"}
simulation_histories: dict[str, list[HistoryPoint]] = {}
SAVE_INTERVAL = 120


def get_training_metadata() -> dict[str, object]:
    """Return the latest training statistics shared with the simulator UI."""

    last_training_at = getattr(auto_pilot, "last_training_at", datetime.now(timezone.utc))
    return {
        "count": getattr(auto_pilot, "training_count", 0),
        "last": last_training_at.isoformat(),
    }

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


def classify_rewards(history_points: list[HistoryPoint]) -> dict[str, int]:
    """Return counts of reward categories for logging purposes."""
    categories = {
        "neg_extreme": 0,
        "neg_high": 0,
        "neg_neutral": 0,
        "neutral": 0,
        "pos_neutral": 0,
        "pos_high": 0,
        "pos_extreme": 0,
    }
    for hp in history_points:
        reward = float(getattr(hp, "reward", 0.0) or 0.0)
        if reward <= -75:
            categories["neg_extreme"] += 1
        elif reward <= -15:
            categories["neg_high"] += 1
        elif reward < 0:
            categories["neg_neutral"] += 1
        elif reward <= 15:
            categories["neutral"] += 1
        elif reward < 75:
            categories["pos_neutral"] += 1
        elif reward < 150:
            categories["pos_high"] += 1
        else:
            categories["pos_extreme"] += 1
    return categories


def log_reward_stats(label: str, total: int, stats: dict[str, int]) -> None:
    """Emit a summary of reward distribution if data is available."""
    if total <= 0:
        logger.info(f"🧮 Stats rewards {label}: aucun point")
        return
    pct = {key: (value / total) * 100 for key, value in stats.items()}
    logger.info(
        "🧮 Stats rewards %s: neutral=%.1f%% | pos[+]=%.1f%% | pos[++]=%.1f%% | pos[+++]=%.1f%% | "
        "neg[-]=%.1f%% | neg[--]=%.1f%% | neg[---]=%.1f%%",
        label,
        pct["neutral"],
        pct["pos_neutral"],
        pct["pos_high"],
        pct["pos_extreme"],
        pct["neg_neutral"],
        pct["neg_high"],
        pct["neg_extreme"],
    )


def clean_history(simulation_history: list[HistoryPoint],
                  tag_probabilities: dict[str, float] | None = None,
                  high_reward_prob: float = 0.02,
                  min_keep_ratios: dict[str, float] | None = None,
                  rng: random.Random | None = None) -> int:
    """
    Supprime probabilistiquement des points de l'historique selon leurs tags.
    Retourne le nombre de points retirés.
    """
    if not simulation_history:
        return 0

    rand = rng.random if rng is not None else random.random
    protected_tags = {"DANGEROUS_SIDE", "NEW_OFFROAD", "STILL_OFFROAD"}
    default_probabilities: dict[str, float] = {
        "CENTERED": 1 / 4,
        "BORING_CENTERED": 0.75,
        "FINISH_LINE": 0.05,
        "SAVING_DANGEROUS_SIDE": 0.05,
        "PANIC_RECENTERING": 0.05,
        "PROGRESSING_CENTER": 0.33,
        "STALLING_CENTERED": 0.5,
        "LOW_FORWARD_SPEED": 0.20,
    }
    if tag_probabilities:
        default_probabilities.update(tag_probabilities)

    total = len(simulation_history)
    center_total = sum(1 for hp in simulation_history if "CENTERED" in (hp.tags or []))
    boring_total = sum(1 for hp in simulation_history if "BORING_CENTERED" in (hp.tags or []))
    center_ratio_before = (center_total / total) * 100 if total else 0.0
    boring_ratio_before = (boring_total / total) * 100 if total else 0.0
    reward_stats_before = classify_rewards(simulation_history)
    log_reward_stats("avant", total, reward_stats_before)
    min_keep_defaults = {
        "CENTERED": 0.15,
        "BORING_CENTERED": 0.05,
    }
    if min_keep_ratios:
        min_keep_defaults.update(min_keep_ratios)

    min_center_keep = math.ceil(total * min_keep_defaults.get("CENTERED", 0.0))
    min_boring_keep = math.ceil(total * min_keep_defaults.get("BORING_CENTERED", 0.0))
    center_remaining = center_total
    boring_remaining = boring_total

    kept: list[HistoryPoint] = []
    removed = 0

    for history_point in simulation_history:
        tags = set(history_point.tags or [])
        is_center = "CENTERED" in tags
        is_boring = "BORING_CENTERED" in tags
        action_mask = history_point.output or []
        left_active = bool(len(action_mask) > IDX_LEFT and action_mask[IDX_LEFT] > 0)
        right_active = bool(len(action_mask) > IDX_RIGHT and action_mask[IDX_RIGHT] > 0)
        forward_active = bool(len(action_mask) > IDX_FORWARD and action_mask[IDX_FORWARD] > 0)
        backward_active = bool(len(action_mask) > IDX_BACKWARD and action_mask[IDX_BACKWARD] > 0)
        lateral_active = left_active or right_active

        if lateral_active:
            kept.append(history_point)
            continue

        if tags & protected_tags:
            kept.append(history_point)
            continue

        if is_center and not is_boring and center_remaining <= min_center_keep:
            kept.append(history_point)
            continue

        if is_boring and boring_remaining <= min_boring_keep:
            kept.append(history_point)
            continue

        only_forward = forward_active and not lateral_active and not backward_active

        if only_forward and history_point.reward < 150 and rand() < 0.05:
            removed += 1
            if is_center:
                center_remaining -= 1
            if is_boring:
                boring_remaining -= 1
            continue

        if not (lateral_active or forward_active or backward_active) and rand() < 0.01:
            removed += 1
            if is_center:
                center_remaining -= 1
            if is_boring:
                boring_remaining -= 1
            continue

        if (
            history_point.reward > 100
            and "CENTERED" in tags
            and "SKIDDING_SIDEWAYS" in tags
        ):
            removed += 1
            if is_center:
                center_remaining -= 1
            if is_boring:
                boring_remaining -= 1
            continue

        if history_point.reward > 50 and rand() < high_reward_prob:
            removed += 1
            if is_center:
                center_remaining -= 1
            if is_boring:
                boring_remaining -= 1
            continue

        removal_probability = 0.0
        for tag in tags:
            prob = default_probabilities.get(tag)
            if prob is None:
                continue
            if tag == "LOW_FORWARD_SPEED" and "CENTERED" in tags:
                continue
            removal_probability = max(removal_probability, prob)

        if removal_probability > 0 and rand() < removal_probability:
            removed += 1
            if is_center:
                center_remaining -= 1
            if is_boring:
                boring_remaining -= 1
            continue

        kept.append(history_point)

    simulation_history[:] = kept
    new_total = len(simulation_history)
    new_center = sum(1 for hp in simulation_history if "CENTERED" in (hp.tags or []))
    new_boring = sum(1 for hp in simulation_history if "BORING_CENTERED" in (hp.tags or []))
    center_ratio_after = (new_center / new_total) * 100 if new_total else 0.0
    boring_ratio_after = (new_boring / new_total) * 100 if new_total else 0.0
    reward_stats_after = classify_rewards(simulation_history)
    if new_total:
        log_reward_stats("après", new_total, reward_stats_after)

    logger.info(
        f"🧹 cleanHistory: removed {removed}/{total} points "
        f"({100 * removed / total:.1f}%)"
    )
    return removed
import threading
def periodic_auto_save(simulation_history):
    """
    Sauvegarde périodiquement la simulation toutes les SAVE_INTERVAL secondes.
    Fonction récursive via threading.Timer.
    """
    save_simulation_history(simulation_history)
    threading.Timer(SAVE_INTERVAL, periodic_auto_save, args=[simulation_history]).start()

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

def log_reward(value: float, reason: str, tags: list[str] | None = None) -> float:
    emoji = reward_emoji(value)
    if log_every("reward"):
        tag_text = f" tags={tags}" if tags else ""
        logger.info(f"{emoji} Reward ({reason}): {value:.2f}{tag_text}")
    return value

def log_reward_update(current: float, delta: float, reason: str, tags: list[str] | None = None) -> float:
    new_value = current + delta
    emoji = reward_emoji(delta)
    total_emoji = reward_emoji(new_value)
    if log_every("reward_update"):
        tag_text = f" tags={tags}" if tags else ""
        logger.info(
            f"{emoji} Reward update ({reason}): {delta:+.2f} -> "
            f"{total_emoji} total {new_value:.2f}{tag_text}"
        )
    return new_value

def compute_reward(historyPoint: HistoryPoint, road_size:dict, car_size:dict, frame_before_interaction:int ,frame_idx:int) -> float:
        """
        Calcule la récompense (reward) entre deux états successifs de simulation.
        - Pénalise fort si la voiture tombe sous la route
        - Récompense le déplacement vers l'avant (axe Z)
        """

        world_result = historyPoint.world_result
        world_input = historyPoint.world_input
        tags: list[str] = []

        def add_tag(tag: str, detail: str | None = None) -> None:
            added = False
            if tag not in tags:
                tags.append(tag)
                added = True
            if added or detail:
                detail_msg = f": {detail}" if detail else ""
                if log_every(f"tag:{tag}", interval=LOG_THROTTLE_FACTOR):
                    logger.info(f"🏷️ Tag {tag}{detail_msg}")

        def finalize(value: float, reason: str) -> float:
            historyPoint.tags = tags
            if tags and log_every("tags_summary", interval=LOG_THROTTLE_FACTOR):
                logger.info(f"🏷️ Tags cumulés: {tags}")
            return value

        historyPoint.tags = []
        
        position_result = world_result.get("positions", {}).get("car", {})
        position_input = world_input.get("positions", {}).get("car", {})

        input_left_sensor = world_input.get("sensors",{}).get("left45Ray",  0)
        input_right_sensor = world_input.get("sensors",{}).get("right45Ray",  0)

        result_left_sensor = world_result.get("sensors",{}).get("left45Ray",  0)
        result_right_sensor = world_result.get("sensors", {}).get("right45Ray", 0)
        # On inverse l’axe Z pour que avancer → reward positif
        result_z_speed = -float(world_result.get("speeds", {}).get("z", 0.0))
        result_z_acceleration = -float(world_result.get("accelerations", {}).get("z", 0.0))

        curr_z_speed = -float(world_input.get("speeds", {}).get("z", 0.0))
        curr_z_acceleration = -float(world_input.get("accelerations", {}).get("z", 0.0))
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
            add_tag("INVALID_POSITIONS")
            return finalize(0.0, "positions invalides")
        if frame_idx <= frame_before_interaction:
            add_tag("EPISODE_WARMUP")
            return finalize(0.0, "début d'épisode")
        road_length = (road_size.get("depth",0) / 2) - 10 
        if(abs(result_position_z) > road_length):
            add_tag("FINISH_LINE")
            return finalize(100, "franchissement de la ligne d'arrivée")
        # Punition si la voiture est tombée sous la route
        if result_position_y < road_pos.get("y", 0):
            add_tag("FALLING_OFF_ROAD")
            return finalize(-10 * abs(result_x_speed), "chute sous la route")

        
        # Si la voiture revient au début, c'est que la prédiction est mauvaise, on punit
        if(input_position_z > 5 and result_position_z < 1):
            add_tag("RETURN_TO_START")
            return finalize(-100, "retour au départ")
        # Road width = 5 
        # Car Width = 1
        min_z_speed = 4
        fast_z_speed = 8
        min_z_acceleration = 0.08
        x_max = (road_width/2) - (car_width/2)
        x_min = x_max/3 if x_max else 0.0
        high_x_speed_threshold = 1.0
        boring_threshold = (x_min / 3) if x_min else 0.0
        dangerous_threshold = (x_max * 0.9) if x_max else None

        if x_min < abs(input_position_x) and log_every("danger_zone_alert", interval=LOG_THROTTLE_FACTOR):
            logger.info("🔴🔴🔴 Zone dangereuse")
        side_proximity_ratio = min(abs(result_position_x) / x_max, 1) if x_max else 1.0
        slight_offset_ratio = min(abs(result_position_x) / x_min, 1.0) if x_min else 0.0

        if dangerous_threshold is not None and abs(result_position_x) >= dangerous_threshold:
            add_tag("DANGEROUS_SIDE")

        if boring_threshold and abs(result_position_x) <= boring_threshold:
            add_tag("CENTERED")
            if abs(result_x_speed) < 0.15:
                add_tag("BORING_CENTERED")

        if x_min and abs(result_position_x) <= x_min and abs(result_x_speed) < 0.1 and abs(result_z_speed) < 0.05:
            add_tag("STALLING_CENTERED")

        if x_min and abs(result_position_x) < x_min and abs(result_x_speed) > high_x_speed_threshold:
            add_tag("SKIDDING_SIDEWAYS")

        if x_min and abs(result_position_x) >= x_min and (result_x_speed * result_position_x) > 0 and abs(result_x_speed) > high_x_speed_threshold:
            add_tag("RUSHING_TO_EDGE")

        if x_min and abs(input_position_x) > x_min and (result_x_speed * result_position_x) < 0 and abs(result_x_speed) > high_x_speed_threshold:
            add_tag("PANIC_RECENTERING")

        if (
            boring_threshold
            and abs(result_position_x) > boring_threshold
            and (result_x_speed * result_position_x) > 0
            and abs(result_x_speed) > 0.3
        ):
            add_tag("SIDE_SPEED_PENALTY")
            offset_factor = max(slight_offset_ratio, 0.2)
            acc_reward = -15 * abs(result_x_speed) * offset_factor
            reward += acc_reward

        # Si la voiture est très centrée mais quasi immobile sur X, on a déjà capturé BORING_CENTERED.
        # Si on était pas dehors mais que l'action fait sortir
        # alors on punit et on ajoute une punition proportionnel à l'engouement vers x
        if(abs(result_position_x) > x_max and abs(input_position_x) < x_max):
            add_tag("NEW_OFFROAD")
            return finalize(-100 - 10 * abs(result_position_x) - abs(input_position_x), "sortie de route")
        # Si on était déjà dehors alors on punit et ajoute une proportionalité à l'engouement vers x
        if(abs(result_position_x) > x_max and abs(input_position_x) > x_max):
            add_tag("STILL_OFFROAD")
            return finalize(-10 - 10 * abs(result_position_x) - abs(input_position_x), "persistance hors route")
        # Si on était pas dans X et qu'on rentre dedans, alors on ajoute une punition avec une proportionnalité de l'engouement vers x
        if(abs(result_position_x)>x_min and abs(input_position_x) < x_min):
            acc_reward = -(10 + 5 * (abs(result_position_x) - abs(input_position_x)))
            add_tag("ENTERING_DANGER_ZONE")
            reward += acc_reward
        # Si on se dirige vers x, et que de base on était dans la zone dangereuse
        # alors on punit de plus en plus qu'on s'approche du bord
        if(abs(result_position_x) > abs(input_position_x) and abs(input_position_x) > x_min):
            add_tag("APPROACHING_EDGE")
            acc_reward = -55*result_z_speed * side_proximity_ratio**3
            reward += acc_reward
        # Si on se pars de x, et que de base on était dans la zone dangereuse
        # alors on récompense proportionnellement à la proximité
        if(abs(result_position_x) < abs(input_position_x) and abs(input_position_x) > x_min):
            add_tag("SAVING_DANGEROUS_SIDE")
            acc_reward = 20*result_z_speed * side_proximity_ratio**2
            reward += acc_reward

        if abs(result_position_x) > x_min and historyPoint.output:
            steer_left = historyPoint.output[IDX_LEFT]
            steer_right = historyPoint.output[IDX_RIGHT]
            direction = -1 if result_position_x < 0 else 1
            center_force = (steer_left - steer_right) * direction
            proximity = side_proximity_ratio if x_max else 0.0
            if center_force < 0:
                acc_reward = -20 * abs(center_force) * proximity
                reward += acc_reward
            elif center_force > 0:
                acc_reward = 15 * center_force * (proximity ** 0.5)
                reward += acc_reward

        # Si on sort de la zone dangereuse on a un petit bonus
        if(abs(input_position_x) > x_min and abs(result_position_x) < x_min):
            add_tag("EXITING_DANGER_ZONE")
            reward += 10
            
        
        # Punition forte en marche arrière, sinon légère si ça avance trop peu
        if result_z_speed < 0:
            add_tag("REVERSING")
            acc_reward = -30 * abs(result_z_speed) - 10
            reward += acc_reward
        elif result_z_speed < 0.2:
            add_tag("LOW_FORWARD_SPEED")
            reward -= 10

        # On récompense par rapport à la vitesse en avant et donc on punit autant si il recule
        centering_factor = 0.0
        recentering_bonus = 0.0
        progress_tag = None
        if x_min:
            centering_factor = max(0.0, 1 - abs(result_position_x) / x_min)
            moving_closer = abs(result_position_x) < abs(input_position_x)
            moving_away = abs(result_position_x) > abs(input_position_x)
            centric_reward = 30 * result_z_speed * centering_factor
            if centric_reward > 0 and moving_closer:
                progress_tag = "PROGRESSING_CENTER"
            if result_z_speed > 0 and moving_closer:
                recentering_bonus = 10 * result_z_speed * max(0.0, 1 - abs(result_position_x) / (x_min * 2))
            if centric_reward > 0 and not moving_closer:
                centric_reward *= 0.2
            x_speed_temper = max(0.0, 1.0 - min(abs(result_x_speed) / (high_x_speed_threshold * 2), 1.0))
            tempered_reward = centric_reward * x_speed_temper
            if tempered_reward != centric_reward:
                add_tag("CENTER_SPEED_TEMPERED")
            if x_min and abs(result_position_x) <= x_min and x_speed_temper < 1.0 and moving_away:
                penalty_factor = 1.0 - x_speed_temper
                lateral_penalty = -25 * abs(result_x_speed) * penalty_factor
                add_tag("CENTER_DRIFT_PENALTY")
                reward += lateral_penalty
            reward += tempered_reward
            if recentering_bonus:
                reward += recentering_bonus
            if progress_tag:
                add_tag(progress_tag)
        else:
            centering_factor = 0.0
        acc_ponderation = 5
        if(curr_z_speed<1):
            acc_ponderation = 100
        elif(curr_z_speed<2):
            acc_ponderation = 50
        elif(curr_z_speed<3):
            acc_ponderation = 20
        elif (curr_z_speed < 4):
            acc_ponderation = 10
        if(result_z_acceleration < min_z_acceleration):
            if(curr_z_speed < min_z_speed):
                acc_ponderation *= 100
            elif(curr_z_speed < fast_z_speed):
                acc_ponderation *= 10
        

        acc_reward = acc_ponderation * (result_z_acceleration - min_z_acceleration)
        reward += torch.clamp(torch.tensor(acc_reward), min=-20.0, max=20.0).item()
        if(curr_z_acceleration < min_z_acceleration and result_z_acceleration < min_z_acceleration):
            reward -=10
            if(result_z_speed < min_z_speed):
                reward += 10*(result_z_speed-min_z_speed)
        if historyPoint.output:
            steer_left = historyPoint.output[IDX_LEFT] if len(historyPoint.output) > IDX_LEFT else 0
            steer_right = historyPoint.output[IDX_RIGHT] if len(historyPoint.output) > IDX_RIGHT else 0
            if steer_left <= 0 and steer_right <= 0 and reward < -100:
                acc_reward = -100 - reward
                reward += acc_reward

        # if(abs(input_position_x) > abs(result_position_x)):
        #     delta = 5*result_z_speed
        #     reward = log_reward_update(reward, delta, "recentrage latéral")

        if "BORING_CENTERED" in tags and reward <= -20:
            tags.remove("BORING_CENTERED")


        return finalize(reward, "cumul")
import math
import torch.optim as optim
optimizer = optim.AdamW(auto_pilot.parameters(), lr=4e-5, weight_decay=1e-2)

from torch.distributions import Bernoulli
def get_action_log_probability(actions, logits):
    dist = Bernoulli(logits=logits)   
    return dist.log_prob(actions).sum(dim=1, keepdim=True) 


SENSOR_MIN = 0.0
SENSOR_MAX = 50.0
SPEED_MAX = 10.0        
ACCEL_MAX = 2.0

def normalize_inputs(values: list[float]) -> torch.Tensor:
    """Normalize raw features into [0,1]. Supports legacy (13) and extended (28) inputs."""
    if len(values) < 13:
        raise ValueError(f"Expected at least 13 features, got {len(values)}")

    sensors = torch.tensor(values[:7], dtype=torch.float32)
    speeds = torch.tensor(values[7:10], dtype=torch.float32)
    accels = torch.tensor(values[10:13], dtype=torch.float32)

    sensors = (sensors - SENSOR_MIN) / (SENSOR_MAX - SENSOR_MIN + 1e-8)
    speeds = (speeds + SPEED_MAX) / (2 * SPEED_MAX + 1e-8)
    accels = (accels + ACCEL_MAX) / (2 * ACCEL_MAX + 1e-8)

    parts: list[torch.Tensor] = [sensors, speeds, accels]

    if len(values) == 28:
        idx = 13
        mem_prev = torch.tensor(values[idx:idx+2], dtype=torch.float32); idx += 2
        mem_short = torch.tensor(values[idx:idx+2], dtype=torch.float32); idx += 2
        mem_long = torch.tensor(values[idx:idx+2], dtype=torch.float32); idx += 2
        action_feats = torch.tensor(values[idx:idx+6], dtype=torch.float32); idx += 6
        ratio_feats = torch.tensor(values[idx:idx+3], dtype=torch.float32)

        mem_prev = (mem_prev - SENSOR_MIN) / (SENSOR_MAX - SENSOR_MIN + 1e-8)
        mem_short = (mem_short - SENSOR_MIN) / (SENSOR_MAX - SENSOR_MIN + 1e-8)
        mem_long = (mem_long - SENSOR_MIN) / (SENSOR_MAX - SENSOR_MIN + 1e-8)

        action_norm = ((action_feats + 1.0) * 0.5).clamp(0.0, 1.0)

        ratio_clamped = torch.clamp(ratio_feats, -5.0, 5.0)
        ratio_norm = (ratio_clamped + 5.0) / 10.0

        parts.extend([mem_prev, mem_short, mem_long, action_norm, ratio_norm])
    elif len(values) != 13:
        raise ValueError(f"Unsupported input length {len(values)}")

    normalized = torch.cat(parts)
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
    rewards = torch.clamp(rewards, -200, 200)

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



def train(simulation_history: list[HistoryPoint],
          batch_size: int = 16,
          save_every: int = 1000,
          resume: bool = False):

    if resume:
        loaded = load_latest_model(auto_pilot)
        if loaded:
            logger.info("🔄 Modèle existant chargé, reprise de l'entraînement.")
        else:
            logger.info("🆕 Aucun modèle trouvé — entraînement à partir de zéro.")

    if not simulation_history:
        logger.warning("⚠️ Impossible d'entraîner — historique vide.")
        return
    clean_history(simulation_history)
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
    auto_pilot.last_training_at = datetime.now(timezone.utc)
    logger.info(f"🚀 Entraînement n°{auto_pilot.training_count} fini")
    logger.info("🖥️" * 30)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_model(auto_pilot, filename=f"autopilot_final_v4_{timestamp}.pt")
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
                world_input=item.get("world_input", item.get("world", {})),
                world_result=item.get("world_result", item.get("result", {})),
                reward=item.get("reward", 0.0),
                tags=item.get("tags", []),
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

# # # Chargement du modèle par défaut au démarrage
# load_model_from_path(STARTUP_MODEL_PATH)

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

        train(simulation_data)
        total_transitions += len(simulation_data)

    logger.info(f"✅ Entraînement terminé sur {len(files)} fichiers ({total_transitions} transitions)")
    # sauvegarde du modèle
    torch.save(auto_pilot.state_dict(), "autopilot_trained.pt")
    logger.info("💾 Modèle sauvegardé : autopilot_trained.pt")

def full_train() -> None:
    """
    Entraîne successivement le modèle sur chaque historique actif en mémoire.
    """
    total_histories = 0
    total_transitions = 0
    for client_id, history in simulation_histories.items():
        if not history:
            logger.info(f"⚠️ Historique vide pour {client_id}, passage.")
            continue
        total_histories += 1
        total_transitions += len(history)
        logger.info(f"🚗 Entraînement sur {client_id} ({len(history)} transitions)")
        train(history)
    logger.info(f"🏁 full_train terminé — {total_histories} sessions, {total_transitions} transitions")

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
        filename = f"autopilot_v4_{timestamp}.pt"

    path = os.path.join(directory, filename)
    torch.save(model.state_dict(), path)
    # logger.info(f"💾 Modèle sauvegardé dans {path}")


async def send_control_message(
    ws: WebSocket,
    client_id: str,
    driving_inputs: list[str],
    reward: float | None = None,
    tags: list[str] | None = None,
) -> None:
    """Send driving instructions enriched with training metadata to the client."""

    response: dict[str, object] = {
        "driving_inputs": driving_inputs,
        "training": get_training_metadata(),
    }
    if reward is not None:
        response["reward"] = reward
    if tags is not None:
        response["tags"] = tags
    await ws.send_json(response)


async def maybe_send_positional_reset(
    ws: WebSocket,
    payload: dict,
    simulation_history: list[HistoryPoint],
    client_id: str,
) -> bool:
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
            await send_control_message(ws, client_id, ["RESET"])
            return True

        road_width = float(payload.get("roadSize", {}).get("width", 0))
        car_width = float(payload.get("carSize", {}).get("width", 0))
        # calcul de x_max identique à compute_reward
        if not (road_width and car_width):
            return False
        x_max = (road_width / 2) - (car_width / 2)

        if abs(result_position_x) > x_max+2 and abs(input_position_x) > x_max:
            logger.info("🔁 Condition bord route détectée — envoi d'un RESET au simulateur")
            await send_control_message(ws, client_id, ["RESET"])
            return True
    except Exception as e:
        logger.warning(f"⚠️ Erreur lors du test de reset automatique: {e}")
    return False
@app.websocket("/ai")
async def websocket_endpoint(ws: WebSocket):
    global predictions_count, predictions_since_training
    await ws.accept()
    prediction_seconds_before_learning = 120*8
    fps = 8
    predictions_before_learning : int = int(prediction_seconds_before_learning * fps)
    initial_reset_sent = False
    # identify client from websocket request (query param "id")
    client_id = ws.query_params.get("id")
    if not client_id:
        # fallback to an anonymous id if none provided
        client_id = f"anon_{random.randint(1, 10**9)}"

    # create session entry if missing, otherwise reuse existing list
    if client_id not in simulation_histories:
        simulation_histories[client_id] = []

    # alias l'historique de cette session pour simplifier la suite du code
    simulation_history = simulation_histories[client_id]
    with client_logging_context(client_id):
        logger.info("✅ WebSocket connection established")
        logger.info(f"🔎 Session sélectionnée: {client_id} ({len(simulation_history)} transitions)")

        try:
            while True:
                if not initial_reset_sent:
                    await send_control_message(ws, client_id, ["RESET"])
                    logger.info("🔁 Envoi d’un reset à la simulation")
                    initial_reset_sent = True
                payload = await ws.receive_json()

                predictions_count = predictions_count + 1
                predictions_since_training = predictions_since_training + 1
                # payload = { "sensors": {...}, "speeds": {...}, "accelerations": {...} }

                new_history_point, chosen = predict_actions(payload, simulation_history, predictions_count)
                predictions_since_training = periodically_train(
                    simulation_history,
                    predictions_before_learning,
                    predictions_since_training,
                )
                if chosen.__contains__("BACKWARD"):
                    logger.info("❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️")
                sent = await maybe_send_positional_reset(ws, payload, simulation_history, client_id)
                if sent:
                    continue
                reward_value = simulation_history[-1].reward if simulation_history else 0.0
                tags_value = simulation_history[-1].tags if simulation_history else []
                await send_control_message(ws, client_id, chosen, reward_value, tags_value)
                simulation_history.append(new_history_point)
        except Exception as e:
            save_simulation_history(simulation_history)
            logger.warning(f"⚠️ WebSocket closed: {e}")

def periodically_train(simulation_history: list[HistoryPoint],
                       predictions_before_learning: int,
                       predictions_since_training: int) -> int:
    if(predictions_since_training >= predictions_before_learning):
        train(simulation_history)
        save_simulation_history(simulation_history)
        return 0
    return predictions_since_training
IDX_LEFT, IDX_FORWARD, IDX_RIGHT, IDX_BACKWARD = 0, 1, 2, 3
def predict_actions(payload, simulation_history: list[HistoryPoint], i: int = 0):
    frame_before_interaction = 200
    # i contient l'indice (ou compteur) de la frame actuelle depuis la boucle websocket
    frame_idx = int(i)
    frame = payload.get("frameId", 0)
    if(len(simulation_history) > 0):
        # Ceci est le résultat de la prédiction précédente
        simulation_history[-1].world_result = payload
        reward = compute_reward(simulation_history[-1], road_size= payload.get("roadSize"), car_size= payload.get("carSize"), frame_before_interaction=frame_before_interaction, frame_idx=frame)
        # Log reward seulement toutes les 30 frames pour réduire le bruit        
        if frame_idx % (1 * LOG_THROTTLE_FACTOR) == 0:
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

            # 3️⃣ Capteurs mémoire (frame précédente)
    def fused_actions(output: list[float] | None) -> tuple[float, float]:
        if not output:
            return 0.0, 0.0
        left = output[IDX_LEFT] if len(output) > IDX_LEFT else 0.0
        forward = output[IDX_FORWARD] if len(output) > IDX_FORWARD else 0.0
        right = output[IDX_RIGHT] if len(output) > IDX_RIGHT else 0.0
        backward = output[IDX_BACKWARD] if len(output) > IDX_BACKWARD else 0.0
        lateral = right - left
        longitudinal = forward - backward
        return lateral, longitudinal

    def speed_ratio(world: dict | None) -> float:
        if not world:
            return 0.0
        speeds_w = world.get("speeds", {})
        z_speed = -float(speeds_w.get("z", 0.0))
        x_speed = float(speeds_w.get("x", 0.0))
        denom = max(abs(x_speed), 1e-3)
        return z_speed / denom

    def sensors_lr(world: dict | None) -> tuple[float, float]:
        sensors_w = (world or {}).get("sensors", {})
        return float(sensors_w.get("left45Ray", 0.0)), float(sensors_w.get("right45Ray", 0.0))

    def aggregate_window(points: list[HistoryPoint]) -> tuple[tuple[float, float], tuple[float, float], float]:
        if not points:
            return (0.0, 0.0), (0.0, 0.0), 0.0
        sum_left = sum_right = 0.0
        sum_lat = sum_long = 0.0
        sum_ratio = 0.0
        count = 0
        for hp in points:
            left, right = sensors_lr(hp.world_input)
            lat, longi = fused_actions(hp.output)
            ratio = speed_ratio(hp.world_input)
            sum_left += left
            sum_right += right
            sum_lat += lat
            sum_long += longi
            sum_ratio += ratio
            count += 1
        if count == 0:
            return (0.0, 0.0), (0.0, 0.0), 0.0
        return (
            (sum_left / count, sum_right / count),
            (sum_lat / count, sum_long / count),
            sum_ratio / count,
        )

    prev_left = prev_right = 0.0
    avg_left = avg_right = 0.0
    long_left = long_right = 0.0
    prev_act_lat = prev_act_long = 0.0
    avg_act_lat = avg_act_long = 0.0
    long_act_lat = long_act_long = 0.0
    prev_ratio = avg_ratio = long_ratio = 0.0
    if simulation_history:
        prev_hp = simulation_history[-1]
        prev_world_input = prev_hp.world_input or {}
        prev_left, prev_right = sensors_lr(prev_world_input)
        prev_act_lat, prev_act_long = fused_actions(prev_hp.output)
        prev_ratio = speed_ratio(prev_world_input)

        short_window = simulation_history[-10:-1]
        (avg_left, avg_right), (avg_act_lat, avg_act_long), avg_ratio = aggregate_window(short_window)

        long_window = simulation_history[-27:-10]
        (long_left, long_right), (long_act_lat, long_act_long), long_ratio = aggregate_window(long_window)

    memory_values = [
        prev_left, prev_right,
        avg_left, avg_right,
        long_left, long_right,
        prev_act_lat, prev_act_long,
        avg_act_lat, avg_act_long,
        long_act_lat, long_act_long,
        prev_ratio, avg_ratio, long_ratio,
    ]

            # 4️⃣ Fusion complète : 13 features courantes + 14 mémoire = 28 features
    values = sensor_values + speed_values + accel_values + memory_values
    normalized_input = normalize_inputs(values).unsqueeze(0).to(device)
    data_input = normalized_input

    with torch.no_grad():
        logits = auto_pilot(data_input)
        chosen_names, action_mask = choose_actions_tensor(logits, threshold=0.2, device=device, frame=frame, frame_before_interaction=frame_before_interaction)


    # Ne logger que toutes les 60 frames pour éviter un trop grand volume de logs
    if frame_idx % (2 * LOG_THROTTLE_FACTOR) == 0:
        # Affiche la position Z de la voiture (curr_z) — déplacé depuis compute_reward
        
        curr_z = float(payload.get("positions", {}).get("car", {}).get("z", 0.0))
        curr_x_speed = float(payload.get("speeds", {}).get("x", 0.0))
            
        # logger.info(f"curr_z: {curr_z:.3f} | curr_x_speed: {curr_x_speed:.3f}")
        # logger.info(f"Inputs: {[round(v, 2) for v in values]}")
        # log normalized inputs            
    new_history_point.input = values
    new_history_point.output = action_mask  
    if frame_idx % (3 * LOG_THROTTLE_FACTOR) == 0:
        logger.info(f"Actions: {chosen_names}")
    return new_history_point,chosen_names

# Indices: 0=LEFT, 1=FORWARD, 2=RIGHT, 3=BACKWARD
IDX_LEFT, IDX_FORWARD, IDX_RIGHT, IDX_BACKWARD = 0, 1, 2, 3

def choose_actions_tensor(logits: torch.Tensor,
                          frame,
                          threshold: float = 0.2,
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



