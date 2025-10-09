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

# NN Module
class NeuralNetwork(nn.Module):
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
    
input_size = 7
hidden_size = 128
output_size = 4

model = NeuralNetwork(input_size,hidden_size,output_size)
x = torch.zeros((1, input_size), device=device)
y = model(x)
print(y.shape)


        