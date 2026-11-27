"""Configuración central: todo lo que se pueda cambiar en vivo está aquí."""
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# Rutas
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
CORPUS_DIR = ROOT / "data" / "corpus"
MODELS_DIR = ROOT / "models"
QDRANT_PATH = ROOT / "qdrant_db"

# Claves (se leen de .env)
ESIOS_TOKEN = os.getenv("ESIOS_TOKEN")
AEMET_KEY = os.getenv("AEMET_KEY")

# Reproducibilidad
SEED = 42

# Datos (se excluye 2020 por la pandemia)
FECHA_INICIO = "2021-01-01"
FECHA_FIN = "2024-12-31"
CIUDADES = {  # lat, lon (Open-Meteo)
    "Madrid": (40.42, -3.70),
    "Barcelona": (41.39, 2.17),
    "Valencia": (39.47, -0.38),
    "Sevilla": (37.39, -5.99),
}

# ML
PERCENTIL_PICO = 0.95  # define "hora pico crítica"

# RAG
EMBEDDING_MODEL = "nomic-embed-text"
COLECCION = "normativa_electrica"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3

# LLM (Ollama)
LLM_MODEL = "gemma3:4b"
TEMPERATURE = 0
LLM_SEED = 5
