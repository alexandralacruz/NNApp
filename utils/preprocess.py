"""
Funciones de preprocesamiento para datos tabulares, de imagen y de audio.
Se usan tanto en entrenamiento como en predicción.
"""
import io
import numpy as np
import pandas as pd
from typing import Tuple

# Opcionales: solo si planeas usar imágenes/audio
from PIL import Image
import librosa


# -----------------------------
# Tabular
# -----------------------------
def preprocess_tabular(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray | None]:
    """
    Limpia y normaliza un DataFrame.
    Si el CSV de entrenamiento incluye una columna 'target', la separa.
    Retorna (X, y) o (X, None) si no hay target.
    """
    df = df.copy()

    y = None
    if "target" in df.columns:
        y = df.pop("target").values

    # Normalización simple 0-1
    X = (df - df.min()) / (df.max() - df.min() + 1e-8)
    return X.values.astype(np.float32), y


# -----------------------------
# Imágenes
# -----------------------------
def preprocess_image(file_bytes: bytes, size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    """
    Abre una imagen, la convierte a RGB, la redimensiona y normaliza a [0,1].
    Retorna un array (H, W, 3).
    """
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(size)
    arr = np.array(img).astype("float32") / 255.0
    return arr


# -----------------------------
# Audio
# -----------------------------
def preprocess_audio(file_bytes: bytes, sr: int = 16000) -> np.ndarray:
    """
    Carga un archivo de audio en memoria, lo resamplea y extrae un mel-spectrogram.
    Retorna un array 2D (frecuencias x frames).
    """
    # librosa.load acepta un file-like object
    audio, _ = librosa.load(io.BytesIO(file_bytes), sr=sr, mono=True)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Normalizamos a [0,1]
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return mel_db.astype("float32")
