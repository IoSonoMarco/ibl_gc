import os
from pathlib import Path

root = Path(__file__).resolve().parents[1]

ROOT_BEHAV_DATA = os.path.join(
    root, "data/behavioral", "behavioral_dataset.csv"
)

ROOT_STAN_MODELS = os.path.join(
    root, "data/behavioral", "stan_models"
)

ROOT_STAN_MODEL_FITS = os.path.join(
    root, "output", "stan_model_fits"
)