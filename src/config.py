"""Experiment configuration for the MIMO-OTFS channel estimator."""
from pathlib import Path
import torch

# -------------------------
# Paths
# -------------------------
DATA_DIR = Path("data/training")
PHI_PATH = Path("data/Phi_pilotWeighted_padded.mat")
SNR_DATA_DIR = Path("data/variable_snr")
CHECKPOINT_DIR = Path("checkpoints")
RESULTS_DIR = Path("results")


M = 256
N = 8
NT = 16
NR = 1


NUM_SAMPLES = 40000
TRAIN_RATIO = 0.90
SPLIT_SEED = 42


EPOCHS = 20
BATCH_SIZE = 16
NUM_WORKERS = 0
PIN_MEMORY = False
LR = 1e-3


THRESHOLD_RATIO = 0.05
GAMMA = 0.05
L1_LAMBDA = 1e-2
SOFT_THRESH_RATIO = 0.01


EVAL_SNRS = [5, 10, 15, 20]
HARD_THRESHOLD_ALPHA = 0.01
USE_TRUE_FOR_TAU = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
