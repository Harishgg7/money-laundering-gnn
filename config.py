"""Central configuration for the Explainable AML GNN pipeline."""
from pathlib import Path

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = ROOT / "checkpoints"
OUTPUT_DIR = ROOT / "outputs"

RAW_TRANS_CSV = DATA_DIR / "HI-Small_Trans.csv"
RAW_ACCOUNTS_CSV = DATA_DIR / "HI-Small_accounts.csv"
GRAPH_PT = PROCESSED_DIR / "aml_graph.pt"          # saved PyG Data object
NODE_MAP_PARQUET = PROCESSED_DIR / "node_map.parquet"  # node_id -> (bank, account)

for _d in (PROCESSED_DIR, CHECKPOINT_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Reproducibility ---------------------------------------------------------
SEED = 42

# --- Data splits (node-level, stratified) ------------------------------------
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# --- Model hyperparameters ---------------------------------------------------
HIDDEN_CHANNELS = 128
NUM_LAYERS = 3            # 3-hop message passing (captures longer laundering chains)
DROPOUT = 0.2
MODEL_TYPE = "sage"      # one of: sage | gat | gcn  (see model.py)

# --- Training ----------------------------------------------------------------
EPOCHS = 60
LEARNING_RATE = 5e-3
WEIGHT_DECAY = 5e-4
FOCAL_ALPHA = 0.75        # up-weight the rare laundering class
FOCAL_GAMMA = 2.0
