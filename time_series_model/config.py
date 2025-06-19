from pathlib import Path
from dataclasses import dataclass

@dataclass
class Paths:
    BASE:   Path = Path("your/base/path/here")  # Replace with your actual base path
    CLEAN:  Path = BASE / "Cleaned Data_with_aug"
    MODELS: Path = BASE / "models"
    TOPK:   Path = BASE / "topks"
    LOG:    Path = BASE / "logs"
    TMP:    Path = BASE / ".tmp"

@dataclass
class Hyper:
    SEED             = 42
    LAG_PERIOD       = 12
    BATCH_SIZE       = 64
    NUM_EPOCHS       = 500
    LR               = 1e-5
    HIDDEN_SIZE      = 256
    NUM_LAYERS       = 2
    K_TOP_STOCKS     = 50