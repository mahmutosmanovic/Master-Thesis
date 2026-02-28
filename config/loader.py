import yaml
from pathlib import Path
from environment import MovementDim, POI_CFG, EE_CFG, CRW_CFG, LPOI_CFG

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"


# Behavior factory
def _build_behavior(name: str):

    behaviors = {
        "POI": POI_CFG,
        "EE": EE_CFG,
        "CRW": CRW_CFG,
        "LPOI": LPOI_CFG,
    }

    try:
        return behaviors[name]()  # instantiate
    except KeyError:
        raise ValueError(f"Unknown behavior: {name}")


# Main loader
def load_config(name: str):
    """
    name = 'train' -> loads config/train.yaml
    """

    config_path = CONFIG_DIR / f"{name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # --- enums ---
    cfg["animal"]["init"]["movement_dim"] = (
        MovementDim[cfg["animal"]["init"]["movement_dim"]]
    )

    # --- behaviors ---
    behavior_name = cfg["animal"]["init"]["behavior"]
    cfg["animal"]["init"]["behavior"] = _build_behavior(behavior_name)

    # useful for logging/debugging
    cfg["_config_name"] = name

    return cfg