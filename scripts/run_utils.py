import yaml
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from environment import MovementDim
from config.loader import _build_behavior

def create_run_dir(config: dict, seed: int):

    project_root = Path(__file__).resolve().parents[1]
    runs_dir = project_root / "runs"
    runs_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    config_name = config.get("_config_name", "exp")

    run_name = f"{config_name}_seed{seed}_{timestamp}"

    run_dir = runs_dir / run_name
    run_dir.mkdir()

    return run_dir

def save_config_snapshot(config: dict, run_dir: Path):
    """
    Saves exact config used for reproducibility.
    """

    cfg_copy = deepcopy(config)
    
    # remove non-serializable python objects
    if "animal" in cfg_copy:
        cfg_copy["animal"]["init"]["behavior"] = \
            str(cfg_copy["animal"]["init"]["behavior"].__class__.__name__)

        cfg_copy["animal"]["init"]["movement_dim"] = \
            cfg_copy["animal"]["init"]["movement_dim"].name

    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg_copy, f)

def resolve_run_dir(run_name: str) -> Path:
    """
    Resolves:
        --run latest
        --run <folder_name>
    """

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    RUNS_DIR = PROJECT_ROOT / "runs"

    if run_name == "latest":

        runs = [p for p in RUNS_DIR.iterdir() if p.is_dir()]

        if not runs:
            raise RuntimeError("No runs found in runs/ directory.")

        # newest by modification time (SAFER than sorting names)
        run_dir = max(runs, key=lambda p: p.stat().st_mtime)

        print(f"[INFO] Using latest run: {run_dir.name}")
        return run_dir

    run_dir = RUNS_DIR / run_name

    if not run_dir.exists():
        raise FileNotFoundError(f"Run not found: {run_dir}")

    return run_dir

def load_run(run_name: str):

    run_dir = resolve_run_dir(run_name)
    config_path = run_dir / "config.yaml"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # enum
    cfg["animal"]["init"]["movement_dim"] = (
        MovementDim[cfg["animal"]["init"]["movement_dim"]]
    )

    # behavior dataclass
    behavior_name = cfg["animal"]["init"]["behavior"]

    # snapshot stores "POI_CFG"
    if behavior_name.endswith("_CFG"):
        behavior_name = behavior_name[:-4]

    cfg["animal"]["init"]["behavior"] = _build_behavior(behavior_name)

    cfg["run_dir"] = str(run_dir)

    return cfg, run_dir