import sys
from pathlib import Path
import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
ROOT = Path(__file__).parent.parent

from config import config, SystemType
from system import RimlessWheelHybridSystem, CompassGaitHybridSystem
from networks import SuspensionNetworks
from visualize import plot_hybrid_suspension_rimless, plot_hybrid_suspension_compass

def replot():
    # Device setup
    if config.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif config.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Replot Rimless Wheel
    print("\n--- Replotting Rimless Wheel ---")
    config.system_type = SystemType.RIMLESS_WHEEL
    hds_rw = RimlessWheelHybridSystem(config)
    model_rw = SuspensionNetworks(config).to(device)
    model_path_rw = ROOT / "runs" / "rimless_wheel" / "model.pt"
    if model_path_rw.exists():
        model_rw.load_state_dict(torch.load(model_path_rw, map_location=device))
        plot_hybrid_suspension_rimless(model_rw, hds_rw, config, np.logspace(0, -3, 100))
    else:
        print(f"Rimless model not found at {model_path_rw}")

    # Replot Compass Gait
    print("\n--- Replotting Compass Gait ---")
    config.system_type = SystemType.COMPASS_GAIT
    hds_cg = CompassGaitHybridSystem(config)
    model_cg = SuspensionNetworks(config).to(device)
    model_path_cg = ROOT / "runs" / "compass_gait" / "model.pt"
    if model_path_cg.exists():
        model_cg.load_state_dict(torch.load(model_path_cg, map_location=device))
        plot_hybrid_suspension_compass(model_cg, hds_cg, config, np.logspace(0, -3, 100))
    else:
        print(f"Compass Gait model not found at {model_path_cg}")

if __name__ == "__main__":
    replot()
