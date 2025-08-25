# Export the trained SimpleResNetEye PyTorch model to ONNX for the web app.
# - Loads weights from: model_registry/eye/0.1.0/weights.pt
# - Writes ONNX to: blinksense_web/static/models/eye_resnet.onnx

from pathlib import Path
import sys
import torch

# Import the eye model
try:
    from src.models.simple_resnet_eye import SimpleResNetEye
except Exception as e:
    print("[ERROR] Could not import SimpleResNetEye. Run from repo root (where 'src/' lives).\n" 
          "Example: python tools/export_onnx_eye.py")
    raise

# Paths
REPO_ROOT = Path(__file__).resolve().parents[1]  # BlinkSense/
CKPT_PATH = REPO_ROOT / "model_registry/eye/0.1.0/weights.pt"
ONNX_PATH = REPO_ROOT / "blinksense_web/static/models/eye_resnet.onnx"
ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)

if not CKPT_PATH.exists():
    print(f"[ERROR] Weights not found at {CKPT_PATH}\n"
          f"Copy your best checkpoint to that path (e.g., from outputs/simple_resnet_eye_best.pt)")
    sys.exit(1)

# Build model and load weights
model = SimpleResNetEye().eval()
state = torch.load(CKPT_PATH, map_location="cpu")

# Support both raw state_dict or {'model': state_dict}
if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]
elif isinstance(state, dict) and "model" in state:
    state = state["model"]

# Strip an optional 'module.' prefix (from DataParallel)
if any(k.startswith("module.") for k in state.keys()):
    state = {k.replace("module.", "", 1): v for k, v in state.items()}

model.load_state_dict(state, strict=True)
print(f"[OK] Loaded weights: {CKPT_PATH}")

# Export to ONNX
# Model expects grayscale 64x64, NCHW
example = torch.randn(1, 1, 64, 64)

torch.onnx.export(
    model,
    example,
    ONNX_PATH.as_posix(),
    input_names=["x"],
    output_names=["logits", "emb"],
    dynamic_axes={"x": {0: "N"}, "logits": {0: "N"}, "emb": {0: "N"}},
    opset_version=13,
)
print(f"[OK] Exported ONNX: {ONNX_PATH}")

# Optional structural check if onnx is installed (no hard dependency)
try:
    import onnx  # type: ignore
    onnx_model = onnx.load(ONNX_PATH.as_posix())
    onnx.checker.check_model(onnx_model)
    print("[OK] ONNX model structure validated.")
except Exception:
    # It's fine if onnx isn't installed; the runtime can still load it in the browser
    pass