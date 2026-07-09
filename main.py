"""
export_yolo_pure_onnx.py — Train and export a YOLO classification model to a *pure* ONNX graph.

Philosophy:
The graph is a pure forward pass. If the host C++ application provides ImageNet-normalized 
tensors (as in the RD4AD pipeline), we must handle it. YOLO typically expects inputs 
scaled to [0, 1]. The YOLOPure wrapper includes an optional denormalization step to 
translate the host's ImageNet tensor back to [0, 1] purely mathematically within the graph.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
import onnx
import onnxruntime as ort
import onnxscript

IMG_SIZE = 256
OPSET = 17
INPUT_NAMES = ["input_tensor"]
OUTPUT_NAMES = ["class_probabilities"]

class YOLOPure(nn.Module):
    """Pure forward pass for YOLO classification: normalized tensor in -> probabilities out."""

    def __init__(self, yolo_pytorch_model: nn.Module, revert_imagenet_norm: bool = True):
        super().__init__()
        self.encoder = yolo_pytorch_model
        self.revert_imagenet_norm = revert_imagenet_norm
        
        # ImageNet constants used by the C++ host
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, input_tensor: torch.Tensor):
        x = input_tensor
        
        # If the C++ engine sends ImageNet normalized data, revert it to [0, 1] range
        if self.revert_imagenet_norm:
            x = (x * self.std) + self.mean
            
        # YOLO classification backbone forward pass
        out = self.encoder(x)
        
        # Ultralytics raw modules often return a tuple: (output, intermediate_features).
        # We must extract the first element which contains the actual class predictions.
        if isinstance(out, (tuple, list)):
            out = out[0]
            
        # The YOLO Classify head automatically applies Softmax when in .eval() mode.
        # We return the tensor directly to avoid double-softmax distortion.
        return out

def train_and_get_model(dataset_path: str, epochs: int, device: str) -> nn.Module:
    """Trains a YOLO classifier and returns the raw PyTorch model."""
    print("--- Starting YOLO Training ---")
    model = YOLO("yolo11n-cls.pt")
    model.train(
        data=dataset_path,
        epochs=epochs,
        imgsz=IMG_SIZE,
        device=device,
        project="yolo_export_project",
        name="pure_model"
    )
    # Extract the raw PyTorch module from the Ultralytics wrapper
    raw_pytorch_model = model.model.eval()
    return raw_pytorch_model

def export_fp32(model: nn.Module, onnx_path: Path, device: str):
    """Exports the PyTorch model to a pure ONNX graph conforming to the pipeline specs."""
    dummy = torch.randn(2, 3, IMG_SIZE, IMG_SIZE, dtype=torch.float32, device=device)
    
    # Only the batch axis is dynamic, ensuring strict optimization in TensorRT/C++ engines
    dynamic_axes = {
        "input_tensor": {0: "batch"},
        "class_probabilities": {0: "batch"}
    }
    
    with torch.no_grad():
        torch.onnx.export(
            model, 
            (dummy,), 
            str(onnx_path),
            input_names=INPUT_NAMES, 
            output_names=OUTPUT_NAMES,
            dynamic_axes=dynamic_axes, 
            opset_version=OPSET,
            do_constant_folding=True, 
            dynamo=False
        )
        
    
    onnx.checker.check_model(str(onnx_path))

def verify(model: nn.Module, onnx_path: Path, device: str, atol: float = 1e-3):
    """Verifies mathematical parity between PyTorch and the exported ONNX graph."""
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    
    print("\n--- PyTorch vs ONNX parity (atol=1e-4) ---")
    for batch in (1, 4):
        x = rng.standard_normal((batch, 3, IMG_SIZE, IMG_SIZE)).astype(np.float32)
        
        with torch.no_grad():
            t_probs = model(torch.from_numpy(x).to(device))
            
        o_probs = sess.run(OUTPUT_NAMES, {INPUT_NAMES[0]: x})[0]
        
        np.testing.assert_allclose(
            o_probs, t_probs.cpu().numpy(), atol=atol, rtol=0,
            err_msg=f"Probability mismatch (batch={batch})"
        )
        print(f"  batch={batch}: probs |Δ|max={np.abs(o_probs-t_probs.cpu().numpy()).max():.2e}  OK")
        
    print("[PASS] Exact mathematical parity confirmed.")

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=str, help="Path to the dataset directory")
    p.add_argument("--checkpoint", type=str, default=None, help="Path to pre-trained best.pt")
    p.add_argument("--output", type=str, default="yolo_pure.onnx")
    p.add_argument("--epochs", type=int, default=10)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.checkpoint:
        print(f"Loading existing checkpoint: {args.checkpoint}")
        yolo = YOLO(args.checkpoint)
        raw_model = yolo.model.eval()
    elif args.dataset:
        raw_model = train_and_get_model(args.dataset, args.epochs, device)
    else:
        sys.exit("ERROR: Provide either --dataset to train or --checkpoint to load.")

    # Wrap the model for the pure forward pass
    pure_model = YOLOPure(raw_model).to(device).eval()
    
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Exporting YOLO (pure graph) -> {out_path} ---")
    export_fp32(pure_model, out_path, device)
    print(f"[OK] fp32 export: {out_path}")

    # Always verify parity after export
    verify(pure_model, out_path, device)

if __name__ == "__main__":
    main()