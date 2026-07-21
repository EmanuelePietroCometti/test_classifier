"""
Export an Ultralytics YOLO classifier (*-cls.pt) to a single self-contained ONNX
file, with outputs in the legacy format (class_id, confidence) and the class
names available both in metadata_props and, optionally, as a string initializer
inside the graph itself.

Exported graph structure
------------------------

    images  [N, 3, H, W]  uint8 or float32, RGB, range [0, 255]
       |
       +-- (Cast to float32, if the input is uint8)
       +-- Div 255
       +-- (Sub mean / Div std, optional)
       |
    backbone + Classify head  ->  probs [N, nc]   (softmax already applied)
       |
       +-- ArgMax    -> class_id    [N]  int64
       +-- ReduceMax -> confidence  [N]  float32
       +-- Gather    -> class_name  [N]  string    (only with --embed-names)
       +-- probs     [N, nc]  float32              (only with --output-probs)

Resizing to H x W and the BGR->RGB conversion are left to the C++ caller,
consistently with the convention used by the other exports in this project.

Usage
-----
    python export_onnx.py --weights runs/.../weights/best.pt --imgsz 192
    python export_onnx.py --weights best.pt --input-dtype uint8 --embed-names
    python export_onnx.py --weights best.pt --static-batch 17 --output-probs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import onnx
import onnxsim
import onnxruntime as ort
from onnx import helper, TensorProto

import numpy as np
import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# Wrapper
# --------------------------------------------------------------------------- #
class ClassifierWrapper(nn.Module):
    """
    Wraps the Ultralytics model, adding input normalization and the argmax/max
    reduction on the output, so that the resulting .onnx is self-contained.
    """

    def __init__(
        self,
        model: nn.Module,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        input_is_uint8: bool,
        output_probs: bool,
    ) -> None:
        super().__init__()
        self.model = model
        self.input_is_uint8 = input_is_uint8
        self.output_probs = output_probs

        # Registered as buffers so they end up as initializers in the ONNX graph.
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor):
        x = images.float() if self.input_is_uint8 else images
        x = x / 255.0
        x = (x - self.mean) / self.std

        out = self.model(x)

        # The Ultralytics Classify head returns `y if self.export else (y, x)`,
        # i.e. a (softmax_probs, raw_logits) tuple unless the export flag is set.
        # set_export_mode() raises that flag, but older/newer head revisions may
        # not expose it, so unwrap defensively as well.
        if isinstance(out, (tuple, list)):
            out = out[0]
        probs = out

        confidence, class_id = probs.max(dim=1)
        class_id = class_id.to(torch.int64)

        if self.output_probs:
            return class_id, confidence, probs
        return class_id, confidence


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def set_export_mode(model: nn.Module) -> None:
    """
    Raise the `export` flag on every submodule that has one. Ultralytics heads
    branch on it to return a bare tensor instead of a (output, intermediate)
    tuple, which is what torch.export can actually trace.
    """
    for module in model.modules():
        if hasattr(module, "export"):
            module.export = True
        if hasattr(module, "format"):
            module.format = "onnx"


def load_model(weights: Path) -> tuple[nn.Module, dict[int, str], int]:
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics is not installed: pip install ultralytics")

    yolo = YOLO(str(weights))

    if getattr(yolo, "task", None) != "classify":
        sys.exit(f"Unsupported task '{yolo.task}': this script only handles 'classify'.")

    core = yolo.model
    core.eval()
    core.float()

    # fuse() merges Conv+BN: smaller graph, and consistent with final validation.
    if hasattr(core, "fuse"):
        core = core.fuse()

    set_export_mode(core)

    for p in core.parameters():
        p.requires_grad_(False)

    names = core.names if isinstance(core.names, dict) else dict(enumerate(core.names))
    names = {int(k): str(v) for k, v in names.items()}

    train_imgsz = None
    args = getattr(core, "args", None)
    if isinstance(args, dict):
        train_imgsz = args.get("imgsz")
    elif args is not None:
        train_imgsz = getattr(args, "imgsz", None)

    return core, names, train_imgsz


def sanity_check_forward(wrapper: nn.Module, dummy: torch.Tensor, n_classes: int) -> None:
    """
    Run the wrapper once in eager mode before handing it to the exporter.
    A failure here is a model-shape problem; a failure inside torch.onnx.export
    would be buried under ~80 frames of tracer internals.
    """
    with torch.no_grad():
        out = wrapper(dummy)

    class_id, confidence = out[0], out[1]
    if class_id.ndim != 1 or confidence.ndim != 1:
        sys.exit(f"Unexpected output shapes: class_id {tuple(class_id.shape)}, "
                 f"confidence {tuple(confidence.shape)}; expected 1-D tensors.")
    if len(out) > 2 and out[2].shape[1] != n_classes:
        sys.exit(f"probs has {out[2].shape[1]} columns but the model declares {n_classes} classes.")

    # Softmax rows must sum to 1: catches a head that returned raw logits.
    if len(out) > 2:
        row_sums = out[2].sum(dim=1)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3):
            print("[warn] probability rows do not sum to 1: the head may be returning "
                  "logits. Enable the softmax in ClassifierWrapper.forward().")

    print(f"[ok]   eager forward pass: class_id {tuple(class_id.shape)}, "
          f"confidence {tuple(confidence.shape)}")


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def run_export(
    wrapper: nn.Module,
    dummy: torch.Tensor,
    onnx_path: Path,
    output_names: list[str],
    opset: int,
    exporter: str,
    static_batch: int | None,
) -> None:
    """
    Torch >= 2.6 defaults to the dynamo exporter. It accepts `dynamic_shapes`
    (not `dynamic_axes`) and tends to produce graphs that need extra patching
    before TensorRT accepts them, so the legacy TorchScript path is the default
    here. Falls back to dynamo if the installed torch no longer honours it.
    """
    common = dict(
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["images"],
        output_names=output_names,
    )
    dynamic = static_batch is None

    if exporter == "dynamo":
        kwargs = dict(common, dynamo=True)
        if dynamic:
            from torch.export import Dim

            kwargs["dynamic_shapes"] = {"images": {0: Dim("batch")}}
        torch.onnx.export(wrapper, (dummy,), str(onnx_path), **kwargs)
        print("[ok]   exported with the dynamo exporter.")
        return

    kwargs = dict(common)
    if dynamic:
        axes: dict[str, dict[int, str]] = {"images": {0: "batch"}}
        for name in output_names:
            axes[name] = {0: "batch"}
        kwargs["dynamic_axes"] = axes

    try:
        torch.onnx.export(wrapper, (dummy,), str(onnx_path), dynamo=False, **kwargs)
        print("[ok]   exported with the legacy TorchScript exporter.")
    except TypeError:
        # `dynamo=False` removed from this torch build: retry without the flag.
        print("[warn] this torch build no longer accepts dynamo=False, "
              "falling back to the default exporter.")
        torch.onnx.export(wrapper, (dummy,), str(onnx_path), **kwargs)


# --------------------------------------------------------------------------- #
# ONNX graph post-processing
# --------------------------------------------------------------------------- #
def embed_class_names_in_graph(onnx_path: Path, names: dict[int, str]) -> None:
    """
    Append a STRING initializer holding the class names plus a Gather node that
    produces the 'class_name' output, indexed by 'class_id'.
    """
    model = onnx.load(str(onnx_path))
    graph = model.graph

    ordered = [names[i] for i in sorted(names)]

    names_init = helper.make_tensor(
        name="class_names",
        data_type=TensorProto.STRING,
        dims=[len(ordered)],
        vals=[n.encode("utf-8") for n in ordered],
    )
    graph.initializer.append(names_init)

    # The dynamo exporter renames graph outputs, so resolve the real tensor name
    # feeding the 'class_id' output instead of assuming the literal string.
    class_id_tensor = next(
        (o.name for o in graph.output if o.name == "class_id"),
        graph.output[0].name,
    )

    gather = helper.make_node(
        "Gather",
        inputs=["class_names", class_id_tensor],
        outputs=["class_name"],
        axis=0,
        name="GatherClassName",
    )
    # class_id is produced by an earlier node, so appending preserves the
    # topological ordering required by ONNX.
    graph.node.append(gather)

    graph.output.append(
        helper.make_tensor_value_info("class_name", TensorProto.STRING, ["batch"])
    )

    onnx.checker.check_model(model)
    onnx.save(model, str(onnx_path))


def write_metadata(onnx_path: Path, meta: dict[str, str]) -> None:
    model = onnx.load(str(onnx_path))
    existing = {p.key for p in model.metadata_props}
    for key, value in meta.items():
        if key in existing:
            for p in model.metadata_props:
                if p.key == key:
                    p.value = value
            continue
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.save(model, str(onnx_path))


def simplify(onnx_path: Path) -> None:
    model = onnx.load(str(onnx_path))
    try:
        model_simp, ok = onnxsim.simplify(model)
    except Exception as exc:
        # onnxsim's shape inference chokes on dynamo-produced local functions.
        print(f"[warn] onnxsim raised {type(exc).__name__}: {exc}")
        print("[warn] keeping the original graph. Try onnx.inliner.inline_local_functions() "
              "if you exported with --exporter dynamo.")
        return

    if ok:
        onnx.save(model_simp, str(onnx_path))
        print("[ok]   graph simplified with onnxsim.")
    else:
        print("[warn] onnxsim validation failed, keeping the original graph.")


# --------------------------------------------------------------------------- #
# Numerical verification
# --------------------------------------------------------------------------- #
def verify(
    onnx_path: Path,
    wrapper: ClassifierWrapper,
    dummy: torch.Tensor,
    names: dict[int, str],
    embed_names: bool,
) -> None:
    with torch.no_grad():
        torch_out = wrapper(dummy)
    torch_id = torch_out[0].cpu().numpy()
    torch_conf = torch_out[1].cpu().numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    ort_out = session.run(None, {input_name: dummy.cpu().numpy()})

    out_names = [o.name for o in session.get_outputs()]
    ort_id = ort_out[out_names.index("class_id")] if "class_id" in out_names else ort_out[0]
    ort_conf = ort_out[out_names.index("confidence")] if "confidence" in out_names else ort_out[1]

    id_match = bool(np.array_equal(torch_id, ort_id))
    conf_delta = float(np.abs(torch_conf - ort_conf).max())

    print(f"[ok]   class_id identical: {id_match}")
    print(f"[ok]   max confidence delta: {conf_delta:.3e}")

    if embed_names and "class_name" in out_names:
        ort_name = ort_out[out_names.index("class_name")]
        decoded = [n.decode("utf-8") if isinstance(n, bytes) else str(n) for n in ort_name]
        expected = [names[int(i)] for i in ort_id]
        print(f"[ok]   class_name consistent: {decoded == expected}  (sample: {decoded[:3]})")

    if not id_match or conf_delta > 1e-4:
        print("[ERROR] numerical divergence above threshold: do not ship this model.")
        sys.exit(1)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ONNX export for Ultralytics YOLO classifiers")
    p.add_argument("--weights", type=Path, required=True, help="Path to best.pt")
    p.add_argument("--output", type=Path, default=None, help="Destination .onnx path")
    p.add_argument("--imgsz", type=int, default=None, help="Square image side (default: the training one)")
    p.add_argument("--opset", type=int, default=18, help="ONNX opset. Torch >= 2.9 implements 18 and up.")
    p.add_argument(
        "--exporter",
        choices=("legacy", "dynamo"),
        default="legacy",
        help="'legacy' = TorchScript tracer, more predictable for TensorRT. "
             "'dynamo' = torch.export based, required for some newer ops.",
    )
    p.add_argument(
        "--input-dtype",
        choices=("float32", "uint8"),
        default="float32",
        help="Input tensor type. uint8 avoids a conversion on the C++ side.",
    )
    p.add_argument(
        "--normalize",
        choices=("none", "imagenet"),
        default="none",
        help="'none' = /255 only (Ultralytics classify default). 'imagenet' = ImageNet mean/std.",
    )
    p.add_argument("--static-batch", type=int, default=None, help="Pin the batch size (e.g. 17). If omitted, batch is dynamic.")
    p.add_argument("--embed-names", action="store_true", help="Add a string-typed 'class_name' output.")
    p.add_argument("--output-probs", action="store_true", help="Also expose the full probability vector.")
    p.add_argument("--no-simplify", action="store_true")
    p.add_argument("--no-verify", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.weights.is_file():
        sys.exit(f"Weights not found: {args.weights}")

    core, names, train_imgsz = load_model(args.weights)
    print(f"[ok]   model loaded, {len(names)} classes: {names}")

    imgsz = args.imgsz or train_imgsz or 192
    if args.imgsz and train_imgsz and args.imgsz != train_imgsz:
        print(f"[warn] requested imgsz ({args.imgsz}) differs from the training one ({train_imgsz}).")

    mean = IMAGENET_MEAN if args.normalize == "imagenet" else (0.0, 0.0, 0.0)
    std = IMAGENET_STD if args.normalize == "imagenet" else (1.0, 1.0, 1.0)

    wrapper = ClassifierWrapper(
        model=core,
        mean=mean,
        std=std,
        input_is_uint8=(args.input_dtype == "uint8"),
        output_probs=args.output_probs,
    ).eval()

    batch = args.static_batch or 1
    if args.input_dtype == "uint8":
        dummy = torch.randint(0, 256, (batch, 3, imgsz, imgsz), dtype=torch.uint8)
    else:
        dummy = torch.rand(batch, 3, imgsz, imgsz) * 255.0

    sanity_check_forward(wrapper, dummy, len(names))

    output_names = ["class_id", "confidence"] + (["probs"] if args.output_probs else [])

    onnx_path = args.output or args.weights.with_suffix(".onnx")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        run_export(
            wrapper=wrapper,
            dummy=dummy,
            onnx_path=onnx_path,
            output_names=output_names,
            opset=args.opset,
            exporter=args.exporter,
            static_batch=args.static_batch,
        )
    print(f"[ok]   written to {onnx_path}")

    if not args.no_simplify:
        simplify(onnx_path)

    if args.embed_names:
        embed_class_names_in_graph(onnx_path, names)
        print("[ok]   class names embedded in the graph ('class_name' output).")

    write_metadata(
        onnx_path,
        {
            "names": json.dumps({str(k): v for k, v in sorted(names.items())}, ensure_ascii=False),
            "names_ordered": json.dumps([names[i] for i in sorted(names)], ensure_ascii=False),
            "num_classes": str(len(names)),
            "task": "classify",
            "imgsz": str(imgsz),
            "input_layout": "NCHW",
            "input_color_order": "RGB",
            "input_range": "0-255",
            "input_dtype": args.input_dtype,
            "normalization": args.normalize,
            "resize_strategy": "external (caller must supply the image already at imgsz x imgsz)",
            "batch": str(args.static_batch) if args.static_batch else "dynamic",
            "exporter": args.exporter,
            "opset": str(args.opset),
            "outputs": ",".join(output_names + (["class_name"] if args.embed_names else [])),
            "source_weights": args.weights.name,
        },
    )
    print("[ok]   metadata_props written.")

    if not args.no_verify:
        verify(onnx_path, wrapper, dummy, names, args.embed_names)

    size_mb = onnx_path.stat().st_size / 1e6
    print(f"\nDone: {onnx_path}  ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()