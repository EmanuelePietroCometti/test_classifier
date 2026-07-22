# test_classifier

Training, evaluation and ONNX export pipeline for a YOLO-based image classifier,
targeted at industrial visual inspection on an edge device (ONNX Runtime + TensorRT, C++ caller).

Reference use case: three-class pin inspection (`_BadPins`, `_GoodPins`, `_NoRivet`).

---

## Contents

- [Requirements](#requirements)
- [Repository layout](#repository-layout)
- [Dataset layout](#dataset-layout)
- [Quick start](#quick-start)
- [Command-line arguments](#command-line-arguments)
- [Evaluation](#evaluation)
- [ONNX export](#onnx-export)
- [C++ integration](#c-integration)
- [Known limitations](#known-limitations)

---

## Requirements

- Python 3.12
- CUDA-capable GPU (developed on an RTX A4000, 16 GB)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install ultralytics scikit-learn scipy seaborn matplotlib
pip install onnx onnxsim onnxruntime-gpu
```

> Install `onnxruntime-gpu`, not `onnxruntime`. Having both in the same
> environment causes provider registration to fail silently and inference falls
> back to CPU. Also check that the ONNX Runtime build matches your CUDA major
> version.

---

## Repository layout

```
test_classifier/
├── main.py                 # Entry point: split -> train -> evaluate -> report
├── export_onnx.py          # Standalone ONNX export
└── src/
    ├── parser.py           # argparse configuration (mirrors Ultralytics train args)
    ├── train.py            # Thin wrapper around YOLO.train()
    ├── eval.py             # Inference over a held-out set, per-class folder walk
    └── utils.py            # Dataset split, metrics, Wilson CI, confusion matrix plot
```

`main.py` runs the full sequence in one shot:

1. `split_and_save_image_dataset()` — stratified 70/20/10 split of `--src_dataset` into `--data`
2. `train()` — Ultralytics training loop
3. `evaluate()` — inference on `--val_dataset` using `best.pt`
4. metrics printout + confusion matrix saved next to the run directory

---

## Dataset layout

The source dataset is a plain folder-per-class structure:

```
src_dataset/
├── _BadPins/
│   ├── img_0001.bmp
│   └── ...
├── _GoodPins/
└── _NoRivet/
```

The split step produces:

```
dataset/
├── train/{_BadPins,_GoodPins,_NoRivet}/
├── val/{_BadPins,_GoodPins,_NoRivet}/
└── test/{_BadPins,_GoodPins,_NoRivet}/
```

**The folder names in `--val_dataset` must match the class names stored in the
model weights**, because `evaluate()` walks `model.names` and looks for a
subfolder per class. A mismatch produces `Warning: Folder 'X' not found ...` for
every class, an empty prediction array, and a downstream crash in the metrics.

Supported extensions: `.bmp`, `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`.

---

## Quick start

```bash
python main.py --src_dataset src_dataset --val_dataset val_dataset --grayscale true \
  --model yolo26n-cls.pt --data dataset --epochs 150 --imgsz 224 --batch 64 \
  --device 0 --workers 1 --cache cache \
  --optimizer MuSGD --lr0 0.01 --lrf 0.01 --cos_lr true --weight_decay 0.0005 \
  --warmup_epochs 3.0 --dropout 0.2 --patience 30 --seed 42 --deterministic true \
  --val true --plots true --save true --save_period 10 \
  --project catene --name exp_musgd_lr1e-2_gray --exist_ok false
```

`--imgsz 224` is the reference configuration for this project. It is not just a
training detail: it becomes part of the deployment contract, since the C++ caller
must resize to 224×224 before feeding the exported graph.

Outputs land in `<project>/<name>/`:

```
weights/best.pt
weights/last.pt
results.csv
confusion_matrix.png
args.yaml
```

> If `--project` is a relative path, Ultralytics prepends its configured
> `runs_dir`, so you end up with `runs/classify/<project>/<name>`. Pass an
> absolute path if you want full control over the destination.

---

## Command-line arguments

`src/parser.py` exposes the Ultralytics training surface plus three pipeline-specific
arguments. Only the non-obvious ones are documented here; everything else maps
1:1 to the Ultralytics `train()` signature.

| Argument | Type | Notes |
|---|---|---|
| `--src_dataset` | str | Raw folder-per-class dataset, split into `--data` |
| `--val_dataset` | str | Held-out set used by the custom evaluation step |
| `--data` | str | Destination of the generated train/val/test split |
| `--batch` | int \| float | Int = fixed batch size. Float in (0,1] = AutoBatch target VRAM fraction |
| `--cache` | bool \| str | `True`, `False`, `'ram'`, `'disk'` |
| `--device` | int \| str \| list | `0`, `'cpu'`, `'0,1'` |
| `--amp` | bool | Parsed via `str2bool`, so `--amp false` works as expected |
| `--patience` | int | Early stopping. `0` disables it |

Boolean flags accept `true/false/yes/no/1/0` (case-insensitive). Mixed-type flags
(`--batch`, `--cache`, `--device`, `--pretrained`) go through `ast.literal_eval`
with a string fallback.

### Batch size

AutoBatch (`--batch 0.7`) picks a batch that saturates VRAM, which on a small
dataset means very few optimizer steps per epoch. With ~2000 training images and
a batch of 537 you get 4 steps per epoch, so early stopping triggers on noise
rather than on convergence. Prefer a fixed `--batch 64` or `--batch 128` unless
you are deliberately benchmarking throughput. At `--imgsz 192` and `--batch 64`,
~2000 training images give roughly 32 optimizer steps per epoch, which is enough
for `--patience 30` to measure real convergence.

### Mixed precision

`--amp false` roughly doubles step time and VRAM for no benefit on a 1.5M-parameter
model. Disable it only if you need bitwise-reproducible runs alongside
`--deterministic true`, and record that reason in the run notes.

### Run naming

The run name is the only human-readable index into the experiment history, and it
is not validated against the actual arguments — `--name exp_adamw_lr1e-3` with
`--optimizer MuSGD` will happily coexist. `args.yaml` inside the run directory is
the source of truth, but a name that contradicts it costs time later. Encode
optimizer, learning rate and image size: `exp_musgd_lr1e-3_img192`.

### Output paths

A relative `--project` is appended to the Ultralytics `runs_dir`, so
`--project runs_pin/classify --name exp_musgd_lr1e-3_img192` resolves to
`runs/classify/runs_pin/classify/exp_musgd_lr1e-3_img192/`. Pass an absolute path
if you want the weights somewhere predictable for the export step.

---

## Evaluation

`src/eval.py` runs inference folder by folder and feeds
`evaluate_imbalanced_predictions()` in `src/utils.py`, which reports:

- number of images and misclassifications
- balanced accuracy, macro precision / recall / F1
- macro AUROC, one-vs-rest
- 99% Wilson confidence interval
- confusion matrix (saved as PNG)

### Multiclass AUROC

With more than two classes, `roc_auc_score` requires the full probability matrix
and an explicit strategy:

```python
auroc = roc_auc_score(
    y_true,
    y_scores,                      # shape (n_samples, n_classes)
    multi_class='ovr',
    average='macro',
    labels=np.arange(n_classes),
)
```

`y_scores` must be the complete softmax output per sample, not the probability of
a single class. `labels=` is mandatory here: without it, a validation set missing
one class raises `Number of classes in y_true not equal to the number of columns
in 'y_score'`. Pass `labels=` to `confusion_matrix()` for the same reason, or the
matrix shrinks and the plot axis labels silently desynchronise.

`ovr` + `average='macro'` is consistent with the other macro metrics in the
report and treats every class equally regardless of support — the right default
for an imbalanced defect dataset.

### Confidence interval

The Wilson interval assumes a binomial proportion over `n` independent trials.
Balanced accuracy is an average of per-class recalls, not such a proportion, so
applying the interval to it is not statistically meaningful. Report either:

- a Wilson interval on plain accuracy over all `n` images, or
- one interval per class, computed on that class's recall and its own support:

```python
recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
supports = np.bincount(y_true, minlength=len(recalls))
per_class_ci = [calculate_wilson_ci(r, int(n), 0.99) for r, n in zip(recalls, supports)]
```

---

## ONNX export

`export_onnx.py` produces a single self-contained `.onnx` with no sidecar file.

```
images  [N, 3, H, W]  uint8 or float32, RGB, range [0, 255]
   |
   +-- Cast to float32 (if uint8 input)
   +-- Div 255
   +-- Sub mean / Div std (optional)
   |
backbone + Classify head  ->  probs [N, nc]   (softmax already applied)
   |
   +-- ArgMax    -> class_id    [N]  int64
   +-- ReduceMax -> confidence  [N]  float32
   +-- Gather    -> class_name  [N]  string    (--embed-names)
   +-- probs     [N, nc]  float32              (--output-probs)
```

Typical invocation for the TensorRT deployment target:

```bash
python export_onnx.py \
    --weights runs/classify/catene/exp_musgd_lr1e-3_img192/weights/best.pt \
    --imgsz 224 \
    --input-dtype float32 \
    --opset 18 \
    --exporter legacy
```

`--imgsz` can be omitted: the script reads the training value from `model.args`
and warns if the value you pass differs. Passing it explicitly is still the safer
habit, since a silent mismatch between the export size and the C++ resize does
not raise anything — the model runs and returns plausible but wrong predictions.

| Flag | Effect |
|---|---|
| `--input-dtype uint8` | Input tensor is `uint8`; the cast happens inside the graph |
| `--exporter {legacy,dynamo}` | `legacy` = TorchScript tracer (default, safer for TensorRT) |
| `--opset N` | Defaults to 18; lower values trigger an automatic downconversion |
| `--static-batch N` | Pins the batch dimension. Omit for a dynamic batch axis |
| `--normalize {none,imagenet}` | `none` = `/255` only, matching the Ultralytics classify default |
| `--embed-names` | Adds a string-typed `class_name` output |
| `--output-probs` | Also exposes the full `[N, nc]` probability vector |
| `--no-simplify` / `--no-verify` | Skip onnxsim / the PyTorch-vs-ORT comparison |

The export ends with a numerical check against the PyTorch model on the dummy
input and exits with code 1 if `class_id` diverges or the confidence delta
exceeds `1e-4`.

### Preprocessing contract

The graph does **not** resize. The caller must supply an image already at
`imgsz × imgsz` — **192×192** for the reference configuration — RGB, NCHW, range
`[0, 255]`. Any hardcoded `224` left over in the C++ side (`AnomalyEngine`,
`WorkerONNX`) is a silent failure: the shapes will not match the static input and
ONNX Runtime throws, or worse, a stale resize constant feeds the wrong crop.
Read `imgsz` from the model metadata instead of hardcoding it. Note that the Ultralytics
validation transform resizes the short side and then centre-crops; a direct
resize on the C++ side is a train/inference mismatch that costs accuracy. Match
whichever strategy you used during validation.

Normalization defaults to `/255` with no mean/std subtraction, because
`classify_transforms()` defaults to `mean=(0,0,0)`, `std=(1,1,1)`. Confirm this
against your installed Ultralytics version before trusting the export.

### Class names

Names are always written to `metadata_props`:

| Key | Value |
|---|---|
| `names` | JSON object, `{"0": "_BadPins", ...}` |
| `names_ordered` | JSON array, index-ordered |
| `num_classes`, `imgsz`, `input_dtype`, `normalization`, `resize_strategy`, `outputs` | Deployment contract |

With `--embed-names` a `Gather` node over a STRING initializer also emits the
predicted name directly. **TensorRT does not support string tensors**, so that
node falls back to another execution provider and fragments the graph into
subgraphs. For a TensorRT deployment, use the metadata and skip `--embed-names`.

### Choosing the exporter

Torch 2.6 and later default to the **dynamo** exporter (`torch.export`-based).
`export_onnx.py` overrides that and uses the legacy TorchScript tracer by
default, because dynamo emits local functions that break `onnxsim`'s shape
inference and often need extra patching before TensorRT accepts the graph.
Switch with `--exporter dynamo` only if the legacy path fails on an unsupported
operator.

Symptoms of the dynamo path, all benign warnings but worth recognising:

```
UserWarning: 'dynamic_axes' is not recommended when dynamo=True
Setting ONNX exporter to use operator set version 18 because the requested
opset_version 17 is a lower version than we have implementations for
```

The first appears because dynamo wants `dynamic_shapes` instead; the script
supplies the right one for the selected exporter. The second is why `--opset`
defaults to 18: asking for 17 triggers an automatic downconversion that may not
succeed.

If `onnxsim` raises during simplification on a dynamo export, inline the local
functions first:

```python
import onnx
model = onnx.inliner.inline_local_functions(onnx.load(path))
```

### Troubleshooting

**`AttributeError: 'tuple' object has no attribute 'max'`**

The Ultralytics `Classify` head returns `y if self.export else (y, x)` — a
`(softmax_probs, raw_logits)` tuple unless the `export` flag is raised. Loading
the weights through `YOLO()` does not raise it; only their own `Exporter` does.
`set_export_mode()` walks the module tree and sets `export = True` wherever the
attribute exists, and `ClassifierWrapper.forward()` additionally unwraps any
tuple defensively, since the flag is not present on every head revision.

**Export fails deep inside the tracer**

`sanity_check_forward()` runs one eager forward pass before the export and
validates output ranks, the `probs` column count against the declared class
count, and that probability rows sum to 1. A model-shape problem surfaces there
with a one-line message instead of ~80 frames of tracer internals. A warning
about rows not summing to 1 means the head returned logits: enable the softmax
in `ClassifierWrapper.forward()`.

**Wrong class names or index order**

The mapping comes from the weights, not from the folders on disk at export time.
Retraining after adding, removing or renaming a class folder silently changes the
index order and invalidates every previously exported `.onnx` and any hardcoded
index on the C++ side.

---

## C++ integration

Reading the class names from the model metadata:

```cpp
Ort::AllocatorWithDefaultOptions allocator;
Ort::ModelMetadata metadata = session.GetModelMetadata();
Ort::AllocatedStringPtr names = metadata.LookupCustomMetadataMapAllocated("names_ordered", allocator);
// -> ["_BadPins","_GoodPins","_NoRivet"]
```

Reading the outputs:

```cpp
const int64_t* class_id   = outputs[0].GetTensorData<int64_t>();
const float*   confidence = outputs[1].GetTensorData<float>();
```

`class_id` is `int64`. TensorRT downgrades it to `int32` internally and logs a
warning; this is harmless. If you prefer a native `int32` output, change
`class_id.to(torch.int64)` to `.to(torch.int32)` in `ClassifierWrapper.forward()`.

A static batch avoids building an optimization profile and yields a leaner
TensorRT engine. Engine build emits VRAM warnings that reflect build-time
workspace probing, not steady-state runtime usage.

---

## Known limitations

- **The generated `dataset/test` split is never used.** `main.py` evaluates on
  the external `--val_dataset` instead. If that set shares acquisitions with
  `--src_dataset`, the reported metrics are leaked and not defensible. Decide
  explicitly which set is the held-out one and document it.
- **`split_and_save_image_dataset()` globs `*.*`**, so any non-image file in a
  class folder (README, `.json`, `Thumbs.db`) is copied into the split and later
  treated as a sample.
- **The split is not deterministic across dataset changes.** `random_state=42`
  fixes the shuffle, but adding or removing images reshuffles every assignment.
  Persist the split manifest if you need run-to-run comparability.
- **The stored class index order is defined by the training run.** Any change to
  the class folders changes the mapping, which silently invalidates already
  exported `.onnx` files and any hardcoded index on the C++ side. Always read the
  mapping from the model metadata rather than hardcoding it.