import argparse
import ast

def str2bool(v):
    """
    Helper function to parse boolean arguments from the command line.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_mixed_type(v):
    """
    Safely evaluates string inputs into their native Python types (int, float, list, bool).
    If evaluation fails (e.g., standard strings like 'ram' or 'cpu'), it returns the raw string.
    """
    try:
        return ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return v

def get_training_parser():
    """
    Builds and returns the comprehensive argument parser for the training configuration.
    """
    parser = argparse.ArgumentParser(description="Full Model Training Configuration Parser")

    # --- Base & IO Configuration ---
    parser.add_argument('--src_dataset', type=str, default=None, 
                        help="Path to source dataset.")
    parser.add_argument('--val_dataset', type=str, default=None, 
                        help="Path to validation dataset.")
    parser.add_argument('--grayscale', type=str2bool, default=False,
                    help="Convert dataset images to grayscale (replicated over 3 channels).")
    parser.add_argument('--model', type=str, default=None, 
                        help="Path to pre-trained .pt model or .yaml config file.")
    parser.add_argument('--data', type=str, default=None, 
                        help="Path to dataset.")
    parser.add_argument('--project', type=str, default=None, 
                        help="Project directory name to save outputs.")
    parser.add_argument('--name', type=str, default=None, 
                        help="Training session name for the sub-directory.")
    parser.add_argument('--exist_ok', type=str2bool, default=False, 
                        help="Allow overwriting an existing project/name directory.")
    parser.add_argument('--save_dir', type=str, default=None, 
                        help="Exact directory to save outputs, bypassing auto-increment.")
    parser.add_argument('--save', type=str2bool, default=True, 
                        help="Enable saving of training checkpoints and final weights.")
    parser.add_argument('--save_period', type=int, default=-1, 
                        help="Checkpoint saving frequency in epochs (-1 disables).")

    # --- Training Execution & Hardware ---
    parser.add_argument('--epochs', type=int, default=100, 
                        help="Total number of training epochs.")
    parser.add_argument('--time', type=float, default=None, 
                        help="Maximum training time in hours (overrides epochs).")
    parser.add_argument('--patience', type=int, default=100, 
                        help="Epochs to wait without improvement before early stopping.")
    # batch can be int (16) or float fraction for auto-batching (0.7)
    parser.add_argument('--batch', type=parse_mixed_type, default=16, 
                        help="Batch size (int), or GPU memory utilization fraction (float).")
    parser.add_argument('--imgsz', type=int, default=640, 
                        help="Target image size for training.")
    # cache can be bool (True/False) or str ('ram', 'disk')
    parser.add_argument('--cache', type=parse_mixed_type, default=False, 
                        help="Cache dataset images: True, False, 'ram', or 'disk'.")
    # device can be int (0), str ('cpu', 'mps'), or list ([0,1])
    parser.add_argument('--device', type=parse_mixed_type, default=None, 
                        help="Computational device(s) to use: '0', '0,1', 'cpu', 'mps', etc.")
    parser.add_argument('--workers', type=int, default=8, 
                        help="Number of worker threads for data loading.")
    parser.add_argument('--pretrained', type=parse_mixed_type, default=True, 
                        help="Start from pre-trained weights (bool) or specify a path to weights (str).")
    parser.add_argument('--optimizer', type=str, default='auto', 
                        help="Optimizer choice (e.g., SGD, Adam, AdamW, auto).")
    parser.add_argument('--seed', type=int, default=0, 
                        help="Random seed for reproducibility.")
    parser.add_argument('--deterministic', type=str2bool, default=True, 
                        help="Force deterministic algorithms for reproducible results.")
    parser.add_argument('--verbose', type=str2bool, default=True, 
                        help="Enable detailed console output.")
    parser.add_argument('--resume', type=str2bool, default=False, 
                        help="Resume training from the last saved checkpoint.")
    parser.add_argument('--amp', type=str2bool, default=True, 
                        help="Enable Automatic Mixed Precision (AMP) training.")
    parser.add_argument('--fraction', type=float, default=1.0, 
                        help="Fraction of the dataset to use for training (0.0 to 1.0).")
    parser.add_argument('--profile', type=str2bool, default=False, 
                        help="Profile ONNX and TensorRT speeds during training.")
    parser.add_argument('--compile', type=parse_mixed_type, default=False, 
                        help="Enable PyTorch 2.x compile (bool or backend mode str).")

    # --- Dataset & Augmentation ---
    parser.add_argument('--single_cls', type=str2bool, default=False,
                        help="Treat all classes in a multi-class dataset as a single class.")
    parser.add_argument('--classes', type=int, nargs='+', default=None,
                        help="List of class IDs to train on (e.g., --classes 0 2 3).")
    parser.add_argument('--rect', type=str2bool, default=False,
                        help="Enable minimal padding strategy (rectangular training).")
    parser.add_argument('--multi_scale', type=float, default=0.0,
                        help="Randomly vary imgsz for each batch by +/- this ratio.")
    parser.add_argument('--close_mosaic', type=int, default=10,
                        help="Disable mosaic data augmentation in the last N epochs.")
    parser.add_argument('--overlap_mask', type=str2bool, default=True,
                        help="Merge overlapping object masks into a single mask.")
    parser.add_argument('--mask_ratio', type=int, default=4,
                        help="Downsampling ratio for segmentation masks.")
    parser.add_argument('--dropout', type=float, default=0.0,
                        help="Dropout rate for regularization in classification tasks.")

    # --- Photometric Augmentation ---
    parser.add_argument('--hsv_h', type=float, default=0.015,
                        help="Hue jitter fraction. Irrelevant for grayscale inputs.")
    parser.add_argument('--hsv_s', type=float, default=0.7,
                        help="Saturation jitter fraction. Irrelevant for grayscale inputs.")
    parser.add_argument('--hsv_v', type=float, default=0.4,
                        help="Brightness jitter fraction. Keep low for low-contrast defects.")
    parser.add_argument('--auto_augment', type=parse_mixed_type, default='randaugment',
                        help="Auto policy for classification: 'randaugment', 'autoaugment', "
                             "'augmix', or None. Use None to preserve micro-contrast.")

    # --- Geometric Augmentation ---
    parser.add_argument('--degrees', type=float, default=0.0,
                        help="Random rotation range in degrees (+/-).")
    parser.add_argument('--translate', type=float, default=0.1,
                        help="Random translation as a fraction of image size (+/-).")
    parser.add_argument('--scale', type=float, default=0.5,
                        help="Random scale/crop jitter. For classification maps to "
                             "RandomResizedCrop(scale=(1-scale, 1.0)).")
    parser.add_argument('--shear', type=float, default=0.0,
                        help="Shear angle range in degrees (+/-).")
    parser.add_argument('--perspective', type=float, default=0.0,
                        help="Perspective distortion factor (0.0 to 0.001).")
    parser.add_argument('--flipud', type=float, default=0.0,
                        help="Probability of vertical flip.")
    parser.add_argument('--fliplr', type=float, default=0.5,
                        help="Probability of horizontal flip.")
    parser.add_argument('--crop_fraction', type=float, default=1.0,
                        help="Center-crop fraction applied at validation/inference.")

    # --- Occlusion & Mixing Augmentation ---
    parser.add_argument('--erasing', type=float, default=0.4,
                        help="Random erasing probability (classification). Set 0.0 when "
                             "defects occupy a small image fraction.")
    parser.add_argument('--mixup', type=float, default=0.0,
                        help="MixUp probability.")
    parser.add_argument('--cutmix', type=float, default=0.0,
                        help="CutMix probability.")
    parser.add_argument('--copy_paste', type=float, default=0.0,
                        help="Copy-paste probability (segmentation).")
    parser.add_argument('--copy_paste_mode', type=str, default='flip',
                        help="Copy-paste strategy: 'flip' or 'mixup'.")
    parser.add_argument('--bgr', type=float, default=0.0,
                        help="Probability of BGR channel swap.")

    # --- Hyperparameters ---
    parser.add_argument('--freeze', type=parse_mixed_type, default=None, 
                        help="Freeze the first N layers (int) or specific layer indices (list).")
    parser.add_argument('--lr0', type=float, default=0.01, 
                        help="Initial learning rate.")
    parser.add_argument('--lrf', type=float, default=0.01, 
                        help="Final learning rate as a fraction of lr0.")
    parser.add_argument('--momentum', type=float, default=0.937, 
                        help="Momentum factor for SGD or beta1 for Adam optimizers.")
    parser.add_argument('--weight_decay', type=float, default=0.0005, 
                        help="L2 regularization term weight decay.")
    parser.add_argument('--warmup_epochs', type=float, default=3.0, 
                        help="Number of epochs for learning rate warmup.")
    parser.add_argument('--warmup_momentum', type=float, default=0.8, 
                        help="Initial momentum for the warmup phase.")
    parser.add_argument('--warmup_bias_lr', type=float, default=0.1, 
                        help="Learning rate for bias parameters during warmup.")
    parser.add_argument('--cos_lr', type=str2bool, default=False, 
                        help="Use a cosine learning rate scheduler.")

    # --- Loss Weights & Tasks ---
    parser.add_argument('--distill_model', type=str, default=None, 
                        help="Path to a teacher model checkpoint for knowledge distillation.")
    parser.add_argument('--dis', type=float, default=6.0, 
                        help="Distillation loss weight.")
    parser.add_argument('--box', type=float, default=7.5, 
                        help="Box loss weight.")
    parser.add_argument('--cls', type=float, default=0.5, 
                        help="Classification loss weight.")
    parser.add_argument('--cls_pw', type=float, default=0.0, 
                        help="Class weighting power for handling class imbalance (0.0 to 1.0).")
    parser.add_argument('--dfl', type=float, default=1.5, 
                        help="Distribution Focal Loss (DFL) weight.")
    parser.add_argument('--pose', type=float, default=12.0, 
                        help="Pose loss weight for pose estimation.")
    parser.add_argument('--kobj', type=float, default=1.0, 
                        help="Keypoint objectness weight.")
    parser.add_argument('--rle', type=float, default=1.0, 
                        help="Residual log-likelihood estimation loss weight.")
    parser.add_argument('--angle', type=float, default=1.0, 
                        help="Angle loss weight for oriented bounding boxes (OBB).")
    parser.add_argument('--nbs', type=int, default=64, 
                        help="Nominal batch size for loss normalization.")

    # --- Validation & Logging ---
    parser.add_argument('--val', type=str2bool, default=True, 
                        help="Enable validation during training.")
    parser.add_argument('--plots', type=str2bool, default=True, 
                        help="Generate and save training/validation plots.")
    parser.add_argument('--max_det', type=int, default=300, 
                        help="Maximum number of objects retained during validation.")

    return parser


if __name__ == '__main__':
    parser = get_training_parser()
    args = parser.parse_args()
    
    # Print the evaluated configuration natively
    print("Updated Training Configuration:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value} (Parsed Type: {type(value).__name__})")