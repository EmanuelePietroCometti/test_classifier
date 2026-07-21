import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.stats as stats
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score
)
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

def split_and_save_image_dataset(
    source_dir: str | Path,
    output_dir: str | Path,
    train_size: float = 0.7,
    val_size: float = 0.2,
    test_size: float = 0.1,
    random_state: int = 42,
    grayscale: bool = False
) -> None:
    """
    Splits an image folder dataset into train, val, and test directories 
    using scikit-learn's train_test_split with stratification.
    """
    # Ensure percentages sum up to 1.0
    if not abs((train_size + val_size + test_size) - 1.0) < 1e-5:
        raise ValueError("The sum of train, val, and test sizes must equal 1.0")

    source_path = Path(source_dir)
    output_path = Path(output_dir)

    if output_path.exists():
        shutil.rmtree(output_path)

    file_paths = []
    labels = []

    # Gather all file paths and corresponding class labels from subdirectory names
    for class_dir in source_path.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            for img_path in class_dir.glob("*.*"):
                file_paths.append(img_path)
                labels.append(class_name)

    # First split: Isolate the test set from the rest of the data
    X_remain, X_test, y_remain, y_test = train_test_split(
        file_paths, 
        labels, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=labels
    )

    # Second split: Divide the remaining data into train and validation sets
    # The relative validation size must be scaled to the remaining subset
    relative_val_size = val_size / (train_size + val_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_remain, 
        y_remain, 
        test_size=relative_val_size, 
        random_state=random_state, 
        stratify=y_remain
    )

    # Dictionary mapping to simplify the physical copy process
    dataset_splits = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test)
    }

    # Physically create directories and copy images
    for split_name, (paths, cls_labels) in dataset_splits.items():
        for path, label in zip(paths, cls_labels):
            target_dir = output_path / split_name / label
            target_dir.mkdir(parents=True, exist_ok=True)
            if grayscale:
                Image.open(path).convert("L").convert("RGB").save(target_dir / path.name)
            else:
                shutil.copy(path, target_dir / path.name)

    print(f"Dataset successfully split! Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

def calculate_wilson_ci(p: float, n: int, confidence: float = 0.99) -> tuple:
    """
    Calculate the Wilson confidence interval for a binomial proportion.
    """
    if n == 0:
        return 0.0, 0.0

    # Calculate the z-score for the desired confidence level (99% default)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    z_squared = z ** 2

    # Wilson score interval formula components
    denominator = 1 + z_squared / n
    center_adjusted_probability = p + z_squared / (2 * n)
    adjusted_standard_deviation = z * np.sqrt((p * (1 - p)) / n + z_squared / (4 * (n ** 2)))

    # Calculate lower and upper bounds
    lower_bound = (center_adjusted_probability - adjusted_standard_deviation) / denominator
    upper_bound = (center_adjusted_probability + adjusted_standard_deviation) / denominator

    return lower_bound, upper_bound

def evaluate_imbalanced_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray) -> dict:
    """
    Evaluate classification metrics tailored for highly imbalanced datasets.
    
    Args:
        y_true: Ground truth labels (e.g., 0 for normal, 1 for defect).
        y_pred: Hard class predictions.
        y_scores: Predicted probabilities for the positive class (required for AUROC).
        
    Returns:
        A dictionary containing all the evaluated metrics.
    """
    # Calculate basic counts
    num_images = len(y_true)
    num_errors = int(np.sum(y_true != y_pred))

    # Calculate balanced metrics ('macro' treats all classes equally regardless of support)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    bal_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    bal_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    bal_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Calculate Area Under the Receiver Operating Characteristic Curve
    n_classes = y_scores.shape[1]
    auroc = roc_auc_score(
        y_true,
        y_scores,
        multi_class='ovr',
        average='macro',
        labels=np.arange(n_classes),
    )

    # Compute Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Calculate 99% Wilson Confidence Interval based on the Balanced Accuracy proportion
    ci_lower, ci_upper = calculate_wilson_ci(bal_acc, num_images, confidence=0.99)

    # Compile the results into a structured dictionary
    metrics = {
        "num_images": num_images,
        "num_errors": num_errors,
        "balanced_accuracy": bal_acc,
        "wilson_ci_99_lower": ci_lower,
        "wilson_ci_99_upper": ci_upper,
        "balanced_precision": bal_precision,
        "balanced_recall": bal_recall,
        "balanced_f1_score": bal_f1,
        "auroc": auroc,
        "confusion_matrix": conf_matrix.tolist()
    }

    return metrics

def print_evaluation_metrics(metrics: dict):
    """
    Prints the evaluation metrics dictionary in a clean, aligned, and readable format.
    
    Args:
        metrics: Dictionary containing the evaluation results.
    """
    print("\n" + "="*45)
    print(" " * 12 + "VALIDATION METRICS")
    print("="*45)
    
    # Print general counts
    print(f"Total Images Processed : {metrics['num_images']}")
    print(f"Total Misclassifications : {metrics['num_errors']}")
    print("-" * 45)
    
    # Print performance metrics (formatted to 4 decimal places)
    print(f"Balanced Accuracy      : {metrics['balanced_accuracy']:.4f}")
    print(f"Balanced Precision     : {metrics['balanced_precision']:.4f}")
    print(f"Balanced Recall        : {metrics['balanced_recall']:.4f}")
    print(f"Balanced F1-Score      : {metrics['balanced_f1_score']:.4f}")
    print(f"AUROC                  : {metrics['auroc']:.4f}")
    print("-" * 45)
    
    # Print Wilson Confidence Interval bounds together for clarity
    ci_lower = metrics['wilson_ci_99_lower']
    ci_upper = metrics['wilson_ci_99_upper']
    print(f"99% Wilson CI Bounds   : [{ci_lower:.4f}, {ci_upper:.4f}]")
    print("="*45 + "\n")

def save_confusion_matrix_plot(
    conf_matrix: list, 
    class_mapping: dict, 
    save_path: str = "confusion_matrix.png"
):
    """
    Generates and saves a visually appealing confusion matrix using Seaborn.
    
    Args:
        conf_matrix: The confusion matrix as a nested list or 2D NumPy array.
        class_mapping: Dictionary mapping class indices to names (e.g., model.names).
        save_path: The file path where the PNG image will be saved.
    """
    # Convert the list back to a NumPy array for Seaborn compatibility
    cm_array = np.array(conf_matrix)
    
    # Extract ordered class names based on the dictionary keys
    # This ensures the labels match the (0, 1, 2...) matrix axes correctly
    ordered_keys = sorted(class_mapping.keys())
    class_names = [class_mapping[k] for k in ordered_keys]
    
    # Initialize a matplotlib figure with a fixed size
    plt.figure(figsize=(8, 6))
    
    # Create the heatmap
    # annot=True: Displays the numbers inside the cells
    # fmt='d': Formats annotations as decimal integers 
    # cmap='Blues': Uses a sequential color map for intuitive readability
    sns.heatmap(
        cm_array, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=class_names, 
        yticklabels=class_names
    )
    
    # Set plot labels and title
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.title('Confusion Matrix')
    
    # Save the figure to disk
    # bbox_inches='tight' prevents axis labels from being cut off in the saved image
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Close the figure to free up memory (crucial if running in a loop)
    plt.close()
    
    print(f"Confusion matrix plot successfully saved to: {save_path}")