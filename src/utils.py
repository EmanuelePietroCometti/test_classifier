import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

def split_and_save_image_dataset(
    source_dir: str | Path,
    output_dir: str | Path,
    train_size: float = 0.7,
    val_size: float = 0.2,
    test_size: float = 0.1,
    random_state: int = 42
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
            shutil.copy(path, target_dir / path.name)

    print(f"Dataset successfully split! Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

if __name__ == "__main__":
    # Execute the structured pipeline
    split_and_save_image_dataset(
        source_dir="dataset",
        output_dir="src_dataset",
        train_size=0.7,
        val_size=0.2,
        test_size=0.1
    )