import os
import glob
import numpy as np
from ultralytics import YOLO
from .utils import evaluate_imbalanced_predictions

def evaluate(
    model_path: str, 
    dataset_dir: str
) -> tuple:
    """
    Run inference using a YOLO classification model by dynamically extracting 
    class mappings directly from the model weights.
    
    Args:
        model_path: Path to the trained YOLO classification weights (e.g., 'best.pt').
        dataset_dir: Path to the validation dataset.
        
    Returns:
        Dictionary with the calculated metrics.
    """
    print("Starting evaluation...")
    # Load the trained YOLO classification model
    model = YOLO(model_path)
    
    # Extract the internal class mapping directly from the .pt file
    class_mapping = model.names
    print(f"Extracted class mapping from model: {class_mapping}")
    
    y_true = []
    y_pred = []
    y_scores = []
    
    # Iterate through the extracted dictionary (label is int, class_name is str)
    for label, class_name in class_mapping.items():
        folder_path = os.path.join(dataset_dir, class_name)
        
        # Add a safety check in case the validation set is missing a class folder
        if not os.path.exists(folder_path):
            print(f"Warning: Folder '{class_name}' not found in {dataset_dir}. Skipping.")
            continue
            
        image_paths = glob.glob(os.path.join(folder_path, "*.bmp")) + \
                      glob.glob(os.path.join(folder_path, "*.png"))
        
        if not image_paths:
            continue
            
        # Run batched inference to optimize execution time
        results = model.predict(source=image_paths, stream=True, verbose=False)
        
        for result in results:
            probs = result.probs.data.cpu().numpy()
            predicted_class = result.probs.top1
            
            # Extract probability for the positive class (assuming index 1 is the anomaly)
            positive_class_score = probs[1] if len(probs) > 1 else 0.0
            
            y_true.append(label)
            y_pred.append(predicted_class)
            y_scores.append(positive_class_score)
            
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    y_scores_np = np.array(y_scores)
    
    # Evaluate imbalanced metrics
    metrics = evaluate_imbalanced_predictions(y_true_np, y_pred_np, y_scores_np)
    print("Evaluation finished!")
    return metrics, class_mapping