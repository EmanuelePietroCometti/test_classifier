from src.parser import get_training_parser
from src.eval import evaluate
from src.train import train
from src.utils import print_evaluation_metrics, save_confusion_matrix_plot, split_and_save_image_dataset
import os
from ultralytics import YOLO

TRAIN_SIZE = 0.7
VAL_SIZE = 0.2
TEST_SIZE = 0.1

def main():
    
    parser = get_training_parser()
    args = parser.parse_args()

    train_args = vars(args)

    src_dataset = train_args.pop("src_dataset", "src_dataset")
    val_dataset = train_args.pop("val_dataset", "val_dataset")
    target_dataset = train_args.get("data", "dataset")

    # Split the raw dataset into train, validation and test split
    split_and_save_image_dataset(
        src_dataset, 
        target_dataset, 
        train_size=TRAIN_SIZE, 
        val_size=VAL_SIZE, 
        test_size=TEST_SIZE
        )

    # Train the model
    save_dir = train(train_args)

    #Evaluate the model
    best_weights = os.path.join(save_dir, "weights", "best.pt")
    metrics, class_mapping = evaluate(
        model_path=best_weights,
        dataset_dir=val_dataset
    )

    # Print metrics
    print_evaluation_metrics(metrics=metrics)

    # Save confusion matrix
    cm_plot_path = os.path.join(save_dir, "confusion_matrix.png")
    save_confusion_matrix_plot(
        conf_matrix=metrics.get('confusion_matrix'),
        class_mapping=class_mapping,
        save_path=cm_plot_path
    )

if __name__ == "__main__":
    main()