import yaml
from ultralytics import YOLO
import os

def train(train_args):
    model_path = train_args.pop('model', 'yolo26n-cls.pt')

    if train_args.get('data') is None:
        raise ValueError("You must specify a dataset path using the --data argument")
    

    print(f"Loading model from: {model_path}")

    model = YOLO(model_path)

    print("Starting training with the following configuration:")
    for k,v in train_args.items():
        print(f"    {k}: {v}")

    model.train(**train_args)

    print("Training completed successfully!")

    return model.trainer.save_dir