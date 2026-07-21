import ultralytics.utils.benchmarks as benchmarks
import ultralytics.engine.exporter as exporter
from ultralytics.utils.benchmarks import benchmark

_original_export_formats = exporter.export_formats

def custom_benchmark_formats():
    """
    Creates a filtered format list exclusively for the benchmark loop.
    Keeps only the PyTorch baseline ('-') and OpenVINO ('openvino').
    """
    data = _original_export_formats()
    target_args = {'-', 'openvino', 'onnx'}
    
    valid_indices = [i for i, arg in enumerate(data['Argument']) if arg in target_args]
    
    filtered_data = {}
    for key, values in data.items():
        filtered_data[key] = tuple(values[i] for i in valid_indices)
        
    return filtered_data

benchmarks.export_formats = custom_benchmark_formats

def main():
    config = {
        "model": "yolo26n-cls.pt",
        "data": "mnist160",
        "imgsz": 224,
        "half": False,
        "device": "cpu",
        "verbose": True
    }

    print(f"Starting benchmark for {config['model']} on {config['device'].upper()}...")
    
    benchmark(**config)

if __name__ == "__main__":
    main()