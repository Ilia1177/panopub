import os
import yaml
from ultralytics import YOLO


def main():
    # 1. Define the dataset path
    # Make sure the script is run from the parent directory of 'my_dataset'
    dataset_path = os.path.abspath("my_dataset")

    # 2. Create a YAML configuration file for YOLOv8 training
    config = {
        'path': dataset_path,
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'billboard',
            1: 'graffiti'
        }
    }

    yaml_path = "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"[INFO] Config file created successfully : {yaml_path}")

    # 3. Load the base YOLOv8n model
    print("[INFO] Loading the base YOLOv8n model...")
    model = YOLO("yolov26s.pt")

    # 4. Train the model on the MacBook Pro GPU (MPS)
    print("[INFO] Training the model on the MacBook Pro GPU (MPS)...")
    model.train(
        data=yaml_path,
        epochs=50,
        batch=4,
        workers=2,
        cache=False,
        imgsz=960,
        device="mps"  # Use "mps" for MacBook Pro GPU
    )
    print("[INFO] Training completed successfully!")
    print("[INFO] Your best model is automatically saved in : runs/detect/train/weights/best.pt")


if __name__ == "__main__":
    main()
