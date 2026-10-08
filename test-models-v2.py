#!/usr/bin/env python3

import sys
from pathlib import Path

from ultralytics import YOLO


TEST_DIR = Path("TEST_IMG")
OUTPUT_DIR = TEST_DIR / "results"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <model1.pt> [model2.pt ...]")
        sys.exit(1)

    model_paths = [Path(path) for path in sys.argv[1:]]

    for model_path in model_paths:
        if not model_path.exists():
            print(f"Error: model not found: {model_path}")
            sys.exit(1)

    if not TEST_DIR.is_dir():
        print(f"Error: directory not found: {TEST_DIR}")
        sys.exit(1)

    images = [
        p for p in TEST_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not images:
        print(f"No images found in {TEST_DIR}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Images: {len(images)}")
    print(f"Models: {len(model_paths)}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    for model_path in model_paths:
        print("=" * 60)
        print(f"Model: {model_path}")
        print("=" * 60)

        model = YOLO(model_path)

        for image in images:
            print(f"  → {image.name}")

            results = model.predict(
                source=str(image),
                conf=0.25,
                save=False,
                verbose=False,
            )

            for result in results:
                # Save manually with model name in filename
                output_name = (
                    f"{image.stem}_{model_path.stem}{image.suffix}"
                )
                output_path = OUTPUT_DIR / output_name

                plotted = result.plot()
                from PIL import Image

                Image.fromarray(plotted[..., ::-1]).save(output_path)

                for box in result.boxes:
                    cls = int(box.cls[0])
                    confidence = float(box.conf[0])
                    name = model.names[cls]

                    print(f"      {name}: {confidence:.2f}")

        print()

    print("Done.")


if __name__ == "__main__":
    main()
