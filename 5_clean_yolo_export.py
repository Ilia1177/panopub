"""Remove annotations of a selected class and compact YOLO class IDs."""

import argparse
import json
from pathlib import Path


def clean_export(root: Path, class_name: str = "delete", apply: bool = False) -> tuple[int, int]:
    """Validate the export before removing annotations and rewriting metadata.

    Keep every image and label file, including labels that become empty.
    Return the number of removed annotations and retained label files.
    """
    classes_path = root / "classes.txt"
    notes_path = root / "notes.json"
    names = classes_path.read_text().splitlines()
    if len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("classes.txt contains empty or duplicate names")
    if class_name not in names:
        raise ValueError(f"Class {class_name!r} is absent; export may already be cleaned")
    removed_id = names.index(class_name)
    mapping = {old: new for new, old in enumerate(
        index for index in range(len(names)) if index != removed_id
    )}
    notes = json.loads(notes_path.read_text())
    categories = notes["categories"]
    if sorted((item["id"], item["name"]) for item in categories) != list(enumerate(names)):
        raise ValueError("notes.json categories do not match classes.txt")
    images_dir = root / "images"
    labels_dir = root / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise ValueError("Export must contain images/ and labels/ directories")
    images: dict[str, list[Path]] = {}
    for image in images_dir.iterdir():
        if image.is_file() and not image.name.startswith("."):
            images.setdefault(image.stem, []).append(image)
    removed_annotations = 0
    rewrites = []
    for label in sorted(labels_dir.glob("*.txt")):
        paired = images.get(label.stem, [])
        if len(paired) != 1:
            raise ValueError(f"{label}: expected exactly one matching image")
        rows = []
        for number, line in enumerate(label.read_text().splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split(maxsplit=1)
            try:
                class_id = int(fields[0])
            except ValueError as error:
                raise ValueError(f"{label}:{number}: invalid class ID") from error
            if class_id not in range(len(names)) or len(fields) != 2:
                raise ValueError(f"{label}:{number}: invalid annotation")
            if class_id == removed_id:
                removed_annotations += 1
            else:
                rows.append(f"{mapping[class_id]} {fields[1]}")
        rewrites.append((label, "\n".join(rows) + ("\n" if rows else "")))
    print(f"Annotations to remove: {removed_annotations}")
    print(f"Label files to retain: {len(rewrites)}")
    print(f"Class mapping: {mapping}")
    if apply:
        for label, content in rewrites:
            label.write_text(content)
        classes_path.write_text("\n".join(name for name in names if name != class_name) + "\n")
        notes["categories"] = [
            {**item, "id": mapping[item["id"]]}
            for item in categories if item["id"] != removed_id
        ]
        notes_path.write_text(json.dumps(notes, indent=2, ensure_ascii=False) + "\n")
        print("Export cleaned.")
    else:
        print("Dry run: no files modified. Add --apply to execute.")
    return removed_annotations, len(rewrites)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path, nargs="?", default=Path("export_label-studio"))
    parser.add_argument("--class-name", default="delete")
    parser.add_argument("--apply", action="store_true", help="Modify the export in place")
    args = parser.parse_args()
    try:
        clean_export(args.export, args.class_name, args.apply)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
