import torch, glob, os
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection

proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
queries = [["a billboard", "an advertising poster", "an advertising sign"]]

os.makedirs("dataset/images", exist_ok=True)
os.makedirs("dataset/labels", exist_ok=True)

for path in glob.glob("training_pictures/*.jpg"):
    name = os.path.splitext(os.path.basename(path))[0]
    if os.path.exists(f"dataset/labels/{name}.txt"):
        continue
    img = Image.open(path)
    print(name, img.size, flush=True)      # shows which image is slow
    img.draft("RGB", (2000, 2000))
    img = img.convert("RGB")
    img.thumbnail((2000, 2000))
    w, h = img.size
    inputs = proc(text=queries, images=img, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    side = max(w, h)
    res = proc.post_process_grounded_object_detection(
        out, threshold=0.3, target_sizes=torch.tensor([[side, side]])
    )[0]

    lines = []
    for box in res["boxes"].tolist():
        x0, y0, x1, y1 = box
        x0, x1 = max(0, x0), min(w, x1)   # clip to the real image
        y0, y1 = max(0, y0), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        xc, yc = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
        bw, bh = (x1 - x0) / w, (y1 - y0) / h
        lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    os.system(f"cp '{path}' dataset/images/{name}.jpg")
    with open(f"dataset/labels/{name}.txt", "w") as f:
        f.write("\n".join(lines))


