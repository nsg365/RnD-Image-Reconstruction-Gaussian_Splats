import os
import cv2
import torch
import timm
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms

# ---------------- CONFIG ---------------- #

IMAGE_DIR = "dataset/images"
TARGET_FILE = "dataset/targets.txt"
RESULT_DIR = "results_global_dino"

TOP_K = 5
MODEL_RES = 224

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(RESULT_DIR, exist_ok=True)

# ---------------- MODEL ---------------- #

model = timm.create_model(
    "vit_small_patch16_224.dino",
    pretrained=True
)

model.eval().to(DEVICE)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((MODEL_RES, MODEL_RES)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# ---------------- FEATURE EXTRACTION ---------------- #

@torch.no_grad()
def extract_cls(img):

    tensor = transform(img).unsqueeze(0).to(DEVICE)

    tokens = model.forward_features(tensor)

    cls = tokens[:,0]

    return cls.cpu().numpy()[0]

# ---------------- MAIN ---------------- #

def main():

    with open(TARGET_FILE) as f:
        targets = [l.strip() for l in f if l.strip()]

    print("🔹 Precomputing embeddings...")

    embeddings = {}
    images = {}

    for fname in tqdm(os.listdir(IMAGE_DIR)):

        if not fname.lower().endswith((".jpg",".png",".jpeg")):
            continue

        path = os.path.join(IMAGE_DIR, fname)

        img = cv2.imread(path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        emb = extract_cls(img_rgb)

        embeddings[fname] = emb
        images[fname] = img

    # ---------------- QUERY ---------------- #

    for target in targets:

        print("\nProcessing", target)

        if target not in embeddings:
            print(" Not found:", target)
            continue

        query_emb = embeddings[target]

        scores = []

        for name, emb in embeddings.items():

            if name == target:
                continue

            sim = cosine_similarity(
                query_emb.reshape(1,-1),
                emb.reshape(1,-1)
            )[0][0]

            scores.append((name, sim))

        scores.sort(key=lambda x: x[1], reverse=True)

        # ---------------- SAVE ---------------- #

        qfolder = os.path.join(
            RESULT_DIR,
            target.split(".")[0]
        )

        os.makedirs(qfolder, exist_ok=True)

        # Save query
        cv2.imwrite(
            os.path.join(qfolder, "query.jpg"),
            images[target]
        )

        vis = [cv2.resize(images[target], (300,300))]

        for i, (name, score) in enumerate(scores[:TOP_K]):

            print(name, score)

            img = images[name]

            cv2.imwrite(
                os.path.join(qfolder, f"top{i+1}.jpg"),
                img
            )

            vis.append(cv2.resize(img, (300,300)))

        summary = cv2.hconcat(vis)

        cv2.imwrite(
            os.path.join(qfolder, "summary.jpg"),
            summary
        )

        print("Saved ", qfolder)


if __name__ == "__main__":
    main()
