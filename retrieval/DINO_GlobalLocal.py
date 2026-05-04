import os
import cv2
import csv
import torch
import timm
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms

# ---------------- CONFIG ---------------- #

@dataclass
class Config:
    image_dir: str = "dataset/images"
    mask_dir: str = "dataset/masks1"   # optional (can be empty)
    target_file: str = "dataset/targets.txt"
    result_root: str = "results_dino_experiments"

    model_name: str = "vit_small_patch16_224.dino"
    model_res: int = 224
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    top_k: int = 5

cfg = Config()
os.makedirs(cfg.result_root, exist_ok=True)

# ---------------- MODEL ---------------- #

model = timm.create_model(cfg.model_name, pretrained=True)
model.eval().to(cfg.device)

base_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((cfg.model_res, cfg.model_res)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- UTILS ---------------- #

def maybe_apply_mask(img_rgb, mask_gray):
    if mask_gray is None:
        return img_rgb
    # assume 255 = hole/damaged region → zero it out
    m = (mask_gray == 255)
    out = img_rgb.copy()
    out[m] = 0
    return out

def get_mask_for(fname):
    base = fname.rsplit(".", 1)[0]
    path = os.path.join(cfg.mask_dir, base + "_mask.png")
    if os.path.exists(path):
        return cv2.imread(path, 0)
    return None

@torch.no_grad()
def forward_tokens(img_rgb):
    t = base_tf(img_rgb).unsqueeze(0).to(cfg.device)
    tokens = model.forward_features(t)   # [1, 1+N, D]
    return tokens

def pool_tokens(tokens, mode="cls"):
    if mode == "cls":
        return tokens[:, 0]                # [1, D]
    elif mode == "mean":
        return tokens[:, 1:].mean(dim=1)  # exclude CLS
    else:
        raise ValueError(mode)

def normalize(v):
    n = np.linalg.norm(v) + 1e-8
    return v / n

def cosine(a, b):
    return float(cosine_similarity(a.reshape(1,-1), b.reshape(1,-1))[0][0])

def multi_scale_embedding(img_rgb, pooling="cls", use_mask=False):
    # two scales (cheap but useful)
    scales = [1.0, 0.75]
    embs = []
    for s in scales:
        if s != 1.0:
            h, w = img_rgb.shape[:2]
            resized = cv2.resize(img_rgb, (int(w*s), int(h*s)))
        else:
            resized = img_rgb

        tokens = forward_tokens(resized)
        emb = pool_tokens(tokens, pooling).cpu().numpy()[0]
        embs.append(emb)

    emb = np.mean(embs, axis=0)
    return emb

# ---------------- DATA ---------------- #

def load_images():
    images = {}
    for f in os.listdir(cfg.image_dir):
        if f.lower().endswith((".jpg",".png",".jpeg")):
            p = os.path.join(cfg.image_dir, f)
            im = cv2.imread(p)
            if im is None: continue
            images[f] = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return images

# ---------------- EXPERIMENTS ---------------- #

def run_experiment(name, images, targets,
                   pooling="cls",
                   normalize_emb=True,
                   use_mask=False,
                   use_multiscale=False):

    out_dir = os.path.join(cfg.result_root, name)
    os.makedirs(out_dir, exist_ok=True)

    # precompute embeddings
    embeddings = {}
    for fname, img in tqdm(images.items(), desc=f"[{name}] embed"):
        m = get_mask_for(fname) if use_mask else None
        proc = maybe_apply_mask(img, m)

        if use_multiscale:
            emb = multi_scale_embedding(proc, pooling=pooling)
        else:
            tokens = forward_tokens(proc)
            emb = pool_tokens(tokens, pooling).cpu().numpy()[0]

        if normalize_emb:
            emb = normalize(emb)

        embeddings[fname] = emb

    # evaluation + saving
    rows = []
    for target in targets:
        if target not in embeddings:
            continue

        q_emb = embeddings[target]
        scores = []

        for name2, emb in embeddings.items():
            if name2 == target: continue
            s = cosine(q_emb, emb)
            scores.append((name2, s))

        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[:cfg.top_k]

        # save visuals
        qfolder = os.path.join(out_dir, target.split(".")[0])
        os.makedirs(qfolder, exist_ok=True)

        # read BGR again for saving
        q_bgr = cv2.imread(os.path.join(cfg.image_dir, target))
        cv2.imwrite(os.path.join(qfolder, "query.jpg"), q_bgr)

        vis = [cv2.resize(q_bgr, (300,300))]
        for i,(n, s) in enumerate(top):
            im = cv2.imread(os.path.join(cfg.image_dir, n))
            cv2.imwrite(os.path.join(qfolder, f"top{i+1}.jpg"), im)
            vis.append(cv2.resize(im, (300,300)))
            rows.append([name, target, n, float(s), i+1])

        summary = cv2.hconcat(vis)
        cv2.imwrite(os.path.join(qfolder, "summary.jpg"), summary)

    # save CSV log
    csv_path = os.path.join(out_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment","target","match","score","rank"])
        w.writerows(rows)

    print(f"✔ Done {name} → {out_dir}")

# ---------------- MAIN ---------------- #

def main():
    with open(cfg.target_file) as f:
        targets = [l.strip() for l in f if l.strip()]

    images = load_images()

    # a small, real ablation grid
    experiments = [
        dict(name="baseline_cls", pooling="cls", normalize_emb=True,  use_mask=False, use_multiscale=False),
        dict(name="mean_pool",    pooling="mean",normalize_emb=True,  use_mask=False, use_multiscale=False),
        dict(name="no_norm",      pooling="cls", normalize_emb=False, use_mask=False, use_multiscale=False),
        dict(name="multiscale",   pooling="cls", normalize_emb=True,  use_mask=False, use_multiscale=True),
        dict(name="mask_aware",   pooling="cls", normalize_emb=True,  use_mask=True,  use_multiscale=False),
    ]

    for exp in experiments:
        run_experiment(**exp, images=images, targets=targets)

if __name__ == "__main__":
    main()
