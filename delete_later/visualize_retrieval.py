# import cv2
# import matplotlib.pyplot as plt
# import numpy as np
# import os


# IMAGE_DIR = "dataset/images"
# MASK_DIR = "dataset/masks1"
# TARGET_FILE = "dataset/targets.txt"
# TOP_K = 5

# # LOAD RETRIEVAL RESULTS

# from tqdm import tqdm

# MAX_SPLATS = 300
# MAX_DIST = 50

# sift = cv2.SIFT_create()

# def extract_gaussian_splats(image, mask=None):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     keypoints = sift.detect(gray, None)

#     splats = []
#     for kp in keypoints:
#         x, y = int(kp.pt[0]), int(kp.pt[1])
#         if mask is not None and mask[y, x] == 255:
#             continue
#         sigma = kp.size / 2.0
#         splats.append((x, y, sigma, kp.response))

#     splats.sort(key=lambda s: s[3], reverse=True)
#     return splats[:MAX_SPLATS]

# def splat_similarity(A, B):
#     score = 0.0
#     for xa, ya, sa, wa in A:
#         best = 0.0
#         for xb, yb, sb, wb in B:
#             if abs(xa - xb) > MAX_DIST or abs(ya - yb) > MAX_DIST:
#                 continue
#             d2 = (xa - xb)**2 + (ya - yb)**2
#             sigma = sa + sb + 1e-6
#             val = np.exp(-d2 / (2 * sigma * sigma)) * wa * wb
#             best = max(best, val)
#         score += best
#     return score

# # LOAD DATA
# with open(TARGET_FILE, "r") as f:
#     targets = [l.strip() for l in f if l.strip()]

# all_images = [
#     f for f in os.listdir(IMAGE_DIR)
#     if f.lower().endswith((".jpg", ".png", ".jpeg"))
# ]

# # Precompute source splats
# print("Precomputing source splats...")
# source_splats = {}
# for name in tqdm(all_images):
#     img = cv2.imread(os.path.join(IMAGE_DIR, name))
#     if img is not None:
#         source_splats[name] = extract_gaussian_splats(img)


# # VISUALIZATION
# for target in targets[:5]:   # visualize only first 5 targets
#     img = cv2.imread(os.path.join(IMAGE_DIR, target))
#     mask = cv2.imread(
#         os.path.join(MASK_DIR, target.rsplit(".",1)[0] + "_mask.png"), 0
#     )

#     target_splats = extract_gaussian_splats(img, mask)

#     scores = []
#     for src_name, src_splats in source_splats.items():
#         if src_name == target:
#             continue
#         score = splat_similarity(target_splats, src_splats)
#         scores.append((src_name, score))

#     scores.sort(key=lambda x: x[1], reverse=True)
#     top_imgs = scores[:TOP_K]

#     # Plot
#     fig, axes = plt.subplots(1, TOP_K + 1, figsize=(18, 4))

#     # Target + mask overlay
#     overlay = img.copy()
#     overlay[mask == 255] = (0.3 * overlay[mask == 255] + 0.7 * np.array([255,0,0])).astype(np.uint8)
#     axes[0].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
#     axes[0].set_title("Target + Mask")
#     axes[0].axis("off")

#     for i, (name, score) in enumerate(top_imgs):
#         simg = cv2.imread(os.path.join(IMAGE_DIR, name))
#         axes[i+1].imshow(cv2.cvtColor(simg, cv2.COLOR_BGR2RGB))
#         axes[i+1].set_title(f"{name}\nscore={score:.3f}")
#         axes[i+1].axis("off")

#     plt.tight_layout()
#     plt.show()


import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.ndimage as ndimage
from tqdm import tqdm

# --- CONFIGURATION ---
IMAGE_DIR = "dataset/images"
MASK_DIR = "dataset/masks1"   
TARGET_FILE = "dataset/targets.txt"
TOP_K = 5
MAX_SPLATS = 300
MAX_DIST = 50

# --- HESSIAN OF GAUSSIAN (HoG) DETECTOR ---
def detect_hessian_blobs(image_gray, mask=None, threshold=2000):
    splats = []
    scales = [1.0, 3.0, 5.0, 7.0] 
    
    for sigma in scales:
        k = int(6 * sigma) | 1 
        if k < 3: k = 3
        
        blurred = cv2.GaussianBlur(image_gray, (k, k), sigma)
        Ixx = cv2.Sobel(blurred, cv2.CV_32F, 2, 0, ksize=3)
        Iyy = cv2.Sobel(blurred, cv2.CV_32F, 0, 2, ksize=3)
        Ixy = cv2.Sobel(blurred, cv2.CV_32F, 1, 1, ksize=3)
        
        det_H = (Ixx * Iyy) - (Ixy ** 2)
        local_max = ndimage.maximum_filter(det_H, size=5)
        peaks = (det_H == local_max) & (det_H > threshold)
        
        y_coords, x_coords = np.where(peaks)
        
        for y, x in zip(y_coords, x_coords):
            if mask is not None and mask[y, x] == 255:
                continue
            weight = det_H[y, x]
            splats.append((int(x), int(y), sigma, float(weight)))
    return splats

def extract_gaussian_splats(image, mask=None):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raw_splats = detect_hessian_blobs(gray, mask, threshold=5000)
    raw_splats.sort(key=lambda s: s[3], reverse=True)
    return raw_splats[:MAX_SPLATS]

# --- SIMILARITY ---
def splat_similarity(A, B):
    score = 0.0
    for xa, ya, sa, wa in A:
        best = 0.0
        for xb, yb, sb, wb in B:
            if abs(xa - xb) > MAX_DIST or abs(ya - yb) > MAX_DIST:
                continue
            d2 = (xa - xb)**2 + (ya - yb)**2
            sigma = sa + sb + 1e-6
            val = np.exp(-d2 / (2 * sigma * sigma)) * wa * wb
            best = max(best, val)
        score += best
    return score

# --- MAIN VISUALIZATION ---
if __name__ == "__main__":
    with open(TARGET_FILE, "r") as f:
        targets = [l.strip() for l in f if l.strip()]

    all_images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".png"))]

    # Precompute source splats
    print("Precomputing source splats (HoG)...")
    source_splats = {}
    for name in tqdm(all_images):
        img = cv2.imread(os.path.join(IMAGE_DIR, name))
        if img is not None:
            source_splats[name] = extract_gaussian_splats(img)

    # Visualize Loop
    for target in targets:  # Iterates through ALL targets
        print(f"Processing {target}...")
        img = cv2.imread(os.path.join(IMAGE_DIR, target))
        mask_path = os.path.join(MASK_DIR, target.rsplit(".",1)[0] + "_mask.png")
        
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, 0)
        else:
            print(f"Mask not found for {target}")
            mask = None

        target_splats = extract_gaussian_splats(img, mask)

        scores = []
        for src_name, src_splats in source_splats.items():
            if src_name == target:
                continue
            score = splat_similarity(target_splats, src_splats)
            scores.append((src_name, score))

        # --- NORMALIZATION LOGIC ---
        scores.sort(key=lambda x: x[1], reverse=True)
        
        if scores:
            max_score = scores[0][1] # The highest score in the batch
            if max_score == 0: max_score = 1.0 # Prevent div by zero
        else:
            max_score = 1.0

        top_imgs = scores[:TOP_K]

        # Plot
        fig, axes = plt.subplots(1, TOP_K + 1, figsize=(18, 4))
        
        # 1. Plot Target + Red Mask
        overlay = img.copy()
        if mask is not None:
            # Red overlay for damaged regions
            overlay[mask == 255] = (0.3 * overlay[mask == 255] + 0.7 * np.array([255,0,0])).astype(np.uint8)
        
        axes[0].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"Target: {target}")
        axes[0].axis("off")

        # 2. Plot Retrieved Images + Blue Mask (if available)
        for i, (name, raw_score) in enumerate(top_imgs):
            simg = cv2.imread(os.path.join(IMAGE_DIR, name))
            
            # Check if retrieved image also has a mask
            ret_mask_path = os.path.join(MASK_DIR, name.rsplit(".",1)[0] + "_mask.png")
            if os.path.exists(ret_mask_path):
                ret_mask = cv2.imread(ret_mask_path, 0)
                # Blue overlay for retrieved masks
                simg[ret_mask == 255] = (0.3 * simg[ret_mask == 255] + 0.7 * np.array([0,0,255])).astype(np.uint8)

            # Normalize Score (0 to 100)
            norm_score = (raw_score / max_score) * 100.0

            axes[i+1].imshow(cv2.cvtColor(simg, cv2.COLOR_BGR2RGB))
            axes[i+1].set_title(f"{name}\nScore: {norm_score:.1f}")
            axes[i+1].axis("off")

        plt.tight_layout()
        plt.show()