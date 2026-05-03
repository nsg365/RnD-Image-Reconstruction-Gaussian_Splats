import os
import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.util import img_as_float

# ---------------- CONFIG ---------------- #
IMAGE_DIR = "dataset/images"
OUT_DIR = "dataset/intra_segments"

NUM_SEGMENTS = 40
COMPACTNESS = 35
MIN_REGION_AREA = 5000
MIN_EDGE_DENSITY = 0.02   # skip flat regions
HOG_SIZE = (128, 128)
# ---------------------------------------- #

os.makedirs(OUT_DIR, exist_ok=True)

def compute_edge_density(gray):
    edges = cv2.Canny(gray, 80, 160)
    return np.sum(edges > 0) / edges.size

def segment_image(image_path, out_base):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_float = img_as_float(img_rgb)

    # SLIC segmentation (coarse + connected)
    segments = slic(
        img_float,
        n_segments=NUM_SEGMENTS,
        compactness=COMPACTNESS,
        start_label=0,
        enforce_connectivity=True,
        min_size_factor=0.5
    )

    unique_segments = np.unique(segments)

    seg_dir = os.path.join(out_base, "hog_regions")
    os.makedirs(seg_dir, exist_ok=True)

    for seg_id in unique_segments:
        mask = (segments == seg_id).astype(np.uint8)

        if np.sum(mask) < MIN_REGION_AREA:
            continue

        # bounding box of segment
        ys, xs = np.where(mask == 1)
        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()

        crop = img[y1:y2+1, x1:x2+1]
        crop_mask = mask[y1:y2+1, x1:x2+1]

        # apply mask (remove background)
        crop[crop_mask == 0] = 0

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # filter flat regions (important for HOG)
        if compute_edge_density(gray) < MIN_EDGE_DENSITY:
            continue

        # resize to HOG window
        hog_ready = cv2.resize(gray, HOG_SIZE)

        out_path = os.path.join(seg_dir, f"region_{seg_id}.png")
        cv2.imwrite(out_path, hog_ready)

def main():
    for img_name in os.listdir(IMAGE_DIR):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(IMAGE_DIR, img_name)
        base_name = os.path.splitext(img_name)[0]
        out_base = os.path.join(OUT_DIR, base_name)

        print(f"[+] Processing {img_name}")
        segment_image(img_path, out_base)

if __name__ == "__main__":
    main()