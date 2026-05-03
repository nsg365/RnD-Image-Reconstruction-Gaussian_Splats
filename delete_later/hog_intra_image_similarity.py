import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# ---------------- CONFIG ---------------- #
SEGMENT_ROOT = "dataset/intra_segments"   # output from segmentation step
HOG_SIZE = (128, 128)

SAVE_SIM_MATRIX = True
VISUALIZE = True
# ---------------------------------------- #

# HOG descriptor (Dalal–Triggs standard)
hog = cv2.HOGDescriptor(
    _winSize=(128, 128),
    _blockSize=(16, 16),
    _blockStride=(8, 8),
    _cellSize=(8, 8),
    _nbins=9
)

def extract_hog_features(region_dir):
    features = []
    names = []

    for fname in sorted(os.listdir(region_dir)):
        if not fname.endswith(".png"):
            continue

        path = os.path.join(region_dir, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        # safety resize (in case)
        if img.shape != HOG_SIZE:
            img = cv2.resize(img, HOG_SIZE)

        feat = hog.compute(img).flatten()
        features.append(feat)
        names.append(fname)

    return np.array(features), names


def compute_similarity(features):
    return cosine_similarity(features)


def visualize_similarity(sim_matrix, image_name):
    plt.figure(figsize=(6, 5))
    plt.imshow(sim_matrix, cmap="hot")
    plt.colorbar(label="Cosine Similarity")
    plt.title(f"Intra-image Similarity (HOG)\n{image_name}")
    plt.xlabel("Segments")
    plt.ylabel("Segments")
    plt.tight_layout()
    plt.show()


def process_image(image_dir):
    region_dir = os.path.join(image_dir, "hog_regions")
    if not os.path.exists(region_dir):
        return

    features, names = extract_hog_features(region_dir)

    if len(features) < 2:
        print("  [-] Not enough valid regions, skipping.")
        return

    sim_matrix = compute_similarity(features)

    if SAVE_SIM_MATRIX:
        out_path = os.path.join(image_dir, "hog_similarity.npy")
        np.save(out_path, sim_matrix)

    if VISUALIZE:
        visualize_similarity(sim_matrix, os.path.basename(image_dir))


def main():
    for image_name in sorted(os.listdir(SEGMENT_ROOT)):
        image_dir = os.path.join(SEGMENT_ROOT, image_name)

        if not os.path.isdir(image_dir):
            continue

        print(f"[+] Computing HOG similarity for {image_name}")
        process_image(image_dir)


if __name__ == "__main__":
    main()