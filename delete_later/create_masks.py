import cv2
import numpy as np
import os

IMAGE_DIR = "dataset/images"
MASK_DIR = "dataset/masks1"
TARGET_FILE = "dataset/targets.txt"

os.makedirs(MASK_DIR, exist_ok=True)

with open(TARGET_FILE, "r") as f:
    targets = [line.strip() for line in f if line.strip()]

for img_name in targets:
    img_path = os.path.join(IMAGE_DIR, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Could not load {img_name}")
        continue

    print(f"\nMasking: {img_name}")
    print("Left-click & drag → paint damaged region")
    print("ENTER → save mask | ESC → skip")

    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    def draw(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(mask, (x, y), 10, 255, -1)
        elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
            cv2.circle(mask, (x, y), 10, 255, -1)

    cv2.namedWindow("Mask Editor")
    cv2.setMouseCallback("Mask Editor", draw)

    while True:
        overlay = img.copy()
        overlay[mask == 255] = (
            0.3 * overlay[mask == 255] +
            0.7 * np.array([0, 0, 255])
        ).astype(np.uint8)

        cv2.imshow("Mask Editor", overlay)
        key = cv2.waitKey(1)

        if key == 13:
            break
        elif key == 27:
            mask = None
            break

    cv2.destroyAllWindows()

    if mask is not None:
        mask_name = img_name.rsplit(".", 1)[0] + "_mask.png"
        cv2.imwrite(os.path.join(MASK_DIR, mask_name), mask)
        print(f"Saved mask: {mask_name}")
    else:
        print(f"Skipped: {img_name}")
