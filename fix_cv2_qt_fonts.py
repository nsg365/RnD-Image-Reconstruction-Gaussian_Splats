"""
fix_cv2_qt_fonts.py
───────────────────
One-time fix for:
  QFontDatabase: Cannot find font directory .../cv2/qt/fonts

The PyPI opencv-python Qt build looks for fonts inside its own qt/fonts/
directory, but ships without any. This script creates that directory and
symlinks the system DejaVu fonts (already installed on Ubuntu/Debian) into it.

Run once, before using reconstruct_splats.py:
    python fix_cv2_qt_fonts.py
"""

import os
import sys
import glob

# ── locate cv2 package ────────────────────────────────────────────────────────
try:
    import cv2
except ImportError:
    sys.exit("ERROR: cv2 not found. Activate your venv first.")

cv2_dir    = os.path.dirname(cv2.__file__)
fonts_dir  = os.path.join(cv2_dir, "qt", "fonts")

print(f"cv2 location : {cv2_dir}")
print(f"Target fonts : {fonts_dir}")

# ── create the fonts directory if missing ─────────────────────────────────────
os.makedirs(fonts_dir, exist_ok=True)
print(f"Created      : {fonts_dir}")

# ── find system DejaVu fonts ──────────────────────────────────────────────────
SEARCH_PATHS = [
    "/usr/share/fonts/truetype/dejavu/*.ttf",
    "/usr/share/fonts/dejavu/*.ttf",
    "/usr/share/fonts/TTF/DejaVu*.ttf",
    "/usr/local/share/fonts/dejavu/*.ttf",
]

found = []
for pattern in SEARCH_PATHS:
    found.extend(glob.glob(pattern))

if not found:
    print("\nWARNING: No DejaVu fonts found in standard locations.")
    print("Install them with:  sudo apt-get install fonts-dejavu-core")
    print("Then re-run this script.")
    sys.exit(1)

print(f"\nFound {len(found)} DejaVu font file(s):")

linked = 0
for src in sorted(found):
    dst = os.path.join(fonts_dir, os.path.basename(src))
    if os.path.exists(dst) or os.path.islink(dst):
        print(f"  [skip]  {os.path.basename(src)}  (already linked)")
        continue
    try:
        os.symlink(src, dst)
        print(f"  [link]  {os.path.basename(src)}")
        linked += 1
    except OSError as e:
        print(f"  [FAIL]  {os.path.basename(src)}: {e}")

print(f"\nDone — linked {linked} font(s) → {fonts_dir}")
print("You can now run reconstruct_splats.py without the Qt font error.")