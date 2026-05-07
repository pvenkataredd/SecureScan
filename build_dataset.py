"""
build_dataset.py
Processes QR code images → extracts lexical + VGG19 features → saves feature_matrix.npy
Run once: python build_dataset.py
"""

import sys
import time

print("[1/7] Importing libraries (cv2, numpy, re, math, urllib)...")
try:
    import cv2
    import numpy as np
    import re
    import math
    import urllib.parse
    from pathlib import Path
    print("      ✓ Standard libraries imported successfully")
except ImportError as e:
    print(f"      ✗ FAILED: {e}")
    print("      Fix: make sure you are in .venv311 and ran pip install opencv-python numpy")
    sys.exit(1)

print("[2/7] Importing pyzbar...")
try:
    from pyzbar.pyzbar import decode
    print("      ✓ pyzbar imported successfully")
except ImportError as e:
    print(f"      ✗ FAILED: {e}")
    print("      Fix: pip install pyzbar  AND  brew install zbar")
    sys.exit(1)

print("[3/7] Importing VGG19...")
try:
    from visual_extraction import extract_features as extract_visual
    print("      ✓ visual_extraction imported successfully")
except ImportError as e:
    print(f"      ✗ FAILED: {e}")
    print("      Fix: pip install torch torchvision pillow")
    print("           make sure visual_extraction.py is in the same folder")
    sys.exit(1)

print("[4/7] Defining lexical feature extractor...")
try:
    SUSPICIOUS_KEYWORDS = [
        "login","signin","verify","secure","account","update","banking",
        "confirm","password","credential","free","prize","winner","click",
        "urgent","alert","suspended","unusual"
    ]

    def safe_parse(url):
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", url):
            url = "https://" + url
        return urllib.parse.urlparse(url), url

    def shannon_entropy(s):
        if not s: return 0.0
        freq = {}
        for c in s: freq[c] = freq.get(c, 0) + 1
        n = len(s)
        return -sum((v/n)*math.log2(v/n) for v in freq.values())

    def extract_lexical(raw_url):
        p, url = safe_parse(raw_url)
        return np.array([
            len(url), len(p.netloc), len(p.path), len(p.query),
            int(p.scheme == "https"), int(p.scheme == "http"),
            int(p.scheme not in ("http","https","")),
            url.count("."), url.count("-"), url.count("_"),
            url.count("/"), url.count("?"), url.count("="),
            url.count("&"), url.count("@"), url.count("%"),
            sum(c.isdigit() for c in url),
            int(bool(re.search(r"redirect|redir|url=|link=|goto=|next=", url, re.I))),
            int(bool(re.search(r"(\d{1,3}\.){3}\d{1,3}", p.netloc))),
            max(0, p.netloc.count(".") - 1),
            int(bool(p.port)),
            shannon_entropy(p.netloc),
            sum(k in url.lower() for k in SUSPICIOUS_KEYWORDS),
            len(re.findall(r"%[0-9A-Fa-f]{2}", url)),
            int("%25" in url),
        ], dtype=np.float32)

    # Quick sanity check on the lexical extractor
    test_lex = extract_lexical("https://google.com")
    assert len(test_lex) == 25, f"Expected 25 lexical features, got {len(test_lex)}"
    print(f"      ✓ Lexical extractor works — outputs {len(test_lex)} features")
except Exception as e:
    print(f"      ✗ FAILED: {e}")
    sys.exit(1)

print("[5/7] Checking VGG19 with a dummy image...")
try:
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    test_vis = extract_visual(dummy_img)
    assert len(test_vis) == 1024, f"Expected 1024 visual features, got {len(test_vis)}"
    print(f"      ✓ VGG19 works — outputs {len(test_vis)} features")
except Exception as e:
    print(f"      ✗ FAILED: {e}")
    print("      Fix: check visual_extraction.py for errors")
    sys.exit(1)

print("[6/7] Checking dataset folder structure...")
DATASET_DIR = Path("QR codes")
OUTPUT_FILE = Path("feature_matrix.npy")

for folder_name in ["benign", "malicious"]:
    folder = DATASET_DIR / folder_name / folder_name
    if not folder.exists():
        print(f"      ✗ FAILED: folder not found: {folder}")
        print(f"      Fix: make sure your dataset is at 'QR codes/benign/benign/' and 'QR codes/malicious/malicious/'")
        sys.exit(1)
    images = list(folder.glob("*.png")) + list(folder.glob("*.jpg"))
    print(f"      ✓ {folder_name}: found {len(images)} images at {folder}")

print("[7/7] Starting feature extraction loop...")
print("      Progress is printed every 500 images.")
print("      Safe to Ctrl+C and restart — but you will lose progress.\n")

rows = []
total_skipped_load   = 0
total_skipped_decode = 0
total_skipped_error  = 0
start_time = time.time()

for label_int, folder_name in [(0, "benign"), (1, "malicious")]:
    folder = DATASET_DIR / folder_name / folder_name
    images = list(folder.glob("*.png")) + list(folder.glob("*.jpg"))

    print(f"  ── {folder_name.upper()} (label={label_int}) ── {len(images)} images")
    class_start = time.time()

    for i, img_path in enumerate(images):

        # ── Load image ────────────────────────────────────────────────
        img = cv2.imread(str(img_path))
        if img is None:
            total_skipped_load += 1
            if total_skipped_load <= 5:
                print(f"      [SKIP] Could not load image: {img_path.name}")
            continue

        # ── Decode QR → URL ───────────────────────────────────────────
        try:
            decoded = decode(img)
        except Exception as e:
            total_skipped_error += 1
            if total_skipped_error <= 5:
                print(f"      [SKIP] pyzbar error on {img_path.name}: {e}")
            continue

        if not decoded:
            total_skipped_decode += 1
            if total_skipped_decode <= 5:
                print(f"      [SKIP] No QR found in image: {img_path.name}")
            continue

        url = decoded[0].data.decode("utf-8")

        # ── Extract features ──────────────────────────────────────────
        try:
            lex_vec = extract_lexical(url)          # (25,)
            vis_vec = extract_visual(img).astype(np.float32)  # (1024,)
            fused   = np.concatenate([lex_vec, vis_vec])      # (1049,)
        except Exception as e:
            total_skipped_error += 1
            if total_skipped_error <= 5:
                print(f"      [SKIP] Feature extraction error on {img_path.name}: {e}")
            continue

        rows.append({"features": fused, "label": label_int})

        # ── Progress report every 500 images ─────────────────────────
        if i > 0 and i % 500 == 0:
            elapsed   = time.time() - class_start
            rate      = i / elapsed
            remaining = (len(images) - i) / rate if rate > 0 else 0
            print(f"      [{folder_name}] {i}/{len(images)} done | "
                  f"{rate:.1f} img/s | "
                  f"~{remaining/60:.1f} min remaining | "
                  f"{len(rows)} rows saved so far")

    class_elapsed = time.time() - class_start
    print(f"  ── {folder_name.upper()} done in {class_elapsed/60:.1f} min "
          f"| {len([r for r in rows if r['label']==label_int])} rows collected\n")

# ── Summary ───────────────────────────────────────────────────────────────────
total_elapsed = time.time() - start_time
print("=" * 60)
print(f"  Total rows collected : {len(rows)}")
print(f"  Skipped (load fail)  : {total_skipped_load}")
print(f"  Skipped (no QR)      : {total_skipped_decode}")
print(f"  Skipped (other error): {total_skipped_error}")
print(f"  Total time           : {total_elapsed/60:.1f} min")
print("=" * 60)

if len(rows) == 0:
    print("  ✗ No rows collected — something went wrong. Check skip messages above.")
    sys.exit(1)

print(f"\nSaving feature matrix to {OUTPUT_FILE} ...")
np.save(OUTPUT_FILE, rows)
print(f"✓ Saved {len(rows)} rows → {OUTPUT_FILE}")
print(f"\nNext step: python classifier.py feature_matrix.npy")
