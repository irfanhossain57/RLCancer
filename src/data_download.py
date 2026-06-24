"""
src/data_download.py
====================
Downloads all required datasets into data/ folder.

CHANGES FROM ORIGINAL:
  - GDSC2 version updated: 15Oct19 → 27Oct23
  - GDSC1 download added (needed for Imatinib, Panobinostat, Quizartinib)
  - GEO GSE220112 FTP download with .gz decompression
  - Cleaner progress reporting

Usage:
    python src/data_download.py              # download everything
    python src/data_download.py --pbmc       # PBMC only (37 MB, pipeline test)
    python src/data_download.py --gdsc       # GDSC1 + GDSC2 only
    python src/data_download.py --geo        # GEO cancer files only (~420 MB)
"""

import os
import sys
import gzip
import shutil
import argparse
import urllib.request
import urllib.error
import yaml

# ── Config ───────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")

with open(os.path.join(PROJECT_ROOT, "configs", "config.yaml")) as f:
    cfg = yaml.safe_load(f)


# ── Progress bar ─────────────────────────────────────────────
def _progress(block, block_size, total):
    done = min(block * block_size, total) if total > 0 else block * block_size
    mb   = done / 1_048_576
    if total > 0:
        pct  = done / total * 100
        tot  = total / 1_048_576
        bar  = "█" * int(30 * done / total) + "░" * (30 - int(30 * done / total))
        print(f"\r  [{bar}] {pct:5.1f}%  {mb:.1f}/{tot:.1f} MB", end="", flush=True)
    else:
        print(f"\r  Downloaded {mb:.1f} MB...", end="", flush=True)


# ── Core download helper ──────────────────────────────────────
def download_file(url, dest_path, label, compressed=False, gz_dest=None):
    """
    Download url → dest_path, with optional .gz decompression.
    Skips entirely if dest_path already exists.
    Returns True on success, False on failure.
    """
    if os.path.exists(dest_path):
        mb = os.path.getsize(dest_path) / 1_048_576
        print(f"  [SKIP]  {os.path.basename(dest_path)} already exists ({mb:.0f} MB)")
        return True

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"\n  Downloading: {label}")
    print(f"  URL:  {url}")

    try:
        if compressed:
            # Download to .gz temp path then decompress
            tmp = gz_dest or (dest_path + ".gz")
            urllib.request.urlretrieve(url, tmp, reporthook=_progress)
            print()
            print("  Decompressing...", end="", flush=True)
            with gzip.open(tmp, "rb") as f_in, open(dest_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(tmp)
            print(" done.")
        else:
            urllib.request.urlretrieve(url, dest_path, reporthook=_progress)
            print()

        mb = os.path.getsize(dest_path) / 1_048_576
        print(f"  [OK]    {os.path.basename(dest_path)} ({mb:.0f} MB)")
        return True

    except urllib.error.HTTPError as e:
        print(f"\n  [ERROR] HTTP {e.code}: {e.reason}")
        return False
    except urllib.error.URLError as e:
        print(f"\n  [ERROR] Network: {e.reason}")
        return False
    except Exception as e:
        print(f"\n  [ERROR] {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def verify(path, min_mb, label):
    if not os.path.exists(path):
        print(f"  [MISSING] {label}: {path}")
        return False
    mb = os.path.getsize(path) / 1_048_576
    if mb < min_mb:
        print(f"  [CORRUPT] {label} — {mb:.1f} MB (expected >{min_mb} MB)")
        return False
    print(f"  [OK]    {label} ({mb:.0f} MB)")
    return True


# ── GEO manual instructions ───────────────────────────────────
def print_geo_manual():
    print()
    print("  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  MANUAL DOWNLOAD — GEO GSE220112")
    print("  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()
    print("  If auto-download failed, do this in your browser:")
    print("  1. Go to: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE220112")
    print("  2. Scroll to 'Supplementary files'")
    print("  3. Download both .h5.gz files:")
    print("       GSE220112_RS411_DMSO10h_filtered_feature_bc_matrix.h5.gz")
    print("       GSE220112_RS411_Wee1i10h_filtered_feature_bc_matrix.h5.gz")
    print()
    print("  4. Decompress:")
    print("       Windows: right-click → Extract (or 7-Zip)")
    print("       Linux/Mac: gunzip GSE220112_RS411_*.h5.gz")
    print()
    print("  5. Move both .h5 files into:")
    print(f"       {os.path.join(DATA_DIR, 'gse220112', '')}")
    print()
    print("  NOTE: For Kaggle training, upload the .h5 files directly as")
    print("        a private Kaggle dataset — do NOT process locally.")
    print("  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()


# ════════════════════════════════════════════════════════════
# DATASET 1 — PBMC Multiome 3K (local pipeline test)
# ════════════════════════════════════════════════════════════
def download_pbmc():
    print("\n" + "=" * 58)
    print("DATASET 1: PBMC Multiome 3K  (local pipeline test)")
    print("=" * 58)
    dest = os.path.join(DATA_DIR, "pbmc_multiome_3k", "filtered_feature_bc_matrix.h5")
    ok = download_file(
        url=cfg["pbmc"]["download_url"],
        dest_path=dest,
        label="PBMC multiome 3k filtered_feature_bc_matrix.h5",
    )
    if not ok:
        print("\n  MANUAL FALLBACK:")
        print("  1. Go to: https://www.10xgenomics.com/datasets/"
              "pbmc-granulocyte-sorted-3-k-1-standard-2-0-0")
        print("  2. Download: pbmc_granulocyte_sorted_3k_filtered_feature_bc_matrix.h5")
        print(f"  3. Copy to:  {dest}")
    return ok


# ════════════════════════════════════════════════════════════
# DATASET 2 — GDSC2 2023  (Venetoclax + Dasatinib)
# ════════════════════════════════════════════════════════════
def download_gdsc2():
    print("\n" + "=" * 58)
    print("DATASET 2: GDSC2 Drug Sensitivity (2023 version)")
    print("=" * 58)
    dest = os.path.join(DATA_DIR, "gdsc2", "GDSC2_fitted_dose_response_27Oct23.xlsx")
    ok = download_file(
        url=cfg["gdsc2"]["download_url"],
        dest_path=dest,
        label="GDSC2_fitted_dose_response_27Oct23.xlsx",
    )
    if not ok:
        print("\n  MANUAL FALLBACK:")
        print("  1. Go to: https://www.cancerrxgene.org/downloads/bulk_download")
        print("  2. Download: GDSC2_fitted_dose_response_27Oct23.xlsx")
        print(f"  3. Copy to:  {dest}")
    return ok


# ════════════════════════════════════════════════════════════
# DATASET 3 — GDSC1 2023  (Imatinib + Panobinostat + Quizartinib)
# NEW: added because 3 of the 5 RL drugs are only in GDSC1
# ════════════════════════════════════════════════════════════
def download_gdsc1():
    print("\n" + "=" * 58)
    print("DATASET 3: GDSC1 Drug Sensitivity (2023 version)  [NEW]")
    print("  Required for: Imatinib, Panobinostat, Quizartinib")
    print("=" * 58)
    dest = os.path.join(DATA_DIR, "gdsc1", "GDSC1_fitted_dose_response_27Oct23.xlsx")
    ok = download_file(
        url=cfg["gdsc1"]["download_url"],
        dest_path=dest,
        label="GDSC1_fitted_dose_response_27Oct23.xlsx",
    )
    if not ok:
        print("\n  MANUAL FALLBACK:")
        print("  1. Go to: https://www.cancerrxgene.org/downloads/bulk_download")
        print("  2. Download: GDSC1_fitted_dose_response_27Oct23.xlsx")
        print(f"  3. Copy to:  {dest}")
    return ok


# ════════════════════════════════════════════════════════════
# DATASET 4 — GEO GSE220112  (real AML cancer data, Kaggle only)
# ════════════════════════════════════════════════════════════
def download_geo():
    print("\n" + "=" * 58)
    print("DATASET 4: GEO GSE220112 AML cancer data  (~420 MB)")
    print("  NOTE: For Kaggle training, upload directly — see below")
    print("=" * 58)

    files = [
        {
            "url": ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE220nnn/"
                    "GSE220112/suppl/"
                    "GSE220112_RS411_DMSO10h_filtered_feature_bc_matrix.h5.gz"),
            "dest": os.path.join(DATA_DIR, "gse220112",
                                 "GSE220112_RS411_DMSO10h_filtered_feature_bc_matrix.h5"),
            "gz":   os.path.join(DATA_DIR, "gse220112",
                                 "GSE220112_RS411_DMSO10h_filtered_feature_bc_matrix.h5.gz"),
            "label": "GSE220112 DMSO control (AML sensitive cells)",
        },
        {
            "url": ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE220nnn/"
                    "GSE220112/suppl/"
                    "GSE220112_RS411_Wee1i10h_filtered_feature_bc_matrix.h5.gz"),
            "dest": os.path.join(DATA_DIR, "gse220112",
                                 "GSE220112_RS411_Wee1i10h_filtered_feature_bc_matrix.h5"),
            "gz":   os.path.join(DATA_DIR, "gse220112",
                                 "GSE220112_RS411_Wee1i10h_filtered_feature_bc_matrix.h5.gz"),
            "label": "GSE220112 Wee1i treated (AML resistant cells)",
        },
    ]

    geo_ok = True
    for f in files:
        ok = download_file(
            url=f["url"],
            dest_path=f["dest"],
            label=f["label"],
            compressed=True,
            gz_dest=f["gz"],
        )
        if not ok:
            geo_ok = False

    if not geo_ok:
        print_geo_manual()

    return geo_ok


# ════════════════════════════════════════════════════════════
# VERIFICATION
# ════════════════════════════════════════════════════════════
def verify_all():
    print("\n" + "=" * 58)
    print("VERIFICATION — final status")
    print("=" * 58)

    checks = [
        (os.path.join(DATA_DIR, "pbmc_multiome_3k", "filtered_feature_bc_matrix.h5"),
         "PBMC multiome 3k", 30),
        (os.path.join(DATA_DIR, "gdsc2", "GDSC2_fitted_dose_response_27Oct23.xlsx"),
         "GDSC2 (Venetoclax + Dasatinib)", 20),
        (os.path.join(DATA_DIR, "gdsc1", "GDSC1_fitted_dose_response_27Oct23.xlsx"),
         "GDSC1 (Imatinib + Panobinostat + Quizartinib)", 20),
        (os.path.join(DATA_DIR, "gse220112",
                      "GSE220112_RS411_DMSO10h_filtered_feature_bc_matrix.h5"),
         "GSE220112 DMSO (sensitive)", 100),
        (os.path.join(DATA_DIR, "gse220112",
                      "GSE220112_RS411_Wee1i10h_filtered_feature_bc_matrix.h5"),
         "GSE220112 Wee1i (resistant)", 100),
    ]

    all_ok = True
    for path, label, min_mb in checks:
        ok = verify(path, min_mb, label)
        all_ok = all_ok and ok

    print()
    if all_ok:
        print("  All datasets ready.")
        print("  Next: python src/train.py --config configs/config.yaml")
    else:
        print("  Some files missing — see instructions above.")
    return all_ok


# ════════════════════════════════════════════════════════════
# GITKEEP  — ensure tracked empty folders exist
# ════════════════════════════════════════════════════════════
def ensure_gitkeeps():
    for folder in ["data", "models", "artifacts", "screenshots",
                   os.path.join("data", "pbmc_multiome_3k"),
                   os.path.join("data", "gdsc2"),
                   os.path.join("data", "gdsc1"),
                   os.path.join("data", "gse220112")]:
        full = os.path.join(PROJECT_ROOT, folder)
        os.makedirs(full, exist_ok=True)
        gk = os.path.join(full, ".gitkeep")
        if not os.path.exists(gk):
            open(gk, "w").close()


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="SC-RLOT dataset downloader")
    parser.add_argument("--pbmc", action="store_true", help="PBMC only (37 MB)")
    parser.add_argument("--gdsc", action="store_true", help="GDSC1 + GDSC2 only")
    parser.add_argument("--geo",  action="store_true", help="GEO GSE220112 only (~420 MB)")
    args = parser.parse_args()

    ensure_gitkeeps()

    # Default: all datasets when no flag given
    do_pbmc = args.pbmc or not (args.gdsc or args.geo)
    do_gdsc = args.gdsc or not (args.pbmc or args.geo)
    do_geo  = args.geo  or not (args.pbmc or args.gdsc)

    if do_pbmc:
        download_pbmc()
    if do_gdsc:
        download_gdsc2()
        download_gdsc1()
    if do_geo:
        download_geo()

    # Full summary only on complete run
    if do_pbmc and do_gdsc:
        verify_all()


if __name__ == "__main__":
    main()