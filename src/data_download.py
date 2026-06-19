"""
src/data_download.py
====================
Verifies that all required data files are present.
Size thresholds are adjusted to realistic values.
"""

import os
import yaml

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

def verify_file(path, min_mb, label):
    if not os.path.exists(path):
        return False, None
    size = os.path.getsize(path) / 1024 / 1024
    if size < min_mb:
        return False, size
    return True, size

def verify_any(paths, min_mb, label):
    for p in paths:
        ok, size = verify_file(p, min_mb, label)
        if ok:
            print(f"  ✅ OK       {label}: {p} ({size:.1f} MB)")
            return True
    print(f"  ❌ MISSING  {label}: tried {paths}")
    return False

print("\n" + "=" * 60)
print("DATA VERIFICATION (manual placement)")
print("=" * 60)

# PBMC
pbmc_path = cfg["pbmc"]["local_path"]
ok1 = verify_any([pbmc_path], 30, "PBMC Multiome 3K")

# GSE DMSO
dmso_path = "data/gse220112/GSE220112_RS411_DMSO10h_filtered_feature_bc_matrix.h5"
ok2 = verify_any([dmso_path], 100, "GSE220112 DMSO")

# GSE Wee1i (allow both with/without (1))
wee1i_candidates = [
    "data/gse220112/GSE220112_RS411_Wee1i10h_filtered_feature_bc_matrix.h5",
    "data/gse220112/GSE220112_RS411_Wee1i10h_filtered_feature_bc_matrix (1).h5"
]
ok3 = verify_any(wee1i_candidates, 100, "GSE220112 Wee1i")

# GDSC2 – try both possible filenames, min size lowered to 10 MB
gdsc2_candidates = [
    cfg["gdsc2"]["local_path"],          # from config (15Oct19)
    "data/gdsc2/GDSC2_fitted_dose_response_27Oct23.xlsx"
]
ok4 = verify_any(gdsc2_candidates, 10, "GDSC2 Drug Sensitivity")   # 👈 threshold lowered

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
if ok1 and ok4:
    print("  ✅ PBMC and GDSC2 are ready for local training.")
else:
    print("  ❌ Missing or corrupt files – check paths above.")
if ok2 and ok3:
    print("  ✅ GSE220112 is also present locally (optional).")
else:
    print("  ℹ️  GSE220112 can be used via Kaggle dataset if not local.")

print("\nNext: python src/train.py --config configs/config.yaml")