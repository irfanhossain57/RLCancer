# fix_adata_numpy.py
import h5py, anndata, numpy as np, scipy.sparse as sp, os, shutil

src = "models/processed_adata.h5ad"
tmp = "models/processed_adata_tmp.h5ad"
dst = "data/processed_adata.h5ad"

# Step 1 — copy file so we don't touch the original
print("Copying file...")
shutil.copy2(src, tmp)

# Step 2 — delete the problematic key directly in the HDF5 file
print("Patching HDF5 (removing incompatible keys)...")
with h5py.File(tmp, "a") as f:
    # Delete the log1p entry that your anndata version can't read
    if "uns/log1p" in f:
        del f["uns/log1p"]
        print("  Removed uns/log1p")
    # Also check for other common problem keys
    if "uns/neighbors" in f:
        del f["uns/neighbors"]
        print("  Removed uns/neighbors")
    # Print what's left so we can debug if needed
    print(f"  Remaining uns keys: {list(f['uns'].keys()) if 'uns' in f else 'none'}")
    print(f"  obsm keys: {list(f['obsm'].keys()) if 'obsm' in f else 'none'}")
    print(f"  obs keys:  {list(f['obs'].keys()) if 'obs' in f else 'none'}")

# Step 3 — now anndata can load it
print("\nLoading patched file...")
adata = anndata.read_h5ad(tmp)
print(f"  Loaded: {adata.n_obs} cells x {adata.n_vars} genes")
print(f"  obsm keys: {list(adata.obsm.keys())}")
print(f"  obs cols:  {list(adata.obs.columns)}")

# Step 4 — re-cast arrays to current numpy
if sp.issparse(adata.X):
    adata.X = adata.X.astype(np.float32)
else:
    adata.X = np.array(adata.X, dtype=np.float32)

for key in adata.obsm.keys():
    adata.obsm[key] = np.array(adata.obsm[key], dtype=np.float32)

# Step 5 — save clean copy
os.makedirs("data", exist_ok=True)
adata.write_h5ad(dst)
print(f"\nDone — saved to {dst}")

# Step 6 — clean up temp file
os.remove(tmp)
print("Temp file removed.")