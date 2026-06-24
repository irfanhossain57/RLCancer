import scanpy as sc
import anndata
import numpy as np
import torch
import torch.nn as nn
import os

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Load and preprocess data
adata = sc.datasets.pbmc3k()
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata = adata[:, adata.var.highly_variable]
sc.pp.scale(adata)

# Proper VAE with encoder AND decoder
class MiniVAE(nn.Module):
    def __init__(self, input_dim, lat=32):
        super().__init__()
        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(input_dim, 128), 
            nn.ReLU(), 
            nn.Linear(128, lat*2)
        )
        # Decoder
        self.dec = nn.Sequential(
            nn.Linear(lat, 128), 
            nn.ReLU(), 
            nn.Linear(128, input_dim)  # Output must match input dimension
        )
    
    def encode(self, x):
        h = self.enc(x)
        mu, logvar = h.chunk(2, -1)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.dec(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

# Convert to tensor - FIXED: after scale(), .X is already dense
if hasattr(adata.X, 'toarray'):
    X = torch.tensor(adata.X.toarray(), dtype=torch.float32)
else:
    X = torch.tensor(adata.X, dtype=torch.float32)

print(f"Data shape: {X.shape}")
print(f"Input dimension: {X.shape[1]}")

# Initialize VAE
vae = MiniVAE(input_dim=X.shape[1], lat=32)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

# Train VAE
print("\nTraining VAE...")
for ep in range(50):  # Increased epochs for better training
    optimizer.zero_grad()
    recon_x, mu, logvar = vae(X)
    
    # Reconstruction loss (MSE) - now both are same dimension (2700, 2000)
    recon_loss = nn.MSELoss()(recon_x, X)
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalize KL loss by batch size
    kl_loss = kl_loss / X.shape[0]
    
    # Total loss (beta-VAE)
    beta = 0.001
    loss = recon_loss + beta * kl_loss
    
    loss.backward()
    optimizer.step()
    
    if (ep + 1) % 10 == 0:
        print(f"Epoch {ep+1:3d}, Loss: {loss.item():.4f}, Recon: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f}")

# Extract latent representation
with torch.no_grad():
    mu, _ = vae.encode(X)  # Use encode method to get latent representation

# Add to AnnData
adata.obsm["X_vae"] = mu.detach().numpy()

# Compute neighbors and clustering
sc.pp.neighbors(adata, use_rep="X_vae")
sc.tl.leiden(adata, resolution=0.5)

# Add metadata
adata.obs["subclone"] = adata.obs["leiden"].astype(str)
adata.obs["resistant"] = (adata.obs["leiden"].astype(int) % 4 == 0).astype(str)
adata.obs["resistant"] = adata.obs["resistant"].map({'True': 'resistant', 'False': 'sensitive'})

# Save
adata.write_h5ad("data/pbmc3k_processed.h5ad")
print("\n" + "="*50)
print("Data saved to data/pbmc3k_processed.h5ad")
print(f"Data shape: {adata.shape}")
print(f"Number of cells: {adata.n_obs}")
print(f"Number of genes: {adata.n_vars}")
print(f"Number of leiden clusters: {len(adata.obs['leiden'].unique())}")
print(f"\nCluster distribution:")
print(adata.obs['leiden'].value_counts().sort_index())
print(f"\nResistant distribution:")
print(adata.obs['resistant'].value_counts())
print("="*50)