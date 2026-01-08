import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.metrics import cohen_kappa_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import time
from tabulate import tabulate
from tqdm import tqdm
from copy import deepcopy

# Try importing ptflops
try:
    from ptflops import get_model_complexity_info
    HAS_PTFLOPS = True
except Exception:
    HAS_PTFLOPS = False

# ------------------------------
# 0) CONFIGURATION
# ------------------------------
PCA_VARIANTS = [1, 2, 3, 5, 10]  # Ablation Study Values
PATCH = 32
SIAMESE_EPOCHS = 5               # Reduced slightly for speed during ablation
SIAMESE_LR = 2e-4
EMBED_DIM = 128
TRUNK_OUT_CH = 64
CHANGE_RATIO_LABEL = 0.30
CHANGE_TOP_FRACTION = 0.30
UNCHANGED_BOTTOM_FRACTION = 0.30
BATCH_SIZE_SIAMESE = 64
BATCH_SIZE_SEG = 4
EPOCHS_SEG = 200                 # Reduced slightly for speed
SEG_LR = 1e-4
RANDOM_SEED = 123
N_BASES = 6
OUT_DIR = "./outputs_ablation_pca"

os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------
# 1) DATA UTILS
# ------------------------------
def load_santa_barbara_pairs(root_dir: str = "./", year1: str = "before", year2: str = "after"):
    """Load Santa-Barbara HypeRvieW cubes and GT map."""
    try:
        d1 = sio.loadmat(f"{root_dir}/river_{year1}.mat")
        d2 = sio.loadmat(f"{root_dir}/river_{year2}.mat")
        gt = sio.loadmat(f"{root_dir}/river_groundtruth.mat")
        x1 = d1["river_before"].astype(np.float32)
        x2 = d2["river_after"].astype(np.float32)
        y_raw = gt["lakelabel_v1"].squeeze()
        y = np.zeros_like(y_raw, dtype=np.int64)
        y[y_raw == 255] = 1
        y[y_raw == 0] = 0
        return x1, x2, y
    except FileNotFoundError:
        print("Error: Dataset files not found in current directory.")
        exit()

def pca_k(cube: np.ndarray, k: int):
    """Return k-channel PCA projection scaled to [0,1]."""
    H_, W_, B_ = cube.shape
    X = cube.reshape(-1, B_)
    X = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    PCs = X @ Vt[:k].T
    # Normalize to [0, 1]
    PCs = PCs - PCs.min(0, keepdims=True)
    PCs = PCs / (PCs.max(0, keepdims=True) + 1e-8)
    return PCs.reshape(H_, W_, k).astype(np.float32)

def set_seed(s=RANDOM_SEED):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

# ------------------------------
# 2) MODEL ARCHITECTURES
# ------------------------------
class SiameseTrunk(nn.Module):
    def __init__(self, in_ch, out_ch=TRUNK_OUT_CH):
        super().__init__()
        # Note: First conv input channels = k (PCA components)
        self.trunk = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.trunk(x)

class SiameseHead(nn.Module):
    def __init__(self, in_ch=TRUNK_OUT_CH, emb_dim=EMBED_DIM):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(in_ch, emb_dim)
    def forward(self, feat_map):
        x = self.pool(feat_map).flatten(1)
        return F.normalize(self.fc(x), dim=-1)

class SiameseEncoder(nn.Module):
    def __init__(self, in_ch, out_ch=TRUNK_OUT_CH, emb_dim=EMBED_DIM):
        super().__init__()
        self.trunk = SiameseTrunk(in_ch=in_ch, out_ch=out_ch)
        self.head  = SiameseHead(in_ch=out_ch, emb_dim=emb_dim)
    def forward(self, x): return self.head(self.trunk(x))

class KANExpand2d(nn.Module):
    def __init__(self, in_channels, out_channels, n_bases=N_BASES):
        super().__init__()
        self.n_bases = n_bases
        self.centers = nn.Parameter(torch.linspace(0.0, 1.0, n_bases).view(1,1,n_bases,1,1).repeat(1,in_channels,1,1,1))
        self.scale   = nn.Parameter(torch.ones(1,in_channels,n_bases,1,1))
        self.mix = nn.Conv2d(in_channels * n_bases, out_channels, kernel_size=1, bias=False)
        self.bn  = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        B, C, H, W = x.shape
        z = x.unsqueeze(2) - self.centers.to(x.device)
        phi = (F.relu(z) ** 2) * self.scale.to(x.device)
        phi = phi.permute(0,1,3,4,2).contiguous().view(B, C*self.n_bases, H, W)
        return self.bn(self.mix(phi))

class UKANBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.expand = KANExpand2d(in_ch, out_ch)
        self.mix = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch, bias=False), # Depthwise
            nn.Conv2d(out_ch, out_ch, 1, bias=False),                           # Pointwise
            nn.BatchNorm2d(out_ch)
        )
        self.act = nn.GELU()
        self.short = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
    def forward(self, x):
        return self.act(self.mix(self.act(self.expand(x))) + self.short(x))

class UKAN(nn.Module):
    def __init__(self, in_channels, out_channels=2):
        super().__init__()
        f = [48, 96, 192, 256]
        self.pool = nn.MaxPool2d(2)
        self.enc1 = UKANBlock(in_channels, f[0])
        self.enc2 = UKANBlock(f[0], f[1])
        self.enc3 = UKANBlock(f[1], f[2])
        self.enc4 = UKANBlock(f[2], f[3])
        self.bot  = UKANBlock(f[3], f[3])
        
        self.up4, self.dec4 = nn.ConvTranspose2d(f[3], f[2], 2, 2), UKANBlock(f[3]+f[2], f[2])
        self.up3, self.dec3 = nn.ConvTranspose2d(f[2], f[1], 2, 2), UKANBlock(f[2]+f[1], f[1])
        self.up2, self.dec2 = nn.ConvTranspose2d(f[1], f[0], 2, 2), UKANBlock(f[1]+f[0], f[0])
        self.up1, self.dec1 = nn.ConvTranspose2d(f[0], f[0], 2, 2), UKANBlock(f[0]+f[0], f[0])
        self.out = nn.Conv2d(f[0], out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bot(self.pool(e4))
        
        d4 = self.dec4(torch.cat([e4, F.interpolate(self.up4(b), size=e4.shape[-2:])], 1))
        d3 = self.dec3(torch.cat([e3, F.interpolate(self.up3(d4), size=e3.shape[-2:])], 1))
        d2 = self.dec2(torch.cat([e2, F.interpolate(self.up2(d3), size=e2.shape[-2:])], 1))
        d1 = self.dec1(torch.cat([e1, F.interpolate(self.up1(d2), size=e1.shape[-2:])], 1))
        return self.out(d1)

# ------------------------------
# 3) UTILITY FUNCTIONS
# ------------------------------
def contrastive_loss(emb1, emb2, y, margin=1.0):
    d = torch.norm(emb1 - emb2, p=2, dim=1)
    return ((1-y)*(d**2) + y*(F.relu(margin-d)**2)).mean()

def fused_embeddings(trunk, t1, t2):
    with torch.no_grad():
        f1, f2 = trunk(t1), trunk(t2)
    fused = torch.cat([torch.abs(f2-f1), f1, f2], dim=1)
    return F.interpolate(fused, size=(32,32), mode='bilinear', align_corners=False)

def augment_pair(img1, img2, lbl):
    imgs1, imgs2, lbls = [img1], [img2], [lbl]
    for k in [1,2,3]:
        imgs1.append(np.rot90(img1, k)); imgs2.append(np.rot90(img2, k)); lbls.append(np.rot90(lbl, k))
    imgs1.append(np.flip(img1, 1)); imgs2.append(np.flip(img2, 1)); lbls.append(np.flip(lbl, 1))
    return imgs1, imgs2, lbls

class PatchDataset(Dataset):
    def __init__(self, t1, t2, y, is_float_y=False):
        self.t1 = torch.from_numpy(t1).permute(0,3,1,2).float()
        self.t2 = torch.from_numpy(t2).permute(0,3,1,2).float()
        dtype = torch.float32 if is_float_y else torch.long
        self.y  = torch.from_numpy(y).to(dtype)
    def __len__(self): return len(self.t1)
    def __getitem__(self, i): return self.t1[i], self.t2[i], self.y[i]

# ------------------------------
# 4) MAIN ABLATION LOOP
# ------------------------------
def run_ablation():
    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load Raw Data
    X1_raw, X2_raw, Y_gt = load_santa_barbara_pairs()
    H, W, _ = X1_raw.shape
    
    ablation_results = []
    
    print("\n" + "="*50)
    print(f"STARTING PCA ABLATION STUDY: {PCA_VARIANTS}")
    print("="*50 + "\n")

    for k_pca in PCA_VARIANTS:
        print(f"\n>>> Running for PCA Components = {k_pca} <<<")
        
        # A. Preprocessing
        t1_k = pca_k(X1_raw, k_pca)
        t2_k = pca_k(X2_raw, k_pca)
        
        # B. Patch Extraction
        patches_t1, patches_t2, patches_y = [], [], []
        for r in range(0, H, PATCH):
            for c in range(0, W, PATCH):
                r_e, c_e = min(r+PATCH, H), min(c+PATCH, W)
                p1 = np.zeros((PATCH, PATCH, k_pca), dtype=np.float32)
                p2 = np.zeros((PATCH, PATCH, k_pca), dtype=np.float32)
                py = np.zeros((PATCH, PATCH), dtype=np.int64)
                
                ph, pw = r_e - r, c_e - c
                p1[:ph, :pw] = t1_k[r:r_e, c:c_e]
                p2[:ph, :pw] = t2_k[r:r_e, c:c_e]
                py[:ph, :pw] = Y_gt[r:r_e, c:c_e]
                
                patches_t1.append(p1); patches_t2.append(p2); patches_y.append(py)
                
        patches_t1 = np.stack(patches_t1)
        patches_t2 = np.stack(patches_t2)
        patches_y  = np.stack(patches_y)
        
        # Weak labels for Siamese
        y_weak = ((patches_y == 1).mean(axis=(1,2)) >= CHANGE_RATIO_LABEL).astype(np.float32)
        
        # C. Train Siamese
        siamese = SiameseEncoder(in_ch=k_pca).to(device)
        opt = torch.optim.Adam(siamese.parameters(), lr=SIAMESE_LR)
        loader = DataLoader(PatchDataset(patches_t1, patches_t2, y_weak, True), batch_size=BATCH_SIZE_SIAMESE, shuffle=True)
        
        siamese.train()
        for ep in range(SIAMESE_EPOCHS):
            for t1, t2, y in loader:
                t1, t2, y = t1.to(device), t2.to(device), y.to(device)
                loss = contrastive_loss(siamese(t1), siamese(t2), y)
                opt.zero_grad(); loss.backward(); opt.step()
                
        # D. Patch Mining
        siamese.eval()
        dists = []
        # Process in chunks to avoid memory issues
        with torch.no_grad():
            for i in range(0, len(patches_t1), 128):
                t1 = torch.from_numpy(patches_t1[i:i+128]).permute(0,3,1,2).to(device)
                t2 = torch.from_numpy(patches_t2[i:i+128]).permute(0,3,1,2).to(device)
                d = torch.norm(siamese(t1)-siamese(t2), dim=1).cpu().numpy()
                dists.append(d)
        dists = np.concatenate(dists)
        
        th_h = np.quantile(dists, 1 - CHANGE_TOP_FRACTION)
        th_l = np.quantile(dists, UNCHANGED_BOTTOM_FRACTION)
        sel_idx = np.where((dists >= th_h) | (dists <= th_l))[0]
        
        # E. Augmentation & Dataset Split
        aug_t1, aug_t2, aug_y = [], [], []
        for i in sel_idx:
            a1, a2, ay = augment_pair(patches_t1[i], patches_t2[i], patches_y[i])
            aug_t1.extend(a1); aug_t2.extend(a2); aug_y.extend(ay)
        
        full_ds = PatchDataset(np.stack(aug_t1), np.stack(aug_t2), np.stack(aug_y))
        n_train = int(0.9 * len(full_ds))
        train_ds, test_ds = random_split(full_ds, [n_train, len(full_ds)-n_train])
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE_SEG, shuffle=True)
        
        # F. Train UKAN
        trunk = deepcopy(siamese.trunk).eval()
        ukan = UKAN(in_channels=3*TRUNK_OUT_CH).to(device)
        opt_seg = torch.optim.Adam(ukan.parameters(), lr=SEG_LR)
        crit = nn.CrossEntropyLoss()
        
        # Compute params & flops
        n_params = sum(p.numel() for p in ukan.parameters())
        if HAS_PTFLOPS:
            macs, _ = get_model_complexity_info(ukan, (3*TRUNK_OUT_CH, 32, 32), as_strings=False, print_per_layer_stat=False)
            flops = macs * 2
        else:
            flops = 0

        # Training Loop
        for ep in tqdm(range(EPOCHS_SEG), desc=f"Training PCA={k_pca}", leave=False):
            ukan.train()
            for t1, t2, y in train_dl:
                t1, t2, y = t1.to(device), t2.to(device), y.to(device)
                logits = ukan(fused_embeddings(trunk, t1, t2))
                loss = crit(logits, y)
                opt_seg.zero_grad(); loss.backward(); opt_seg.step()
        
        # G. Evaluation (Full Image Reconstruction)
        ukan.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(patches_t1), 64):
                t1 = torch.from_numpy(patches_t1[i:i+64]).permute(0,3,1,2).to(device)
                t2 = torch.from_numpy(patches_t2[i:i+64]).permute(0,3,1,2).to(device)
                lg = ukan(fused_embeddings(trunk, t1, t2))
                preds.append(torch.argmax(lg, dim=1).cpu().numpy())
        preds = np.concatenate(preds)
        
        # Reassemble map
        pmap = np.zeros((H, W), dtype=np.int64)
        idx = 0
        for r in range(0, H, PATCH):
            for c in range(0, W, PATCH):
                if idx < len(preds):
                    r_e, c_e = min(r+PATCH, H), min(c+PATCH, W)
                    pmap[r:r_e, c:c_e] = preds[idx][:r_e-r, :c_e-c]
                    idx += 1
        
        # Calculate Metrics
        oa = (pmap == Y_gt).mean()
        kappa = cohen_kappa_score(Y_gt.flatten(), pmap.flatten())
        tp = ((pmap==1) & (Y_gt==1)).sum()
        fp = ((pmap==1) & (Y_gt==0)).sum()
        fn = ((pmap==0) & (Y_gt==1)).sum()
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        
        # Store Result
        res = [k_pca, oa, kappa, f1, n_params, flops]
        ablation_results.append(res)
        
        # Save map for this variant
        plt.imsave(os.path.join(OUT_DIR, f"pred_pca_{k_pca}.png"), pmap, cmap='gray')
        print(f"  -> OA: {oa:.4f} | Kappa: {kappa:.4f} | F1: {f1:.4f}")

    return ablation_results

# ------------------------------
# 5) REPORTING
# ------------------------------
if __name__ == "__main__":
    results = run_ablation()
    
    # Format table data
    table_data = []
    for r in results:
        # PCA, OA, Kappa, F1, Params(M), GFLOPs
        table_data.append([
            r[0], 
            f"{r[1]:.4f}", 
            f"{r[2]:.4f}", 
            f"{r[3]:.4f}", 
            f"{r[4]/1e6:.2f}M", 
            f"{r[5]/1e9:.3f}G"
        ])
    
    headers = ["PCA Components", "OA", "Kappa", "F1 Score", "Params", "FLOPs"]
    
    print("\n\n" + "="*60)
    print("FINAL ABLATION STUDY RESULTS")
    print("="*60)
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))
    
    # Save to file
    with open(os.path.join(OUT_DIR, "ablation_results.txt"), "w") as f:
        f.write(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    print(f"\nResults saved to {OUT_DIR}/ablation_results.txt")