"""
byol_features.py  ——  BYOL: Self-supervised Image Feature Extraction

"""

import gc
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from copy import deepcopy

try:
    from tqdm.notebook import tqdm
except Exception:
    from tqdm import tqdm


# ══════════════════════════════════════════════════════
# 1. Pathological Image Data Augmentation
# ══════════════════════════════════════════════════════

def make_augmentation(patch_size: int = 96):
    return transforms.Compose([
        transforms.RandomResizedCrop(patch_size, scale=(0.5, 1.0),
                                     interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.1, hue=0.02)
        ], p=0.8),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 1.5))
        ], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ══════════════════════════════════════════════════════
# 2. Spot Patch Dataset
# ══════════════════════════════════════════════════════

class SpotPatchDataset(Dataset):
    def __init__(self, img_np, rows, cols, radius, patch_size=96):
        self.img    = img_np
        self.rows   = rows.astype(int)
        self.cols   = cols.astype(int)
        self.radius = max(int(radius), patch_size // 2)
        self.H, self.W = img_np.shape[:2]
        self.aug    = make_augmentation(patch_size)

    def __len__(self):
        return len(self.rows)

    def _get_patch(self, idx):
        r, c = self.rows[idx], self.cols[idx]
        r0 = max(0, r - self.radius); r1 = min(self.H, r + self.radius)
        c0 = max(0, c - self.radius); c1 = min(self.W, c + self.radius)
        patch = self.img[r0:r1, c0:c1]
        if patch.shape[0] < 8 or patch.shape[1] < 8:
            patch = np.zeros((self.radius*2, self.radius*2, 3), dtype=np.uint8)
        return Image.fromarray(patch.astype(np.uint8))

    def __getitem__(self, idx):
        pil = self._get_patch(idx)
        return self.aug(pil), self.aug(pil)


# ══════════════════════════════════════════════════════
# 3. BYOL Network
# Dimension rule: The outputs of both projector and predictor are "dim".
# Both must be the same for the "byol_loss" to be calculated correctly
# ══════════════════════════════════════════════════════

class ProjectionMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x):
        return self.net(x)


class OnlineNetwork(nn.Module):
    """encoder → projector(→dim) → predictor(→dim)"""
    def __init__(self, encoder, feat_dim: int, dim: int = 128):
        super().__init__()
        self.encoder   = encoder
        self.projector = ProjectionMLP(feat_dim, 512, dim)
        self.predictor = ProjectionMLP(dim,      256, dim)  # output=dim

    def forward(self, x):
        z = self.encoder(x)          # (B, feat_dim)
        p = self.projector(z)        # (B, dim)
        q = self.predictor(p)        # (B, dim)  ← Same dimension as the target output
        return q, p.detach()


class TargetNetwork(nn.Module):
    """encoder → projector(→dim)，update EMA"""
    def __init__(self, encoder, feat_dim: int, dim: int = 128):
        super().__init__()
        self.encoder   = encoder
        self.projector = ProjectionMLP(feat_dim, 512, dim)

    @torch.no_grad()
    def forward(self, x):
        return self.projector(self.encoder(x))   # (B, dim)


def build_networks(device, dim: int = 128):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        print(f"[BYOL] GPU free: {free/1024**3:.2f}GB / {total/1024**3:.2f}GB")

    backbone = models.mobilenet_v2(pretrained=True)
    feat_dim = backbone.classifier[1].in_features  # 1280
    backbone.classifier = nn.Identity()

    online = OnlineNetwork(backbone,           feat_dim, dim).to(device)
    target = TargetNetwork(deepcopy(backbone), feat_dim, dim).to(device)

    for p in target.parameters():
        p.requires_grad_(False)

    print(f"[BYOL] Network Ready: MobileNetV2，dim={dim}")
    return online, target


# ══════════════════════════════════════════════════════
# 4. EMA & Loss
# ══════════════════════════════════════════════════════

@torch.no_grad()
def ema_update(online, target, tau: float):
    for po, pt in zip(online.parameters(), target.parameters()):
        pt.data.mul_(tau).add_(po.data, alpha=1.0 - tau)


def byol_loss(q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Both q and t have a shape of (B, dim). Compute the cosine similarity loss."""
    q = F.normalize(q, dim=-1)
    t = F.normalize(t, dim=-1)
    return 2.0 - 2.0 * (q * t).sum(dim=-1).mean()


# ══════════════════════════════════════════════════════
# 5. training
# ══════════════════════════════════════════════════════

def train_byol(img_np, rows, cols, radius=48, patch_size=96,
               n_epochs=300, batch_size=32, lr=3e-4,
               tau_base=0.996, dim=128, device='cuda'):

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dev = torch.device(device if torch.cuda.is_available() else 'cpu')

    dataset    = SpotPatchDataset(img_np, rows, cols, radius, patch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=2,
                            pin_memory=(dev.type == 'cuda'),
                            drop_last=True)

    online, target = build_networks(dev, dim)
    optimizer = torch.optim.Adam(online.parameters(), lr=lr, weight_decay=1e-6)

    total_steps = n_epochs * len(dataloader)
    global_step = 0

    online.train()
    print(f"[BYOL] training start：{n_epochs} epochs，{len(dataloader)} steps/epoch")

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for v1, v2 in dataloader:
            v1, v2 = v1.to(dev), v2.to(dev)

            q1, _ = online(v1)   # q1: (B, dim)
            q2, _ = online(v2)   # q2: (B, dim)

            with torch.no_grad():
                t1 = target(v2)  # t1: (B, dim)  ← Same dimension as q1
                t2 = target(v1)  # t2: (B, dim)  ← Same dimension as q2 

            loss = (byol_loss(q1, t1) + byol_loss(q2, t2)) * 0.5

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress = global_step / max(total_steps - 1, 1)
            tau = 1.0 - (1.0 - tau_base) * (np.cos(np.pi * progress) + 1) / 2
            ema_update(online, target, tau)

            epoch_loss  += loss.item()
            global_step += 1

        if (epoch + 1) % max(1, n_epochs // 10) == 0:
            print(f"  Epoch {epoch+1:>4d}/{n_epochs}  "
                  f"loss={epoch_loss/len(dataloader):.4f}  tau={tau:.4f}")

    print("[BYOL] finish training")
    return online


# ══════════════════════════════════════════════════════
# 6. Feature extraction
# ══════════════════════════════════════════════════════

@torch.no_grad()
def extract_features(online, img_np, rows, cols, radius,
                     patch_size=96, out_dim=128,
                     batch_size=128, device='cuda'):
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    infer_tf = transforms.Compose([
        transforms.Resize((patch_size, patch_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    H, W   = img_np.shape[:2]
    rows_i = rows.astype(int)
    cols_i = cols.astype(int)
    rad    = max(int(radius), patch_size // 2)

    patches = []
    for i in range(len(rows_i)):
        r, c = rows_i[i], cols_i[i]
        r0 = max(0, r-rad); r1 = min(H, r+rad)
        c0 = max(0, c-rad); c1 = min(W, c+rad)
        p  = img_np[r0:r1, c0:c1]
        if p.shape[0] < 8 or p.shape[1] < 8:
            p = np.zeros((rad*2, rad*2, 3), dtype=np.uint8)
        patches.append(infer_tf(Image.fromarray(p.astype(np.uint8))))

    online.eval().to(dev)
    feats = []
    for i in range(0, len(patches), batch_size):
        batch = torch.stack(patches[i:i+batch_size]).to(dev)
        feats.append(online.encoder(batch).cpu().numpy())

    feats = np.concatenate(feats, axis=0)   # (n_spots, 1280)

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    if feats.shape[1] > out_dim:
        feats = PCA(n_components=out_dim, random_state=0).fit_transform(
            StandardScaler().fit_transform(feats)
        ).astype(np.float32)

    print(f"[BYOL] Feature extraction completed, shape: {feats.shape}")
    return feats


# ══════════════════════════════════════════════════════
# 7. External interface
# ══════════════════════════════════════════════════════

def get_byol_features(img, locs, rad=55.0, patch_size=96,
                      n_epochs=300, batch_size=32, out_dim=128,
                      device='cuda', save_path=None):
    if save_path and os.path.exists(save_path):
        arr = np.load(save_path)
        print(f"[BYOL] Loading cache: {save_path}, shape={arr.shape}")
        return arr

    img_np = np.array(img)
    if img_np.ndim == 3 and img_np.shape[-1] == 4:
        img_np = img_np[..., :3]

    rows   = locs['4'].values.astype(float)
    cols   = locs['5'].values.astype(float)
    radius = max(int(rad), patch_size // 2)

    online = train_byol(
        img_np=img_np, rows=rows, cols=cols,
        radius=radius, patch_size=patch_size,
        n_epochs=n_epochs, batch_size=batch_size,
        dim=out_dim, device=device,
    )

    image_emb = extract_features(
        online=online, img_np=img_np,
        rows=rows, cols=cols,
        radius=radius, patch_size=patch_size,
        out_dim=out_dim, batch_size=batch_size*2,
        device=device,
    )

    if save_path:
        np.save(save_path, image_emb)
        print(f"[BYOL] Cached to: {save_path}")

    return image_emb