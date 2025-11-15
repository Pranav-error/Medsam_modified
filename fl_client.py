"""Federated learning client for MedSAM hackathon demo.

Each instance of this script represents one "hospital" node. It connects to
an FL server (fl_server.py), trains a small segmentation model on local data,
and sends updates. This does **not** modify any existing MedSAM inference code.

Usage (on each hospital laptop):
    # Example for Hospital 0
    python fl_client.py --client-id 0 --server YOUR_IP:8080

    # Hospital 1
    python fl_client.py --client-id 1 --server YOUR_IP:8080

Data:
    By default this looks for preprocessed .npy data in `data/fl/` with the
    structure created by your existing `fl_data_loader.py`. If that directory
    is missing, it falls back to synthetic data so the demo still runs.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterator, List, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from fl_data_loader import get_data_loaders
except ImportError:
    get_data_loaders = None  # type: ignore[misc]


@dataclass
class FLConfig:
    client_id: int
    server_address: str
    data_dir: str
    batch_size: int = 2
    epochs_per_round: int = 1
    device: str = "cpu"


class SimpleSegNet(nn.Module):
    """Lightweight U-Net style segmentation network for FL demo.

    This is intentionally small so it can train quickly on CPU/MPS.
    """

    def __init__(self) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Conv2d(16, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        b = self.bottleneck(p2)
        u2 = self.up2(b)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        out = self.out_conv(d1)
        return out


class SyntheticSegDataset(Dataset):
    """Fallback synthetic dataset when real FL data is not available."""

    def __init__(self, length: int = 32, image_size: int = 128) -> None:
        self.length = length
        self.image_size = image_size

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Simple circles as synthetic lesions
        img = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)
        mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)

        # Random circle
        rr, cc = np.ogrid[: self.image_size, : self.image_size]
        center_x = np.random.randint(self.image_size // 4, 3 * self.image_size // 4)
        center_y = np.random.randint(self.image_size // 4, 3 * self.image_size // 4)
        radius = np.random.randint(self.image_size // 8, self.image_size // 5)
        circle = (rr - center_y) ** 2 + (cc - center_x) ** 2 <= radius**2
        mask[circle] = 1.0
        img[circle, 0] = 1.0  # red channel

        img_t = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)
        mask_t = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)
        return img_t, mask_t


def get_dataloaders_for_client(cfg: FLConfig) -> Tuple[DataLoader, DataLoader]:
    """Load real data if available, otherwise fall back to synthetic.

    Real path uses your existing `fl_data_loader.get_data_loaders` which expects
    preprocessed .npy MedSAM data. If `cfg.data_dir` does not exist or the
    import is not available, we create a synthetic dataset so the FL demo still
    runs end-to-end.
    """

    if get_data_loaders is not None and os.path.isdir(cfg.data_dir):
        print(f"[FL-CLIENT {cfg.client_id}] Using real data from {cfg.data_dir}")
        train_loader, val_loader = get_data_loaders(cfg.client_id, cfg.data_dir, batch_size=cfg.batch_size)
        return train_loader, val_loader

    print(f"[FL-CLIENT {cfg.client_id}] No real data found at {cfg.data_dir}, using synthetic data")
    train_ds = SyntheticSegDataset(length=32, image_size=128)
    val_ds = SyntheticSegDataset(length=16, image_size=128)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    return train_loader, val_loader


def dice_coeff(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = torch.sigmoid(pred)
    pred_bin = (pred > 0.5).float()
    target = target.float()
    intersection = (pred_bin * target).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (2 * intersection + eps) / (union + eps)


class MedSegClient(fl.client.NumPyClient):
    def __init__(self, cfg: FLConfig) -> None:
        self.cfg = cfg
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.cfg.device = str(device)
        self.model = SimpleSegNet().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.train_loader, self.val_loader = get_dataloaders_for_client(cfg)

    # Flower interface
    def get_parameters(self, config):  # type: ignore[override]
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        for p, new_p in zip(self.model.parameters(), parameters):
            p.data = torch.from_numpy(new_p).to(p.device)

    def fit(self, parameters, config):  # type: ignore[override]
        self.set_parameters(parameters)
        self.model.train()
        device = next(self.model.parameters()).device

        for _ in range(self.cfg.epochs_per_round):
            for images, masks in self._limited_batches(self.train_loader, max_batches=5):
                images = images.to(device)
                masks = masks.to(device)
                self.optimizer.zero_grad()
                logits = self.model(images)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, masks)
                loss.backward()
                self.optimizer.step()

        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):  # type: ignore[override]
        self.set_parameters(parameters)
        self.model.eval()
        device = next(self.model.parameters()).device

        total_loss = 0.0
        total_dice = 0.0
        n_samples = 0

        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(device)
                masks = masks.to(device)
                logits = self.model(images)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, masks)
                dice = dice_coeff(logits, masks).mean()

                bs = images.size(0)
                total_loss += loss.item() * bs
                total_dice += dice.item() * bs
                n_samples += bs

        avg_loss = total_loss / max(n_samples, 1)
        avg_dice = total_dice / max(n_samples, 1)

        metrics = {
            "dice": float(avg_dice),
            "hospital_id": int(self.cfg.client_id),
        }

        print(
            f"[FL-CLIENT {self.cfg.client_id}] Eval loss={avg_loss:.4f}, dice={avg_dice:.4f} on {n_samples} samples (device={self.cfg.device})"
        )

        return avg_loss, n_samples, metrics

    @staticmethod
    def _limited_batches(loader: DataLoader, max_batches: int) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            yield batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Federated Learning Client (Flower)")
    parser.add_argument("--client-id", type=int, required=True, help="Client/hospital ID (0, 1, 2, ...)")
    parser.add_argument(
        "--server",
        type=str,
        default="127.0.0.1:8080",
        help="FL server address host:port (default: 127.0.0.1:8080)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/fl",
        help="Directory with FL .npy data (imgs/gts). If missing, synthetic data is used.",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size (default: 2)")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs per FL round (default: 1)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = FLConfig(
        client_id=args.client_id,
        server_address=args.server,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        epochs_per_round=args.epochs,
    )

    print(
        f"[FL-CLIENT {cfg.client_id}] Connecting to {cfg.server_address} with data_dir={cfg.data_dir}, "
        f"batch_size={cfg.batch_size}, epochs_per_round={cfg.epochs_per_round}"
    )

    client = MedSegClient(cfg)
    fl.client.start_numpy_client(server_address=cfg.server_address, client=client)


if __name__ == "__main__":  # pragma: no cover
    main()
