"""
Training pipeline for PhysioFusion model on CASIA-FASD / Replay-Attack / OULU-NPU.
"""
import sys, os, json, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models as tv_models

from physiofusion import CrossAttentionFusion, PhysioFusionTrainer


# ── Synthetic training dataset ──────────────────────────────────
class SyntheticSignalDataset(Dataset):
    """Generates synthetic rPPG/motion/depth signals for testing the training pipeline.
    Replace with real data loader for actual training.
    """
    def __init__(self, size=200, seq_len=90):
        self.size = size
        self.seq_len = seq_len

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        live = torch.rand(1) > 0.5
        t = torch.linspace(0, 4*np.pi, self.seq_len)
        # rPPG: ~1.2 Hz pulse for live, flat for spoof
        if live:
            rppg = 0.5 * torch.sin(1.2 * 2*np.pi * t / self.seq_len + torch.randn(1)*0.3)\
                 + 0.05 * torch.randn(self.seq_len)
        else:
            rppg = 0.02 * torch.randn(self.seq_len) + 0.01 * torch.sin(60 * t / self.seq_len)
        # micro-motion: natural sway for live, static for spoof
        if live:
            motion = 0.3 * torch.sin(0.8 * 2*np.pi * t / self.seq_len + torch.randn(1)*0.2)\
                    + 0.02 * torch.randn(self.seq_len)
        else:
            motion = 0.01 * torch.randn(self.seq_len)
        # depth consistency: natural curvature for live, flat for spoof
        if live:
            depth = 0.4 * torch.sin(1.5 * np.pi * t / self.seq_len) + 0.6 + 0.02 * torch.randn(self.seq_len)
        else:
            depth = 0.9 * torch.ones(self.seq_len) + 0.01 * torch.randn(self.seq_len)

        return (rppg.float().unsqueeze(0),
                motion.float().unsqueeze(0),
                depth.float().unsqueeze(0),
                torch.tensor(1 if live else 0).long())


# ── Real dataset placeholders ────────────────────────────────────
class CASIADataset(Dataset):
    """Placeholder for CASIA-FASD dataset loading."""
    def __init__(self, root, split='train', seq_len=90):
        self.seq_len = seq_len
        self.samples = []  # [(video_path, label), ...]

    def __len__(self):
        return max(len(self.samples), 1)

    def __getitem__(self, idx):
        if len(self.samples) == 0:
            return SyntheticSignalDataset(1, self.seq_len)[0]
        # TODO: implement real video loading + signal extraction
        raise NotImplementedError("Real dataset loader: implement `_extract_signals(video_path)`")


class ReplayAttackDataset(CASIADataset):
    pass


class OULUNPUDataset(CASIADataset):
    pass


# ─────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for rppg, motion, depth, labels in loader:
        rppg = rppg.to(device)
        motion = motion.to(device)
        depth = depth.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits, _, _, _ = model(rppg, motion, depth)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for rppg, motion, depth, labels in loader:
        rppg, motion, depth, labels = [x.to(device) for x in (rppg, motion, depth, labels)]
        logits, _, _, _ = model(rppg, motion, depth)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seq-len', type=int, default=90)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output-dir', default='checkpoints')
    parser.add_argument('--dataset', choices=['synthetic', 'casia', 'replay', 'oulu'], default='synthetic')
    parser.add_argument('--data-root', default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    if args.device != 'auto':
        device = torch.device(args.device)
    print(f'Device: {device}')

    os.makedirs(args.output_dir, exist_ok=True)

    # Dataset
    if args.dataset == 'synthetic':
        train_ds = SyntheticSignalDataset(500, args.seq_len)
        val_ds = SyntheticSignalDataset(100, args.seq_len)
    elif args.dataset == 'casia':
        train_ds = CASIADataset(args.data_root, 'train', args.seq_len)
        val_ds = CASIADataset(args.data_root, 'test', args.seq_len)
    elif args.dataset == 'replay':
        train_ds = ReplayAttackDataset(args.data_root, 'train', args.seq_len)
        val_ds = ReplayAttackDataset(args.data_root, 'test', args.seq_len)
    elif args.dataset == 'oulu':
        train_ds = OULUNPUDataset(args.data_root, 'train', args.seq_len)
        val_ds = OULUNPUDataset(args.data_root, 'test', args.seq_len)
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    model = CrossAttentionFusion(signal_dim=64, num_heads=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0.0
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_acc = evaluate(model, val_loader, device)
        scheduler.step()

        print(f'Epoch {epoch+1:3d}/{args.epochs}  loss={train_loss:.4f}  val_acc={val_acc:.4f}')

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = os.path.join(args.output_dir, f'physiofusion_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'config': {'signal_dim': 64, 'num_heads': 4},
            }, ckpt_path)
            print(f'  → Saved {ckpt_path} (acc={val_acc:.4f})')

        # Save last checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
        }, os.path.join(args.output_dir, 'physiofusion_last.pt'))

    print(f'\nTraining done. Best val_acc={best_acc:.4f}')
    print(f'Checkpoints in {args.output_dir}/')


if __name__ == '__main__':
    main()
