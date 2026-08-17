import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from config import *
from dataset import (
    PreloadedSamplesFirstDataset,
    make_batch_collate_fn,
    make_train_val_indices,
    derive_pilot_coords_from_Phi,
)
from loss import WeightedComplexMSELoss
from model import ChannelEst2DNet
from utils import load_phi, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--phi", type=Path, default=PHI_PATH)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "best_model.pth")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(SPLIT_SEED)

    print("Device:", DEVICE)

    # CHANGE: discover the dataset once and create one shared deterministic split.
    # This fixes the original independent train/val randomization issue.
    probe = PreloadedSamplesFirstDataset(
        args.data_dir,
        indices=None,
        verbose=True,
    )
    train_idx, val_idx = make_train_val_indices(
        probe.total_samples, TRAIN_RATIO, SPLIT_SEED
    )

    ds_train = PreloadedSamplesFirstDataset(
        args.data_dir, indices=train_idx, verbose=True
    )
    ds_val = PreloadedSamplesFirstDataset(
        args.data_dir, indices=val_idx, verbose=True
    )

    print("Train samples:", len(ds_train), "Val samples:", len(ds_val))

    phi = load_phi(args.phi)
    if phi.shape[0] != ds_train.numDD:
        phi = phi.T.copy()
    if phi.shape[0] != ds_train.numDD:
        raise RuntimeError(f"Phi shape {phi.shape} incompatible with numDD={ds_train.numDD}")

    pilot_rows, pilot_cols = derive_pilot_coords_from_Phi(
        phi, ds_train.M, ds_train.N, ds_train.Nt
    )
    collate_fn = make_batch_collate_fn(
        phi, ds_train.M, ds_train.N, ds_train.Nt, pilot_rows, pilot_cols
    )

    loader_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_fn,
    )
    loader_val = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_fn,
    )

    sample_x, _ = next(iter(loader_train))
    in_channels = sample_x.shape[1]
    if in_channels != 2 * ds_train.Nt:
        raise RuntimeError("Unexpected model input channel count")

    model = ChannelEst2DNet(in_channels=in_channels, Nt=ds_train.Nt).to(DEVICE)
    criterion = WeightedComplexMSELoss(
        threshold_ratio=THRESHOLD_RATIO,
        gamma=GAMMA,
        l1_lambda=L1_LAMBDA,
        soft_thresh_ratio=SOFT_THRESH_RATIO,
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        n_train = 0
        start = time.time()

        for X, Y in loader_train:
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            optimizer.zero_grad()
            prediction = model(X)
            loss = criterion(prediction, Y)
            loss.backward()
            optimizer.step()

            batch_size = X.size(0)
            running_loss += loss.item() * batch_size
            n_train += batch_size

        train_loss = running_loss / max(n_train, 1)

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        total_err = 0.0
        total_power = 0.0

        with torch.no_grad():
            for Xv, Yv in loader_val:
                Xv, Yv = Xv.to(DEVICE), Yv.to(DEVICE)
                Ypv = model(Xv)

                val_loss = criterion(Ypv, Yv)
                val_loss_sum += val_loss.item() * Xv.size(0)
                val_count += Xv.size(0)

                pred = Ypv.cpu().numpy()
                target = Yv.cpu().numpy()
                pred_c = pred[:, :ds_train.Nt] + 1j * pred[:, ds_train.Nt:]
                target_c = target[:, :ds_train.Nt] + 1j * target[:, ds_train.Nt:]
                total_err += float(np.sum(np.abs(pred_c - target_c) ** 2))
                total_power += float(np.sum(np.abs(target_c) ** 2))

        val_loss = val_loss_sum / max(val_count, 1)
        nmse_linear = total_err / max(total_power, 1e-12)
        nmse_db_value = 10.0 * math.log10(nmse_linear)
        elapsed = time.time() - start

        print(
            f"Epoch {epoch + 1}/{args.epochs}: "
            f"train_loss={train_loss:.6e} "
            f"val_loss={val_loss:.6e} "
            f"val_NMSE={nmse_db_value:.3f} dB "
            f"time={elapsed:.1f}s"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": {
                        "M": ds_train.M,
                        "N": ds_train.N,
                        "Nt": ds_train.Nt,
                        "train_samples": len(ds_train),
                        "val_samples": len(ds_val),
                        "split_seed": SPLIT_SEED,
                    },
                },
                args.checkpoint,
            )
            print(f"Saved best model: {args.checkpoint}")


if __name__ == "__main__":
    main()
