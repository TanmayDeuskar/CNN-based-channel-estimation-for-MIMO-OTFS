import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from config import *
from dataset import make_batch_collate_fn, derive_pilot_coords_from_Phi
from metrics import complex_from_channels, hard_threshold_per_antenna_batch
from model import ChannelEst2DNet
from utils import load_phi, load_state_dict


class NpyChunkDataset(Dataset):
    def __init__(self, y_memmap, hadd_memmap, layout):
        self.y = y_memmap
        self.hadd = hadd_memmap
        self.layout = layout
        if self.y.shape[0] != self.hadd.shape[0]:
            raise ValueError("HADD and yDD sample counts differ")

    def __len__(self):
        return int(self.y.shape[0])

    def __getitem__(self, idx):
        y = self.y[idx].astype(np.complex64)
        hadd = self.hadd[idx].astype(np.complex64)
        if self.layout == "P,Nt,N,M":
            hadd = np.transpose(hadd, (2, 1, 0))
        return y, hadd.astype(np.complex64)


def infer_layout(hadd):
    if hadd.ndim != 4:
        raise RuntimeError(f"Unexpected HADD shape: {hadd.shape}")
    if hadd.shape[1] <= 64 and hadd.shape[3] > 20:
        return int(hadd.shape[3]), int(hadd.shape[2]), int(hadd.shape[1]), "P,Nt,N,M"
    if hadd.shape[3] <= 64:
        return int(hadd.shape[1]), int(hadd.shape[2]), int(hadd.shape[3]), "P,M,N,Nt"
    return int(hadd.shape[3]), int(hadd.shape[2]), int(hadd.shape[1]), "P,Nt,N,M"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--snr-dir", type=Path, default=SNR_DATA_DIR)
    p.add_argument("--phi", type=Path, default=PHI_PATH)
    p.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "best_model.pth")
    p.add_argument("--output", type=Path, default=RESULTS_DIR / "eval_by_snr.csv")
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    phi_raw = load_phi(args.phi)
    state = load_state_dict(args.checkpoint, DEVICE)
    model = None
    rows = []

    hadd_files = sorted(args.snr_dir.glob("*_HADD.npy"))
    pairs = []
    for h in hadd_files:
        y = h.with_name(h.name.replace("_HADD.npy", "_yDD.npy"))
        if y.exists():
            pairs.append((h.stem.replace("_HADD", ""), h, y))

    if not pairs:
        raise RuntimeError(f"No HADD/yDD pairs found in {args.snr_dir}")

    for base, hadd_path, y_path in pairs:
        hadd_mm = np.load(hadd_path, mmap_mode="r")
        y_mm = np.load(y_path, mmap_mode="r")
        M, N, Nt, layout = infer_layout(hadd_mm)
        phi = phi_raw if phi_raw.shape[0] == M * N else phi_raw.T.copy()

        pilot_rows, pilot_cols = derive_pilot_coords_from_Phi(phi, M, N, Nt)
        collate_fn = make_batch_collate_fn(phi, M, N, Nt, pilot_rows, pilot_cols)
        ds = NpyChunkDataset(y_mm, hadd_mm, layout)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=NUM_WORKERS, collate_fn=collate_fn)

        if model is None:
            model = ChannelEst2DNet(2 * Nt, Nt).to(DEVICE)
            try:
                model.load_state_dict(state)
            except RuntimeError:
                from utils import strip_module_prefix
                model.load_state_dict(strip_module_prefix(state))
            model.eval()

        total_err = 0.0
        total_err_thr = 0.0
        total_power = 0.0
        total_zeroed = 0
        total_locations = 0

        with torch.no_grad():
            for X, Y in loader:
                pred = model(X.to(DEVICE)).cpu().numpy()
                target = Y.numpy()
                pred_c = complex_from_channels(pred)
                target_c = complex_from_channels(target)
                total_err += float(np.sum(np.abs(pred_c - target_c) ** 2))
                total_power += float(np.sum(np.abs(target_c) ** 2))

                pred_bm = np.transpose(pred_c, (0, 2, 3, 1))
                target_bm = np.transpose(target_c, (0, 2, 3, 1))
                pred_thr, mask = hard_threshold_per_antenna_batch(
                    pred_bm, target_bm, HARD_THRESHOLD_ALPHA, USE_TRUE_FOR_TAU
                )
                total_err_thr += float(np.sum(np.abs(pred_thr - target_bm) ** 2))
                total_zeroed += int(np.sum(~mask))
                total_locations += mask.size

        nmse = total_err / max(total_power, 1e-12)
        nmse_thr = total_err_thr / max(total_power, 1e-12)
        match = re.search(r"snr[_-]?(\d+)", base, flags=re.I)
        snr = int(match.group(1)) if match else None

        rows.append({
            "chunk": base,
            "snr_db": snr,
            "num_samples": len(ds),
            "nmse_db": 10 * np.log10(nmse),
            "nmse_thresholded_db": 10 * np.log10(nmse_thr),
            "pct_zeroed": 100 * total_zeroed / max(total_locations, 1),
        })

    result = pd.DataFrame(rows).sort_values("snr_db")
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
