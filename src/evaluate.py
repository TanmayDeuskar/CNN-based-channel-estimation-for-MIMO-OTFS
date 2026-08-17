import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import *
from dataset import (
    PreloadedSamplesFirstDataset,
    make_batch_collate_fn,
    make_train_val_indices,
    derive_pilot_coords_from_Phi,
)
from metrics import complex_from_channels, nmse_db, hard_threshold_per_antenna_batch
from model import ChannelEst2DNet
from utils import complex_to_str_grid, load_phi, load_state_dict, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--phi", type=Path, default=PHI_PATH)
    p.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "best_model.pth")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--alpha", type=float, default=HARD_THRESHOLD_ALPHA)
    p.add_argument("--save-sample", action="store_true")
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--antenna", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(SPLIT_SEED)

    probe = PreloadedSamplesFirstDataset(args.data_dir, indices=None, verbose=False)
    _, val_idx = make_train_val_indices(probe.total_samples, TRAIN_RATIO, SPLIT_SEED)
    ds_val = PreloadedSamplesFirstDataset(args.data_dir, indices=val_idx, verbose=True)

    phi = load_phi(args.phi)
    phi = phi if phi.shape[0] == ds_val.numDD else phi.T.copy()
    pilot_rows, pilot_cols = derive_pilot_coords_from_Phi(
        phi, ds_val.M, ds_val.N, ds_val.Nt
    )
    collate_fn = make_batch_collate_fn(
        phi, ds_val.M, ds_val.N, ds_val.Nt, pilot_rows, pilot_cols
    )

    loader = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_fn,
    )

    model = ChannelEst2DNet(2 * ds_val.Nt, ds_val.Nt).to(DEVICE)
    state = load_state_dict(args.checkpoint, DEVICE)
    try:
        model.load_state_dict(state)
    except RuntimeError:
        from utils import strip_module_prefix
        model.load_state_dict(strip_module_prefix(state))
    model.eval()

    total_err = 0.0
    total_power = 0.0
    total_err_thr = 0.0
    total_zeroed = 0
    total_locations = 0

    with torch.no_grad():
        for X, Y in loader:
            prediction = model(X.to(DEVICE)).cpu().numpy()
            target = Y.numpy()
            pred_c = complex_from_channels(prediction)
            target_c = complex_from_channels(target)

            total_err += float(np.sum(np.abs(pred_c - target_c) ** 2))
            total_power += float(np.sum(np.abs(target_c) ** 2))

            # Hard thresholding is an evaluation/post-processing step.
            pred_bm = np.transpose(pred_c, (0, 2, 3, 1))
            target_bm = np.transpose(target_c, (0, 2, 3, 1))
            pred_thr, mask = hard_threshold_per_antenna_batch(
                pred_bm,
                Htrue_batch=target_bm,
                alpha_per_ant=args.alpha,
                use_true=USE_TRUE_FOR_TAU,
            )
            total_err_thr += float(np.sum(np.abs(pred_thr - target_bm) ** 2))
            total_zeroed += int(np.sum(~mask))
            total_locations += mask.size

    raw_nmse = 10 * np.log10(total_err / max(total_power, 1e-12))
    thr_nmse = 10 * np.log10(total_err_thr / max(total_power, 1e-12))
    pct_zeroed = 100 * total_zeroed / max(total_locations, 1)

    print(f"Validation samples: {len(ds_val)}")
    print(f"Raw NMSE:           {raw_nmse:.3f} dB")
    print(f"Thresholded NMSE:   {thr_nmse:.3f} dB")
    print(f"Zeroed:             {pct_zeroed:.3f}%")

    if args.save_sample:
        if not (0 <= args.sample_index < len(ds_val)):
            raise ValueError("sample-index is outside the validation split")
        if not (0 <= args.antenna < ds_val.Nt):
            raise ValueError("antenna is outside the valid range")

        y, htrue = ds_val[args.sample_index]
        X, Y = collate_fn([(y, htrue)])
        with torch.no_grad():
            pred = model(X.to(DEVICE)).cpu().numpy()
        pred_c = complex_from_channels(pred)[0]
        hcap = np.transpose(pred_c, (1, 2, 0))
        ant = args.antenna

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(complex_to_str_grid(hcap[:, :, ant])).to_csv(
            RESULTS_DIR / f"Hcap_val_sample{args.sample_index}_ant{ant}.csv",
            index=False,
            header=False,
        )
        pd.DataFrame(complex_to_str_grid(htrue[:, :, ant])).to_csv(
            RESULTS_DIR / f"Htrue_val_sample{args.sample_index}_ant{ant}.csv",
            index=False,
            header=False,
        )
        print(f"Saved sample CSVs in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
