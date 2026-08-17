import argparse
from pathlib import Path

import pandas as pd
import torch

from config import *
from dataset import PreloadedSamplesFirstDataset, make_batch_collate_fn, make_train_val_indices, derive_pilot_coords_from_Phi
from metrics import complex_from_channels
from model import ChannelEst2DNet
from utils import complex_to_str_grid, load_phi, load_state_dict, set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--phi", type=Path, default=PHI_PATH)
    p.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "best_model.pth")
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--antenna", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = p.parse_args()

    set_seed(SPLIT_SEED)
    probe = PreloadedSamplesFirstDataset(args.data_dir, indices=None, verbose=False)
    _, val_idx = make_train_val_indices(probe.total_samples, TRAIN_RATIO, SPLIT_SEED)
    ds = PreloadedSamplesFirstDataset(args.data_dir, indices=val_idx, verbose=False)

    if not 0 <= args.sample < len(ds):
        raise ValueError("sample index out of range")
    if not 0 <= args.antenna < ds.Nt:
        raise ValueError("antenna index out of range")

    phi = load_phi(args.phi)
    phi = phi if phi.shape[0] == ds.numDD else phi.T.copy()
    pr, pc = derive_pilot_coords_from_Phi(phi, ds.M, ds.N, ds.Nt)
    collate = make_batch_collate_fn(phi, ds.M, ds.N, ds.Nt, pr, pc)

    model = ChannelEst2DNet(2 * ds.Nt, ds.Nt).to(DEVICE)
    state = load_state_dict(args.checkpoint, DEVICE)
    try:
        model.load_state_dict(state)
    except RuntimeError:
        from utils import strip_module_prefix
        model.load_state_dict(strip_module_prefix(state))
    model.eval()

    y, htrue = ds[args.sample]
    X, _ = collate([(y, htrue)])
    with torch.no_grad():
        pred = model(X.to(DEVICE)).cpu().numpy()
    hpred = complex_from_channels(pred)[0]
    hpred = hpred.transpose(1, 2, 0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ant = args.antenna
    pd.DataFrame(complex_to_str_grid(hpred[:, :, ant])).to_csv(
        args.output_dir / f"Hcap_val_sample{args.sample}_ant{ant}.csv",
        index=False, header=False,
    )
    pd.DataFrame(complex_to_str_grid(htrue[:, :, ant])).to_csv(
        args.output_dir / f"Htrue_val_sample{args.sample}_ant{ant}.csv",
        index=False, header=False,
    )
    print(f"Saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
