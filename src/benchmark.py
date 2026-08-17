"""Latency benchmarking for the CNN and optional preprocessing pipeline."""
import argparse
import glob
import time
from pathlib import Path

import numpy as np
import torch

from config import *
from dataset import make_batch_collate_fn, derive_pilot_coords_from_Phi
from model import ChannelEst2DNet
from utils import load_phi, load_state_dict


def infer_layout(hadd):
    if hadd.shape[1] <= 64 and hadd.shape[3] > 20:
        return int(hadd.shape[3]), int(hadd.shape[2]), int(hadd.shape[1]), "P,Nt,N,M"
    if hadd.shape[3] <= 64:
        return int(hadd.shape[1]), int(hadd.shape[2]), int(hadd.shape[3]), "P,M,N,Nt"
    return int(hadd.shape[3]), int(hadd.shape[2]), int(hadd.shape[1]), "P,Nt,N,M"


def orient_sample(hadd, layout):
    if layout == "P,Nt,N,M":
        return np.transpose(hadd, (2, 1, 0)).astype(np.complex64)
    return hadd.astype(np.complex64)


def stats(values):
    return {
        "mean_s": float(np.mean(values)),
        "median_s": float(np.median(values)),
        "std_s": float(np.std(values)),
        "min_s": float(np.min(values)),
    }


def benchmark(hadd_path, y_path, checkpoint, phi_path, sample_idx=0, repeats=300, warmup=20):
    device = DEVICE
    phi_raw = load_phi(phi_path)
    hadd_mm = np.load(hadd_path, mmap_mode="r")
    y_mm = np.load(y_path, mmap_mode="r")
    M, N, Nt, layout = infer_layout(hadd_mm)
    phi = phi_raw if phi_raw.shape[0] == M * N else phi_raw.T.copy()

    pilot_rows, pilot_cols = derive_pilot_coords_from_Phi(phi, M, N, Nt)
    collate = make_batch_collate_fn(phi, M, N, Nt, pilot_rows, pilot_cols)

    state = load_state_dict(checkpoint, device)
    model = ChannelEst2DNet(2 * Nt, Nt).to(device)
    try:
        model.load_state_dict(state)
    except RuntimeError:
        from utils import strip_module_prefix
        model.load_state_dict(strip_module_prefix(state))
    model.eval()

    y = y_mm[sample_idx].astype(np.complex64)
    h = orient_sample(hadd_mm[sample_idx].astype(np.complex64), layout)
    batch = [(y, h)]

    with torch.no_grad():
        for _ in range(warmup):
            X, _ = collate(batch)
            _ = model(X.to(device))
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    collate_times = []
    model_times = []
    e2e_times = []

    for _ in range(repeats):
        t0 = time.perf_counter()
        X, _ = collate(batch)
        collate_times.append(time.perf_counter() - t0)

        Xd = X.to(device)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.no_grad():
                _ = model(Xd)
            end.record()
            torch.cuda.synchronize()
            model_times.append(start.elapsed_time(end) * 1e-3)
        else:
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(Xd)
            model_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        X2, _ = collate(batch)
        with torch.no_grad():
            _ = model(X2.to(device))
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        e2e_times.append(time.perf_counter() - t0)

    result = {
        "device": device,
        "model": stats(np.array(model_times)),
        "collate": stats(np.array(collate_times)),
        "e2e": stats(np.array(e2e_times)),
    }
    print(result)
    print(f"Model mean latency: {result['model']['mean_s'] * 1e3:.4f} ms")
    print(f"E2E mean latency:   {result['e2e']['mean_s'] * 1e3:.4f} ms")
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--phi", type=Path, default=PHI_PATH)
    p.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "best_model.pth")
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--repeats", type=int, default=300)
    p.add_argument("--warmup", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    hadd_files = sorted(args.data_dir.glob("*_HADD.npy"))
    if not hadd_files:
        raise RuntimeError(f"No HADD files found in {args.data_dir}")
    hadd = hadd_files[0]
    y = hadd.with_name(hadd.name.replace("_HADD.npy", "_yDD.npy"))
    if not y.exists():
        raise RuntimeError(f"Missing yDD pair for {hadd}")
    benchmark(hadd, y, args.checkpoint, args.phi, args.sample, args.repeats, args.warmup)


if __name__ == "__main__":
    main()
