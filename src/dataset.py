import bisect
import os
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset


def discover_chunk_bases(npy_dir):
    npy_dir = os.path.abspath(npy_dir)
    if not os.path.isdir(npy_dir):
        raise RuntimeError(f"Dataset directory does not exist: {npy_dir}")

    hadd_files = sorted(
        f for f in os.listdir(npy_dir) if f.endswith("_HADD.npy")
    )
    bases = []
    for filename in hadd_files:
        base = filename[:-9]
        if os.path.exists(os.path.join(npy_dir, base + "_yDD.npy")):
            bases.append(base)
    if not bases:
        raise RuntimeError(f"No matching HADD/yDD pairs found in {npy_dir}")
    return bases


class PreloadedSamplesFirstDataset(Dataset):
    

    def __init__(
        self,
        data_dir,
        indices=None,
        files=None,
        files_per_group=4,
        groups_in_memory=2,
        verbose=True,
    ):
        self.data_dir = os.path.abspath(data_dir)
        self.npy_dir = os.path.abspath(os.environ.get("PRELOAD_NPY_DIR", self.data_dir))
        self.files_per_group = max(1, int(files_per_group))
        self.groups_in_memory = max(1, int(groups_in_memory))
        self.verbose = verbose

        self._bases_all = files if files is not None else discover_chunk_bases(self.npy_dir)
        self.files = list(self._bases_all)

        if verbose:
            print(f"[Dataset] found {len(self.files)} chunks in {self.npy_dir}")

        self.samples_per_file = []
        for base in self.files:
            path = os.path.join(self.npy_dir, base + "_yDD.npy")
            self.samples_per_file.append(int(np.load(path, mmap_mode="r").shape[0]))

        self.total_samples = int(sum(self.samples_per_file))
        self.cum_counts = np.cumsum([0] + self.samples_per_file)

        if indices is None:
            self.global_indices = list(range(self.total_samples))
        else:
            self.global_indices = [int(i) for i in indices]

        self.numDD = None
        self.Nt = None
        self.N = None
        self.M = None
        self._group_cache = OrderedDict()

        if self.files:
            self._ensure_group_loaded(self._group_start_for_file(0))
            first = next(iter(self._group_cache.values()))
            self.numDD = int(first["y_list"][0].shape[1])
            hadd0 = first["hadd_list"][0]
            if hadd0.ndim != 4:
                raise RuntimeError(f"HADD expected 4D, got {hadd0.shape}")
            self.Nt = int(hadd0.shape[1])
            self.N = int(hadd0.shape[2])
            self.M = int(hadd0.shape[3])

        if verbose:
            print(
                f"[Dataset] samples={len(self.global_indices)}, "
                f"M={self.M}, N={self.N}, Nt={self.Nt}"
            )

    def __len__(self):
        return len(self.global_indices)

    def _group_start_for_file(self, file_idx):
        return (file_idx // self.files_per_group) * self.files_per_group

    def _files_in_group(self, group_start):
        return list(range(group_start, min(group_start + self.files_per_group, len(self.files))))

    def _ensure_group_loaded(self, group_start):
        if group_start in self._group_cache:
            self._group_cache.move_to_end(group_start)
            return self._group_cache[group_start]

        y_list, hadd_list, sizes = [], [], []
        for file_idx in self._files_in_group(group_start):
            base = self.files[file_idx]
            y_path = os.path.join(self.npy_dir, base + "_yDD.npy")
            hadd_path = os.path.join(self.npy_dir, base + "_HADD.npy")
            y_mem = np.load(y_path, mmap_mode="r")
            hadd_mem = np.load(hadd_path, mmap_mode="r")

            y_arr = y_mem if y_mem.dtype == np.complex64 else y_mem.astype(np.complex64)
            hadd_arr = hadd_mem if hadd_mem.dtype == np.complex64 else hadd_mem.astype(np.complex64)
            y_list.append(y_arr)
            hadd_list.append(hadd_arr)
            sizes.append(int(y_arr.shape[0]))

        entry = {
            "bases": self._files_in_group(group_start),
            "sizes": sizes,
            "y_list": y_list,
            "hadd_list": hadd_list,
        }
        self._group_cache[group_start] = entry
        self._group_cache.move_to_end(group_start)

        while len(self._group_cache) > self.groups_in_memory:
            self._group_cache.popitem(last=False)
        return entry

    def _global_to_file_local(self, global_idx):
        file_idx = bisect.bisect_right(self.cum_counts, global_idx) - 1
        local_idx = int(global_idx - self.cum_counts[file_idx])
        return file_idx, local_idx

    def __getitem__(self, idx):
        global_idx = self.global_indices[idx]
        file_idx, local_idx = self._global_to_file_local(global_idx)
        group_start = self._group_start_for_file(file_idx)
        entry = self._ensure_group_loaded(group_start)

        file_pos = entry["bases"].index(file_idx)
        y_arr = entry["y_list"][file_pos]
        hadd_arr = entry["hadd_list"][file_pos]

        y_row = y_arr[local_idx].astype(np.complex64)
        hadd_sample = hadd_arr[local_idx].astype(np.complex64)
        Hadd = np.transpose(hadd_sample, (2, 1, 0)).astype(np.complex64)
        return y_row, Hadd


def make_train_val_indices(total_samples, train_ratio=0.9, seed=42):
    """Create one deterministic, non-overlapping sample-level split."""
    rng = np.random.default_rng(seed)
    indices = np.arange(total_samples, dtype=np.int64)
    rng.shuffle(indices)
    split = int(np.floor(total_samples * train_ratio))
    split = min(max(1, split), total_samples - 1)
    return indices[:split].tolist(), indices[split:].tolist()


def derive_pilot_coords_from_Phi(Phi, M, N, Nt):
    Phi = np.asarray(Phi)
    numDD, Ptot = Phi.shape
    if numDD != M * N:
        raise RuntimeError(f"Phi numDD ({numDD}) != M*N ({M}*{N})")
    if Ptot % Nt != 0:
        raise RuntimeError(f"Phi Ptot ({Ptot}) not divisible by Nt ({Nt})")

    P = Ptot // Nt
    pilot_rows = np.empty(P, dtype=int)
    pilot_cols = np.empty(P, dtype=int)
    for p in range(P):
        col = Phi[:, p]
        idx = int(np.argmax(np.abs(col)))
        r, c = np.unravel_index(idx, (M, N), order="C")
        pilot_rows[p] = r
        pilot_cols[p] = c
    return pilot_rows, pilot_cols


def scatter_batch_feats_to_X(feats, Nt, M, N, pilot_rows, pilot_cols):
    Ptot, B = feats.shape
    if Ptot == Nt * M * N:
        return feats.T.reshape((B, Nt, M, N), order="C")
    if Ptot % Nt != 0:
        raise RuntimeError("Feature count is not divisible by Nt")

    P = Ptot // Nt
    if len(pilot_rows) != P or len(pilot_cols) != P:
        raise RuntimeError("Pilot coordinate count does not match Phi")

    feat3 = feats.reshape((Nt, P, B), order="C")
    X = np.zeros((B, Nt, M, N), dtype=np.complex64)
    for t in range(Nt):
        X[:, t, pilot_rows, pilot_cols] = feat3[t].T
    return X


def make_batch_collate_fn(Phi, M, N, Nt, pilot_rows, pilot_cols):
    Phi = np.asarray(Phi).astype(np.complex64)

    def collate(batch):
        Ys = np.stack([b[0] for b in batch], axis=1).astype(np.complex64)
        Hs = np.stack([b[1] for b in batch], axis=0).astype(np.complex64)

        feats = Phi.conj().T.dot(Ys)
        Xc = scatter_batch_feats_to_X(
            feats, Nt, M, N, pilot_rows, pilot_cols
        )

        X_batch = np.concatenate([Xc.real, Xc.imag], axis=1).astype(np.float32)

        Yc = np.transpose(Hs, (0, 3, 1, 2))
        Y_batch = np.concatenate([Yc.real, Yc.imag], axis=1).astype(np.float32)

        return torch.from_numpy(X_batch), torch.from_numpy(Y_batch)

    return collate
