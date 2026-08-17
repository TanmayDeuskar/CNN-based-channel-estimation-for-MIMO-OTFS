import math
import numpy as np


def complex_from_channels(tensor):
    arr = tensor
    nt = arr.shape[1] // 2
    return arr[:, :nt] + 1j * arr[:, nt:]


def nmse_db(pred_complex, target_complex):
    err = np.sum(np.abs(pred_complex - target_complex) ** 2)
    power = np.sum(np.abs(target_complex) ** 2)
    if power <= 0:
        return float("nan")
    return 10.0 * math.log10(err / power)


def hard_threshold_per_antenna_batch(
    hpred_batch, htrue_batch=None, alpha_per_ant=0.01, use_true=False
):
    B, M, N, Nt = hpred_batch.shape
    if np.isscalar(alpha_per_ant):
        alpha_vec = np.full(Nt, float(alpha_per_ant))
    else:
        alpha_vec = np.asarray(alpha_per_ant, dtype=float)
        if alpha_vec.size != Nt:
            raise ValueError("alpha_per_ant must be scalar or length Nt")

    hpred_thr = np.zeros_like(hpred_batch)
    mask = np.zeros((B, M, N, Nt), dtype=bool)

    for b in range(B):
        for t in range(Nt):
            if use_true:
                if htrue_batch is None:
                    raise ValueError("Htrue is required when use_true=True")
                base = np.abs(htrue_batch[b, :, :, t])
            else:
                base = np.abs(hpred_batch[b, :, :, t])
            tau = alpha_vec[t] * float(base.max() if base.size else 0.0)
            keep = np.abs(hpred_batch[b, :, :, t]) >= tau
            mask[b, :, :, t] = keep
            hpred_thr[b, :, :, t] = hpred_batch[b, :, :, t] * keep

    return hpred_thr, mask
