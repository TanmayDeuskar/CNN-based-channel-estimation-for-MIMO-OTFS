import torch
import torch.nn as nn


class WeightedComplexMSELoss(nn.Module):
   
    def __init__(
        self,
        threshold_ratio=0.01,
        gamma=0.1,
        l1_lambda=1e-2,
        soft_thresh_ratio=0.01,
        eps=1e-12,
    ):
        super().__init__()
        self.threshold_ratio = float(threshold_ratio)
        self.gamma = float(gamma)
        self.l1_lambda = float(l1_lambda)
        self.soft_thresh_ratio = float(soft_thresh_ratio)
        self.eps = float(eps)

    def forward(self, pred, target):
        b, c, m, n = pred.shape
        nt = c // 2

        pred_real = pred[:, :nt]
        pred_imag = pred[:, nt:]
        targ_real = target[:, :nt]
        targ_imag = target[:, nt:]

        targ_mag_sq = torch.sum(
            targ_real * targ_real + targ_imag * targ_imag, dim=1
        )
        targ_mag = torch.sqrt(targ_mag_sq + self.eps)

        max_mag = torch.amax(targ_mag.view(b, -1), dim=1).view(b, 1, 1)
        max_mag = max_mag + self.eps
        thresh = self.threshold_ratio * max_mag
        tau = (self.soft_thresh_ratio * max_mag).view(b, 1, 1, 1)

        pred_mag_sq = pred_real * pred_real + pred_imag * pred_imag
        pred_mag = torch.sqrt(pred_mag_sq + self.eps)
        shrink = torch.relu(pred_mag - tau)
        scale = shrink / (pred_mag + self.eps)

        pred_real_thr = pred_real * scale
        pred_imag_thr = pred_imag * scale

        weights = torch.where(
            targ_mag >= thresh,
            torch.ones_like(targ_mag),
            torch.full_like(targ_mag, self.gamma),
        ).unsqueeze(1)

        se = (pred_real_thr - targ_real) ** 2 + (pred_imag_thr - targ_imag) ** 2
        weighted_mse = torch.mean(weights * torch.sum(se, dim=1))

        pred_mag_sq_raw = torch.sum(
            pred_real * pred_real + pred_imag * pred_imag, dim=1
        )
        pred_mag_raw = torch.sqrt(pred_mag_sq_raw + self.eps)
        small_mask = (targ_mag < thresh).to(dtype=pred_mag_raw.dtype)
        masked_l1 = torch.mean(pred_mag_raw * small_mask)

        return weighted_mse + self.l1_lambda * masked_l1
