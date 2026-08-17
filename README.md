# Lightweight CNN-Based Channel Estimation for MIMO-OTFS

A cleaned, reproducible implementation of the lightweight residual CNN used for MIMO-OTFS channel estimation in the delay-Doppler domain.


## Experiment configuration

| Parameter | Value |
|---|---:|
| OTFS grid | 256 × 8 |
| Transmit antennas | 16 |
| Receive antennas | 1 |
| Total samples | 40,000 |
| Training split | 36,000 |
| Validation split | 4,000 |
| Training SNR | 10 dB |
| Epochs | 20 |
| Batch size | 16 |
| Learning rate | 1e-3 |
| Optimizer | Adam |
| Model parameters | 277,664 |
| Evaluation SNRs | 5, 10, 15, 20 dB |

The model retains the original seven-layer residual-CNN structure and the weighted complex MSE + masked L1 objective.


## Data layout

The training directory is expected to contain paired files such as:

```text
chunk_000_HADD.npy
chunk_000_yDD.npy
chunk_001_HADD.npy
chunk_001_yDD.npy
...
```

The variable-SNR evaluation directory should contain equivalent pairs whose filenames encode the SNR, for example:

```text
chunk_snr05_HADD.npy
chunk_snr05_yDD.npy
chunk_snr10_HADD.npy
chunk_snr10_yDD.npy
...
```

The sensing matrix is expected at:

```text
data/Phi_pilotWeighted_padded.mat
```

Datasets and checkpoints are intentionally **not committed to Git**.

## Installation

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then:

```bash
pip install -r requirements.txt
```

Install a PyTorch build appropriate for the CUDA version on the target machine if GPU acceleration is desired.

## Training

From the repository root:

```bash
python src/train.py --data-dir data/training --phi data/Phi_pilotWeighted_padded.mat
```

The best checkpoint is saved to:

```text
checkpoints/best_model.pth
```

## Validation

```bash
python src/evaluate.py --checkpoint checkpoints/best_model.pth
```

To export one validation sample and antenna:

```bash
python src/evaluate.py --checkpoint checkpoints/best_model.pth --save-sample --sample-index 0 --antenna 0
```

## SNR evaluation

```bash
python src/evaluate_snr.py \
    --snr-dir data/variable_snr \
    --phi data/Phi_pilotWeighted_padded.mat \
    --checkpoint checkpoints/best_model.pth
```

## Latency benchmark

```bash
python src/benchmark.py \
    --data-dir data/variable_snr \
    --phi data/Phi_pilotWeighted_padded.mat \
    --checkpoint checkpoints/best_model.pth \
    --repeats 300 \
    --warmup 20
```

The benchmark reports CNN forward latency separately from the full preprocessing + model path. This distinction is important when comparing the CNN's forward-pass timing with classical algorithms.

