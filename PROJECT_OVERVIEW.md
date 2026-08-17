# Project Technical Overview

## 1. About Project

This project develops a lightweight convolutional neural network (CNN) for **MIMO-OTFS channel estimation in the delay-Doppler (DD) domain**.

The goal is to recover the complex DD-domain channel response from a small set of pilot observations while exploiting the fact that wireless channels are typically **sparse in delay and Doppler**.

The overall pipeline is:

```text
OTFS / MIMO channel simulation in MATLAB
              |
              v
       Generate pilot observations
              |
              v
       Construct sensing matrix Phi
              |
              v
       Generate noisy received DD signal y
              |
              v
             Phi^H y
              |
              v
   Scatter pilot features onto DD grid
              |
              v
      Real/imaginary channels
              |
              v
       Lightweight residual CNN
              |
              v
     Estimated complex DD channel
```

The implemented configuration is:

| Parameter | Value |
|---|---:|
| Delay bins, M | 256 |
| Doppler bins, N | 8 |
| Transmit antennas, Nt | 16 |
| Receive antennas, Nr | 1 |
| DD grid size | 256 × 8 |
| Pilot overhead | 25% |
| Training SNR | 10 dB |
| Training samples | 40,000 |
| Train / validation | 36,000 / 4,000 |
| CNN parameters | 277,664 |

---

## 2. DD Domain

OTFS represents the wireless channel in the delay-Doppler domain rather than directly in the time-frequency domain.

For a MIMO system with $N_t$ transmit antennas, the channel can be represented as

$$
H[m,n,t],
$$

where:

- $m$ is the delay-bin index,
- $n$ is the Doppler-bin index,
- $t$ is the transmit-antenna index.

For this project,

$$
H \in \mathbb{C}^{M\times N\times N_t} = \mathbb{C}^{256\times 8\times 16}.
$$

A useful property of this representation is that practical wireless channels tend to have only a relatively small number of significant delay-Doppler coefficients. The estimator therefore does not need to reconstruct an arbitrary dense tensor; it can exploit the structure and sparsity of the channel.

---

## 3. MATLAB: channel and dataset generation

The MATLAB scripts generate the data used by the neural network.

### 3.1 MIMO channel model

The simulation uses an NR CDL-D channel with:

- carrier frequency: 2.15 GHz,
- velocity: 360 km/h,
- 16 transmit antennas,
- 1 receive antenna,
- a 4 × 4 transmit uniform rectangular array.

The maximum Doppler shift is obtained from the carrier frequency and velocity,

$$
f_D = \frac{v f_c}{c},
$$

where $v$ is the terminal velocity, $f_c$ is the carrier frequency, and $c$ is the speed of light.

The OTFS parameters are:

$$
M=256,\qquad N=8,
$$

with

$$
\Delta f = 15\text{ kHz}
$$

and an FFT size of 1024.

---

### 3.2 Pilot configuration

A 25% pilot overhead is used.

The number of DD-domain pilot positions is therefore

$$
P = 0.25MN = 0.25(256)(8) = 512.
$$

The pilot locations are generated using a fixed random seed so that the measurement configuration is reproducible.

Complex Gaussian pilot symbols are also generated once and reused:

$$
x_p \sim \mathcal{CN}(0,1).
$$

The fixed pilot pattern means that different channel realizations are observed using the same measurement configuration.

---

### 3.3 OTFS modulation and channel probing

For each transmit antenna and pilot location, the MATLAB code creates a unit DD-domain response and passes it through the OTFS modulation/channel/demodulation process.

This is used to construct a sensing matrix

$$
\Phi.
$$

For the implemented configuration,

$$
\Phi \in \mathbb{C}^{MN\times PN_t} = \mathbb{C}^{2048\times8192}.
$$

Each column of $\Phi$ represents the DD-domain response associated with a particular pilot and transmit antenna.

Conceptually,

$$
\Phi = \begin{bmatrix} \phi_{1,1} & \cdots & \phi_{1,P} & \phi_{2,1} & \cdots & \phi_{N_t,P} \end{bmatrix}.
$$

The MATLAB script saves $\Phi$, the pilot locations, pilot symbols, and padding information so that the Python pipeline can reproduce the same measurement model.

---

### 3.4 Channel ground truth

For every channel realization, the MATLAB code probes the channel separately for each transmit antenna and obtains

$$
H_{\mathrm{ADD}} \in \mathbb{C}^{M\times N\times N_t}.
$$

The channel is generated in the DD domain using an OTFS pilot waveform, passed through the CDL channel, aligned using the calculated delay padding, and then demodulated back into the DD domain.

Weak channel coefficients are removed using a power threshold relative to the maximum channel power. This produces the sparse ground-truth channel used for training.

---

## 4. Measurement model

The Python training pipeline uses the complex received DD-domain observation $y$ and the sensing matrix $\Phi$.

The measurement model is

$$
\mathbf{y} = \Phi\mathbf{h} + \mathbf{w},
$$

where:

- $\mathbf{h}$ contains the channel coefficients corresponding to the pilot positions,
- $\mathbf{y}$ is the received DD-domain observation,
- $\Phi$ is the sensing matrix,
- $\mathbf{w}$ is complex AWGN.

For the training dataset, the observation SNR is 10 dB.

The Python code then forms a matched-filter / correlation-style feature:

$$
\mathbf{z} = \Phi^{H}\mathbf{y}.
$$

This produces a vector containing the correlation of the received observation with the sensing responses associated with the pilots.

---

## 5. CNN input

Only the pilot positions are directly observed.

The vector

$$
\mathbf{z}=\Phi^H\mathbf{y}
$$

is therefore scattered back onto the corresponding positions of the DD grid.

For each transmit antenna, this gives a sparse

$$
M\times N
$$

feature map.

The resulting complex feature tensor has dimensions

$$
N_t\times M\times N.
$$

Because the CNN operates on real-valued tensors, the complex input is separated into real and imaginary components:

$$
X = \left[ \Re\{Z\}; \Im\{Z\} \right].
$$

Thus the CNN input has

$$
2N_t = 32
$$

channels and dimensions

$$
32\times256\times8.
$$

The same real/imaginary representation is used for the target channel.

---

## 6. Lightweight CNN architecture

The estimator is a fully convolutional residual CNN.

Its high-level structure is:

```text
Input
32 × 256 × 8
      |
      v
3×3 Conv: 32 channels
      |
      v
3×3 Conv: 64 channels
      |
      v
Residual Block
      |
Residual Block
      |
Residual Block
      |
      v
3×3 Conv: 32 channels
      |
      v
3×3 Conv: 32 channels
      |
      v
Output
32 × 256 × 8
```

Each residual block contains two $3\times3$ convolutions with batch normalization and ReLU activation, followed by an identity skip connection:

$$
\mathbf{x}_{out} = \mathrm{ReLU} \left( F(\mathbf{x})+\mathbf{x} \right).
$$

The network does not use fully connected layers. Consequently, the spatial DD dimensions are preserved throughout the network.

The final 32 channels correspond to

$$
2N_t = 32
$$

real-valued channels:

- 16 real-valued channel maps,
- 16 imaginary-valued channel maps.

These are recombined to form

$$
\hat{H} = \hat{H}_{\mathrm{Re}} + j\hat{H}_{\mathrm{Im}}.
$$

The network contains **277,664 trainable parameters**.

---

## 7. Training objective

The network is trained with a weighted complex-domain reconstruction loss.

The target magnitude is first aggregated across transmit antennas:

$$
|H[m,n]| = \sqrt{ \sum_{t=1}^{N_t} \left( H_{\mathrm{Re}}[m,n,t]^2+ H_{\mathrm{Im}}[m,n,t]^2 \right) }.
$$

A threshold is then calculated for each sample:

$$
\tau = \alpha \max_{m,n}|H[m,n]|.
$$

Strong DD locations receive full weight while weak locations receive a reduced weight:

$$
w[m,n]= \begin{cases} 1, & |H[m,n]|\geq\tau,\\ \gamma, & |H[m,n]|<\tau. \end{cases}
$$

This focuses the reconstruction loss on significant channel paths.

---

### 7.1 Soft thresholding

Before the reconstruction MSE is calculated, the predicted complex coefficients undergo soft thresholding.

For a predicted complex coefficient $z$,

$$
\mathcal{S}_{\tau}(z) = \max(|z|-\tau,0) \frac{z}{|z|+\epsilon}.
$$

This encourages small predicted coefficients to shrink toward zero while retaining the phase of larger coefficients.

---

### 7.2 Weighted complex MSE

The real and imaginary errors are combined as

$$
e = (\hat{H}_{\mathrm{Re}}-H_{\mathrm{Re}})^2 + (\hat{H}_{\mathrm{Im}}-H_{\mathrm{Im}})^2.
$$

The weighted reconstruction term is approximately

$$
L_{\mathrm{MSE}} = \mathrm{mean} \left( w[m,n] \sum_t e[m,n,t] \right).
$$

---

### 7.3 Sparsity term

A masked L1 penalty is additionally applied to predicted energy at locations where the target is weak:

$$
L_1 = \mathrm{mean} \left( |\hat{H}[m,n]|\,\mathbf{1}_{|H[m,n]|<\tau} \right).
$$

The total training objective is

$$
L = L_{\mathrm{MSE}} + \lambda L_1
$$

with the implemented training configuration using:

$$
\alpha=0.05,\qquad \gamma=0.05,\qquad \lambda=10^{-2},
$$

and a soft-threshold ratio of 0.01.

---

## 8. Training

The model is trained using Adam with:

| Training parameter | Value |
|---|---:|
| Samples | 40,000 |
| Training samples | 36,000 |
| Validation samples | 4,000 |
| SNR | 10 dB |
| Epochs | 20 |
| Batch size | 16 |
| Learning rate | $10^{-3}$ |
| Optimizer | Adam |

The cleaned repository uses a **single deterministic train/validation split** so that the two sets are guaranteed to be disjoint.

---

## 9. Evaluation

The model produces a complex channel estimate

$$
\hat{H}\in\mathbb{C}^{256\times8\times16}.
$$

Performance is measured using normalized mean squared error:

$$
\mathrm{NMSE} = \frac{ \|\hat{H}-H\|_2^2 }{ \|H\|_2^2 }.
$$

The value is reported in decibels:

$$
\mathrm{NMSE}_{dB} = 10\log_{10}(\mathrm{NMSE}).
$$

The project also evaluates a per-antenna hard-thresholding operation. Predicted coefficients below a fraction of the maximum predicted magnitude are set to zero, allowing the estimator to produce a sparse channel representation.

---

## 10. SNR generalization

After training at 10 dB, the same network is evaluated on independently generated datasets at:

$$
5,\ 10,\ 15,\ 20\text{ dB}.
$$

This tests whether the learned channel estimator remains effective when the observation SNR differs from the training condition.

The current experimental results are approximately:

| SNR | NMSE |
|---:|---:|
| 5 dB | −12.27 dB |
| 10 dB | −13.64 dB |
| 15 dB | −14.24 dB |
| 20 dB | −14.46 dB |

---

## 11. Computational characteristics

The estimator is intentionally small compared with large image-processing CNNs.

The network has:

$$
277,664
$$

trainable parameters and uses only convolutional operations.

On the NVIDIA Tesla P100 used in the original experiment, the CNN forward-pass latency was approximately **1.5 ms**.

The implementation also contains an optimized GPU preprocessing path that can perform the feature/scattering operation before CNN inference, enabling an approximately **2.3 ms end-to-end GPU pipeline** in the benchmark environment.

These latency values are hardware-dependent and should therefore be treated as reference measurements rather than universal inference times.

---

## 12. Repository workflow

The repository separates the project into three main stages.

### MATLAB

```text
matlab/
    generate_dataset.m
    generate_dataset_varying_snr.m
```

These scripts generate the OTFS/MIMO channel realizations, sensing matrix, pilot observations, and datasets.

### Python training

```text
src/
    dataset.py
    model.py
    loss.py
    train.py
```

These files load the generated data, construct $\Phi^H y$, form the CNN input, train the residual CNN, and save the best checkpoint.

### Python evaluation

```text
src/
    evaluate.py
    evaluate_snr.py
    benchmark.py
```

These scripts evaluate NMSE, sparsity after thresholding, performance across SNR, and inference latency.

---

## 13. Reproducibility note

The repository currently uses the **256 × 8 DD-grid, 16-transmit-antenna, 40,000-sample configuration** represented by the implementation.

---

## 14. Key equation

At a high level, the entire estimator can be viewed as learning

$$
\hat{H} = f_{\theta} \left( \mathrm{Scatter} \left( \Phi^H \left( \Phi h+w \right) \right) \right),
$$

where:

- $\Phi h+w$ represents the noisy pilot observation,
- $\Phi^H y$ produces correlation-based pilot features,
- `Scatter` places those features on the DD grid,
- $f_\theta(\cdot)$ is the lightweight residual CNN,
- $\hat H$ is the estimated MIMO DD-domain channel.

The CNN therefore learns to infer the **full sparse DD-domain channel from partial, noisy pilot-derived observations**.