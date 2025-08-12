# Dual-Window Convergence Detection for Early Stopping

## 1. Motivation

In iterative machine learning training, stopping too early risks underfitting, while stopping too late wastes computation and can lead to overfitting.  
Most practical approaches—such as **patience-based early stopping**—focus on whether the *best-so-far* validation score has improved recently.  
This approach has two weaknesses:

1. **Insensitivity to convergence rate**: A model may still be improving slightly, but at a negligible pace that does not justify further training.
2. **Noise vulnerability**: Random fluctuations can falsely trigger stops or prolong training unnecessarily.

We propose **Dual-Window Convergence Detection (DWCD)**, a statistically tunable stopping rule that detects when training has *converged* in terms of validation performance, using a short-term vs. long-term trend comparison.

---

## 2. Method

### 2.1 Core Idea

Maintain two rolling averages over the validation metric:
- **Short window** \( M_s(t) \): recent performance trend over \( w_s \) epochs.
- **Long window** \( M_\ell(t) \): historical baseline over \( w_\ell \) epochs (\( w_s < w_\ell \)).

We declare *convergence* when the short-term average is no longer **significantly better** than the long-term average.

---

### 2.2 Formal Definition

Let \( y_t \) be the validation metric at epoch \( t \) (higher-is-better; reverse inequality for loss):

\[
M_s(t) = \frac{1}{w_s} \sum_{i=0}^{w_s-1} y_{t-i}
\]
\[
M_\ell(t) = \frac{1}{w_\ell} \sum_{i=0}^{w_\ell-1} y_{t-i}
\]

**Stopping condition** (accuracy-style metric):

\[
M_s(t) \le M_\ell(t) + \delta_t
\]

---

### 2.3 Margin Calibration

To make \(\delta_t\) interpretable and tunable:

1. Estimate short-term noise:
\[
\hat\sigma_t = \text{stdev of residuals in long window}
\]
2. Set:
\[
\delta_t = z_\alpha \cdot \hat\sigma_t \cdot \sqrt{\frac{1}{w_s} + \frac{1}{w_\ell}}
\]
where \( z_\alpha \) is the one-sided z-score for the desired false-stop probability (e.g., \( z_{0.05} \approx 1.64 \)).

This converts the margin into a statistical test:  
- \(H_0\): metric is stable (no improvement).  
- \(H_1\): metric is still improving.

---

### 2.4 Optional Enhancements

- **Patience overlay**: Require the stop condition for \( p \) consecutive epochs.
- **Exponential Moving Averages (EMA)**: Replace simple averages with EMA for faster responsiveness.
- **Adaptive windows**: Increase \( w_\ell \) over time for a more stable baseline in late training.

---

## 3. Algorithm (Pseudo-code)

```

Initialize short\_window\_buffer (size w\_s)
Initialize long\_window\_buffer (size w\_l)

For each epoch t:
Train one epoch
y\_t = validation metric
Update M\_s(t), M\_l(t) in O(1) time
σ\_hat = std(long\_window\_residuals)
δ\_t = z\_alpha \* σ\_hat \* sqrt(1/w\_s + 1/w\_l)

```
If M_s(t) <= M_l(t) + δ_t for p consecutive epochs:
    Stop training
    Save checkpoint

---

## 4. Experimental Plan

### 4.1 Baselines
- Patience-based early stopping (with/without smoothing)
- EMA + patience
- Training-loss plateau detection
- Sequential statistical tests (SPRT/Bayesian)
- Fixed-epoch (oracle-tuned)

### 4.2 Benchmarks
- **Vision**: CIFAR-10/100 (ResNet-18), Tiny-ImageNet (WRN-28-10)
- **NLP**: SST-2, AG-News (DistilBERT fine-tuning)
- **Tabular**: UCI datasets with XGBoost/LightGBM
- **Time Series (optional)**: M4 subset (LSTM/TFT)

### 4.3 Evaluation Metrics

**Effectiveness**
- Test metric at stop
- Generalization gap at stop
- Stopping regret (gap to oracle stop)

**Efficiency**
- Epochs/wall-clock time to stop
- Compute saved at equal performance

**Robustness**
- Variance across seeds
- Premature/late stop rates
- Sensitivity to \( w_s, w_\ell, z_\alpha \)

---

## 5. Key Contributions

1. **A convergence-focused stopping criterion**: Detects when recent performance is statistically indistinguishable from historical baseline.
2. **Statistical tuning**: Margin \(\delta_t\) linked to noise estimate, enabling false-stop control.
3. **Practicality**: Low computation overhead, easy to integrate into existing ML frameworks.
4. **Generality**: Applicable to any model/metric with an iterative training process.

---

## 6. Relationship to Existing Work

DWCD differs from:
- **Patience rules**: Focus on best-so-far improvement rather than trend convergence.
- **Smoothing + patience**: Uses two distinct horizons instead of one smoothed series.
- **Change-point detection in statistics**: Inspired by CUSUM/two-sample tests, but adapted for online ML validation monitoring.

