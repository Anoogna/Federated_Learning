# Scientific Limitations and Threat to Validity

In adherence to research integrity, this document outlines the fundamental constraints, assumptions, and potential threats to validity of this study.

---

## 1. Surrogate vs. Large Foundation Models

### Context
To enable rapid, zero-crash execution on consumer-grade hardware (CPU execution with 8 GB RAM and no CUDA GPU), our local verification environment leverages `TinyCausalLM` (~5M parameters) as a surrogate transformer with identical target attention projections (`q_proj`, `v_proj`).

### Limitations
- **Loss Landscapes**: While low-rank reparameterization, gradient clipping, Gaussian perturbation, and Top-$k$ sparsification behave identically from an algorithmic standpoint, the loss landscape of a 490M parameter pre-trained model (`Qwen2.5-0.5B`) is significantly smoother than an un-pretrained surrogate.
- **Pre-trained Generalization**: Pre-trained weights exhibit structured gradient sparsity naturally, which may enable even higher compression ratios without utility degradation compared to randomly initialized models.

---

## 2. Simulated Federated Optimization vs. Physical Device Deployment

### Context
All clients are executed sequentially or in simulated batches within an orchestrator loop. Communication is measured through exact bit-level and tensor-level serialization byte counters.

### Limitations
- **Network Asymmetry & Latency**: Real-world federated deployments across edge smartphones or cross-silo hospitals experience packet drops, asymmetric uplink bandwidth (uplink is typically $5\times$ to $10\times$ slower than downlink), and unpredictable device stragglers.
- **Compute Heterogeneity**: Our simulation assumes clients compute local updates in synchronized rounds. In practice, heterogeneous edge devices complete local epochs at vastly different rates.

---

## 3. Privacy Scope: Client-Level vs. Example-Level Differential Privacy

### Context
Our implementation provides an exact, mathematically proven **Client-Level Differential Privacy** guarantee using global $L_2$ update delta clipping and calibrated Gaussian noise injection with Rényi Differential Privacy (RDP) accounting.

### Limitations
- **Threat Model**: Client-level DP protects against the server or an eavesdropper inferring whether a specific *client institution* participated in training. It does not bound the contribution of a single *record* within a client's dataset if that client's total update is under the clipping threshold $C$.
- **Record-Level DP Overhead**: Enforcing strict example-level DP (DP-SGD) requires computing per-sample gradient norms before batch averaging, which increases peak memory footprint by $B\times$ (where $B$ is batch size).

---

## 4. Hyperparameter Sensitivity of Compression Bounds

### Context
AdaComp-DP dynamically scales the retention ratio $p_t$ between $[p_{min}, p_{max}]$ based on training round progression and empirical update energy.

### Limitations
- Setting $p_{min}$ excessively low ($< 1\%$) in the presence of strong DP noise ($\sigma \ge 2.0$) can cause model divergence if error accumulation is disabled.
- The energy threshold $\rho$ requires calibration if transferred to entirely different model architectures or downstream tasks.
