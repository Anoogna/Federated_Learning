# Research Methodology: Adaptive Communication Compression for Privacy-Preserving Federated LLM Fine-Tuning (AdaComp-DP)

## 1. Problem Formulation & Theoretical Foundations

Federated instruction tuning enables decentralized participants to collaboratively fine-tune a foundation Large Language Model (LLM) on private, heterogeneous instruction-following datasets without exchanging raw text. Formally, let $M$ denote the number of simulated federated clients, where each client $i \in \{1, \dots, M\}$ possesses a private dataset $\mathcal{D}_i = \{(x_k, y_k)\}_{k=1}^{n_i}$ drawn from local distribution $\mathcal{P}_i$. The global federated objective is:

$$\min_{\theta} \mathcal{F}(\theta) \triangleq \sum_{i=1}^M \frac{n_i}{N} f_i(\theta), \quad \text{where } f_i(\theta) = \mathbb{E}_{(x, y) \sim \mathcal{P}_i} \left[ \ell(y \mid x; W_0 + \Delta W(\theta)) \right]$$

where $N = \sum_{i=1}^M n_i$, $W_0$ represents the frozen pre-trained base model weights, and $\Delta W(\theta)$ is the low-rank parameter-efficient adapter parameterized by $\theta$.

---

## 2. Parameter-Efficient Low-Rank Adaptation (LoRA)

Rather than communicating or updating full base model matrices $W_0 \in \mathbb{R}^{d_{in} \times d_{out}}$, LoRA reparameterizes the weight delta as:

$$\Delta W = \frac{\alpha}{r} B A$$

where $A \in \mathbb{R}^{r \times d_{in}}$ is initialized via a Gaussian/Kaiming uniform distribution and $B \in \mathbb{R}^{d_{out} \times r}$ is initialized to zero. At initialization ($t=0$), $\Delta W = 0$, preserving the base model's pre-trained knowledge.
During local optimization on client $i$, base weights $W_0$ remain strictly frozen, and gradients are propagated solely into the factor matrices:

$$\theta = \bigcup_{l \in \mathcal{L}} \{A_l, B_l\}$$

where $\mathcal{L}$ indexes target projection layers (e.g., query, key, value, and output projections).

---

## 3. Differential Privacy Mechanism & Accounting

### 3.1 Sensitivity Bounding via Global $L_2$ Clipping
To bound the sensitivity of each client's contribution, client $i$ computes the parameter update delta:

$$\Delta_i^{(t)} = \theta_i^{(t)} - \theta^{(t)}$$

The global $L_2$ norm across all adapter tensors is computed as:

$$\|\Delta_i^{(t)}\|_2 = \sqrt{\sum_{p \in \theta} \|\Delta_{i, p}^{(t)}\|_2^2}$$

Updates are clipped to a maximum norm bound $C$:

$$\overline{\Delta}_i^{(t)} = \Delta_i^{(t)} \cdot \min\left(1, \frac{C}{\|\Delta_i^{(t)}\|_2}\right)$$

### 3.2 Calibrated Perturbation
Calibrated Gaussian noise is injected into each coordinate:

$$\widetilde{\Delta}_i^{(t)} = \overline{\Delta}_i^{(t)} + \mathcal{N}\left(0, \sigma^2 C^2 I\right)$$

where $\sigma$ is the noise multiplier and sensitivity $\Delta f \le C$.

### 3.3 Rényi Differential Privacy (RDP) Accounting
By the properties of the Gaussian Mechanism, the privacy guarantee at order $\alpha > 1$ per communication round is:

$$\epsilon_{RDP}(\alpha) = \frac{\alpha}{2 \sigma^2}$$

Under composition across $T$ rounds with sampling ratio $q = \frac{m}{M}$:

$$\epsilon_{RDP}^{(T)}(\alpha) = \sum_{t=1}^T \epsilon_{RDP}^{(t)}(\alpha)$$

Conversion to canonical $(\epsilon, \delta)$-Differential Privacy is evaluated by taking the infimum over all permissible orders $\alpha \in (1, \infty)$:

$$\epsilon(\delta) = \min_{\alpha > 1} \left\{ \epsilon_{RDP}^{(T)}(\alpha) + \frac{\ln(1 / \delta)}{\alpha - 1} \right\}$$

---

## 4. Proposed Adaptive Compression: AdaComp-DP

### 4.1 Post-Processing Differential Privacy Preservation
> **Theorem (Post-Processing Immunity)**: If a mechanism $\mathcal{M}$ satisfies $(\epsilon, \delta)$-Differential Privacy, then for any data-independent or post-processing mapping $g$, the composed mechanism $g \circ \mathcal{M}$ also satisfies $(\epsilon, \delta)$-Differential Privacy.

Because AdaComp-DP operates strictly on the privatized update $\widetilde{\Delta}_i^{(t)}$, the compression stage does not consume any additional privacy budget or leak sensitive coordinate locations.

### 4.2 Error Feedback Mechanism (EF21)
To prevent error accumulation caused by biased contractive compressors (such as Top-$k$), each client maintains a local residual error accumulator $e_i^{(t)} \in \mathbb{R}^d$. The compensated update is:

$$v_i^{(t)} = \widetilde{\Delta}_i^{(t)} + e_i^{(t)}$$

### 4.3 Convergence and SNR-Guided Dynamic Sparsification
Static Top-$k$ sparsification uses an invariant retention ratio $p$, which either stunts early exploration or wastes bandwidth transmitting late-stage noise. AdaComp-DP dynamically scales the retention ratio $p_t$ based on two continuous signals:

1. **Training Round Schedule**: Early rounds benefit from broader parameter motion, whereas late rounds converge toward the optimum:
   $$s(t) = \frac{1}{2} \left(1 + \cos\left(\frac{\pi t}{T}\right)\right) \in [0, 1]$$
   $$p_t = p_{min} + (p_{max} - p_{min}) \cdot s(t)$$

2. **Update Energy Modulation (Signal-to-Noise Ratio)**:
   $$\eta_i^{(t)} = \text{clip}\left(\frac{\|v_i^{(t)}\|_2}{C + \epsilon_{num}}, 0.2, 1.5\right)$$
   $$k_{i, p}^{(t)} = \text{clip}\left(\left\lfloor p_t \cdot \eta_i^{(t)} \cdot d_p \right\rfloor, \lfloor p_{min} d_p \rfloor, \lceil p_{max} d_p \rceil\right)$$

The top-$k$ operator is applied:

$$\widehat{v}_i^{(t)} = \text{Top-}k_{i}^{(t)}\left(v_i^{(t)}\right)$$

Residual uncommunicated coordinates are saved for the next round:

$$e_i^{(t+1)} = v_i^{(t)} - \widehat{v}_i^{(t)}$$

---

## 5. Wire Serialization & Bandwidth Accounting

To ensure rigorous communication measurement, logical sparsity is translated into actual wire payload bytes:
- **Coordinate List (COO)**: $8 + 4 \cdot \text{dims} + k \cdot (\text{bytes\_per\_index} + \text{bytes\_per\_value})$
- **1-Bit Bitmask**: $8 + 4 \cdot \text{dims} + \lceil d / 8 \rceil + k \cdot \text{bytes\_per\_value}$
- The serializer dynamically chooses the representation minimizing physical bytes for the active sparsity level $k/d$.

---

## 6. Server Aggregation (Sample-Weighted FedAvg)

The central server aggregates compatible sparse updates weighted by client dataset sizes:

$$\theta^{(t+1)} = \theta^{(t)} + \sum_{i \in \mathcal{S}_t} \frac{n_i}{\sum_{j \in \mathcal{S}_t} n_j} \widehat{v}_i^{(t)}$$
