# Scientific Experiment Protocol: Federated LLM Fine-Tuning Benchmarking

## 1. Overview & Experimental Principles
This protocol outlines the procedure for executing, replicating, and analyzing federated fine-tuning experiments across both local CPU verification mode and Google Colab GPU execution mode.

All experiments are governed by four principles:
1. **Identical Partitions**: The dataset is deterministically curated and partitioned into training, validation, and test splits before client allocation. Zero evaluation data is accessible during local client optimization.
2. **Deterministic Reproducibility**: Random seeds are explicitly tracked across Python (`random.seed`), NumPy (`np.random.seed`), and PyTorch (`torch.manual_seed`).
3. **True Wire Communication Accounting**: Communication volume is measured in physical serialized payload bytes (including COO/Bitmask headers and metadata) rather than hypothetical parameter counts.
4. **Valid Differential Privacy Bounds**: Privacy budgets $(\epsilon, \delta)$ are computed strictly using Rényi Differential Privacy (RDP) composition rather than unverified heuristic assertions.

---

## 2. Experimental Execution Modes

### Mode A: Local CPU Smoke Test & Pipeline Verification
Designed for resource-constrained environments (e.g., local machines without CUDA GPUs and limited RAM).
- **Surrogate Architecture**: `TinyCausalLM` (~5M parameters), with target attention linear projections (`q_proj`, `v_proj`).
- **Dataset**: Curated Dolly 15K subset (50–100 instruction-response pairs).
- **Runtime**: $< 10$ seconds per run.
- **Command**:
  ```bash
  python -m src.experiments.run_experiment --config configs/smoke_test_cpu.yaml
  ```

### Mode B: Full GPU Research Experiments (Google Colab / Dedicated Server)
Designed for evaluating foundation language models on NVIDIA GPUs (T4 16GB or A100 40GB).
- **Base Model**: `Qwen/Qwen2.5-0.5B-Instruct` (~490M parameters).
- **Dataset**: Full curated Dolly 15K instruction-following dataset (1,000 samples).
- **Precision**: FP16 / BF16 mixed precision.
- **Command**:
  ```bash
  python -m src.experiments.run_experiment --config configs/full_experiment_gpu.yaml
  ```

---

## 3. Step-by-Step Replication Protocol

### Step 1: Environment Setup
```bash
# Clone or navigate to the repository
cd D:/college/Major_Project

# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# Linux/Colab:
source venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

### Step 2: Verification of Unit & Integration Tests
Run the automated test suite to ensure mathematical correctness of partitioning, serialization, compression, and differential privacy:
```bash
pytest tests/ -v
```
Expected output: **16/16 tests passing**.

### Step 3: Running the Full Baseline Benchmark Suite
To execute all 6 federated configurations simultaneously and generate comparative charts:
```bash
python -m src.experiments.baselines --rounds 3 --output_dir outputs/baselines_suite
```

### Step 4: Inspecting Outputs
Outputs are automatically written to `outputs/`:
- `baseline_results_summary.md`: Formatted Markdown table.
- `plots/convergence_comparison.png`: Global Validation Loss & Perplexity curves.
- `plots/pareto_efficiency.png`: Communication Volume (KB) vs. Model Perplexity Pareto frontier.
- `<experiment_id>_results.json`: Full machine-readable telemetry JSON.
