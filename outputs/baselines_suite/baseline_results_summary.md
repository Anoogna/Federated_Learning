# Empirical Experimental Results: Federated LLM Fine-Tuning Benchmarks

| Strategy / Baseline | DP ($\epsilon, \delta$) | Compression Policy | Final Test Loss | Final Test PPL | Transmitted Wire KB | Comm. Ratio | Bandwidth Savings |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FedAvg-Dense** | None | `none` | 120.9537 | 485165195.41 | 97.1 KB | 1.00x | 0.0% |
| **FedAvg-FixedTopK** | None | `fixed_topk` | 120.9554 | 485165195.41 | 57.1 KB | 1.70x | 41.2% |
| **FedAvg-AdaComp** | None | `adaptive_adacomp` | 120.9559 | 485165195.41 | 54.2 KB | 1.79x | 44.2% |
| **DP-FedAvg-Dense** | Yes (eps=17.6753) | `none` | 36.6582 | 485165195.41 | 97.1 KB | 1.00x | 0.0% |
| **DP-FedAvg-FixedTopK** | Yes (eps=17.6753) | `fixed_topk` | 54.9363 | 485165195.41 | 57.1 KB | 1.70x | 41.2% |
| **AdaComp-DP-Proposed** | Yes (eps=17.6753) | `adaptive_adacomp` | 48.5129 | 485165195.41 | 60.4 KB | 1.61x | 37.8% |
