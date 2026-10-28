"""Reporting tables module.

Generates formatted Markdown and publication-ready LaTeX tables comparing baseline and
adaptive federated fine-tuning experiments across accuracy, wire bytes, and privacy metrics.
"""

from __future__ import annotations

from typing import Any, Dict, List


def generate_comparison_markdown_table(results_dict_map: Dict[str, Dict[str, Any]]) -> str:
    """Generate Markdown summary table from a dictionary of experiment results."""
    lines = [
        "# Empirical Experimental Results: Federated LLM Fine-Tuning Benchmarks",
        "",
        r"| Strategy / Baseline | DP ($\epsilon, \delta$) | Compression Policy | Final Test Loss | Final Test PPL | Transmitted Wire KB | Comm. Ratio | Bandwidth Savings |",
        r"| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for label, res in results_dict_map.items():
        cfg = res.get("config", {})
        comm = res.get("communication_summary", {})
        final_eval = res.get("final_test_eval", {})
        priv = res.get("privacy_summary", {})

        dp_str = f"Yes (eps={priv.get('spent_epsilon', 'N/A')})" if cfg.get("privacy", {}).get("enabled") else "None"
        comp_str = cfg.get("compression", {}).get("strategy", "none")
        test_loss = f"{final_eval.get('loss', 0.0):.4f}"
        test_ppl = f"{final_eval.get('perplexity', 0.0):.2f}"
        wire_kb = f"{(comm.get('cumulative_total_wire_bytes', 0) / 1024.0):.1f} KB"
        ratio = f"{comm.get('overall_compression_ratio', 1.0):.2f}x"
        savings = f"{comm.get('overall_savings_percent', 0.0):.1f}%"

        lines.append(
            f"| **{label}** | {dp_str} | `{comp_str}` | {test_loss} | {test_ppl} | {wire_kb} | {ratio} | {savings} |"
        )

    lines.append("")
    return "\n".join(lines)


def generate_comparison_latex_table(results_dict_map: Dict[str, Dict[str, Any]]) -> str:
    """Generate publication-ready LaTeX tabular snippet for final B.Tech thesis."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Comparison of Differentially Private Federated LLM Fine-Tuning Strategies under Non-IID Client Data.}",
        r"\label{tab:fed_llm_results}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Strategy} & \textbf{Differential Privacy} & \textbf{Compression} & \textbf{Test Loss} $\downarrow$ & \textbf{Test PPL} $\downarrow$ & \textbf{Payload (KB)} $\downarrow$ & \textbf{Savings (\%)} $\uparrow$ \\",
        r"\midrule",
    ]

    for label, res in results_dict_map.items():
        cfg = res.get("config", {})
        comm = res.get("communication_summary", {})
        final_eval = res.get("final_test_eval", {})
        priv = res.get("privacy_summary", {})

        dp_str = f"$\\epsilon={priv.get('spent_epsilon', 'N/A')}$" if cfg.get("privacy", {}).get("enabled") else "None"
        comp_str = cfg.get("compression", {}).get("strategy", "none").replace("_", r"\_")
        test_loss = f"{final_eval.get('loss', 0.0):.4f}"
        test_ppl = f"{final_eval.get('perplexity', 0.0):.2f}"
        wire_kb = f"{(comm.get('cumulative_total_wire_bytes', 0) / 1024.0):.1f}"
        savings = f"{comm.get('overall_savings_percent', 0.0):.1f}\\%"

        lines.append(
            f"{label} & {dp_str} & {comp_str} & {test_loss} & {test_ppl} & {wire_kb} & {savings} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    return "\n".join(lines)
