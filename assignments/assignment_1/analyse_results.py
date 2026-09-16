"""Summarise experiment history and generate the required convergence plot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    """Turn raw run histories into report-ready statistics and a plot.
    We make a convergence plot with avarages of the seeds.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("history", type=Path)
    parser.add_argument("--plot", type=Path, default=Path("convergence.png"))
    parser.add_argument("--summary", type=Path, default=Path("summary.csv"))
    args = parser.parse_args()
    #read the experiment records back into memory.
    rows = [json.loads(line) for line in args.history.read_text().splitlines()]
    variants = sorted({row["variant"] for row in rows})
    #this table is the small numerical result table for the report.
    summary = []
    for variant in variants:
        #compare only the final-generation best score from each independent run.
        final = [r["best"] for r in rows if r["variant"] == variant and r["generation"] == max(x["generation"] for x in rows)]
        summary.append((variant, float(np.mean(final)), float(np.std(final)), len(final)))
    args.summary.write_text("variant,mean_final_best,std_final_best,runs\n" + "\n".join(
        f"{v},{mean:.6f},{std:.6f},{n}" for v, mean, std, n in summary
    ) + "\n")

    #this creates the plot
    plt.figure(figsize=(8, 5))
    for variant in variants:
        variant_rows = [r for r in rows if r["variant"] == variant]
        generations = sorted({r["generation"] for r in variant_rows})
        #at each generation, combine the independent seeds into mean and spread.
        means = [np.mean([r["best"] for r in variant_rows if r["generation"] == g]) for g in generations]
        spreads = [np.std([r["best"] for r in variant_rows if r["generation"] == g]) for g in generations]
        plt.plot(generations, means, label=variant)
        plt.fill_between(generations, np.array(means) - spreads, np.array(means) + spreads, alpha=0.15)
    plt.xlabel("Generation")
    plt.ylabel("Best fitness (lower is better)")
    plt.legend()
    plt.tight_layout()
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.plot, dpi=180)
    print(args.summary)
    print(args.plot)


if __name__ == "__main__":
    main()
