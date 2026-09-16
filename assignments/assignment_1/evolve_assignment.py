from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Callable

import networkx as nx
import numpy as np

from ariel.ec.genotypes.tree.operators import (
    crossover_subtree,
    mutate_replace_node,
    mutate_subtree_replacement,
    random_tree,
)
from ariel.ec.genotypes.tree.tree_genome import TreeGenome

from tree_edit_distance import mean_plus_std_tree_edit_distance
from A1_template_2026 import load_targets

ROOT = Path(__file__).parent
TARGET_DIR = ROOT / "target_bodies"


def seed_everything(seed: int) -> None:
    """
    This function makes sure that we can reproduce the same random numbers for a seed so we can replicate results for seeds.
    """
    random.seed(seed)
    np.random.seed(seed)


def score(genome: TreeGenome, targets: list[nx.DiGraph]) -> float:
    """This function converts one genome to a body and calculates the fitness.
    """
    body = genome.to_networkx()
    return float(mean_plus_std_tree_edit_distance(body, targets))


def tournament(population: list[tuple[TreeGenome, float]], size: int = 3) -> TreeGenome:
    """Select a parent using tournament selection.
    """
    contestants = random.sample(population, min(size, len(population)))
    return min(contestants, key=lambda item: item[1])[0]


def make_child(
    population: list[tuple[TreeGenome, float]],
    targets: list[nx.DiGraph],
    mutation: Callable[[TreeGenome], None],
    max_modules: int,
    crossover_probability: float,
    mutation_probability: float,
) -> tuple[TreeGenome, float]:
    """Create, mutate, and evaluate one offspring.
    """
    #1: choose two parents from the current population.
    parent_a = tournament(population)
    parent_b = tournament(population)
    #2: crossover combines genetic material from both parents.
    if random.random() < crossover_probability:
        child, _ = crossover_subtree(copy.deepcopy(parent_a), copy.deepcopy(parent_b))
    else:
        #if crossover does not happen, continue with a copy of parent A.
        child = copy.deepcopy(parent_a)
    #3: mutation changes the child and creates new possible solutions.
    if random.random() < mutation_probability:
        if mutation is mutate_subtree_replacement:
            mutation(child, max_modules=max_modules)
        else:
            mutation(child)
    #4: score the child before it enters the next population.
    return child, score(child, targets)


def run_ea(
    variant: str,
    seed: int,
    targets: list[nx.DiGraph],
    population_size: int,
    generations: int,
    max_modules: int,
    crossover_probability: float = 0.8,
    mutation_probability: float = 0.2,
) -> list[dict[str, float | int | str]]:
    """Run one complete evolutionary experiment and return its history.
    Each generation keeps the best individual
    """
    seed_everything(seed)
    #variant A changes one module, variant B replaces a complete subtree.
    mutation = mutate_replace_node if variant == "point_mutation" else mutate_subtree_replacement
    #the initial population is thefirst set of candidate bodies.
    population = [
        (genome, score(genome, targets))
        for genome in (random_tree(max_modules=max_modules) for _ in range(population_size))
    ]
    #every fitness calculation counts toward the comparison budget.
    evaluations = len(population)
    history: list[dict[str, float | int | str]] = []

    for generation in range(generations + 1):
        #these values show the current population for the convergence plot.
        fitnesses = [fit for _, fit in population]
        history.append({
            "variant": variant,
            "seed": seed,
            "generation": generation,
            "evaluations": evaluations,
            "best": min(fitnesses),
            "mean": float(np.mean(fitnesses)),
            "std": float(np.std(fitnesses)),
            "worst": max(fitnesses),
            "best_modules": len(min(population, key=lambda item: item[1])[0].nodes),
        })
        #do not go over the set number of generations
        if generation == generations:
            break
        #prevent best solution so far from being lost
        elite = min(population, key=lambda item: item[1])
        #all other individuals are newly generated offspring.
        offspring = [make_child(population, targets, mutation, max_modules,
                                crossover_probability, mutation_probability)
                     for _ in range(population_size - 1)]
        evaluations += len(offspring)
        #keep the best one and all the offspring
        population = [elite, *offspring]
        population.sort(key=lambda item: item[1])
    return history


def run_random_search(
    seed: int,
    targets: list[nx.DiGraph],
    population_size: int,
    generations: int,
    max_modules: int,
) -> list[dict[str, float | int | str]]:
    """Run random search using exactly the same evaluation budget as the EA.
    Random search is the control condition.
    """
    seed_everything(seed)
    #match the EA numbers: one initial population plus one population per step.
    total_evaluations = population_size * (generations + 1)
    best = float("inf")
    history = []
    for evaluation in range(1, total_evaluations + 1):
        candidate = random_tree(max_modules=max_modules)
        best = min(best, score(candidate, targets))
        if evaluation % population_size == 0 or evaluation == total_evaluations:
            history.append({
                "variant": "random_search",
                "seed": seed,
                "generation": evaluation // population_size - 1,
                "evaluations": evaluation,
                "best": best,
                "mean": best,
                "std": 0.0,
                "worst": best,
                "best_modules": len(candidate.nodes),
            })
    return history


def main() -> None:
    """Run all requested variants and write their histories to JSONL.
    """
    #add the variables that can be set from the command line with defaults
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=["point_mutation", "subtree_mutation", "random_search"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
    parser.add_argument("--population", type=int, default=50)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--max-modules", type=int, default=20)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "history.jsonl")
    args = parser.parse_args()
    targets = load_targets(TARGET_DIR)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for variant in args.variants:
            for seed in args.seeds:
                if variant == "random_search":
                    rows = run_random_search(seed, targets, args.population, args.generations, args.max_modules)
                else:
                    rows = run_ea(variant, seed, targets, args.population, args.generations, args.max_modules)
                #store the history immediately so long experiments produce usable data.
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
                print(f"completed {variant} seed={seed}")


if __name__ == "__main__":
    main()
