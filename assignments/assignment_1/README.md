# Assignment 1 experiment

Run experiments:

You can see the possible variables to start with in the function in evolve_assignments.py

```bash
uv run assignments/assignment_1/evolve_assignment.py
uv run assignments/assignment_1/analyse_results.py \
  assignments/assignment_1/results/history.jsonl \
  --plot assignments/assignment_1/results/convergence.png \
  --summary assignments/assignment_1/results/summary.csv
```
