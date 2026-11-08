# A pruning path is not a search over every possible tree

TreeLab minimizes squared error at each greedy binary split. For a region A,
the fitting mean minimizes sum_i (y_i-c)^2. Each distinct adjacent sorted
feature value defines one candidate training partition. Centered cumulative
sums and squared sums evaluate its two leaf SSEs. A midpoint defines the split;
when binary64 rounding reaches the upper adjacent value, the lower observed
value is used instead. Feature-index/threshold order breaks equal losses.
Splits must strictly improve fitting SSE, obey minimum leaf size 5 and stop
at depth 20. These rules are fixed before held-out evaluation.

## Weakest link

Write R(T) = sum of terminal-node SSE / N_fit and L(T) for the number of leaves.
The objective is R(T) + alpha L(T), alpha >= 0. For an internal node t,

    alpha_eff(t) = [R(t) - R(T_t)] / [L(T_t) - 1].

Compute risk with the original root fitting count. Dividing by each subtree's
row count would change the objective and is not this algorithm. At the minimum
effective alpha, cut every exactly tied weakest link. An ancestor cut includes
its descendants. Retain counts, means and SSEs when replacing a branch with a
leaf. Repeat to the root. Alpha breakpoints are made nondecreasing if binary64
summation causes a tiny inversion; nonpositive numerator roundoff is clamped
to zero. No exact-rational claim is made for this production path.

Node means and SSEs use Python `math.fsum` rather than a hardware-dependent
BLAS dot-product reduction. Prefix candidates still use binary64 cumulative
sums. Exact floating ties are not mathematical real-number ties. This matters:
the preliminary dot-product version had 307 states, while the final stable
node-statistics version has 290 because more mathematically tied cuts occur
together. Both versions selected the same 19-leaf predictions and test metrics.
Only the final version is the saved/published protocol and report.

Independent tests enumerate every subtree of 12 depth-three fitted trees using
Fraction risks calculated directly from their training rows, rather than from
saved floating node SSEs. Breakpoints, interval interiors, zero and a large
alpha compare the selected path and `prune` objective with this exhaustive
oracle. This is a finite audit, not a proof over arbitrary floating data.
Forty separate stump controls recompute every candidate SSE by direct means.

## Selection, controls and leakage

Select the smallest validation MSE on the full path. Exact score ties prefer
fewer leaves, then earlier path index. No test curve participates. The selected
model keeps its original fitting means, with no full-data refit. A changed-test-
target control regenerates the entire experiment and checks identical fitting,
pruning, validation selection and model. A second control verifies that test
predictions begin only after validation selection.

Exact feature duplicates stay together in a target-blind deterministic hash
split. This is stricter than a naive row shuffle for exact-copy leakage, but
cannot rule out near duplicates or unrecorded production batches. All source
rows, split indices and predictions are retained. Model and report regeneration
uses finite absolute tolerance 2e-9 for floating values, exact keys, lengths,
discrete fields and hashes. Corruption, nonfinite and missing-key controls
check that the tolerance is not an excuse to ignore material changes.

Mean, median and ordinary least-squares controls use the same fitting rows.
The linear control uses fitting-only standardized features, intercept and
`numpy.linalg.lstsq`; its stronger held-out result is retained. This report is
one historical dataset and one split. No significance, causal benefit from
pruning, superiority to ensembles or general wine-quality assessment follows.

## Sources read for implementation

- UCI, Wine Quality source, variables, citation and CC BY 4.0 license:
  https://archive.ics.uci.edu/dataset/186/wine%2Bquality
- Original CSV and names:
  https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv
  https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality.names
- University of Toronto STA450 notes, regression means, greedy SSE splits and
  the cost-complexity objective: https://utstat.utoronto.ca/reid/sta450/mar9.pdf
- Scikit-learn tree guide, greedy limitations and weakest-link definition:
  https://scikit-learn.org/stable/modules/tree.html
- Scikit-learn pruning example, full path and alpha interpretation:
  https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html
  Its displayed test-score example is not copied as a test-selection protocol.
- TensorFlow Datasets, source description and ordinal expert scores:
  https://www.tensorflow.org/datasets/catalog/wine_quality
- Community derivation, independent algebraic cross-check, not primary authority:
  https://stats.stackexchange.com/questions/193538/how-to-choose-alpha-in-cost-complexity-pruning
- CC BY 4.0 legal terms: https://creativecommons.org/licenses/by/4.0/legalcode

No code is copied from those implementations. No scikit-learn or TensorFlow
runtime is required. NumPy provides arrays, sorting, sums and the explicitly
identified least-squares control.
