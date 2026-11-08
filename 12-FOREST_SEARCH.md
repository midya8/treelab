# Constrained ensemble search with objective bounds

TreeLab extends its single-CART counterfactual engine to mean ensembles using best-first branch-and-bound over compatible split boxes. A witness is scored on the whole ensemble at one input. Completed searches return a global weighted-L1 optimum or infeasibility; an exhausted state budget returns explicit bounds and unfinished status, with or without an incumbent.

This is established tree-region search, not a new algorithm or a scalable solver. The frozen all-feature cases do **not** finish at 4,096 expansions. Keeping those incomplete results is part of the audit.

## Run offline

```sh
python -m treelab forest-demo
python -m treelab forest-audit --output /tmp/tree-forest
python -m treelab forest-search \
  --model treelab/artifacts/bagging-model.json.gz \
  --input examples/forest-search.json
python examples/export_forest.py
node --test tests/test_forest_browser.cjs
python -m http.server 8000 --directory docs
```

Open `forest.html` on the local server. The page displays all 36 frozen requests, exact rational bounds, complete input witnesses, every member output/leaf ID, the three budget outcomes, provenance hashes, copy and local JSON download. No browser optimizer, new inference, API, accounts, analytics or uploads. Existing one-axis and single-tree pages remain available.

## Request and model semantics

Required JSON: `row`, closed output `target: [low,high]`, one independent closed feature `bounds` pair per coordinate. Optional positive `weights`, distinct `immutable` feature indices, `max_states` (0 through 100,000, default 10,000) and `seed_inputs` (at most 128 full vectors). `--trees N` restricts the original ensemble prefix. A single CART is also accepted. Unknown request fields are rejected.

All inputs are converted to finite binary64 within +/-1e100; model split thresholds and leaf means must already be exactly binary64-representable. Input widths, constraints, domains, prefix and budgets are validated. An immutable source outside its supplied domain makes the request infeasible. Supplied seeds must satisfy width, bounds and immutable features, but are never assumed to reach the target. Each is evaluated on the full ensemble before it can improve the incumbent.

The output function for one input is the binary64 NumPy mean of the member outputs in original order, matching the native single-input `bagging.predict` route. A final witness is checked by that public inference function as well as individual `tree.predict` calls. This is not exact-rational ensemble prediction: rational arithmetic is used for costs, not for the model's reduction. Batch reduction layouts may differ at rounding level; the guarantee names the single-input route. Finite leaf values and model sizes here keep the mean reduction far below overflow.

## State, admissible bounds and completion

A state contains a feature box and a current node for every member. For `x[j] <= t` the left upper bound includes `t`; the right lower bound is `nextafter(t,+inf)`, inclusive in the finite binary64 domain. Empty intersections are pruned. Repeated splits keep the strongest bounds.

Before branching, propagate any split forced by the whole box across every member. If the entire feature interval lies left or right, descend without creating a new search split. This makes fixed-feature policies tractable without changing their geometry.

For a nonempty box, coordinate-wise projection of the source onto its intervals minimizes positive weighted L1 on that box. Compute this objective exactly with `Fraction` values of the stored binary64 source, projection and weights. It is an admissible lower bound for every descendant box, because later intersections only remove inputs.

Precomputed minimum/maximum descendant leaf means at each current node give an output range per member. Applying the same ordered NumPy mean to the lower vector and upper vector conservatively bounds all outputs below this state. Floating addition/division is monotone here and values cannot overflow. Incompatibility among members can make this range loose; it cannot justify treating their separate extrema as jointly attainable. A range outside the requested band prunes that state.

A heap orders remaining states by exact objective lower bound, then deterministic insertion order. Expand the first unresolved member's split. Full-ensemble prediction at each popped state's projection may supply a better incumbent. Child states intersect the common box, propagate forced branches, check output bounds and enter the frontier only if their exact lower cost can improve the incumbent.

- `optimal`: a verified witness exists and no frontier box can improve its exact cost. Return one minimizer, not all optima or a canonical leaf-ID tie.
- `infeasible`: the frontier is exhausted without any feasible witness. Null objective bounds mean no finite feasible optimum, not numeric zero.
- `budget_exhausted`: frontier remains after the expansion budget. Return its minimum exact cost as the lower bound and the incumbent cost as the upper bound, if one exists. A missing upper bound is not infeasibility. With an incumbent, `absolute_gap_exact = upper - lower`.

Once a feasible cost equals the best frontier lower bound, the proof can finish even if no more expansion budget remains. Equal-cost boxes need not be searched to return one optimal witness. Rational numerator/denominator strings retain tiny positive costs and distinguish rounded false ties. Display floats are rounded, not used for ordering.

The budget counts popped/expanded states, not preprocessing, seed scoring, generated children, heap work, full inference calls or wall time. Counts expose generated/expanded/frontier/peak states, seed evaluations and pruning categories. Binary branching limits peak frontier to at most expanded states plus one. Time/memory can still be large; the implementation is serial Python with NumPy means and rational objective arithmetic. No throughput or asymptotic speedup benchmark is claimed.

## Warm starts and the frozen protocol

Four fixed initial points are tried: source projected onto constraints, lower domain corner, upper corner and a clamped binary64 midpoint. Duplicate vectors are removed. Clamping the midpoint matters for subnormal singleton domains, where halving each endpoint can underflow outside the interval. Each optional supplied seed is also re-scored on the full ensemble. Warm starts improve upper bounds only, never prove optimality or infeasibility.

The real demo uses the unchanged 32-tree group-bagging model and first three original historical test rows 1, 8 and 11, including the exact-feature duplicate pair. Cross target bands `[6,7]` and `[7,8]`, all-feature versus alcohol-only constraints, and budgets `[0,128,4096]`: 36 fixed requests. Fitting-only feature min/max and reciprocal-range weights. No fitting, test-label choice or new accuracy evaluation.

To avoid hiding an already-known easy witness, the demo supplies the closest target point on each available frozen one-axis ensemble slice as an additional seed (only alcohol for the alcohol-only policy). Vectors are retained in each request and re-evaluated on the full ensemble. The frozen sweep report's hash is displayed along with the ensemble and search report hashes. The general search API does not silently read external reports or invent background data.

For CSV row 1:

| Target | Policy | Budget | Result | Expanded | Cost lower / upper |
| --- | --- | ---: | --- | ---: | --- |
| [6,7] | all features | 0 | budget exhausted, witness known | 0 | 0 / 0.36290323 |
| [6,7] | all features | 4096 | budget exhausted, not optimal | 4096 | 0.01838427 / 0.36290323 |
| [6,7] | alcohol only | 4096 | optimal | 12 | 0.36290323 / 0.36290323 |
| [7,8] | all features | 4096 | budget exhausted, coarse corner witness | 4096 | 0.07727250 / 8.59221214 |
| [7,8] | alcohol only | 4096 | infeasible | 5 | no finite feasible objective |

The [6,7] witness predicts 6.05953621031746 at alcohol 11.750000000000002, with the other source coordinates fixed. Its one-axis optimality does not prove all-feature optimality. The [7,8] all-feature corner witness predicts 7.124739583333334 and is expensive; its large gap is preserved, not advertised as a good or realistic change. The all-feature search still has unseen regions at the fixed budget.

## Independent checks

- Tiny random forests: exhaustive Cartesian grid formed independently from source values, domain endpoints, every split threshold and representable successor; compare global optimum and every finite-budget lower/upper bound.
- A single-tree cross-check against complete target-leaf enumeration, plus native whole-ensemble witness re-scoring.
- Opposite stumps each separately reach 10, but their common-input mean is always 5; target [9,10] is infeasible. No sum of independent member optima.
- Budget exhaustion with no witness and with a nonzero-gap incumbent; exact positive subnormal cost despite float display zero; midpoint-underflow domain control; immutable/source-outside/directional bounds; deterministic repeats and no model mutation.
- Every alcohol-only frozen result checked against the independent complete one-axis sweep target-cost oracle. All-feature upper bounds cannot discard the known feasible one-axis seed.
- Full report reproduction with fitting forbidden, all three CLI modes, strict request fields/prefixes and report/model/warm-start source hashes.
- Real Chrome: all 36 cases, all three statuses, exact rational bound JSON, every returned member output/leaf, budget selection, copy denial/success, download, 320/390/1280 expanded layouts and 44px controls.

## Scope and attribution

This solves a finite binary64 predictor problem under independent feature domains. It has no causal model, joint chemistry restrictions, discrete feature constraints, intervention cost study, density/plausibility check or outcome guarantee in reality. A counterfactual explanation is not a recommended intervention. Quality is ordinal but model outputs are treated numerically. Original training, historical metrics, selected CART, ensemble, reports and audit figures are unchanged.

Code MIT. Data and derived results: Cortez et al. (2009), [UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine%2Bquality), DOI 10.24432/C56S3T, CC BY 4.0. No endorsement implied.

Established methods and context, not implementations copied or performance claims reproduced:

- [An exact counterfactual-example-based approach to tree-ensemble models interpretability](https://arxiv.org/pdf/2105.14820), 2021. Compatible-region decomposition, projection lower bounds and branch-and-bound.
- Parmentier and Vidal, [Optimal Counterfactual Explanations in Tree Ensembles](https://proceedings.mlr.press/v139/parmentier21a.html), ICML 2021. Model-based guarantees and separate plausibility/actionability constraints. This implementation is not OCEAN/MIP.
- Khouna, Emine, Ferry and Vidal, [Optimal Counterfactual Search in Tree Ensembles: A Study Across Modeling and Solution Paradigms](https://arxiv.org/html/2605.06561), 2026. Threshold-induced finite-domain formulations and solver comparisons; this implementation is not CPCF, MaxSAT or a reproduction of their benchmarks.
- Carreira-Perpinan and Hada, [Very fast, approximate counterfactual explanations for decision forests](https://export.arxiv.org/pdf/2303.02883v1.pdf), 2023. Forest-region explosion and the distinction between approximation and exact search.
- Karimi, Scholkopf and Valera, [Algorithmic Recourse: from Counterfactual Explanations to Interventions](https://ar5iv.labs.arxiv.org/html/2002.06278), 2020. Why changing a predictor does not establish an effective intervention.
- [Python math documentation](https://docs.python.org/3/library/math.html), representable successors; [NumPy mean documentation](https://numpy.org/doc/stable/reference/generated/numpy.mean.html), reduction precision. Rational objective comparison does not remove model-output rounding.
