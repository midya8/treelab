# Constrained multi-feature model counterfactuals

Which input is closest to this row while making one frozen CART output a value in a requested closed band? TreeLab enumerates every leaf, intersects its split path with user bounds and immutable inputs, projects the row onto each feasible target box, and ranks those witnesses by weighted L1 cost.

This is established closest-leaf counterfactual geometry, implemented here for TreeLab's own single-tree format. It is not a new algorithm, a causal intervention, a plausible-wine generator or an optimizer for the bagging ensemble.

[Browser demo](https://midya8.github.io/treelab/counterfactual.html) | [one-axis explorer](https://midya8.github.io/treelab/)

## Run it

```sh
python -m unittest discover -s tests -v
python -m treelab counterfactual-demo
python -m treelab counterfactual-audit --output /tmp/tree-counterfactual
python -m treelab counterfactual \
  --model treelab/artifacts/selected-model.json \
  --input examples/counterfactual.json
python examples/export_counterfactual.py
node --test tests/test_counterfactual_browser.cjs
```

A request contains `row`, closed prediction `target: [low, high]`, and one closed `bounds: [low, high]` pair per feature. Optional `weights` must be finite and strictly positive; optional `immutable` contains distinct zero-based feature indices. Everything is converted to binary64, must be finite and within +/-1e100. Unknown CLI fields, nonfinite numbers, reversed ranges, malformed models and ensemble inputs are rejected. No silent target relaxation or fallback witness.

Directional changes can be expressed through bounds: to permit only increases, set that feature's lower bound to its source value. Bounds need not include the source. If an immutable source coordinate lies outside its domain, the request is infeasible. A source that already meets the target is zero-cost only if it also meets every constraint.

## Geometry and guarantee

For `x[j] <= threshold`, the left branch's upper bound is inclusive. The right branch requires `x[j] > threshold`. In the finite binary64 input domain, replace this strict lower bound with `math.nextafter(threshold, +inf)`, inclusive. Repeated splits intersect componentwise. No arbitrary epsilon is added. A box with lower > upper in any coordinate is empty.

For each leaf whose mean is in the target band, intersect its path box with the supplied domains and immutable coordinates. For a nonempty box, clamp each source coordinate to the corresponding closed interval. For a positive-weight, separable L1 objective this minimizes the cost within that box. Every finite input routes to one leaf; enumerating all target leaves and selecting their cheapest projected witness therefore gives a global optimum under these constraints.

Costs are compared as exact Python `Fraction` arithmetic over the stored binary64 source, witness and weights:

```text
sum_j Fraction(weight[j]) * abs(Fraction(witness[j]) - Fraction(source[j]))
```

The JSON gives the reduced numerator and denominator as decimal strings, plus a rounded float display cost. This prevents a rounded false tie or a subnormal product underflow from changing the selected leaf. It does not claim exact arithmetic for model training, decimal inputs before conversion, or ideal reciprocal weights. Exact-cost ties break by smallest leaf ID, an arbitrary deterministic convention, not a claim of uniqueness. Every feasible target leaf is returned with its projected input, effective inclusive box, exact cost, changed indices and routing path. Each witness is independently routed through `tree.predict` before it is emitted.

`status: infeasible` with a null solution means no finite binary64 input under the supplied constraints reaches the target band in this stored tree. The report distinguishes no target-output leaves from target leaves excluded by the constraints. This is an exhaustive search result, not a timeout or failure to sample a witness.

Traversal is linear in tree nodes with dimension-sized bounds copied at splits. Candidate projection costs O(leaves * features) rational operations, followed by sorting and routing returned witnesses. Rational arithmetic and complete candidate output add cost; this is not a scalable forest solver. Signed zeros are equal for routing and cost, not distinct actionable states.

## Frozen real-data protocol

No fitting or new scoring. Use the unchanged validation-selected 19-leaf CART and the first three original historical test CSV rows (zero-based 1, 8, 11), including the duplicate-feature pair. Test labels do not define the targets or costs. Cross all three rows with fixed target bands `[6,7]`, `[7,8]`, `[9,9]` and four policies: all features, alcohol only, alcohol immutable, every feature immutable. All 36 outcomes are retained, including infeasible requests.

Domains are feature-wise fitting-row minima/maxima only. Weights are their stored binary64 reciprocal ranges (weight 1 for a constant feature), so the objective approximately measures change in fractions of fitting ranges. It is not an empirically measured cost of changing wine chemistry. Independent min/max limits do not preserve dependencies or ensure support under the data distribution.

For CSV row 1, base output 5.315789473684211:

| Target | Constraints | Result |
| --- | --- | --- |
| [6,7] | all features | 6.3628318584070795; cost about 0.18145161; alcohol changes |
| [6,7] | alcohol immutable | infeasible |
| [7,8] | all features | 7.647058823529412; cost about 0.27508758; fixed acidity, volatile acidity and alcohol change |
| [7,8] | alcohol only | infeasible |
| [9,9] | all features | infeasible: no leaf outputs 9 |

These are properties of one stored predictor. Lower alcohol can increase its output in a particular multi-feature leaf combination. That is not a claim that reducing alcohol improves wine quality. Do not translate the witness into production advice.

The browser displays the complete frozen report, not an optimizer or arbitrary-input predictor. It includes every feasible candidate, exact rational costs, full witness JSON, download/copy and explicit clipboard-denial fallback. SHA-256 report and model provenance can be checked against the source files. The checksum is not a signature. No accounts, analytics, paid services, APIs or uploads.

## Checks

- Independent small-tree Cartesian-grid oracle constructed from domains, source values, thresholds and successor floats; compare minimum exact cost and leaf ID without using the production box enumeration.
- Multi-feature targets; immutable features; directional/closed bounds; source outside bounds; already-satisfied targets; absent target outputs; constant trees; singleton domains; repeated split constraints.
- Adjacent binary64 inputs, exact-threshold left routing, subnormal positive exact cost with displayed float zero, false rounded cost ties, and exact-cost leaf-ID ties.
- All 36 real-data requests reproduced byte-for-byte with fitting forbidden; every feasible leaf witness re-routed and checked against its box, target, cost and immutable inputs.
- Strict CLI requests, all three CLI modes and installed-wheel artifact/module coverage.
- Real Chrome: every request and candidate matches bundled data; changed-feature counts; copy success/denial; download name; 320/390/1280 layouts with expanded disclosures and 44px controls.

## Limits and attribution

A counterfactual explanation answers where a model changes, not how to intervene on a world. Real actionability can require causal relationships, discrete and joint constraints, feasible paths and distribution support. None are established here. Wine quality is ordinal but treated numerically by CART; tree means are not observed label values, calibrated uncertainty or guaranteed outcomes. There is no robustness-to-refit guarantee and no ensemble optimum. Prior test scores are historical descriptive results, not fresh validation of counterfactual quality.

Original code is MIT. Data and derived results are CC BY 4.0: Cortez, Cerdeira, Almeida, Matos and Reis (2009), [UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine%2Bquality), DOI 10.24432/C56S3T. Models and prior artifacts remain unchanged. No endorsement is implied.

Sources used for method and scope:

- Carreira-Perpinan and Hada, [Counterfactual Explanations for Oblique Decision Trees: Exact, Efficient Algorithms](https://ojs.aaai.org/index.php/AAAI/article/download/16851/16658), AAAI 2021. Leaf-wise constrained optimization; not reproduced oblique-tree code.
- Hada and Carreira-Perpinan, [Exploring counterfactual explanations for classification and regression trees](https://faculty.ucmerced.edu/mcarreira-perpinan/papers/ecml21-xkdd.pdf), 2021. Regression-band and feature-restricted queries.
- Carreira-Perpinan and Hada, [Very fast, approximate counterfactual explanations for decision forests](https://export.arxiv.org/pdf/2303.02883v1.pdf), 2023. Single-tree boxes and exponential forest-region growth; no claim to implement LIRE.
- Karimi, Scholkopf and Valera, [Algorithmic Recourse: from Counterfactual Explanations to Interventions](https://ar5iv.labs.arxiv.org/html/2002.06278), 2020. Why nearest counterfactuals are not causal recommendations.
- [Python math documentation](https://docs.python.org/3/library/math.html). `nextafter` and representable float boundaries.
- [treecf implementation explanation](https://wlazlod.github.io/treecf/how-it-works/). Independent engineering reference for witness re-scoring and explicit constraints. This project does not use treecf code, its genetic search or its plausibility machinery.
