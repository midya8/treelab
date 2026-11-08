# One-feature response geometry, with attainable witnesses

TreeLab v0.3 adds ICE-style one-feature response slices for the frozen selected
CART model and group-bagging ensemble. The concept is established individual
conditional expectation analysis. The implementation enumerates the stored
binary64 tree partitions instead of approximating the response on a grid.

These are model-output functions, not an estimated conditional outcome
population, causal interventions, uncertainty intervals or realistic-wine
recommendations. Correlated features can create impossible combinations when
one input varies while all others remain fixed.

## What is partitioned

Take a stored tree/ensemble, one complete row, a feature index, and an explicit
closed finite domain `[low, high]`. Keep all other inputs fixed. Every tree
split routes left when `value <= threshold`, right otherwise.

Walk each member tree. At non-axis splits follow only the branch taken by the
fixed row. At axis splits intersect the leaf's axis domain: left adds an
inclusive upper bound; right adds an exclusive lower bound. This produces
reachable leaf intervals `(lower, upper]`. Leaves incompatible with the fixed
complement are not part of the response.

Intersect their boundary union with the requested domain. The first segment's
lower endpoint is inclusive, all later lower endpoints are exclusive, and all
upper endpoints are inclusive. If a threshold equals the requested low, save
the isolated singleton at low before moving to its right. For an open lower
bound use `math.nextafter(lower, +infinity)` as the witness. Do not use an
arbitrary epsilon or a midpoint that can round onto the wrong branch. Omit
intervals that contain no representable binary64 value.

For each segment save both endpoints, inclusion flags, an actual binary64
witness, every member's prediction and leaf ID, and the ensemble arithmetic
mean in stored member order. A model can cross a leaf boundary without
changing its output; the leaf partition is retained rather than coalesced.
The SVG joins adjacent steps as visual guides; exact inclusivity is in JSON.

## Attainable extrema, not separate member bounds

Choose min/max over this common segment partition, with the first segment
winning exact prediction ties. Return the axis witness and complete input row
that attains each reported value. All non-axis inputs remain unchanged.

Independent member extrema cannot generally be added to obtain an attainable
ensemble extremum. Example: two opposite stumps, one predicting 0 then 10 and
the other 10 then 0. Both individually attain a maximum of 10, but their mean is
5 everywhere on the common domain. The test suite includes this failure
control. These are one-axis extrema, not a joint-feature robustness bound.

"Exact" here refers to the finite stored binary64 routing partition and its
representable input domain. It does not mean rational arithmetic, an all-input
numerical proof, a globally optimal model or a statistical guarantee. The
implementation does not cast inputs to float32. Frozen trees and all original
metrics remain unchanged, and no fitting, tuning or test scoring is performed.

## Fixed real-data demonstration

Use the first three original test rows, CSV indices **1, 8, 11** (zero based),
all **11** features and both unchanged models. Axis domains are feature-wise
min/max of original fitting rows only. No response-driven row/feature choice,
feature ranking or new accuracy claim. This produces **66 slices and 1,028
segments**. Rows 1 and 8 are exact copies and yield the same curves, retained
rather than filtered for a more varied-looking demonstration.

The overview plots the first row's full alcohol response for both models, with
all **3 CART segments** and **50 ensemble common segments**, not a resampled
smooth curve. Its table includes all eleven first-row feature slices. Full
results for the other two rows are in `treelab/artifacts/sweep-report.json`.

For row 1, the alcohol slice gives selected-CART outputs **5.31579 to 6.36283**
and bagging outputs **5.09926 to 6.54844**. These are attainable model outputs
on the specified fitting-range slice. They are not predictions with confidence
limits, causal gains from changing alcohol, advice for winemaking or evidence
that every input combination exists in the source population. Numeric treatment
of ordinal quality scores remains a modeling choice.

## APIs and CLI

```bash
python -m treelab sweep-demo > /tmp/saved-sweeps.json
python -m treelab sweep-audit --output /tmp/sweep-audit
python -m treelab sweep --model treelab/artifacts/selected-model.json --input examples/sweep.json
python -m treelab sweep --model treelab/artifacts/bagging-model.json.gz --input examples/sweep.json
# Optionally restrict a saved ensemble to its first four trees:
python -m treelab sweep --model treelab/artifacts/bagging-model.json.gz --input examples/sweep.json --trees 4
```

Input has exactly `row`, `feature`, `low`, `high`; the row must match the model
width. Domains are ordered, finite numeric values in +/-1e100, and may be
singletons. Feature and prefix must be genuine integers, not booleans. The
base row's axis value need not be in the requested domain: the result states
whether it is. JSON rejects duplicate keys and nonfinite constants. Prefix
restriction applies only to ensembles. Both plain JSON and gzip models work.

Python entry point: `treelab.sweep.response(model, row, feature, low, high,
prefix=None)`. No missing/categorical inputs or simultaneous-feature sweep.

## Independent checks

Tests enumerate full multidimensional leaf boxes separately and intersect
fixed complements, then compare reachable leaves. Direct traversal probes
every real-data segment at its witness, inclusive upper, nextafter lower,
near-upper and interior point, and independently checks member leaf IDs and
all complete extrema inputs. Random small trees/ensembles, prefix restriction,
constant trees, unused axes, repeated-axis nested splits, singleton domains,
left-endpoint thresholds, adjacent representable floats and signed zero are
covered. The opposite-stump example rejects sum-of-member-extrema reasoning.

Full report reproduction runs with fitting disabled and checks all fitting-only
domains. Corrupt models, invalid widths, booleans, nonfinite/reversed domains,
duplicate JSON and unsupported prefix use are rejected. The generated SVG
retains all displayed slice data in metadata and tests exact curve point
counts, summary extrema, warnings and regeneration. The installed wheel
exercises all **seven modules and thirteen CLI modes** outside the source tree.

## Attribution and sources

- Goldstein, Kapelner, Bleich and Pitkin (2015), [Peeking Inside the Black Box](https://arxiv.org/abs/1309.6392), primary ICE research. [Author/citation page](https://kapelner.github.io/ICEbox/authors.html). Individual response curves are not a newly invented explanation method.
- scikit-learn, [PDP and ICE documentation](https://scikit-learn.org/stable/modules/partial_dependence.html): fixed individual complements versus averaged partial dependence, and correlated-feature caveats. Not a code dependency.
- Molnar, [Interpretable Machine Learning, ICE chapter](https://christophm.github.io/interpretable-ml-book/ice.html): one curve per instance and limitations under feature dependence.
- Dassen, Hou and Kronseder, [LMU limitations chapter](https://slds-lmu.github.io/iml_methods_limitations/pdp.html): unrealistic combinations and heterogeneity hidden by averages.
- scikit-learn, [Decision Trees](https://scikit-learn.org/stable/modules/tree.html): piecewise-constant model structure and extrapolation limits.
- [Regression trees](https://mlweb.loria.fr/book/en/regressiontree.html): rectangular leaf partitions and feature-threshold routing.
- Cortez et al. (2009), [UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine%2Bquality), DOI 10.24432/C56S3T, CC BY 4.0 for data and derived outputs. Code MIT. No endorsement or causal claim.
