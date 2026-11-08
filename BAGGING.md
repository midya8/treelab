# Group-bootstrap bagging, with exclusions you can audit

TreeLab v0.2 adds established bootstrap aggregation of regression trees. It
samples exact-feature groups, not independent rows, and averages tree outputs.
Every draw, complete fitted tree and member prediction is saved. No feature
subsampling, random-forest claim, pruning of bootstrap members, paid API or
scikit-learn estimator.

## Fixed protocol

Keep v0.1's unmodified UCI white-wine data and target-blind exact-feature-group
split: 2,921 fitting rows in 2,376 groups; 1,008 validation rows; 969 test rows.
Grouping serializes the binary64 features only, never targets. Known identical
features stay together even if targets disagree. These groups are not known
producer, batch, patient or time-independent experimental units.

Before the first v0.2 result, fix seed **20261108**, 32 grown trees, minimum leaf
size **5**, depth limit **20**, and candidate ensemble prefixes **[1,4,16,32]**.
For each tree draw G group indices G times independently with replacement using
NumPy's default_rng. Expand every drawn group to all its original rows and
repeat those rows for each occurrence of its group. All feature columns remain
available at every greedy CART split. Minimum leaf size counts expanded rows,
including bootstrap multiplicity, not unique original observations.

This is a group bootstrap, not ordinary iid row bootstrap. Each group has equal
draw probability. Unequal group sizes give a variable number of expanded rows:
**2,856 to 2,951** in the saved run. Expected expanded count is N because every
group has expected multiplicity one. An entire group's per-tree absence
probability is `(1 - 1/G)^G`; a given row's chance of zero OOB votes after B
independent trees is `(1 - (1 - 1/G)^G)^B`. These are sampling probabilities,
not a proof of scientific independence or of unbiased error estimates.

Score only validation MSE to select the prefix. Exact ties prefer fewer trees.
Do not use OOB or test scores for selection, and do not refit on validation.
This one split/seed is an implementation demonstration, not a seed search.
The historical test set was already used in v0.1, so these version comparisons
are descriptive and are not a new blind benchmark.

## Group OOB, explicit support

A fitting row receives a member prediction only if that member drew its entire
feature group zero times. For each prefix save the vote count, average of
eligible tree outputs, target, supported/unsupported row count and row-weighted
MSE on supported rows only. A row with no votes has a **null** prediction.
Never treat zero votes as a numeric zero, invent support or score missing rows.
All fitting, validation and test member predictions are retained separately.

The saved model binds original fitting-feature and target arrays by SHA-256
of their JSON numeric arrays, plus the group partition and feature-group keys.
The OOB API rejects changed arrays, row orders or group bindings. This prevents
applying membership offsets to unrelated data, not malicious model forgery:
hashes are provenance checks, not authentication signatures.

| Prefix | Validation MSE | OOB supported / 2,921 | OOB MSE on supported rows |
|---:|---:|---:|---:|
| 1 | 0.884515 | 1,089 | 0.884641 |
| 4 | 0.607652 | 2,439 | 0.756681 |
| 16 | 0.519944 | 2,920 | 0.567121 |
| **32, selected** | **0.501465** | **2,921** | **0.531137** |

Each prefix's OOB estimate averages fewer, row-dependent trees and may score a
different supported subset. These are not fixed-set like-for-like error
curves. OOB does not establish calibrated predictive uncertainty or replace
properly designed external evaluation. No tree-spread confidence interval is
reported.

## A concrete duplicate witness, not a dataset-wide effect estimate

Use features `[0,0,1,1]`, targets `[0,0,10,10]`, and row-bootstrap indices
`[1,1,2,2]`. Row 0 is absent, but its identical feature copy at row 1 is in-bag.
A grown tree predicts 0 for row 0, a naive row-OOB squared error of zero.
Group exclusion refuses this prediction. This is a finite failure witness for
row-absence logic, not a causal estimate of optimism in the wine dataset.

## Preserve the controls

All controls fit the same original fitting rows. v0.1 artifacts are untouched.

| Model | Test RMSE / score units | Test MAE / score units |
|---|---:|---:|
| Fitting mean | 0.889429 | 0.673294 |
| Fitting median | 0.902353 | 0.644995 |
| Single grown CART | 0.882552 | 0.663626 |
| Validation-selected CART | 0.775952 | 0.607582 |
| Ordinary least squares | 0.734217 | 0.573791 |
| Validation-selected group bagging, 32 trees | **0.705212** | **0.545678** |

All 969 test predictions and 32 member outputs per row are saved. Ordinary
least squares still beats the pruned tree; bagging has the lowest error among
these saved controls on this split. This does not establish significance,
feature causality, an isolated mechanism, a general bagging advantage,
state-of-the-art wine prediction, future-distribution accuracy or clinical/
safety/economic meaning. Scores are ordinal but squared error treats them as
numeric. No near-copy, batch or time-forward independence guarantee.

## Tests and reproduction

The suite checks reproducible draws and prefix stability, actual group
multiplicity against direct tree refits, every real fitting/validation/test
member output by independent traversal, all OOB membership and averages,
zero-support single-group behavior, conflicting-target duplicate grouping,
a duplicate witness, strict model and CLI corruption checks, exact validation
ties, complete refits, test-target mutation, selection-before-test prediction
ordering and figure data. Installed-wheel tests exercise all ten CLI modes
outside the source checkout, including real-data bagging refit.

```bash
python -m treelab bag-demo > /tmp/saved-bagging-report.json
python -m treelab bag-audit --output /tmp/bagging-refit
python -m treelab bag-fit --input examples/train.json --output /tmp/ensemble.json --trees 4 --min-leaf 1
# bag-predict input contains only features, with the fitting feature width.
# bag-oob input contains the original fitting features and targets in original order.
python -m treelab bag-oob --model /tmp/ensemble.json --input examples/train.json
```

## Sources, credit and scope

- Breiman, [Bagging Predictors, technical report 421, September 1994](https://www.stat.berkeley.edu/~breiman/bagging.pdf): bootstrap replicas and averaging for regression.
- Breiman, [Out-of-Bag Estimation](https://www.stat.berkeley.edu/pub/users/breiman/OOBestimation.pdf): excluding the observed case from contributing bootstrap learners. Row-bootstrap claims are not automatically group-bootstrap guarantees.
- Penn State, [11.11 Bagging](https://online.stat.psu.edu/stat857/node/180/): replacement sampling and grown trees; its classification vote example is not our regression averaging rule.
- scikit-learn, [BaggingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html): averaging and sample/feature resampling distinctions, not a code or numerical dependency.
- scikit-learn, [cross-validation of grouped data](https://scikit-learn.org/stable/modules/cross_validation.html): whole-group exclusion and domain-specific dependence.
- Cross Validated, [OOB and dependent rows](https://stats.stackexchange.com/questions/207815/out-of-bag-error-makes-cv-unnecessary-in-random-forests): community caution about dependence and smaller OOB ensembles, corroborative rather than statistical authority.
- Cortez et al. (2009), [UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine%2Bquality), DOI 10.24432/C56S3T, CC BY 4.0 for bundled data and derived results. Code MIT. Established CART, pruning, least squares and bagging are not claimed inventions. No source endorsement.

The complete frozen ensemble is bundled as deterministic `bagging-model.json.gz`
(gzip of the full JSON, zero timestamp). CLI model loading accepts `.json.gz`
as well as plain JSON. This changes storage only, not trees, draws or predictions.
