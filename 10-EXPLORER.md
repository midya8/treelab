# The frozen response explorer

TreeLab v0.4 adds a dependency-free static browser explorer for v0.3's exact
stored response report. Choose any of the three original test rows, either
unchanged model and any of eleven varied features. Inspect the complete
piecewise-constant curve, each exact interval, every member output/leaf ID and
full input witnesses for attainable extrema.

This does not add a new model or arbitrary-input inference. It reads the frozen
66 slices and 1,028 leaf segments. No fit, API calls, analytics, accounts, remote
model service, conditional distribution, new accuracy claim or data uploads.
The initial view is row 1 / 32-tree ensemble / alcohol, not a best-case search.

## Using the page

1. Choose row, model and varied feature. Other row inputs stay fixed. The
   fitting-only axis domain, original row prediction and member/segment counts
   are shown. Changing selection resets to the segment containing the stored
   row's original axis value, or the first segment if outside the domain.
2. Use the segment slider, Previous/Next, or an interval in the complete list.
   The selected segment is highlighted on the full curve. The slider supports
   keyboard arrows. All controls have text labels and visible focus rings.
3. Inspect attainable minimum/maximum to select that exact common-axis segment.
   Open the witness disclosure for all input coordinates, exact axis value,
   predicted output, model, row, domain and interpretation warning.
4. Copy witness JSON or download it as a local file. Clipboard denial does not
   claim success: it shows an explicit fallback to selecting or downloading the
   JSON. No witness is transmitted to a server.
5. Expand the member list to see all 32 tree predictions/leaf IDs, or the single
   selected CART output. The complete interval list gives unrounded endpoints
   and outputs. Rounded chart ticks are for orientation only.

Mobile charts reflow rather than shrinking 960px text to unreadable labels.
320px, 390px and 1280px layouts have no page-level horizontal overflow. All
selects, buttons, slider and disclosure controls support keyboard use; primary
controls and disclosure rows are at least 44px tall. No animations are required.
This is tested behavior, not a complete accessibility certification or a claim
of testing every assistive technology.

## Reproduce and verify

```bash
python examples/export_explorer.py
python -m unittest discover -s tests -v
node --test tests/test_explorer.cjs tests/test_explorer_browser.cjs
python -m http.server 8000 --directory docs
# Then open the local server in a browser.
```

The Node tests require Node 22+ and Chrome. Set `CHROME_BIN` if Chrome is not
`google-chrome`. No npm dependencies. Tests use Chrome DevTools only in the
local test process; the published page has no debugger or external requests.

`examples/export_explorer.py` copies the full frozen report to `docs/data.js`
with its source path and SHA-256 of the original report bytes. It does not
resample, round, drop member outputs, compute new predictions or fit anything.
The wrapper's checksum is shown on the page. A hash is provenance, not a
signature. Existing data/models/metrics/reports and all three audit figures are
unchanged. The Python package retains seven modules and thirteen CLI modes.

Two export/source tests, two Node geometry tests and a real-Chrome interaction
test cover the explorer. All 66 UI selections and 1,028 segment witnesses,
member lists and displayed extrema match the frozen report. Geometry tests
check exact inclusive/exclusive boundaries and all witnesses. Browser tests
exercise keyboard arrows, previous/next, min/max, clipboard success/denial,
local download names, expanded disclosures and 320/390/1280px overflow/touch
sizes. Publication adds an actual live-site smoke check and byte comparison.
CI has Python 3.10-3.13, installed wheel, plus Node 22 / real stable Chrome.

## Interpretation and credit

The method and all limitations remain in [SWEEPS.md](SWEEPS.md). This is
ICE-style model-response geometry. Holding one correlated feature apart from
its complement can create implausible combinations. The displayed extrema are
not predictive uncertainty, causal effects, plausible-wine guarantees or
joint-feature robustness bounds. Separate member extrema need not be attainable
at one shared input. No test-set accuracy is recomputed here.

Established CART, bagging and individual conditional expectation are not
claimed inventions. Goldstein, Kapelner, Bleich and Pitkin (2015) developed ICE
plots. Data and derived outputs: Cortez et al. (2009), UCI Wine Quality,
DOI 10.24432/C56S3T, CC BY 4.0. Code MIT. No endorsement implied.
