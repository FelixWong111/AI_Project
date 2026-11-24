# P06 candidate geometric measurement

## Current screw candidate and line review

Open `output/p06_screw_review/index.html` in a desktop browser. Keep the adjacent
`crops/` directory with the page. All 24 crops come from different Train images,
eight per screw type. The first 18 support development and the last six are
reserved for later review. Only raw crops appear on the annotation page. Click
the shoulder centre and tip centre for length, then opposite crests in one
interior cross-section for diameter. Mark either dimension unclear when needed.
Zoom before clicking small features. Export JSON before clearing browser storage
or moving the page. The file preserves partial work and original display pixel
coordinates. Caliper measurements must be linked to physical specimen IDs.

The default screw method estimates a shoulder from the width profile, seeds
edges from a pooled interior profile and picks a section with two supported
gradients. The Stage 1 comparison explicitly selects `screw_method='stage1'`.
Washers and nuts use their existing methods.

```powershell
conda run -n ai-p06-yolo python -B tools/p06_screw_review.py --output output/p06_screw_review_NEW
conda run -n ai-p06-yolo python -B tools/p06_screw_review.py --compare-marks "$HOME/Downloads/p06_screw_marks.json"
```

The second command reads the default review folder unless supplied with
`--output`. It prints per-dimension comparisons, including pending and unclear
judgements. Diameter lines may be at different shaft positions. The signed
length difference uses algorithm minus human and the candidate scale.

## Exterior-edge and aperture checks

An opt-in screw check (`screw_method='edge_audit'`) tests two-sided edge contrast
and continuity across three regions of the shaft. Two exposed tracks verify the
under-head seating plane; three tracks seek the last supported transition into
background at the tip. The opt-in washer check (`radial_washer=True`) samples
72 rays from the outer ellipse centre, fits supported aperture points and checks
concentricity, angular support and agreement with any contour-based aperture.
Both variants can refuse a measurement. Train comparisons and ten side-by-side
overlays are in `output/p06_edge_audit_final/`. Yellow S denotes the shoulder,
pink T the tip, turquoise the shaft axis and blue the diameter. A failed audit
leaves the audit side of the overlay blank.

```powershell
conda run -n ai-p06-yolo python -B tools/p06_edge_audit.py --output output/p06_edge_audit_NEW
conda run -n ai-p06-yolo python -B tools/p06_screw_review.py --compare-marks "$HOME/Downloads/p06_screw_marks.json" --edge-results output/p06_edge_audit_final/records.json
```

## Previous Stage 1 revision: Train only

The Stage 1 screw method adds sharp per-section shaft edges, dominant-track
consistency and two-sided under-head edge checks. A failed under-head check may
retain a diameter **candidate** in `diagnostics.partial_dimensions`; aggregate
status remains UNMEASURABLE and the main `dimensions` stays empty. Such a line
is for manual review, not a usable complete screw specification.

Bright washers can have nested dark rims and a bright face. Centred odd-depth
descendant apertures are now considered, with nearly full observed angular
support and robust ellipse fits. Primary diameter lines show the **visible fitted
ellipse major axis**; minor axes are retained. This replaces the previous fitted
circle's horizontal diameter and is not a correction of perspective or a claim
of true physical diameter. Do not treat a new value closer to nominal as accuracy
improvement. Original contour residuals and fit-support fractions remain available.

The initial development selection is 13 Train images: the preceding 12-image
selection plus pass01.jpg. Full-Train image review then exposed an additional
head/shaft confusion case, fail113_006; this Train-only regression prompted a
half-diameter head-clearance guard. It is recorded separately, not claimed as
an untouched validation example. Use `tools/p06_geometry_stage1.py` for before/after
checks. The original module is preserved as a small source-only snapshot under
`output/p06_geometry_stage1/before/`; this experiment does not copy the dataset.
See `output/p06_geometry_stage1/report.md` and `stage1_frozen_code_final.json` for the
current record. This revision has **no new Validation result**. Earlier
`output/p06_geometry/evaluation_02` and supplier comparison reports retain their
historical code and numerical definitions. Existing frozen hash guards are kept;
new Validation checks require a newly recorded candidate version after review.

```powershell
conda run -n ai-p06-yolo python -B tools/p06_geometry_stage1.py --full-train --output output/p06_geometry_stage1/train_NEW
```

This diagnostic entry depends on the preserved baseline snapshot and selection
manifest. It accepts no Validation/Test input argument and refuses existing output.

This is an independent experiment. Frozen `p06_baseline3.json`, YOLO inference,
P07/P08 interfaces, original GT/images and final PASS/FAIL behaviour are unchanged.
Supplier product dimensions and owner-proposed allowances are now recorded in
`config/p06_part_specs_candidate.json`; exact ISO/DIN identity and caliper
validation remain unresolved. See [size_references.md](size_references.md).
`tools/p06_check_part_specs.py` checks source hashes and computes descriptive
differences from existing measurements. It never changes measurement parameters
or makes product decisions. The washer reference is 5×15×1.2 mm, not an assumed
5.3 mm ISO-series hole. Provisional allowances are separate from measurement
uncertainty and remain outside the frozen YOLO config.

## API and measurements

`src.p06_geometry.measure_object(crop_bgr, class_id, (sx, sy), parameters=None)`
accepts an isolated uint8 BGR object crop and separate positive x/y px-per-mm
scales. It does not infer class, search a standard-size table or load a model.
ID 0 returns `NON_CATALOGUE_NOT_MEASURED`; invalid inputs raise `ValueError`.

- Screw: fill the silhouette, estimate/refine the shaft axis, locate a broad
  head-to-shaft transition, then measure from that under-head plane to the visible
  tip. The current candidate uses paired normal gradients on interior sections.
  These are estimates from visible silhouettes, not a thread gauge. Shadows,
  oblique axes, chamfers and unconfirmed head types can bias both dimensions.
- Washer: use the binary contour hierarchy to require a centred odd-depth
  descendant aperture inside an annular outer boundary. Fit ellipses in
  scale-corrected coordinates; retain major/minor axes, residuals and support.
  Off-centre metal highlights are excluded from aperture candidates.
- Nut: require six visible flat sides; fit straight sides and measure perpendicular
  separations of three opposing pairs. Report the median pair and all three
  estimates, never a bounding-box side or across-corners diameter.

Results contain `dimensions` (measurement lines, original-crop pixel lengths,
approximate mm), `contours_xy`, diagnostics, status and explicit failure reason.
`MEASURED_CANDIDATE` means the algorithm produced a geometry estimate; it does
not mean size compliance or physical accuracy. Every result has `decision=null`.
Geometric quality checks (circle residual, side parallelism, separation consistency)
are algorithm validity gates, **not product tolerances**.

## Scale and coordinates

The ruler photos use their own x/y scales from
`output/p06_measure_candidate/scale_fits.csv`. Development images use the
owner-approved candidate mean of photos 1/4/5/6: x=34.4690736763,
y=34.0255490097 px/mm, conditional on equal capture geometry, no unknown
resizing/cropping, and approximate coplanarity. No lens-distortion or height
compensation is applied; these limitations remain in every experiment report.
The 1280×1707 development image is refused for mm measurement; resizing history
has not been confirmed. No photo scale is automatically applied to a live camera.

The API uses crop-local coordinates. The runner adds the crop origin to produce
full EXIF-corrected original-image coordinates. Saved contour diagnostics are
simplified at 0.75 px; this does not change the points used in computation.
Washer lines represent visible fitted-ellipse major diameters. Nut and screw lines show the
specific perpendicular/axial measurement construction.

## Reproducibility and separation

```powershell
conda run -n ai-p06-yolo python -B tools/p06_measure_geometry.py tune --output output/p06_geometry/train_tuning_NEW
conda run -n ai-p06-yolo python -B tools/p06_measure_geometry.py evaluate --frozen output/p06_geometry/train_tuning_NEW/frozen_geometry.json --output output/p06_geometry/evaluation_NEW
conda run -n ai-p06-yolo python -B -m unittest discover -s tests -p "test_p06_geometry.py" -v
```

The training-only selection is deterministic: 12 full-resolution Train images
containing IDs 1–5, ordered by SHA-256(image_id). Three background offsets are
compared for geometric availability, without optimising against nominal sizes.
All other algorithm changes are checked on synthetic/Train cases first. The
final module/runner/scale/table hashes are frozen before processing Validation.
Evaluation refuses changed code or tables; output directories cannot be reused.
The runner has no Test source argument and uses only the fixed development root.

The evaluation uses **human GT boxes and class IDs** to isolate geometry. It
does not demonstrate model localisation, classification, open-set rejection or
end-to-end size decisions. ID 0 objects and truly empty scenes are excluded from
target-dimension denominators. Do not call extraction coverage dimensional
accuracy, and do not use class-wise size distributions as caliper ground truth.

Outputs include Train/Validation summaries, all target-object records, pixel/mm
lines, failure reasons, compact preview contact sheets, source SHA-256 records
and the frozen experimental parameters. Standards and measured dimensions are
kept separate. Final physical error validation requires caliper GT for identified
specimens; repeatability requires repeated captures/measurements, not just many
different photographs.

## Rebuilding the Chinese Stage 1 report

Run the UTF-8 source file directly; do not pipe Chinese Python source through
Windows PowerShell 5.1. Its default `$OutputEncoding` can be US-ASCII, replacing
Chinese before Python receives it. Changing only the final file encoding cannot
recover that loss.

```powershell
conda run -n ai-p06-yolo python -B tools/p06_report_geometry_stage1.py --replace
```

The generator checks `train_full_final/records.json`, `summary.json` and
`comparison.csv` against each other, then writes UTF-8 atomically. It never uses
the damaged Markdown as input and does not rerun geometry or open source images.
Regression tests include an ASCII console with Python UTF-8 mode disabled,
Chinese JSON/CSV identifiers, corruption replacement, and mismatched evidence.
