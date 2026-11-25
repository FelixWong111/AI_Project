# P06 supplier references and provisional size allowances

## Current supplier handoff (2026-11-24)

The owner has now provided `data/p06_part_specs_handoff/README_for_Codex.md`,
`p06_supplier_specs.json` and five original product images. Image text agrees
with JSON and all five image SHA-256 values match. The active independent
reference configuration is `config/p06_part_specs_candidate.json`.

| ID | Supplier product | Candidate measured reference (mm) | Owner-proposed allowance (mm) |
|---|---|---|---|
| 1 | M4×25 large flat/truss-style cross-recess screw | Under-head length 25; thread nominal diameter 4 | ±1.0; ±0.3 |
| 2 | M5×10 same supplier head family | Under-head length 10; thread nominal diameter 5 | ±1.0; ±0.3 |
| 3 | M4×10 same supplier head family | Under-head length 10; thread nominal diameter 4 | ±1.0; ±0.3 |
| 4 | M4 hex nut | Across flats 7 | ±0.5 |
| 5 | 5×15×1.2 flat washer | Outer diameter 15; inner diameter **5** | ±0.75; ±0.5 |

Screw head diameters are supplier references 9.2/11.4/9.2 mm, head thicknesses
2.4/2.9/2.4 mm. Nut thickness is 3.1 mm; washer thickness 1.2 mm.
Thicknesses are not automatically measured from overhead images. Head diameters
are descriptive only and do not add acceptance gates. Packaging counts are not
inspection expected counts; material 304 is a supplier claim.

The head family is now identified by the supplier's description, **not certified
as ISO 7045, DIN 7985 or any other unlisted standard**. ISO/DIN numbers remain
null. The intended washer is 5×15×1.2; do not substitute the generic ISO-table
hole 5.3 mm. These supplier dimensions take precedence over the historical
generic alternatives below for this project catalogue.

Allowances come from the owner's explicit proposal, not from the supplier's
manufacturing specification. Instructor confirmation and actual caliper error
validation remain pending. The independent config disables formal decisions;
frozen YOLO configuration and measurement algorithms are unchanged. Comparison
to supplier values is a descriptive difference, not measured physical error.

## Historical ISO/DIN background — not the active product configuration

The following section records the earlier catalogue-only investigation, before
the supplier handoff. Its outstanding identification statements describe that
earlier state. Product descriptions/dimensions above supersede them; exact
standard identity, physical specimen correspondence and caliper validation are
still unresolved.

Checked 2026-11-24. These are documentary references, **not product acceptance limits**.
The course catalogue and `team_plan_ai.md` establish M4×25, M5×10, M4×10,
M4 hex nut and M5 flat washer. Neither identifies the purchased standard, edition,
head style, washer series, nut style, thread tolerance class, material or coating.
The owner confirmed that procurement/packaging evidence is currently unavailable.
No photo or measured pixel size has been used to declare compliance with a standard.

## Project catalogue and outstanding identification

| ID | Project specification | Established | Still required |
|---|---|---|---|
| 1 | long_screw, M4×25 | Nominal thread size and stated length | Actual head/drive, standard/edition, whether 25 is under-head length, pitch and tolerance class |
| 2 | thick_screw, M5×10 | Nominal thread size and stated length | Same identification as ID 1 |
| 3 | thin_screw, M4×10 | Nominal thread size and stated length | Same identification as ID 1 |
| 4 | hex_nut, M4 | Mating nominal thread designation | Regular/thin/flanged/locking style, standard/edition; actual across-flats and thickness |
| 5 | flat_washer, M5 | Intended nominal screw size | Small/normal/large series, standard/edition, actual hole/outer diameter and thickness |

ID 0 has no single nominal geometry. Geometry selected using a known GT class
does not validate unknown-part rejection or certify that a specimen belongs to that class.

## Conditional dimensional references (mm)

| Family / reference | Relevant nominal size | Documentary dimension | Status for our specimens |
|---|---|---|---|
| ISO 7045 pan head, H/Z cross recess; historical DIN 7985 comparison requires edition checks | M4 | Manufacturer table: head diameter **maximum** 8.0, head height maximum 3.1; coarse pitch 0.7 | Candidate family only; not an identified purchased head |
| Same family | M5 | Head diameter maximum 9.5, head height maximum 3.7; coarse pitch 0.8 | Candidate family only |
| DIN 934 regular hex nut; manufacturer marks approximate ISO 4032 relationship | M4 | Across flats `s=7`; manufacturer's DIN table height maximum 3.2 | Candidate only; thread designation alone does not establish nut style |
| ISO 7089 / historical DIN 125 A, normal-series plain washer | M5 | Hole `d1=5.3`, outer diameter `d2=10` | Candidate series, not a project default |
| ISO 7092 / historical DIN 433, small series | M5 | Hole 5.3, outer diameter 9 | Candidate series |
| Historical DIN 9021 / approximate ISO 7093, large series | M5 | Hole 5.3, outer diameter 15 | Candidate series; appearance/rough size cannot confirm identity |

The symbols M4/M5 denote thread nominal sizes, not nut across-flats or washer outer
diameter. Nominal screw diameters 4/5 mm are not guaranteed measured crest diameters.
The stated lengths 25/10/10 mm remain course catalogue references, not caliper GT.
Do not scale or relabel specimens to make measured results match this table.

ISO 4032:2023 is titled hexagon regular nuts (style 1). Its principal scope starts
at M5; smaller sizes are referred to Annex A. Therefore the M4 DIN manufacturer
row above is not represented as a freshly verified ISO 4032:2023 dimensional row.
Legacy DIN/ISO comparisons marked `~` are not declarations of full equivalence.
Other heads, nut styles and washer series remain possible until procurement is confirmed.

## Source register

All numeric rows are sourced from manufacturers' own catalogues, including
manufacturer-authored sheets hosted by distributors. ISO catalogue pages confirm
scope and edition; the full paid ISO dimensional standards were not obtained.

1. [ISO 7045:2011 official scope](https://www.iso.org/standard/57372.html): pan head screws with H/Z cross recess.
2. [BSN Fastener ISO 7045 Type H dimensional sheet](https://www.bsnfastener.com/pdf/Section01/catalog_P1-4_A4-50_HE7045.pdf), one-page manufacturer table. The maximum head dimensions above must not be relabelled nominal or universal head dimensions.
3. [ISO 4032:2023 official scope](https://www.iso.org/standard/75016.html).
4. [Bossard BN 629 DIN 934 sheet](https://media.distrelec.com/Web/Downloads/_t/ds/1241613_eng_tds.pdf), M4 row, dated 2020-04-02: distinguishes `s`, `e min` and `m max`. Across-corners `e` is not the across-flats measurement.
5. [Bossard BN 602 DIN 125 A / ~ISO 7089](https://www.tme.eu/Document/515331066d6ede6b62c48015b5dff1fd/BN_602_1_Bossard_1.pdf), M5 row, dated 2020-09-12.
6. [Lederer DIN 433 / ISO 7092 catalogue](https://www.lederer-online.com/fileadmin/users/redakteur/redakteur_upload/ISO7092.pdf), PDF page 1, M5 row.
7. [Bossard BN 729 DIN 9021 / ~ISO 7093](https://www.tme.eu/Document/975441008a0df4b6895daf3ab99decfc/B5_BN729.pdf), M5 article 1762273. This source documents a DIN product; its thickness is not silently transferred to a current ISO series.

## Confirmation and small caliper check

Provide a purchase link/BOM/packaging designation and identify which physical
specimen appears in each reference photo. Record at least repeated measurements
of screw under-head length and thread crest diameter, washer ID/OD, and nut
three opposing flat pairs; include instrument resolution and specimen ID.
Nut thickness and washer thickness help identify series but are not measured
from the overhead image by this module. Head photos help distinguish head styles.

Catalogue identification and a few caliper checks complement each other:
the former establishes the intended specification; the latter estimates actual
image-measurement bias. Neither sets the course's permitted product tolerance.
Instructor confirmation of the catalogue, imaging conditions and tolerances is
still needed before formal size acceptance/rejection evaluation.
