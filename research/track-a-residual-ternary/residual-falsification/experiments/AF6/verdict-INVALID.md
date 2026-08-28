# Verdict — EXP-AF-006 — INVALID (verification-gate miscalibration)

**Date:** 2026-08-28
**Status:** INVALID — no experiment cells were run. Record retained
per OPERATING-PLAN §6 (corrections get new IDs; the old record is
retained). Corrected experiment: **EXP-AF-006b**.

## What happened

The preregistered measurement-sanity gate fired at the verification
pass, before any cell ran, exactly as designed:

- FP16 base `corpus_ppl(wikitext-103 test)` = **22.155**, outside the
  preregistered sanity band [12, 15].
- Companion measurements (same instrument): FP16 owt_test 14.016,
  damaged wt_test 75.13, damaged owt_test 93.01. (OWT test is the
  easier distribution for this model — consistent with OLMo
  pretraining on web text.)

## Diagnosis (root cause)

The [12, 15] band was calibrated at PROPOSE against the program's
lm-eval ladder value (wikitext ppl 13.09) on the assumption that
lm-eval's `wikitext` task measures wikitext-103 at token level.
**That assumption is false.** Verified 2026-08-28 against the
installed lm-eval 0.4.3 task config:

- lm-eval `wikitext` = `EleutherAI/wikitext_document_level`,
  **wikitext-2-raw-v1**, per-document **word_perplexity**.

So the ladder instrument and the new corpus_ppl instrument differ in
BOTH dataset (wikitext-2 test vs wikitext-103 test) and normalization
(per-document word-level vs concatenated-stream token-level, 64
windows × 256 tokens). The 22.155-vs-13.09 gap and the
damage-ratio divergence (corpus_ppl 75.13/22.155 = 3.39× vs lm-eval
425.76/13.09 = 32.5×) are instrument differences, not an eval-path
bug: the corpus_ppl path is pinned by unit tests (perfect model =
1.0, uniform model = vocab, deterministic window contract) and its
FP16-vs-damaged ordering is correct on both corpora.

## Findings retained

1. **Measurement literacy (program-wide):** every "wikitext ppl"
   number in the registry since EXP-A-001 is wikitext-2-raw-v1
   document-level word perplexity. Training corpora are
   wikitext-103 train. All cross-cell comparisons in the program
   remain valid (frozen instrument); the label in future reports
   should read "wikitext-2 ppl (lm-eval document-level)".
2. The corpus_ppl instrument is sound for **within-instrument**
   comparisons (FP16 vs damaged vs trained on identical windows) —
   which is all AF6's Q2(b) recovery ratio needs.
3. The sanity gate did its job: a miscalibrated acceptance band was
   caught before any claim-bearing data existed.

## Disposition

EXP-AF-006 is INVALID (design defect in the verification band, not a
run defect). EXP-AF-006b is preregistered with a corrected
verification design: the corpus_ppl sanity band is anchored to the
instrument's own measured scale (22.155 ± 10%) and a
within-instrument damage anchor (damaged/FP16 ratio > 1.5), both
frozen in the AF6b manifest before any AF6b cell runs. All other
design elements (cells, thresholds, bars, interpretation rule) are
unchanged.
