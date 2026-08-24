# RPM-001 / RPM-002 / RPM-006 Analysis (Stage 1 post-hoc)

Generated from runs/r/EXP-RPM-D{0..5}/<ts>/af2d/seed-{1,2,3}/{arm}/eval.summary.json after post-hoc eval of random_t2_ternary + random_lora arms.

## RPM-001 per-regime T2 vs next-best trained arm

| Regime | Task | T2 value | Next-best arm | Next-best value | T2 - Next |
|---|---|---|---|---|---|
| EXP-RPM-D0 | wikitext | 13.0990 | lora | 13.1034 | +0.0044 (better) |
| EXP-RPM-D0 | arc_easy | 0.5633 | int8_residual | 0.5916 | -0.0283 (worse) |
| EXP-RPM-D0 | lambada_openai | 0.6111 | dense_adapter | 0.6096 | +0.0015 (better) |
| EXP-RPM-D1 | wikitext | 24.1368 | int8_residual | 17.7464 | -6.3903 (worse) |
| EXP-RPM-D1 | arc_easy | 0.5916 | dense_adapter | 0.6256 | -0.0340 (worse) |
| EXP-RPM-D1 | lambada_openai | 0.5629 | lora | 0.5700 | -0.0071 (worse) |
| EXP-RPM-D2 | wikitext | 23.6611 | int8_residual | 18.2832 | -5.3779 (worse) |
| EXP-RPM-D2 | arc_easy | 0.5997 | dense_adapter | 0.6277 | -0.0279 (worse) |
| EXP-RPM-D2 | lambada_openai | 0.5655 | dense_adapter | 0.5735 | -0.0080 (worse) |
| EXP-RPM-D3 | wikitext | 26.8306 | int8_residual | 18.9892 | -7.8414 (worse) |
| EXP-RPM-D3 | arc_easy | 0.5930 | dense_adapter | 0.6218 | -0.0288 (worse) |
| EXP-RPM-D3 | lambada_openai | 0.5604 | dense_adapter | 0.5698 | -0.0094 (worse) |
| EXP-RPM-D4 | wikitext | 26.9107 | int8_residual | 18.5722 | -8.3385 (worse) |
| EXP-RPM-D4 | arc_easy | 0.5826 | lora | 0.6229 | -0.0403 (worse) |
| EXP-RPM-D4 | lambada_openai | 0.5522 | lora | 0.5628 | -0.0107 (worse) |
| EXP-RPM-D5 | wikitext | 17.3248 | int8_residual | 18.7532 | +1.4284 (better) |
| EXP-RPM-D5 | arc_easy | 0.6094 | lora | 0.6153 | -0.0059 (worse) |
| EXP-RPM-D5 | lambada_openai | 0.5498 | dense_adapter | 0.5624 | -0.0126 (worse) |

## RPM-002 cross-regime trained-vs-random separation

| Task | mean gap | min | max | n_regimes | signs |
|---|---|---|---|---|---|
| wikitext | -942.8010 | -1526.0074 | +0.0053 | 6 | +----- |
| arc_easy | +0.0937 | -0.0053 | +0.1221 | 6 | -+++++ |
| lambada_openai | +0.2689 | +0.0000 | +0.3340 | 6 | -+++++ |

## RPM-006 per-regime trained-vs-random z-score

| Regime | wikitext | arc_easy | lambada_openai |
|---|---|---|---|
| EXP-RPM-D0 | +1.97σ | -5.43σ | +0.00σ |
| EXP-RPM-D1 | -1093.82σ | +116.00σ | +163.98σ |
| EXP-RPM-D2 | -1682.51σ | +58.79σ | +252.83σ |
| EXP-RPM-D3 | -580.20σ | +65.81σ | +168.81σ |
| EXP-RPM-D4 | -247.81σ | +21.50σ | +78.49σ |
| EXP-RPM-D5 | -1079.47σ | +64.01σ | +236.98σ |
