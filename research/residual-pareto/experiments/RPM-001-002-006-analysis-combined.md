# RPM-001 / RPM-002 / RPM-006 Analysis (Stage 1 + Stage 1.5 combined)

Generated from per-(regime, seed, arm) eval.summary.json.

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
| EXP-RPM-D0p | wikitext | 13.0983 | dense_adapter | 13.0992 | +0.0009 (better) |
| EXP-RPM-D0p | arc_easy | 0.5671 | int8_residual | 0.5776 | -0.0105 (worse) |
| EXP-RPM-D0p | lambada_openai | 0.6100 | dense_adapter | 0.6103 | -0.0003 (worse) |
| EXP-RPM-D1p | wikitext | 17.8517 | int8_residual | 19.2975 | +1.4458 (better) |
| EXP-RPM-D1p | arc_easy | 0.5689 | dense_adapter | 0.5988 | -0.0299 (worse) |
| EXP-RPM-D1p | lambada_openai | 0.5336 | lora | 0.5542 | -0.0206 (worse) |
| EXP-RPM-D2p | wikitext | 17.0749 | int8_residual | 18.7648 | +1.6899 (better) |
| EXP-RPM-D2p | arc_easy | 0.5854 | lora | 0.5978 | -0.0123 (worse) |
| EXP-RPM-D2p | lambada_openai | 0.5423 | lora | 0.5581 | -0.0158 (worse) |
| EXP-RPM-D3p | wikitext | 17.0120 | int8_residual | 17.9701 | +0.9581 (better) |
| EXP-RPM-D3p | arc_easy | 0.6066 | dense_adapter | 0.5939 | +0.0128 (better) |
| EXP-RPM-D3p | lambada_openai | 0.5499 | dense_adapter | 0.5639 | -0.0140 (worse) |
| EXP-RPM-D4p | wikitext | 18.4269 | int8_residual | 17.9936 | -0.4333 (worse) |
| EXP-RPM-D4p | arc_easy | 0.5996 | dense_adapter | 0.6139 | -0.0143 (worse) |
| EXP-RPM-D4p | lambada_openai | 0.5495 | dense_adapter | 0.5644 | -0.0149 (worse) |
| EXP-RPM-D5p | wikitext | 21.1205 | int8_residual | 19.3157 | -1.8047 (worse) |
| EXP-RPM-D5p | arc_easy | 0.5930 | dense_adapter | 0.6059 | -0.0129 (worse) |
| EXP-RPM-D5p | lambada_openai | 0.5478 | dense_adapter | 0.5669 | -0.0191 (worse) |

## RPM-002 cross-regime trained-vs-random separation

| Task | mean gap | min | max | n_regimes | signs |
|---|---|---|---|---|---|
| wikitext | -607.6271 | -1526.0074 | +0.0053 | 12 | +-----+----- |
| arc_easy | +0.0873 | -0.0053 | +0.1221 | 12 | -+++++-+++++ |
| lambada_openai | +0.2558 | -0.0011 | +0.3340 | 12 | 0+++++-+++++ |

## RPM-006 per-regime trained-vs-random z-score

| Regime | wikitext | arc_easy | lambada_openai |
|---|---|---|---|
| EXP-RPM-D0 | +1.97σ | -5.43σ | +0.00σ |
| EXP-RPM-D1 | -1093.82σ | +116.00σ | +163.98σ |
| EXP-RPM-D2 | -1682.51σ | +58.79σ | +252.83σ |
| EXP-RPM-D3 | -580.20σ | +65.81σ | +168.81σ |
| EXP-RPM-D4 | -247.81σ | +21.50σ | +78.49σ |
| EXP-RPM-D5 | -1079.47σ | +64.01σ | +236.98σ |
| EXP-RPM-D0p | +2.55σ | -0.38σ | -3.40σ |
| EXP-RPM-D1p | -465.20σ | +18.81σ | +155.11σ |
| EXP-RPM-D2p | -379.62σ | +47.68σ | +78.61σ |
| EXP-RPM-D3p | -2449.75σ | +39.96σ | +261.81σ |
| EXP-RPM-D4p | -355.80σ | +19.68σ | +113.38σ |
| EXP-RPM-D5p | -588.79σ | +27.87σ | +62.12σ |
